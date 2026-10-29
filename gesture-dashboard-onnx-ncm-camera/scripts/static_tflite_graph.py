"""Small, auditable ONNX-MLP to TFLite builtin graph writer.

The supplied graph has only three dense layers. Writing its FlatBuffer directly
avoids the TensorFlow compiler's multi-GB installation on memory-limited hosts.
The FP32 graph keeps all source arithmetic. The quantized candidate uses INT8
dense activations/weights, with FP32 normalization and the open-set output tail.
It is deliberately described as mixed precision, not a fully integer graph.
"""
from __future__ import annotations

import flatbuffers
import numpy as np
import tflite as schema


class Graph:
    def __init__(self):
        self.tensors = []
        self.operators = []

    def tensor(self, name, shape, value=None, dtype=np.float32, quant=None):
        index = len(self.tensors)
        self.tensors.append((name, shape, None if value is None else np.asarray(value, dtype=dtype).tobytes(), np.dtype(dtype), quant))
        return index

    def op(self, code, inputs, output, option, attributes=None, version=1):
        self.operators.append((code, inputs, [output], option, attributes or {}, version))
        return output

    def finish(self, input_index, outputs):
        b = flatbuffers.Builder(100_000)

        def offsets(values):
            b.StartVector(4, len(values), 4)
            for value in reversed(values):
                b.PrependUOffsetTRelative(value)
            return b.EndVector()

        def integers(values):
            return b.CreateNumpyVector(np.asarray(values, dtype=np.int32))

        buffers = []
        schema.BufferStart(b)
        buffers.append(schema.BufferEnd(b))
        tensors = []
        for name, shape, value, dtype, quant in self.tensors:
            buffer_index = 0
            if value is not None:
                data = b.CreateByteVector(value)
                schema.BufferStart(b)
                schema.BufferAddData(b, data)
                buffer_index = len(buffers)
                buffers.append(schema.BufferEnd(b))
            name_offset = b.CreateString(name)
            shape_offset = integers(shape)
            quant_offset = None
            if quant is not None:
                scales, zeros, dimension = quant
                scale_offset = b.CreateNumpyVector(np.asarray(scales, dtype=np.float32).reshape(-1))
                zero_offset = b.CreateNumpyVector(np.asarray(zeros, dtype=np.int64).reshape(-1))
                schema.QuantizationParametersStart(b)
                schema.QuantizationParametersAddScale(b, scale_offset)
                schema.QuantizationParametersAddZeroPoint(b, zero_offset)
                schema.QuantizationParametersAddQuantizedDimension(b, dimension)
                quant_offset = schema.QuantizationParametersEnd(b)
            schema.TensorStart(b)
            schema.TensorAddName(b, name_offset)
            schema.TensorAddShape(b, shape_offset)
            schema.TensorAddBuffer(b, buffer_index)
            schema.TensorAddType(b, {np.dtype(np.float32): schema.TensorType.FLOAT32, np.dtype(np.int8): schema.TensorType.INT8, np.dtype(np.int32): schema.TensorType.INT32}[dtype])
            if quant_offset is not None:
                schema.TensorAddQuantization(b, quant_offset)
            tensors.append(schema.TensorEnd(b))

        opcodes = []
        opcode_indices = {}
        operators = []
        for code, inputs, out, option, attributes, version in self.operators:
            key = (code, version)
            if key not in opcode_indices:
                schema.OperatorCodeStart(b)
                schema.OperatorCodeAddBuiltinCode(b, code)
                schema.OperatorCodeAddDeprecatedBuiltinCode(b, min(code, 127))
                schema.OperatorCodeAddVersion(b, version)
                opcode_indices[key] = len(opcodes)
                opcodes.append(schema.OperatorCodeEnd(b))
            getattr(schema, option + "Start")(b)
            for attr, value in attributes.items():
                getattr(schema, option + "Add" + attr)(b, value)
            options = getattr(schema, option + "End")(b)
            input_offset, output_offset = integers(inputs), integers(out)
            schema.OperatorStart(b)
            schema.OperatorAddOpcodeIndex(b, opcode_indices[key])
            schema.OperatorAddInputs(b, input_offset)
            schema.OperatorAddOutputs(b, output_offset)
            schema.OperatorAddBuiltinOptionsType(b, getattr(schema.BuiltinOptions, option))
            schema.OperatorAddBuiltinOptions(b, options)
            operators.append(schema.OperatorEnd(b))

        tensor_offset, operator_offset = offsets(tensors), offsets(operators)
        input_offset, output_offset = integers([input_index]), integers(outputs)
        graph_name = b.CreateString("static_gesture_mlp")
        schema.SubGraphStart(b)
        schema.SubGraphAddName(b, graph_name)
        schema.SubGraphAddTensors(b, tensor_offset)
        schema.SubGraphAddInputs(b, input_offset)
        schema.SubGraphAddOutputs(b, output_offset)
        schema.SubGraphAddOperators(b, operator_offset)
        subgraph = schema.SubGraphEnd(b)
        graph_offset = offsets([subgraph])
        code_offset, buffer_offset = offsets(opcodes), offsets(buffers)
        description = b.CreateString("Exact current static MLP; generated by static_tflite_graph.py")
        schema.ModelStart(b)
        schema.ModelAddVersion(b, 3)
        schema.ModelAddDescription(b, description)
        schema.ModelAddSubgraphs(b, graph_offset)
        schema.ModelAddOperatorCodes(b, code_offset)
        schema.ModelAddBuffers(b, buffer_offset)
        model = schema.ModelEnd(b)
        b.Finish(model, file_identifier=b"TFL3")
        return bytes(b.Output())


def quant_range(values):
    low, high = min(0., float(np.min(values))), max(0., float(np.max(values)))
    scale = max((high - low) / 255., 1e-8)
    zero = int(np.clip(np.rint(-128 - low / scale), -128, 127))
    return np.float32(scale), zero


def build_graph(w, calibration, quantized=False):
    g = Graph()
    code = schema.BuiltinOperator
    inp = g.tensor("landmark_features", [1, 76])
    mean = g.tensor("mean", [76], w["mean"])
    scale = g.tensor("scale", [76], w["scale"])
    centered = g.tensor("centered", [1, 76])
    g.op(code.SUB, [inp, mean], centered, "SubOptions")
    previous = g.tensor("scaled", [1, 76])
    g.op(code.DIV, [centered, scale], previous, "DivOptions")
    calibration = (calibration - w["mean"]) / w["scale"]
    previous_quant = None
    if quantized:
        previous_quant = quant_range(calibration)
        q = g.tensor("scaled_int8", [1, 76], dtype=np.int8, quant=(*previous_quant, 0))
        g.op(code.QUANTIZE, [previous], q, "QuantizeOptions")
        previous = q
    for i in range(3):
        matrix = w[f"w{i}"].T.copy()
        bias = w[f"b{i}"]
        calibration = calibration @ matrix.T + bias
        activation = schema.ActivationFunctionType.RELU if i < 2 else schema.ActivationFunctionType.NONE
        if i < 2:
            calibration = np.maximum(calibration, 0.)
        if quantized:
            next_quant = quant_range(calibration)
            weight_scale = np.float32(max(float(np.abs(matrix).max()) / 127., 1e-8))
            matrix_q = np.clip(np.rint(matrix / weight_scale), -127, 127).astype(np.int8)
            bias_scale = np.float32(previous_quant[0] * weight_scale)
            bias_q = np.rint(bias / bias_scale).astype(np.int32)
            matrix_index = g.tensor(f"w{i}", matrix.shape, matrix_q, np.int8, (weight_scale, 0, 0))
            bias_index = g.tensor(f"b{i}", bias.shape, bias_q, np.int32, (bias_scale, 0, 0))
            output = g.tensor(f"layer{i}_int8", [1, len(bias)], dtype=np.int8, quant=(*next_quant, 0))
            previous_quant = next_quant
        else:
            matrix_index = g.tensor(f"w{i}", matrix.shape, matrix)
            bias_index = g.tensor(f"b{i}", bias.shape, bias)
            output = g.tensor(f"layer{i}", [1, len(bias)])
        g.op(code.FULLY_CONNECTED, [previous, matrix_index, bias_index], output, "FullyConnectedOptions", {"FusedActivationFunction": activation}, version=4 if quantized else 1)
        previous = output
    if quantized:
        output = g.tensor("logits", [1, 14])
        g.op(code.DEQUANTIZE, [previous], output, "DequantizeOptions", version=2)
        previous = output
    all_probabilities = g.tensor("all_probabilities", [1, 14])
    g.op(code.SOFTMAX, [previous], all_probabilities, "SoftmaxOptions", {"Beta": 1.0})
    indices = g.tensor("known_indices", [13], w["known_indices"], np.int32)
    known = g.tensor("known", [1, 13])
    g.op(code.GATHER, [all_probabilities, indices], known, "GatherOptions", {"Axis": 1})
    axis = g.tensor("reduce_axis", [1], [1], np.int32)
    mass = g.tensor("known_gesture_mass", [1, 1])
    g.op(code.SUM, [known, axis], mass, "ReducerOptions", {"KeepDims": True})
    epsilon = g.tensor("epsilon", [1], w["epsilon"])
    safe = g.tensor("safe_known", [1, 13])
    g.op(code.ADD, [known, epsilon], safe, "AddOptions")
    denominator = g.tensor("denominator", [1, 1])
    g.op(code.SUM, [safe, axis], denominator, "ReducerOptions", {"KeepDims": True})
    probabilities = g.tensor("probabilities", [1, 13])
    g.op(code.DIV, [safe, denominator], probabilities, "DivOptions")
    return g.finish(inp, [probabilities, mass])
