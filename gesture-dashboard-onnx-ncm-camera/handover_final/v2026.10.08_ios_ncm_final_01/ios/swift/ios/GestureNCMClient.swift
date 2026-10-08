import Foundation
import Network

public enum GestureNCMState: String {
    case stopped
    case connecting
    case connected
    case reconnecting
}

public enum GestureNCMError: Error, CustomStringConvertible {
    case invalidPort
    case connection(String)

    public var description: String {
        switch self {
        case .invalidPort:
            return "Invalid NCM TCP port."
        case .connection(let message):
            return message
        }
    }
}

/// Receives the board's TCP/JLIP stream and emits only CRC-checked JPEG payloads.
/// Callbacks run on this client's private serial queue. The app should retain only
/// the latest JPEG and admit inference no faster than once every 100 ms.
public final class GestureNCMClient {
    public var onState: ((GestureNCMState) -> Void)?
    public var onJPEG: ((Data, UInt32, UInt32) -> Void)?
    public var onError: ((Error) -> Void)?

    private let queue = DispatchQueue(label: "gesture.ncm.jlip")
    private var connection: NWConnection?
    private var parser = JLIPParser()
    private var stopped = true
    private var generation: UInt64 = 0
    private var host = "192.168.50.2"
    private var port: UInt16 = 5000
    private var clientSequence: UInt32 = 0

    public init() {}

    public func start(host: String = "192.168.50.2", port: UInt16 = 5000) {
        queue.async { [weak self] in
            guard let self else { return }
            self.stopLocked(notify: false)
            self.host = host
            self.port = port
            self.stopped = false
            self.generation &+= 1
            self.connect(generation: self.generation, reconnecting: false)
        }
    }

    public func stop() {
        queue.async { [weak self] in self?.stopLocked(notify: true) }
    }

    private func stopLocked(notify: Bool) {
        stopped = true
        generation &+= 1
        connection?.stateUpdateHandler = nil
        connection?.cancel()
        connection = nil
        parser.reset()
        if notify { onState?(.stopped) }
    }

    private func connect(generation expected: UInt64, reconnecting: Bool) {
        guard !stopped, generation == expected else { return }
        guard let nwPort = NWEndpoint.Port(rawValue: port) else {
            onError?(GestureNCMError.invalidPort)
            stopLocked(notify: true)
            return
        }
        parser.reset()
        onState?(reconnecting ? .reconnecting : .connecting)
        let parameters = NWParameters.tcp
        parameters.allowLocalEndpointReuse = true
        let next = NWConnection(host: NWEndpoint.Host(host), port: nwPort, using: parameters)
        connection = next
        next.stateUpdateHandler = { [weak self, weak next] state in
            guard let self, let next else { return }
            self.queue.async {
                guard !self.stopped, self.generation == expected, self.connection === next else { return }
                switch state {
                case .ready:
                    self.onState?(.connected)
                    self.receive(on: next, generation: expected)
                case .failed(let error):
                    self.onError?(GestureNCMError.connection(error.localizedDescription))
                    self.scheduleReconnect(generation: expected)
                case .cancelled:
                    if !self.stopped { self.scheduleReconnect(generation: expected) }
                default:
                    break
                }
            }
        }
        next.start(queue: queue)
    }

    private func receive(on current: NWConnection, generation expected: UInt64) {
        current.receive(minimumIncompleteLength: 1, maximumLength: 64 * 1024) {
            [weak self, weak current] content, _, isComplete, error in
            guard let self, let current else { return }
            self.queue.async {
                guard !self.stopped, self.generation == expected, self.connection === current else { return }
                if let content, !content.isEmpty {
                    for packet in self.parser.feed(content) {
                        self.handle(packet, connection: current)
                    }
                }
                if let error {
                    self.onError?(GestureNCMError.connection(error.localizedDescription))
                    self.scheduleReconnect(generation: expected)
                } else if isComplete {
                    self.scheduleReconnect(generation: expected)
                } else {
                    self.receive(on: current, generation: expected)
                }
            }
        }
    }

    private func handle(_ packet: JLIPPacket, connection: NWConnection) {
        switch packet.type {
        case 0x03: // Firmware/reference receiver uses heartbeat type 0x03 for the reply.
            let reply = JLIPParser.buildPacket(
                type: 0x03,
                sequence: clientSequence,
                timestampMS: UInt32(truncatingIfNeeded: UInt64(Date().timeIntervalSince1970 * 1000)),
                payload: Data()
            )
            clientSequence &+= 1
            connection.send(content: reply, completion: .contentProcessed { [weak self] error in
                if let error { self?.onError?(GestureNCMError.connection(error.localizedDescription)) }
            })
        case 0x10:
            if let jpeg = JLIPParser.validJPEG(packet.payload) {
                onJPEG?(jpeg, packet.sequence, packet.timestampMS)
            }
        default:
            break
        }
    }

    private func scheduleReconnect(generation expected: UInt64) {
        guard !stopped, generation == expected else { return }
        generation &+= 1
        let reconnectGeneration = generation
        connection?.stateUpdateHandler = nil
        connection?.cancel()
        connection = nil
        parser.reset()
        onState?(.reconnecting)
        queue.asyncAfter(deadline: .now() + 1) { [weak self] in
            self?.connect(generation: reconnectGeneration, reconnecting: true)
        }
    }
}

private struct JLIPPacket {
    let type: UInt8
    let sequence: UInt32
    let timestampMS: UInt32
    let payload: Data
}

private struct JLIPParser {
    private static let magic = Data([0x4A, 0x4C, 0x49, 0x50]) // JLIP
    private static let headerSize = 24
    private static let maximumPayload = 200 * 1024
    private var buffer = Data()

    mutating func reset() { buffer.removeAll(keepingCapacity: true) }

    mutating func feed(_ bytes: Data) -> [JLIPPacket] {
        buffer.append(bytes)
        var packets: [JLIPPacket] = []
        while true {
            guard let magicRange = buffer.range(of: Self.magic) else {
                if buffer.count > 3 { buffer.removeFirst(buffer.count - 3) }
                break
            }
            if magicRange.lowerBound > 0 { buffer.removeFirst(magicRange.lowerBound) }
            guard buffer.count >= Self.headerSize else { break }
            let version = buffer[4]
            let type = buffer[5]
            let headerBytes = Int(readUInt16(buffer, 6))
            let sequence = readUInt32(buffer, 8)
            let payloadBytes = Int(readUInt32(buffer, 12))
            let expectedCRC = readUInt32(buffer, 16)
            let timestamp = readUInt32(buffer, 20)
            guard version == 1, headerBytes == Self.headerSize,
                  payloadBytes >= 0, payloadBytes <= Self.maximumPayload else {
                buffer.removeFirst()
                continue
            }
            let packetBytes = headerBytes + payloadBytes
            guard buffer.count >= packetBytes else { break }
            let payload = Data(buffer[headerBytes..<packetBytes])
            buffer.removeFirst(packetBytes)
            guard crc32(payload) == expectedCRC else { continue }
            packets.append(JLIPPacket(type: type, sequence: sequence,
                                      timestampMS: timestamp, payload: payload))
        }
        return packets
    }

    static func validJPEG(_ payload: Data) -> Data? {
        guard payload.count >= 4, payload[0] == 0xFF, payload[1] == 0xD8 else { return nil }
        var end: Int?
        var index = payload.count - 2
        while index >= 0 {
            if payload[index] == 0xFF, payload[index + 1] == 0xD9 {
                end = index + 2
                break
            }
            if index == 0 { break }
            index -= 1
        }
        guard let end else { return nil }
        return Data(payload[0..<end])
    }

    static func buildPacket(type: UInt8, sequence: UInt32,
                            timestampMS: UInt32, payload: Data) -> Data {
        var result = Data()
        result.append(magic)
        result.append(1)
        result.append(type)
        appendUInt16(UInt16(headerSize), to: &result)
        appendUInt32(sequence, to: &result)
        appendUInt32(UInt32(payload.count), to: &result)
        appendUInt32(crc32(payload), to: &result)
        appendUInt32(timestampMS, to: &result)
        result.append(payload)
        return result
    }

    private static func readUInt16(_ data: Data, _ offset: Int) -> UInt16 {
        (UInt16(data[offset]) << 8) | UInt16(data[offset + 1])
    }

    private static func readUInt32(_ data: Data, _ offset: Int) -> UInt32 {
        (UInt32(data[offset]) << 24) | (UInt32(data[offset + 1]) << 16)
            | (UInt32(data[offset + 2]) << 8) | UInt32(data[offset + 3])
    }

    private static func appendUInt16(_ value: UInt16, to data: inout Data) {
        data.append(UInt8((value >> 8) & 0xFF))
        data.append(UInt8(value & 0xFF))
    }

    private static func appendUInt32(_ value: UInt32, to data: inout Data) {
        data.append(UInt8((value >> 24) & 0xFF))
        data.append(UInt8((value >> 16) & 0xFF))
        data.append(UInt8((value >> 8) & 0xFF))
        data.append(UInt8(value & 0xFF))
    }

    private static func crc32(_ data: Data) -> UInt32 {
        var crc: UInt32 = 0xFFFF_FFFF
        for byte in data {
            crc ^= UInt32(byte)
            for _ in 0..<8 {
                crc = (crc & 1) == 1 ? (crc >> 1) ^ 0xEDB8_8320 : crc >> 1
            }
        }
        return crc ^ 0xFFFF_FFFF
    }
}
