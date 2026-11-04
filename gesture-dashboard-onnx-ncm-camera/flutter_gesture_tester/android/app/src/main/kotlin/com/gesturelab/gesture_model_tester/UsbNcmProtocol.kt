package com.gesturelab.gesture_model_tester

import java.io.IOException
import java.util.ArrayDeque
import java.util.zip.CRC32

/** Wire codecs have no Android dependency so malformed USB/network input can be host-tested. */
internal object NcmWire {
    fun u8(b: ByteArray, p: Int) = b[p].toInt() and 255
    fun le16(b: ByteArray, p: Int) = u8(b, p) or (u8(b, p + 1) shl 8)
    fun be16(b: ByteArray, p: Int) = (u8(b, p) shl 8) or u8(b, p + 1)
    fun le32(b: ByteArray, p: Int): Long = (0..3).fold(0L) { n, i -> n or (u8(b, p + i).toLong() shl (8 * i)) }
    fun be32(b: ByteArray, p: Int): Long = (0..3).fold(0L) { n, i -> (n shl 8) or u8(b, p + i).toLong() }
    fun putLe16(b: ByteArray, p: Int, n: Int) { b[p] = n.toByte(); b[p + 1] = (n ushr 8).toByte() }
    fun putBe16(b: ByteArray, p: Int, n: Int) { b[p] = (n ushr 8).toByte(); b[p + 1] = n.toByte() }
    fun putLe32(b: ByteArray, p: Int, n: Long) { for (i in 0..3) b[p + i] = (n ushr (i * 8)).toByte() }
    fun putBe32(b: ByteArray, p: Int, n: Long) { for (i in 0..3) b[p + i] = (n ushr ((3 - i) * 8)).toByte() }
    fun checksum(b: ByteArray, start: Int = 0, size: Int = b.size, initial: Long = 0): Int {
        var sum = initial
        var p = start
        while (p + 1 < start + size) { sum += be16(b, p); p += 2 }
        if (p < start + size) sum += u8(b, p) shl 8
        while (sum ushr 16 != 0L) sum = (sum and 65535) + (sum ushr 16)
        return sum.inv().toInt() and 65535
    }
    fun distance(a: Long, b: Long) = ((a - b) and 0xffffffffL).toInt()
    fun next(n: Long, count: Int) = (n + count) and 0xffffffffL
}

internal data class NcmFunction(val control: Int, val data: Int, val macString: Int,
                                val maxSegment: Int, val capabilities: Int) {
    companion object {
        fun parse(raw: ByteArray, controlId: Int): NcmFunction {
            var p = 0; var current = -1; var data = -1; var mac = -1; var segment = 1514; var caps = -1
            while (p + 2 <= raw.size) {
                val length = NcmWire.u8(raw, p)
                require(length >= 2 && p + length <= raw.size) { "Malformed USB descriptor at $p" }
                when (NcmWire.u8(raw, p + 1)) {
                    4 -> { require(length >= 9); current = NcmWire.u8(raw, p + 2) }
                    0x24 -> if (current == controlId && length >= 3) {
                        when (NcmWire.u8(raw, p + 2)) {
                            6 -> { require(length >= 5); require(NcmWire.u8(raw, p + 3) == controlId); data = NcmWire.u8(raw, p + 4) }
                            15 -> { require(length >= 13); mac = NcmWire.u8(raw, p + 3); segment = NcmWire.le16(raw, p + 8) }
                            26 -> { require(length >= 6); caps = NcmWire.u8(raw, p + 5) }
                        }
                    }
                }
                p += length
            }
            require(data >= 0 && mac > 0 && caps >= 0) { "CDC-NCM union/Ethernet/NCM functional descriptors missing" }
            require(segment in 576..16300) { "Unsupported Ethernet segment size $segment" }
            return NcmFunction(controlId, data, mac, segment, caps)
        }
    }
}

internal data class NcmParameters(val inputSize: Int, val outputSize: Int,
                                  val divisor: Int, val remainder: Int, val alignment: Int) {
    companion object {
        fun parse(b: ByteArray): NcmParameters {
            require(b.size >= 28 && NcmWire.le16(b, 0) >= 28) { "Short GET_NTB_PARAMETERS" }
            require(NcmWire.le16(b, 2) and 1 != 0) { "Camera does not support NTB16" }
            val input = NcmWire.le32(b, 4).coerceAtMost(16384).toInt()
            val outputRaw = NcmWire.le32(b, 16)
            val output = (if (outputRaw == 0L) 16384 else outputRaw.coerceAtMost(16384)).toInt()
            val divisor = NcmWire.le16(b, 20).coerceAtLeast(1)
            val remainder = NcmWire.le16(b, 22)
            val alignment = NcmWire.le16(b, 24).coerceAtLeast(1)
            require(input >= 2048 && output >= 2048 && divisor <= 4096 && alignment <= 4096 && remainder < divisor) {
                "Unsupported NCM transfer/alignment parameters"
            }
            return NcmParameters(input, output, divisor, remainder, alignment)
        }
    }
}

/** NTB16 transmit and bounded reassembly of split/concatenated USB reads. */
internal class NcmNtbCodec(private val parameters: NcmParameters, private val maxPacket: Int) {
    private var sequence = 0
    private var pending = ByteArray(0)
    fun encode(frame: ByteArray): ByteArray {
        require(frame.size in 14..16000)
        val ndp = ((12 + parameters.alignment - 1) / parameters.alignment) * parameters.alignment
        var data = ndp + 16
        while ((data + 14) % parameters.divisor != parameters.remainder) data++
        var length = data + frame.size.coerceAtLeast(60)
        if (length % maxPacket == 0) length++ // End with a short USB packet without relying on a ZLP.
        require(length <= parameters.outputSize) { "Ethernet frame exceeds negotiated NTB output size" }
        val b = ByteArray(length)
        NcmWire.putLe32(b, 0, 0x484d434e); NcmWire.putLe16(b, 4, 12)
        NcmWire.putLe16(b, 6, sequence++ and 65535); NcmWire.putLe16(b, 8, length); NcmWire.putLe16(b, 10, ndp)
        NcmWire.putLe32(b, ndp, 0x304d434e); NcmWire.putLe16(b, ndp + 4, 16)
        NcmWire.putLe16(b, ndp + 8, data); NcmWire.putLe16(b, ndp + 10, frame.size.coerceAtLeast(60))
        frame.copyInto(b, data)
        return b
    }
    fun feed(bytes: ByteArray): List<ByteArray> {
        require(pending.size + bytes.size <= parameters.inputSize * 3) { "NCM receive buffer overflow" }
        pending += bytes
        val frames = mutableListOf<ByteArray>()
        while (pending.size >= 12) {
            if (NcmWire.le32(pending, 0) != 0x484d434eL) {
                // Firmware may append padding after wBlockLength.
                var next = 1
                while (next + 4 <= pending.size && NcmWire.le32(pending, next) != 0x484d434eL) next++
                pending = pending.copyOfRange(next, pending.size); continue
            }
            require(NcmWire.le16(pending, 4) == 12) { "Invalid NTB16 header length" }
            val length = NcmWire.le16(pending, 8)
            require(length in 12..parameters.inputSize) { "Invalid NTB16 block length $length" }
            if (pending.size < length) break
            val block = pending.copyOfRange(0, length)
            pending = pending.copyOfRange(length, pending.size)
            var ndp = NcmWire.le16(block, 10)
            val visited = mutableSetOf<Int>()
            while (ndp != 0) {
                require(visited.add(ndp) && ndp >= 12 && ndp + 8 <= length) { "Invalid NDP chain" }
                val signature = NcmWire.le32(block, ndp)
                require(signature == 0x304d434eL || signature == 0x314d434eL) { "Unsupported NDP signature" }
                val tableLength = NcmWire.le16(block, ndp + 4)
                require(tableLength >= 12 && tableLength % 4 == 0 && ndp + tableLength <= length) { "Invalid NDP length" }
                var terminated = false
                for (entry in ndp + 8 until ndp + tableLength step 4) {
                    val start = NcmWire.le16(block, entry); val size = NcmWire.le16(block, entry + 2)
                    if (start == 0 && size == 0) { terminated = true; break }
                    require(start >= 12 && size >= 14 && start + size <= length &&
                        (start + size <= ndp || start >= ndp + tableLength)) { "Invalid NCM datagram range" }
                    var frame = block.copyOfRange(start, start + size)
                    if (signature == 0x314d434eL) {
                        require(frame.size >= 18)
                        val crc = CRC32().also { it.update(frame, 0, frame.size - 4) }.value
                        require(crc == NcmWire.le32(frame, frame.size - 4)) { "NCM datagram CRC mismatch" }
                        frame = frame.copyOf(frame.size - 4)
                    }
                    frames += frame
                }
                require(terminated) { "Unterminated NDP table" }
                ndp = NcmWire.le16(block, ndp + 6)
            }
        }
        return frames
    }
}

/** One bounded IPv4/TCP client for the directly attached board; no global Android network changes. */
internal class NcmCameraTcp(
    private val mac: ByteArray, private val localIp: Long, private val boardIp: Long,
    private val localPort: Int, private val boardPort: Int, initialSequence: Long,
    private val clock: () -> Long, private val transmit: (ByteArray) -> Unit,
    private val maximumMss: Int = 1400,
) {
    enum class State { ARP, SYN, ESTABLISHED, CLOSED }
    @Volatile var state = State.ARP; private set
    @Volatile var error: String? = null; private set
    var receivedBytes = 0L; private set
    private var peer: ByteArray? = null
    private var sendNext = initialSequence and 0xffffffffL
    private var receiveNext = 0L
    private var peerWindow = 65535
    private var peerMss = maximumMss
    private var lastArp = -1000L
    private var arpRetries = 0
    private var packetId = 0
    private var lastProbe = 0L
    private val received = ArrayDeque<ByteArray>()
    private var buffered = 0
    private val outgoing = ArrayDeque<ByteArray>()
    private var queued = 0
    private data class Sent(val seq: Long, val flags: Int, val data: ByteArray, val end: Long,
                            var time: Long, var attempts: Int = 1)
    private var outstanding: Sent? = null

    @Synchronized fun tick() {
        if (state == State.CLOSED) return
        val now = clock()
        if (state == State.ARP) {
            if (peer != null) {
                state = State.SYN
                sendTracked(2, ByteArray(0))
            } else if (now - lastArp >= 1000) {
                if (++arpRetries > 8) { fail("USB-NCM initialized, but board 192.168.50.2 did not answer ARP"); return }
                lastArp = now; arp(1, ByteArray(6) { 255.toByte() }, ByteArray(6), boardIp)
            }
            return
        }
        val waiting = outstanding
        if (waiting != null && now - waiting.time >= (400L shl (waiting.attempts - 1).coerceAtMost(3))) {
            if (waiting.attempts++ >= 6) { fail("Board TCP acknowledgement timed out"); return }
            waiting.time = now; tcp(waiting.seq, waiting.flags, waiting.data)
        }
        if (state == State.ESTABLISHED && outstanding == null && outgoing.isNotEmpty()) {
            if (peerWindow > 0) {
                val bytes = outgoing.removeFirst()
                val count = minOf(bytes.size, peerMss, peerWindow)
                if (count < bytes.size) outgoing.addFirst(bytes.copyOfRange(count, bytes.size))
                queued -= count
                sendTracked(0x18, bytes.copyOf(count))
            } else if (now - lastProbe > 1000) {
                lastProbe = now; tcp(NcmWire.next(sendNext, -1), 0x10, ByteArray(0))
            }
        }
    }
    @Synchronized fun write(bytes: ByteArray): Boolean {
        if (state != State.ESTABLISHED || queued + bytes.size > 65535) return false
        if (bytes.isNotEmpty()) { outgoing += bytes.copyOf(); queued += bytes.size }
        tick(); return true
    }
    @Synchronized fun read(): ByteArray? {
        if (received.isEmpty()) return null
        val bytes = received.removeFirst(); buffered -= bytes.size
        if (state == State.ESTABLISHED) ack()
        return bytes
    }
    @Synchronized fun input(frame: ByteArray) {
        if (state == State.CLOSED || frame.size < 14) return
        val type = NcmWire.be16(frame, 12)
        if (type == 0x0806) { inputArp(frame); return }
        if (type != 0x0800 || peer == null || frame.size < 54 || !frame.copyOfRange(0, 6).contentEquals(mac)) return
        val ip = 14
        val header = (NcmWire.u8(frame, ip) and 15) * 4
        val length = NcmWire.be16(frame, ip + 2)
        if (NcmWire.u8(frame, ip) ushr 4 != 4 || header < 20 || length < header + 20 || ip + length > frame.size ||
            NcmWire.checksum(frame, ip, header) != 0 || NcmWire.be16(frame, ip + 6) and 0x3fff != 0 ||
            NcmWire.u8(frame, ip + 9) != 6 || NcmWire.be32(frame, ip + 12) != boardIp ||
            NcmWire.be32(frame, ip + 16) != localIp) return
        val t = ip + header
        val size = length - header
        val tcpHeader = (NcmWire.u8(frame, t + 12) ushr 4) * 4
        if (tcpHeader < 20 || tcpHeader > size || NcmWire.be16(frame, t) != boardPort ||
            NcmWire.be16(frame, t + 2) != localPort) return
        val pseudo = pseudoSum(boardIp, localIp, size)
        if (NcmWire.checksum(frame, t, size, pseudo) != 0) return
        val seq = NcmWire.be32(frame, t + 4); val acknowledgement = NcmWire.be32(frame, t + 8)
        val flags = NcmWire.u8(frame, t + 13)
        if (flags and 4 != 0) {
            if ((state == State.SYN && flags and 16 != 0 && acknowledgement == sendNext) ||
                (state == State.ESTABLISHED && seq == receiveNext)) fail("Board reset/refused TCP port $boardPort")
            return
        }
        if (state == State.SYN) {
            if (flags and 0x12 != 0x12 || acknowledgement != sendNext) return
            var p = t + 20
            while (p < t + tcpHeader) {
                val option = NcmWire.u8(frame, p)
                if (option == 0) break
                if (option == 1) { p++; continue }
                if (p + 2 > t + tcpHeader) return
                val n = NcmWire.u8(frame, p + 1)
                if (n < 2 || p + n > t + tcpHeader) return
                if (option == 2 && n == 4) peerMss = minOf(maximumMss, NcmWire.be16(frame, p + 2).coerceAtLeast(64))
                p += n
            }
            receiveNext = NcmWire.next(seq, 1); outstanding = null
            peerWindow = NcmWire.be16(frame, t + 14); state = State.ESTABLISHED; ack()
            val early = frame.copyOfRange(t + tcpHeader, t + size)
            if (early.isNotEmpty()) {
                received += early; buffered += early.size; receivedBytes += early.size
                receiveNext = NcmWire.next(receiveNext, early.size); ack()
            }
            return
        }
        if (state != State.ESTABLISHED) return
        if (flags and 2 != 0) { ack(); return }
        if (flags and 16 == 0) return
        val waiting = outstanding
        if (flags and 16 != 0 && NcmWire.distance(acknowledgement, sendNext) <= 0 &&
            (waiting == null || NcmWire.distance(acknowledgement, waiting.seq) >= 0)) {
            peerWindow = NcmWire.be16(frame, t + 14)
            if (waiting != null && acknowledgement == waiting.end) outstanding = null
        }
        val data = frame.copyOfRange(t + tcpHeader, t + size)
        var trim = -NcmWire.distance(seq, receiveNext)
        if (trim < 0) { ack(); return } // Out-of-order segments are re-requested with the current cumulative ACK.
        trim = minOf(trim, data.size)
        val fresh = data.size - trim
        if (fresh > 65535 - buffered) { ack(); return }
        if (fresh > 0) {
            val bytes = data.copyOfRange(trim, data.size)
            received += bytes; buffered += bytes.size; receivedBytes += bytes.size
            receiveNext = NcmWire.next(receiveNext, bytes.size)
        }
        if (flags and 1 != 0 && NcmWire.next(seq, data.size) == receiveNext) {
            receiveNext = NcmWire.next(receiveNext, 1); ack(); state = State.CLOSED
        } else if (data.isNotEmpty()) ack()
        tick()
    }
    private fun inputArp(b: ByteArray) {
        if (b.size < 42 || NcmWire.be16(b, 14) != 1 || NcmWire.be16(b, 16) != 0x0800 ||
            NcmWire.u8(b, 18) != 6 || NcmWire.u8(b, 19) != 4) return
        val operation = NcmWire.be16(b, 20)
        val sender = NcmWire.be32(b, 28); val target = NcmWire.be32(b, 38)
        if (target != localIp || sender != boardIp || operation !in 1..2) return
        val address = b.copyOfRange(22, 28)
        if (NcmWire.u8(address, 0) and 1 != 0 || address.all { it == 0.toByte() }) return
        peer = address
        if (operation == 1) arp(2, address, address, sender)
        tick()
    }
    private fun arp(operation: Int, dest: ByteArray, targetMac: ByteArray, targetIp: Long) {
        val b = ByteArray(60); dest.copyInto(b); mac.copyInto(b, 6)
        NcmWire.putBe16(b, 12, 0x0806); NcmWire.putBe16(b, 14, 1); NcmWire.putBe16(b, 16, 0x0800)
        b[18] = 6; b[19] = 4; NcmWire.putBe16(b, 20, operation)
        mac.copyInto(b, 22); NcmWire.putBe32(b, 28, localIp); targetMac.copyInto(b, 32); NcmWire.putBe32(b, 38, targetIp)
        transmit(b)
    }
    private fun ack() = tcp(sendNext, 16, ByteArray(0))
    private fun sendTracked(flags: Int, data: ByteArray) {
        val seq = sendNext; sendNext = NcmWire.next(sendNext, data.size + if (flags and 2 != 0) 1 else 0)
        outstanding = Sent(seq, flags, data, sendNext, clock())
        tcp(seq, flags, data)
    }
    private fun tcp(seq: Long, flags: Int, data: ByteArray) {
        val tcpSize = 20 + (if (flags and 2 != 0) 4 else 0) + data.size
        val b = ByteArray((14 + 20 + tcpSize).coerceAtLeast(60))
        peer!!.copyInto(b); mac.copyInto(b, 6); NcmWire.putBe16(b, 12, 0x0800)
        b[14] = 0x45; NcmWire.putBe16(b, 16, 20 + tcpSize); NcmWire.putBe16(b, 18, packetId++ and 65535)
        NcmWire.putBe16(b, 20, 0x4000); b[22] = 64; b[23] = 6
        NcmWire.putBe32(b, 26, localIp); NcmWire.putBe32(b, 30, boardIp)
        NcmWire.putBe16(b, 24, NcmWire.checksum(b, 14, 20))
        val t = 34
        NcmWire.putBe16(b, t, localPort); NcmWire.putBe16(b, t + 2, boardPort)
        NcmWire.putBe32(b, t + 4, seq); NcmWire.putBe32(b, t + 8, if (flags and 2 != 0) 0 else receiveNext)
        b[t + 12] = ((if (flags and 2 != 0) 6 else 5) shl 4).toByte(); b[t + 13] = flags.toByte()
        NcmWire.putBe16(b, t + 14, 65535 - buffered)
        if (flags and 2 != 0) { b[t + 20] = 2; b[t + 21] = 4; NcmWire.putBe16(b, t + 22, maximumMss) }
        data.copyInto(b, t + tcpSize - data.size)
        NcmWire.putBe16(b, t + 16, NcmWire.checksum(b, t, tcpSize, pseudoSum(localIp, boardIp, tcpSize)))
        transmit(b)
    }
    private fun pseudoSum(src: Long, dst: Long, n: Int) =
        (src ushr 16) + (src and 65535) + (dst ushr 16) + (dst and 65535) + 6 + n
    private fun fail(message: String) { error = message; state = State.CLOSED }
    @Synchronized fun checkAlive() { if (error != null) throw IOException(error) }
    @Synchronized fun abort() {
        if (peer != null && state != State.CLOSED) tcp(sendNext, if (state == State.ESTABLISHED) 0x14 else 4, ByteArray(0))
        state = State.CLOSED
    }
}
