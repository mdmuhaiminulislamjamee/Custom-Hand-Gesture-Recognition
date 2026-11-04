package com.gesturelab.gesture_model_tester

private fun hex(s: String) = s.chunked(2).map { it.toInt(16).toByte() }.toByteArray()
private fun ByteArray.hex() = joinToString("") { "%02x".format(it) }
private fun rejects(block: () -> Unit) { check(runCatching(block).isFailure) { "Expected malformed input to be rejected" } }
private val hostMac = hex("020000000001")
private val boardMac = hex("020000000002")
private const val hostIp = 0xc0a83201L
private const val boardIp = 0xc0a83202L

private class Fixture(initial: Long = 100, peerInitial: Long = 1000, window: Int = 65535, mss: Int = 1400, establish: Boolean = true) {
    var now = 0L
    val sent = mutableListOf<ByteArray>()
    val tcp = NcmCameraTcp(hostMac, hostIp, boardIp, 51000, 5000, initial, { now }, { sent += it })
    var peerSeq = NcmWire.next(peerInitial, 1)
    var hostNext = NcmWire.next(initial, 1)
    init {
        tcp.tick()
        val arp = sent.last().copyOf()
        hostMac.copyInto(arp); boardMac.copyInto(arp, 6); NcmWire.putBe16(arp, 20, 2)
        boardMac.copyInto(arp, 22); NcmWire.putBe32(arp, 28, boardIp)
        hostMac.copyInto(arp, 32); NcmWire.putBe32(arp, 38, hostIp)
        tcp.input(arp)
        check(tcp.state == NcmCameraTcp.State.SYN)
        val options = byteArrayOf(2, 4, (mss ushr 8).toByte(), mss.toByte())
        if (establish) {
            tcp.input(packet(0x12, peerInitial, hostNext, window = window, options = options))
            check(tcp.state == NcmCameraTcp.State.ESTABLISHED)
        }
    }
    fun packet(flags: Int = 16, seq: Long = peerSeq, ack: Long = hostNext,
               bytes: ByteArray = ByteArray(0), window: Int = 65535,
               options: ByteArray = ByteArray(0)): ByteArray {
        require(options.size % 4 == 0)
        val n = 20 + options.size + bytes.size
        val b = ByteArray(34 + n)
        hostMac.copyInto(b); boardMac.copyInto(b, 6); NcmWire.putBe16(b, 12, 0x0800)
        b[14] = 0x45; NcmWire.putBe16(b, 16, 20 + n); b[22] = 64; b[23] = 6
        NcmWire.putBe32(b, 26, boardIp); NcmWire.putBe32(b, 30, hostIp)
        NcmWire.putBe16(b, 24, NcmWire.checksum(b, 14, 20))
        NcmWire.putBe16(b, 34, 5000); NcmWire.putBe16(b, 36, 51000)
        NcmWire.putBe32(b, 38, seq); NcmWire.putBe32(b, 42, ack)
        b[46] = (((20 + options.size) / 4) shl 4).toByte(); b[47] = flags.toByte()
        NcmWire.putBe16(b, 48, window)
        options.copyInto(b, 54); bytes.copyInto(b, 54 + options.size)
        val pseudo = (boardIp ushr 16) + (boardIp and 65535) + (hostIp ushr 16) + (hostIp and 65535) + 6 + n
        NcmWire.putBe16(b, 50, NcmWire.checksum(b, 34, n, pseudo))
        return b
    }
    fun receive(s: String) {
        tcp.input(packet(0x18, bytes = s.toByteArray()))
        peerSeq = NcmWire.next(peerSeq, s.length)
    }
    fun acknowledge() {
        val frame = sent.last { NcmWire.be16(it, 12) == 0x0800 && it.size >= 54 }
        val count = NcmWire.be16(frame, 16) - 20 - ((NcmWire.u8(frame, 46) ushr 4) * 4)
        hostNext = NcmWire.next(NcmWire.be32(frame, 38), count)
        tcp.input(packet())
    }
}

fun main() {
    val parameters = NcmParameters(16384, 16384, 4, 0, 4)
    val codec = NcmNtbCodec(parameters, 64)
    val payload = ByteArray(60) { it.toByte() }
    val ntb = codec.encode(payload)
    check(ntb.copyOfRange(0, 30).hex() == "4e434d480c0000005a000c004e434d30100000001e003c00000000000000")
    check(NcmNtbCodec(parameters, 64).feed(ntb).single().contentEquals(payload))
    println("WIRE_NTB=${ntb.hex()}")
    println("PASS: NTB16 golden byte layout, NDP table, Ethernet/IP alignment")

    val fragments = NcmNtbCodec(parameters, 64)
    val decoded = ntb.flatMap { fragments.feed(byteArrayOf(it)) }
    check(decoded.size == 1 && decoded.single().contentEquals(payload))
    val concatenated = NcmNtbCodec(parameters, 64).feed(ntb + ByteArray(17) + ntb)
    check(concatenated.size == 2)
    println("PASS: split reads, concatenated NTBs and padding resynchronization")

    val invalidSize = ntb.copyOf().also { NcmWire.putLe16(it, 8, 65535) }
    rejects { NcmNtbCodec(parameters, 64).feed(invalidSize) }
    val loop = ntb.copyOf().also { NcmWire.putLe16(it, 18, 12) }
    rejects { NcmNtbCodec(parameters, 64).feed(loop) }
    val badData = ntb.copyOf().also { NcmWire.putLe16(it, 20, 89) }
    rejects { NcmNtbCodec(parameters, 64).feed(badData) }
    val badTable = ntb.copyOf().also { NcmWire.putLe16(it, 16, 14) }
    rejects { NcmNtbCodec(parameters, 64).feed(badTable) }
    rejects { NcmNtbCodec(parameters, 64).feed(ByteArray(50000)) }
    println("PASS: invalid NTB lengths, pointer loops, datagram bounds and bounded buffers")

    val crcBlock = ntb.copyOf(ntb.size + 4)
    NcmWire.putLe32(crcBlock, 12, 0x314d434e)
    NcmWire.putLe16(crcBlock, 8, crcBlock.size); NcmWire.putLe16(crcBlock, 22, 64)
    NcmWire.putLe32(crcBlock, ntb.size, java.util.zip.CRC32().also { it.update(payload) }.value)
    check(NcmNtbCodec(parameters, 64).feed(crcBlock).single().contentEquals(payload))
    rejects { NcmNtbCodec(parameters, 64).feed(crcBlock.copyOf().also { it[it.lastIndex] = 1 }) }
    println("PASS: NCM1 CRC verification and FCS removal")

    val exactPacket = NcmNtbCodec(parameters, 64).encode(ByteArray(98))
    check(exactPacket.size == 129 && NcmWire.le16(exactPacket, 8) == 129)
    val oddAlignment = NcmParameters(16384, 16384, 16, 3, 16)
    val aligned = NcmNtbCodec(oddAlignment, 64).encode(payload)
    val ndp = NcmWire.le16(aligned, 10); val data = NcmWire.le16(aligned, ndp + 8)
    check(ndp % 16 == 0 && (data + 14) % 16 == 3)
    println("PASS: short-packet termination and non-default negotiated alignment")

    val descriptors = hex("0904010001020d000005240601020d240f0300000000ea0500000006241a000110")
    val function = NcmFunction.parse(descriptors, 1)
    check(function.data == 2 && function.macString == 3 && function.maxSegment == 1514 && function.capabilities == 16)
    rejects { NcmFunction.parse(descriptors.copyOf(descriptors.size - 1), 1) }
    rejects { NcmFunction.parse(descriptors, 0) }
    val raw = ByteArray(28)
    NcmWire.putLe16(raw, 0, 28); NcmWire.putLe16(raw, 2, 1)
    NcmWire.putLe32(raw, 4, 65536); NcmWire.putLe32(raw, 16, 16384)
    NcmWire.putLe16(raw, 20, 4); NcmWire.putLe16(raw, 24, 4)
    check(NcmParameters.parse(raw) == parameters)
    rejects { NcmParameters.parse(raw.copyOf().also { NcmWire.putLe16(it, 2, 2) }) }
    rejects { NcmParameters.parse(raw.copyOf().also { NcmWire.putLe16(it, 22, 7) }) }
    println("PASS: union/MAC/NCM functional descriptors and negotiated transfer limits")

    val f = Fixture()
    val syn = f.sent.first { NcmWire.be16(it, 12) == 0x0800 }
    println("WIRE_SYN=${syn.hex()}")
    check(NcmWire.u8(syn, 47) == 2 && NcmWire.be16(syn, 56) == 1400)
    check(NcmWire.be32(f.sent.last(), 42) == 1001L)
    println("PASS: ARP resolution and TCP three-way handshake with MSS")

    val synRetry = Fixture(establish = false)
    val firstSyn = synRetry.sent.last()
    synRetry.now += 401; synRetry.tcp.tick()
    check(synRetry.sent.last().copyOfRange(34, firstSyn.size).contentEquals(firstSyn.copyOfRange(34, firstSyn.size)))
    synRetry.tcp.input(synRetry.packet(flags = 0x12, seq = 1000, ack = 99))
    check(synRetry.tcp.state == NcmCameraTcp.State.SYN)
    synRetry.tcp.input(synRetry.packet(flags = 0x12, seq = 1000, bytes = "early".toByteArray()))
    check(synRetry.tcp.state == NcmCameraTcp.State.ESTABLISHED && String(synRetry.tcp.read()!!) == "early")
    val refused = Fixture(establish = false)
    refused.tcp.input(refused.packet(flags = 0x14))
    check(refused.tcp.state == NcmCameraTcp.State.CLOSED)
    println("PASS: SYN loss/retry, invalid SYN acknowledgement, early data and refused connection")

    f.receive("abcd")
    check(String(f.tcp.read()!!) == "abcd")
    f.tcp.input(f.packet(seq = 1001, bytes = "abcd".toByteArray()))
    check(f.tcp.read() == null)
    f.tcp.input(f.packet(seq = 1003, bytes = "cdef".toByteArray()))
    check(String(f.tcp.read()!!) == "ef")
    check(f.tcp.receivedBytes == 6L)
    f.peerSeq = 1007
    f.tcp.input(f.packet(seq = 1010, bytes = "later".toByteArray()))
    check(f.tcp.read() == null && NcmWire.be32(f.sent.last(), 42) == 1007L)
    f.receive("123")
    check(String(f.tcp.read()!!) == "123")
    f.receive("later")
    check(String(f.tcp.read()!!) == "later")
    println("PASS: cumulative ACK, duplicate/overlap suppression and out-of-order recovery")

    val corrupt = f.packet(bytes = "bad".toByteArray()).also { it[it.lastIndex] = (it.last().toInt() xor 1).toByte() }
    f.tcp.input(corrupt)
    check(f.tcp.read() == null)
    val badIp = f.packet(bytes = "bad".toByteArray()).also { it[22] = 1 }
    f.tcp.input(badIp); check(f.tcp.read() == null)
    val wrongPort = f.packet(bytes = "bad".toByteArray()).also { NcmWire.putBe16(it, 34, 5001) }
    f.tcp.input(wrongPort); check(f.tcp.read() == null)
    f.tcp.input(f.packet(flags = 8, bytes = "noACK".toByteArray())); check(f.tcp.read() == null)
    println("PASS: TCP/IP checksum rejection and exact endpoint filtering")

    val retry = Fixture()
    check(retry.tcp.write("heartbeat".toByteArray()))
    val original = retry.sent.last()
    retry.now += 401; retry.tcp.tick()
    check(retry.sent.last().copyOfRange(34, original.size).contentEquals(original.copyOfRange(34, original.size)))
    retry.acknowledge()
    val acknowledgedCount = retry.sent.size
    retry.now += 4000; retry.tcp.tick()
    check(retry.sent.size == acknowledgedCount)
    println("PASS: lost application-data ACK retransmission and cancellation after ACK")

    val segmented = Fixture(mss = 512)
    check(segmented.tcp.write(ByteArray(1600) { it.toByte() }))
    val sizes = mutableListOf<Int>()
    repeat(4) {
        val packet = segmented.sent.last()
        sizes += NcmWire.be16(packet, 16) - 40
        segmented.acknowledge()
    }
    check(sizes == listOf(512, 512, 512, 64))
    println("PASS: outbound MSS segmentation and stop-and-wait sequence progression")

    val flow = Fixture(window = 0)
    check(flow.tcp.write(ByteArray(100)))
    flow.now = 1100; flow.tcp.tick()
    check(NcmWire.be16(flow.sent.last(), 16) == 40 && NcmWire.be32(flow.sent.last(), 38) == 100L)
    flow.tcp.input(flow.packet(window = 50))
    check(NcmWire.be16(flow.sent.last(), 16) - 40 == 50)
    println("PASS: remote zero-window probe and window-limited transmission")

    val bounded = Fixture()
    repeat(65) {
        bounded.tcp.input(bounded.packet(bytes = ByteArray(1000)))
        bounded.peerSeq += 1000
    }
    bounded.tcp.input(bounded.packet(bytes = ByteArray(535))); bounded.peerSeq += 535
    check(NcmWire.be16(bounded.sent.last(), 48) == 0)
    bounded.tcp.input(bounded.packet(bytes = ByteArray(100)))
    check(NcmWire.be32(bounded.sent.last(), 42) == bounded.peerSeq)
    check(bounded.tcp.read()!!.size == 1000 && NcmWire.be16(bounded.sent.last(), 48) == 1000)
    bounded.tcp.input(bounded.packet(bytes = ByteArray(100)))
    check(bounded.tcp.receivedBytes == 65635L)
    println("PASS: bounded receive window, backpressure, and window-update ACK")

    val wrapped = Fixture(initial = 0xffffffffL, peerInitial = 0xfffffffeL)
    wrapped.receive("wrap")
    check(String(wrapped.tcp.read()!!) == "wrap" && NcmWire.be32(wrapped.sent.last(), 42) == 3L)
    check(wrapped.tcp.write("ok".toByteArray()))
    check(NcmWire.be32(wrapped.sent.last(), 38) == 0L)
    wrapped.acknowledge(); check(wrapped.hostNext == 2L)
    println("PASS: 32-bit sequence wrap for handshake, receive and transmit")

    val fin = Fixture()
    fin.tcp.input(fin.packet(flags = 0x19, bytes = "last".toByteArray()))
    check(fin.tcp.state == NcmCameraTcp.State.CLOSED && String(fin.tcp.read()!!) == "last")
    check(NcmWire.be32(fin.sent.last(), 42) == 1006L)
    val reset = Fixture()
    reset.tcp.input(reset.packet(flags = 4, seq = reset.peerSeq + 1))
    check(reset.tcp.state == NcmCameraTcp.State.ESTABLISHED)
    reset.tcp.input(reset.packet(flags = 4))
    check(reset.tcp.state == NcmCameraTcp.State.CLOSED)
    rejects { reset.tcp.checkAlive() }
    val stopped = Fixture(); stopped.tcp.abort()
    check(stopped.tcp.state == NcmCameraTcp.State.CLOSED && NcmWire.u8(stopped.sent.last(), 47) == 0x14)
    println("PASS: FIN drains final payload, validated RST and stop resets board session")

    var clock = 0L
    val unreachable = NcmCameraTcp(hostMac, hostIp, boardIp, 51000, 5000, 1, { clock }, {})
    repeat(10) { unreachable.tick(); clock += 1001 }
    check(unreachable.state == NcmCameraTcp.State.CLOSED)
    rejects { unreachable.checkAlive() }
    println("PASS: bounded ARP retry deadline and explicit connection failure")
}
