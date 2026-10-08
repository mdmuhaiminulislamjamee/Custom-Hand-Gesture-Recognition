package com.example.gesturehandover

import android.net.Network
import java.net.InetSocketAddress
import java.net.Socket
import java.util.zip.CRC32

data class NcmJpeg(val bytes: ByteArray, val boardSequence: Long, val boardTimestampMs: Long)
enum class NcmState { STOPPED, CONNECTING, CONNECTED, RECONNECTING }

/** TCP stream parser. TCP reads are not packet boundaries. */
class JlipParser {
    private var pending = ByteArray(0)

    fun reset() { pending = ByteArray(0) }

    fun feed(bytes: ByteArray, length: Int = bytes.size): List<JlipPacket> {
        require(length in 0..bytes.size)
        pending += bytes.copyOf(length)
        val packets = ArrayList<JlipPacket>()
        while (true) {
            val magic = findMagic()
            if (magic < 0) {
                pending = pending.takeLast(minOf(3, pending.size)).toByteArray()
                break
            }
            if (magic > 0) pending = pending.copyOfRange(magic, pending.size)
            if (pending.size < 24) break
            val version = unsigned(4)
            val type = unsigned(5)
            val headerSize = (unsigned(6) shl 8) or unsigned(7)
            val sequence = u32(8)
            val payloadSize = u32(12)
            val expectedCrc = u32(16)
            val timestamp = u32(20)
            if (version != 1 || headerSize != 24 || payloadSize > MAX_PAYLOAD) {
                pending = pending.copyOfRange(1, pending.size)
                continue
            }
            val packetSize = headerSize + payloadSize.toInt()
            if (pending.size < packetSize) break
            val payload = pending.copyOfRange(headerSize, packetSize)
            pending = pending.copyOfRange(packetSize, pending.size)
            val crc = CRC32().also { it.update(payload) }.value
            if (crc == expectedCrc) packets.add(JlipPacket(type, sequence, timestamp, payload))
        }
        return packets
    }

    private fun findMagic(): Int {
        for (i in 0..pending.size - 4) {
            if (pending[i] == 0x4a.toByte() && pending[i + 1] == 0x4c.toByte() &&
                pending[i + 2] == 0x49.toByte() && pending[i + 3] == 0x50.toByte()) return i
        }
        return -1
    }

    private fun unsigned(i: Int) = pending[i].toInt() and 0xff
    private fun u32(i: Int): Long = (unsigned(i).toLong() shl 24) or
        (unsigned(i + 1).toLong() shl 16) or (unsigned(i + 2).toLong() shl 8) or unsigned(i + 3).toLong()

    companion object {
        const val MAX_PAYLOAD = 200 * 1024L

        fun validJpeg(payload: ByteArray): ByteArray? {
            if (payload.size < 4 || (payload[0].toInt() and 0xff) != 0xff ||
                (payload[1].toInt() and 0xff) != 0xd8) return null
            for (i in payload.size - 2 downTo 0) {
                if ((payload[i].toInt() and 0xff) == 0xff &&
                    (payload[i + 1].toInt() and 0xff) == 0xd9) return payload.copyOfRange(0, i + 2)
            }
            return null
        }

        fun heartbeat(sequence: Long, timestampMs: Long): ByteArray {
            val p = ByteArray(24)
            p[0] = 0x4a; p[1] = 0x4c; p[2] = 0x49; p[3] = 0x50
            p[4] = 1; p[5] = 0x03; p[6] = 0; p[7] = 24
            fun put32(offset: Int, value: Long) {
                for (i in 0..3) p[offset + i] = ((value ushr (24 - 8 * i)) and 0xff).toByte()
            }
            put32(8, sequence)
            put32(12, 0)
            put32(16, 0) // CRC32 of an empty payload
            put32(20, timestampMs)
            return p
        }
    }
}

data class JlipPacket(val type: Int, val sequence: Long, val timestampMs: Long, val payload: ByteArray)

/** Android OS must expose a route to the board. Supply its Network when cellular/Wi-Fi is default. */
class GestureNcmClient(
    private val host: String = "192.168.50.2",
    private val port: Int = 5000,
    private val network: Network? = null,
    private val onState: (NcmState) -> Unit,
    private val onJpeg: (NcmJpeg) -> Unit,
    private val onError: (Throwable) -> Unit,
) : AutoCloseable {
    @Volatile private var generation = 0L
    @Volatile private var socket: Socket? = null
    private var worker: Thread? = null

    @Synchronized fun start() {
        stop()
        val run = generation
        worker = Thread({ receiveLoop(run) }, "gesture-ncm").also { it.start() }
    }

    @Synchronized fun stop() {
        generation++
        socket?.close()
        socket = null
        onState(NcmState.STOPPED)
    }

    private fun receiveLoop(run: Long) {
        var reconnect = false
        var clientSequence = 0L
        while (run == generation) {
            onState(if (reconnect) NcmState.RECONNECTING else NcmState.CONNECTING)
            val parser = JlipParser()
            var current: Socket? = null
            try {
                val connected = network?.socketFactory?.createSocket() ?: Socket()
                current = connected
                if (run != generation) break
                socket = connected
                connected.connect(InetSocketAddress(host, port), 3000)
                connected.soTimeout = 10000
                onState(NcmState.CONNECTED)
                val input = connected.getInputStream()
                val output = connected.getOutputStream()
                val chunk = ByteArray(8192)
                while (run == generation) {
                    val count = input.read(chunk)
                    if (count < 0) break
                    for (packet in parser.feed(chunk, count)) {
                        when (packet.type) {
                            0x03 -> {
                                output.write(JlipParser.heartbeat(clientSequence++, System.currentTimeMillis()))
                                output.flush()
                            }
                            0x10 -> JlipParser.validJpeg(packet.payload)?.let {
                                onJpeg(NcmJpeg(it, packet.sequence, packet.timestampMs))
                            }
                        }
                    }
                }
            } catch (error: Exception) {
                if (run == generation) onError(error)
            } finally {
                current?.close()
                if (socket === current) socket = null
                parser.reset()
            }
            if (run != generation) break
            reconnect = true
            try { Thread.sleep(1000) } catch (_: InterruptedException) { break }
        }
    }

    override fun close() = stop()
}
