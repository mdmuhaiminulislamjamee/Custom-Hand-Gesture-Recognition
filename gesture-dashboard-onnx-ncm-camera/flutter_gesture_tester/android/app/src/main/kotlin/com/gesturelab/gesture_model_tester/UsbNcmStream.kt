package com.gesturelab.gesture_model_tester

import android.app.PendingIntent
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.hardware.usb.UsbConstants
import android.hardware.usb.UsbDevice
import android.hardware.usb.UsbDeviceConnection
import android.hardware.usb.UsbEndpoint
import android.hardware.usb.UsbInterface
import android.hardware.usb.UsbManager
import android.hardware.usb.UsbRequest
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import java.io.Closeable
import java.io.IOException
import java.io.InputStream
import java.io.OutputStream
import java.security.SecureRandom
import java.nio.ByteBuffer
import java.util.concurrent.CompletableFuture
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.TimeoutException

internal interface NcmBoardStream : Closeable {
    val input: InputStream
    val output: OutputStream
    fun connect(timeout: Int)
}

/** Permission survives an Activity's temporary inactive state while Android shows its dialog. */
internal class NcmUsbPermission(private val context: Context) : Closeable {
    private val manager = context.getSystemService(Context.USB_SERVICE) as UsbManager
    private val action = "${context.packageName}.USB_NCM_PERMISSION"
    private val main = Handler(Looper.getMainLooper())
    private var pending: CompletableFuture<Boolean>? = null
    private var device: UsbDevice? = null
    private var denied: String? = null
    private var closed = false
    private val receiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            if (intent.action != action) return
            synchronized(this@NcmUsbPermission) {
                val target = device ?: return
                val granted = manager.hasPermission(target)
                if (!granted) denied = target.deviceName
                pending?.complete(granted)
            }
        }
    }
    init {
        if (Build.VERSION.SDK_INT >= 33) context.registerReceiver(receiver, IntentFilter(action), Context.RECEIVER_NOT_EXPORTED)
        else @Suppress("DEPRECATION") context.registerReceiver(receiver, IntentFilter(action))
    }
    fun await(target: UsbDevice, cancelled: () -> Boolean) {
        if (manager.hasPermission(target)) return
        val future = synchronized(this) {
            check(!closed) { "USB permission manager closed" }
            if (denied == target.deviceName) throw IOException("USB permission denied. Tap Reconnect and choose Allow in Android's USB dialog.")
            if (pending == null || pending!!.isDone) {
                device = target
                pending = CompletableFuture()
                val created = pending!!
                main.post {
                    synchronized(this) {
                        if (!closed && pending === created && !cancelled()) {
                            try {
                                val intent = Intent(action).setPackage(context.packageName)
                                val permission = PendingIntent.getBroadcast(context, target.deviceId,
                                    intent, PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE)
                                manager.requestPermission(target, permission)
                            } catch (error: Exception) { created.completeExceptionally(error) }
                        } else created.complete(false)
                    }
                }
            }
            require(device?.deviceName == target.deviceName) { "Another USB permission request is active" }
            pending!!
        }
        val deadline = SystemClock.elapsedRealtime() + 120000
        while (SystemClock.elapsedRealtime() < deadline) {
            if (cancelled()) throw IOException("USB connection cancelled")
            try {
                if (!future.get(250, TimeUnit.MILLISECONDS)) throw IOException("USB permission denied. Tap Reconnect and allow camera USB access.")
                if (!manager.hasPermission(target)) throw IOException("USB access was not granted")
                return
            } catch (_: TimeoutException) { /* The user may still be reading the system dialog. */ }
        }
        throw IOException("USB permission prompt timed out. Tap Reconnect and allow access.")
    }
    @Synchronized fun retry() { denied = null }
    @Synchronized override fun close() {
        closed = true; pending?.complete(false)
        runCatching { context.unregisterReceiver(receiver) }
    }
}

/** Direct NTB16/ARP/IPv4/TCP transport for the verified 1a86:df07 board, 192.168.50.2:5000. */
internal class UsbNcmStream(
    private val manager: UsbManager, private val device: UsbDevice,
    private val stage: (String) -> Unit,
) : NcmBoardStream {
    @Volatile private var closed = false
    @Volatile private var finished = false
    @Volatile private var failure: Throwable? = null
    @Volatile private var connection: UsbDeviceConnection? = null
    private var control: UsbInterface? = null
    private var data: UsbInterface? = null
    private var idle: UsbInterface? = null
    private var endpointIn: UsbEndpoint? = null
    private var endpointOut: UsbEndpoint? = null
    private var codec: NcmNtbCodec? = null
    private var tcp: NcmCameraTcp? = null
    private val receiveWorker = Executors.newSingleThreadExecutor()
    private val notificationWorker = Executors.newSingleThreadExecutor()
    @Volatile private var notificationRequest: UsbRequest? = null
    private val txLock = Any()
    private val signal = Object()

    override fun connect(timeout: Int) {
        check(!closed)
        stage("USB permission granted; opening CDC-NCM")
        val conn = manager.openDevice(device) ?: throw IOException("Android could not open the USB-NCM camera")
        synchronized(this) {
            if (closed) { conn.close(); throw IOException("USB connection cancelled") }
            connection = conn
        }
        try {
            val interfaces = (0 until device.interfaceCount).map(device::getInterface)
            val ctl = interfaces.first { it.interfaceClass == 2 && it.interfaceSubclass == 13 }
            val function = NcmFunction.parse(conn.rawDescriptors, ctl.id)
            val active = interfaces.first { it.id == function.data && it.alternateSetting == 1 && it.interfaceClass == 10 }
            val inactive = interfaces.first { it.id == function.data && it.alternateSetting == 0 }
            control = ctl; data = active; idle = inactive
            stage("Claiming NCM control ${ctl.id} and data ${active.id} with force=true")
            check(conn.claimInterface(ctl, true)) { "Could not claim NCM control interface; another USB owner may be active" }
            check(conn.claimInterface(active, true)) { "Could not claim NCM data interface; another USB owner may be active" }
            check(conn.setInterface(inactive)) { "Could not select NCM alternate setting 0" }
            val endpoints = (0 until active.endpointCount).map(active::getEndpoint)
            endpointIn = endpoints.first { it.type == UsbConstants.USB_ENDPOINT_XFER_BULK && it.direction == UsbConstants.USB_DIR_IN }
            endpointOut = endpoints.first { it.type == UsbConstants.USB_ENDPOINT_XFER_BULK && it.direction == UsbConstants.USB_DIR_OUT }
            stage("Reading CDC-NCM parameters on control ${ctl.id}, data ${active.id}/alt1")
            val paramsBytes = ByteArray(28)
            check(conn.controlTransfer(0xa1, 0x80, 0, ctl.id, paramsBytes, 28, 1500) >= 28) { "GET_NTB_PARAMETERS failed" }
            val parameters = NcmParameters.parse(paramsBytes)
            stage("NTB16 parameters: IN=${parameters.inputSize}, OUT=${parameters.outputSize}, divisor=${parameters.divisor}, remainder=${parameters.remainder}, alignment=${parameters.alignment}")
            // Format selection is optional for NTB16-only functions; those
            // devices may legitimately stall GET/SET_NTB_FORMAT.
            if (NcmWire.le16(paramsBytes, 2) and 2 != 0) {
                val formatResult = command(0x84, 0)
                if (formatResult < 0) {
                    val format = ByteArray(2)
                    check(conn.controlTransfer(0xa1, 0x83, 0, ctl.id, format, 2, 1000) == 2 && NcmWire.le16(format, 0) == 0) {
                        "Camera did not accept NTB16 format"
                    }
                }
            }
            if (function.capabilities and 16 != 0) check(command(0x8a, 0) >= 0) { "Could not disable NCM CRC mode" }
            val inputSize = ByteArray(4).also { NcmWire.putLe32(it, 0, parameters.inputSize.toLong()) }
            check(command(0x86, 0, inputSize) == 4) { "SET_NTB_INPUT_SIZE failed" }
            if (function.capabilities and 8 != 0) {
                val segment = ByteArray(2).also { NcmWire.putLe16(it, 0, minOf(function.maxSegment, 1514)) }
                check(command(0x88, 0, segment) == 2) { "SET_MAX_DATAGRAM_SIZE failed" }
            }
            val mac = readMac(function.macString)
            if (command(0x43, 12) < 0 && function.capabilities and 1 != 0) throw IOException("SET_ETHERNET_PACKET_FILTER failed")
            codec = NcmNtbCodec(parameters, endpointOut!!.maxPacketSize)
            check(conn.setInterface(active)) { "Could not activate NCM alternate setting 1" }
            val notification = (0 until ctl.endpointCount).map(ctl::getEndpoint).firstOrNull {
                it.type == UsbConstants.USB_ENDPOINT_XFER_INT && it.direction == UsbConstants.USB_DIR_IN
            }
            if (notification != null) notificationWorker.execute { notifications(conn, notification) }
            val random = SecureRandom()
            tcp = NcmCameraTcp(mac, 0xc0a83201L, 0xc0a83202L, 49152 + random.nextInt(16384), 5000,
                random.nextInt().toLong() and 0xffffffffL, SystemClock::elapsedRealtime, ::sendFrame,
                minOf(1400, function.maxSegment - 54))
            stage("USB-NCM active; resolving 192.168.50.2 from 192.168.50.1 (ARP)")
            receiveWorker.execute { receive(parameters.inputSize) }
            val deadline = SystemClock.elapsedRealtime() + 18000
            while (!closed && tcp!!.state != NcmCameraTcp.State.ESTABLISHED) {
                failure?.let { throw IOException("USB-NCM receive failed: ${it.message}", it) }
                tcp!!.checkAlive()
                if (tcp!!.state == NcmCameraTcp.State.CLOSED) throw IOException("Board closed TCP during USB-NCM connection setup")
                if (SystemClock.elapsedRealtime() > deadline) throw IOException("USB-NCM ${tcp!!.state} timed out; check board IP 192.168.50.2 and TCP service 5000")
                synchronized(signal) { signal.wait(25) }
            }
            if (closed) throw IOException("USB connection cancelled")
            stage("Native USB-NCM TCP connected to 192.168.50.2:5000")
        } catch (error: Exception) { stage("USB-NCM setup failed: ${error.message}"); close(); throw error }
    }
    private fun command(request: Int, value: Int, bytes: ByteArray? = null): Int =
        connection!!.controlTransfer(0x21, request, value, control!!.id, bytes, bytes?.size ?: 0, 1500)

    private fun readMac(index: Int): ByteArray {
        val b = ByteArray(255)
        val conn = connection!!
        val languageCount = conn.controlTransfer(0x80, 6, 0x0300, 0, b, b.size, 1000)
        val language = if (languageCount >= 4) NcmWire.le16(b, 2) else 0x0409
        val n = conn.controlTransfer(0x80, 6, 0x0300 or index, language, b, b.size, 1000)
        check(n >= 2 && NcmWire.u8(b, 1) == 3) { "Could not read CDC Ethernet MAC string" }
        val length = minOf(n, NcmWire.u8(b, 0))
        check(length >= 2 && length % 2 == 0) { "Invalid CDC MAC string descriptor" }
        val text = String(b, 2, length - 2, Charsets.UTF_16LE).replace(":", "").trim()
        check(text.matches(Regex("[0-9a-fA-F]{12}"))) { "Invalid CDC Ethernet MAC address: $text" }
        val mac = ByteArray(6) { text.substring(it * 2, it * 2 + 2).toInt(16).toByte() }
        check(NcmWire.u8(mac, 0) and 1 == 0 && mac.any { it != 0.toByte() }) { "Invalid multicast/zero host MAC" }
        return mac
    }
    private fun sendFrame(frame: ByteArray) = synchronized(txLock) {
        if (closed) throw IOException("USB connection closed")
        val block = codec!!.encode(frame)
        val count = connection!!.bulkTransfer(endpointOut, block, block.size, 1500)
        if (count != block.size) throw IOException("USB bulk OUT failed: $count/${block.size}")
    }
    private fun notifications(conn: UsbDeviceConnection, endpoint: UsbEndpoint) {
        val request = UsbRequest()
        notificationRequest = request
        try {
            check(request.initialize(conn, endpoint)) { "NCM interrupt endpoint initialization failed" }
            val b = ByteBuffer.allocate(endpoint.maxPacketSize.coerceAtLeast(16))
            while (!closed) {
                b.clear()
                @Suppress("DEPRECATION")
                val queued = if (Build.VERSION.SDK_INT >= 26) request.queue(b) else request.queue(b, b.capacity())
                if (!queued || conn.requestWait() == null) break
                val size = b.position()
                if (size >= 8 && b.get(0).toInt() and 255 == 0xa1) {
                    val type = b.get(1).toInt() and 255
                    if (type == 0) stage("CDC-NCM link notification: ${if (b.get(2).toInt() != 0) "up" else "down"}")
                }
            }
        } catch (error: Exception) { if (!closed) stage("NCM notification endpoint: ${error.message}") }
        finally { runCatching { request.close() } }
    }
    private fun receive(size: Int) {
        try {
            val bytes = ByteArray(size)
            var lastState = NcmCameraTcp.State.ARP
            var payloadSeen = false
            while (!closed && !finished) {
                val n = connection!!.bulkTransfer(endpointIn, bytes, bytes.size, 100)
                if (n > 0) for (frame in codec!!.feed(bytes.copyOf(n))) tcp!!.input(frame)
                else if (manager.deviceList[device.deviceName]?.deviceId != device.deviceId) throw IOException("Camera USB disconnected")
                tcp!!.tick(); tcp!!.checkAlive()
                if (tcp!!.state != lastState) {
                    lastState = tcp!!.state; stage("USB-NCM TCP state: $lastState")
                }
                if (!payloadSeen && tcp!!.receivedBytes > 0) {
                    payloadSeen = true; stage("Board TCP payload received; forwarding to JLIP decoder")
                }
                if (tcp!!.state == NcmCameraTcp.State.CLOSED) finished = true
                synchronized(signal) { signal.notifyAll() }
            }
        } catch (error: Exception) {
            if (!closed) { failure = error; stage("USB-NCM stream failed: ${error.message}") }
        } finally { finished = true; synchronized(signal) { signal.notifyAll() } }
    }
    override val input = object : InputStream() {
        private var pending = ByteArray(0)
        private var offset = 0
        override fun read(): Int { val b = ByteArray(1); return if (read(b, 0, 1) < 0) -1 else b[0].toInt() and 255 }
        override fun read(b: ByteArray, off: Int, len: Int): Int {
            require(off >= 0 && len >= 0 && off + len <= b.size)
            if (len == 0) return 0
            while (offset == pending.size) {
                failure?.let { throw IOException(it.message, it) }
                val next = tcp?.read()
                if (next != null) { pending = next; offset = 0; break }
                if (closed || finished) return -1
                synchronized(signal) { signal.wait(25) }
            }
            val count = minOf(len, pending.size - offset)
            pending.copyInto(b, off, offset, offset + count); offset += count
            return count
        }
    }
    override val output = object : OutputStream() {
        override fun write(b: Int) = write(byteArrayOf(b.toByte()), 0, 1)
        override fun write(b: ByteArray, off: Int, len: Int) {
            require(off >= 0 && len >= 0 && off + len <= b.size)
            var p = off
            while (p < off + len) {
                val end = minOf(p + 1400, off + len)
                val bytes = b.copyOfRange(p, end)
                while (tcp?.write(bytes) != true) {
                    failure?.let { throw IOException(it.message, it) }
                    if (closed || finished) throw IOException("USB-NCM TCP stream closed")
                    synchronized(signal) { signal.wait(25) }
                }
                p = end
            }
        }
    }
    @Synchronized override fun close() {
        if (closed) return
        runCatching { tcp?.abort() }
        closed = true
        val conn = connection
        // Closing the fd cancels bounded bulk transfers even if the board was unplugged.
        runCatching { notificationRequest?.cancel() }
        runCatching { idle?.let { conn?.setInterface(it) } }
        runCatching { data?.let { conn?.releaseInterface(it) } }
        runCatching { control?.let { conn?.releaseInterface(it) } }
        runCatching { conn?.close() }
        receiveWorker.shutdownNow()
        notificationWorker.shutdownNow()
        synchronized(signal) { signal.notifyAll() }
    }
}
