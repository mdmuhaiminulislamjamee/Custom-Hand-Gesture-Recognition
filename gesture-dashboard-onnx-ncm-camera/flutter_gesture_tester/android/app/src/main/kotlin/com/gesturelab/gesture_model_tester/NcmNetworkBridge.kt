package com.gesturelab.gesture_model_tester

import android.content.Context
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import android.hardware.usb.UsbConstants
import android.hardware.usb.UsbManager
import android.os.Handler
import android.os.Looper
import io.flutter.plugin.common.BinaryMessenger
import io.flutter.plugin.common.MethodChannel
import java.io.Closeable
import java.io.InputStream
import java.io.OutputStream
import java.net.Inet4Address
import java.net.InetAddress
import java.net.InetSocketAddress
import java.net.ServerSocket
import java.net.Socket
import java.security.MessageDigest
import java.security.SecureRandom
import java.util.concurrent.Executors

/** Binds only the board socket. Dart retains JLIP parsing and heartbeats. */
class NcmNetworkBridge(context: Context, messenger: BinaryMessenger) : Closeable {
    private val connectivity = context.getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager
    private val channel = MethodChannel(messenger, "gesture_lab/ncm_network")
    private val handler = Handler(Looper.getMainLooper())
    private val workers = Executors.newCachedThreadPool()
    private val usbManager = context.getSystemService(Context.USB_SERVICE) as UsbManager
    private val usbPermission = NcmUsbPermission(context.applicationContext)
    private val usbTrace = java.util.ArrayDeque<String>()
    private var active: Relay? = null
    @Volatile private var generation = 0
    @Volatile private var disposed = false

    init {
        channel.setMethodCallHandler { call, result ->
            when (call.method) {
                "describeUsb" -> {
                    try {
                        val description = describeUsbDevices(context)
                        description.forEach { android.util.Log.i("USB_DESCRIPTOR", it) }
                        result.success(description)
                    } catch (error: Exception) {
                        result.error("USB_INSPECTION_FAILED", error.message, null)
                    }
                }
                "open" -> {
                    val host = call.argument<String>("host") ?: "192.168.50.2"
                    val port = call.argument<Int>("port") ?: 5000
                    val timeout = (call.argument<Int>("timeoutMs") ?: 3000).coerceIn(100, 15000)
                    if (!host.matches(Regex("[0-9]{1,3}(\\.[0-9]{1,3}){3}")) || port !in 1..65535) {
                        result.error("NCM_ADDRESS_INVALID", "Invalid board IPv4 address or TCP port.", null)
                    } else {
                        stop()
                        val epoch = generation
                        workers.execute { open(host, port, timeout, epoch, result) }
                    }
                }
                "close" -> { stop(); usbPermission.retry(); result.success(null) }
                else -> result.notImplemented()
            }
        }
    }

    private fun describeUsbDevices(context: Context): List<String> {
        val manager = context.getSystemService(Context.USB_SERVICE) as UsbManager
        val devices = manager.deviceList.values.toList()
        val rows = mutableListOf("USB devices currently attached: ${devices.size}",
            "Transport revision: v2.0.13, direct USB-NCM fallback with forced interface claims enabled")
        if (devices.isEmpty()) rows += "No USB device is currently enumerated. Reconnect the powered camera with OTG enabled, then inspect again."
        rows += devices.flatMap { device ->
            val rows = mutableListOf(
                "device=${device.deviceName} vid=${device.vendorId} pid=${device.productId} " +
                    "vidHex=0x${device.vendorId.toString(16)} pidHex=0x${device.productId.toString(16)} " +
                    "class=${device.deviceClass} subclass=${device.deviceSubclass} protocol=${device.deviceProtocol} " +
                    "permission=${manager.hasPermission(device)}"
            )
            for (index in 0 until device.interfaceCount) {
                val usbInterface = device.getInterface(index)
                rows += "interface=$index id=${usbInterface.id} alternate=${usbInterface.alternateSetting} class=${usbInterface.interfaceClass} " +
                    "subclass=${usbInterface.interfaceSubclass} protocol=${usbInterface.interfaceProtocol} " +
                    "endpoints=${usbInterface.endpointCount}"
                for (endpointIndex in 0 until usbInterface.endpointCount) {
                    val endpoint = usbInterface.getEndpoint(endpointIndex)
                    val direction = if (endpoint.direction == UsbConstants.USB_DIR_IN) "IN" else "OUT"
                    val type = when (endpoint.type) {
                        UsbConstants.USB_ENDPOINT_XFER_BULK -> "BULK"
                        UsbConstants.USB_ENDPOINT_XFER_INT -> "INTERRUPT"
                        UsbConstants.USB_ENDPOINT_XFER_ISOC -> "ISOCHRONOUS"
                        else -> "CONTROL"
                    }
                    rows += "endpoint=$endpointIndex type=$type direction=$direction " +
                        "address=${endpoint.address} maxPacket=${endpoint.maxPacketSize}"
                }
            }
            rows
        }
        synchronized(usbTrace) { rows += usbTrace.toList() }
        return rows
    }

    private fun usbStage(message: String) {
        android.util.Log.i("USB_NCM", message)
        synchronized(usbTrace) {
            if (usbTrace.size >= 24) usbTrace.removeFirst()
            usbTrace.addLast(message)
        }
    }

    private fun open(host: String, port: Int, timeout: Int, epoch: Int, result: MethodChannel.Result) {
        var relay: Relay? = null
        var delivered = false
        try {
            val address = InetAddress.getByName(host)
            require(address is Inet4Address) { "Board requires an IPv4 address" }
            val candidates = connectivity.allNetworks.mapNotNull { network ->
                val caps = connectivity.getNetworkCapabilities(network) ?: return@mapNotNull null
                val links = connectivity.getLinkProperties(network) ?: return@mapNotNull null
                if (!caps.hasTransport(NetworkCapabilities.TRANSPORT_ETHERNET)) return@mapNotNull null
                Triple(network, links, links.routes.filter { it.matches(address) }
                    .maxOfOrNull { it.destination.prefixLength } ?: -1)
            }
            val diagnostic = candidates.joinToString("; ") { (_, links, _) ->
                "${links.interfaceName}: ${links.linkAddresses.joinToString()} routes=${links.routes.joinToString()}"
            }
            val selected = candidates.filter { it.third >= 0 }.maxByOrNull { it.third }
            val remote: NcmBoardStream
            val interfaceName: String
            val addresses: List<String>
            val networkName: String
            if (selected != null) {
                val socket = selected.first.socketFactory.createSocket()
                remote = object : NcmBoardStream {
                    override val input: InputStream get() = socket.getInputStream()
                    override val output: OutputStream get() = socket.getOutputStream()
                    override fun connect(timeout: Int) { socket.tcpNoDelay = true; socket.connect(InetSocketAddress(address, port), timeout) }
                    override fun close() { socket.close() }
                }
                interfaceName = selected.second.interfaceName ?: "Ethernet"
                addresses = selected.second.linkAddresses.map { it.toString() }
                networkName = selected.first.toString()
            } else {
                val devices = usbManager.deviceList.values.filter { device ->
                    device.vendorId == 0x1a86 && device.productId == 0xdf07 &&
                        (0 until device.interfaceCount).any { i -> device.getInterface(i).let { it.interfaceClass == 2 && it.interfaceSubclass == 13 } }
                }
                if (devices.size != 1 || host != "192.168.50.2" || port != 5000) throw RouteFailure(
                    "NCM_NETWORK_UNAVAILABLE", "No Ethernet route or single supported USB-NCM camera (1a86:df07) is attached. $diagnostic")
                val device = devices.single()
                usbStage("No Android Ethernet route; using native USB-NCM for ${device.deviceName}")
                if (!usbManager.hasPermission(device)) usbStage("Waiting for Android USB permission; choose Allow")
                usbPermission.await(device) { disposed || epoch != generation }
                remote = UsbNcmStream(usbManager, device, ::usbStage)
                interfaceName = "Native USB-NCM (1a86:df07)"
                addresses = listOf("192.168.50.1/30")
                networkName = "userspace USB"
            }
            relay = Relay(remote)
            synchronized(this) {
                check(!disposed && epoch == generation) { "NCM connection cancelled" }
                active = relay
            }
            remote.connect(timeout)
            val listener = ServerSocket()
            relay.attachListener(listener)
            listener.bind(InetSocketAddress(InetAddress.getByName("127.0.0.1"), 0), 1)
            listener.soTimeout = 5000
            val token = ByteArray(32).also { SecureRandom().nextBytes(it) }
                .joinToString("") { "%02x".format(it) }
            val opened = relay
            val endpointPort = listener.localPort
            delivered = true
            handler.post {
                if (disposed || epoch != generation) {
                    opened.close()
                    result.error("NCM_CANCELLED", "NCM connection cancelled", null)
                } else result.success(mapOf(
                    "port" to endpointPort, "token" to token,
                    "interface" to interfaceName,
                    "addresses" to addresses,
                    "board" to "$host:$port", "network" to networkName
                ))
            }
            val local = listener.accept()
            relay.attachLocal(local)
            listener.close()
            local.soTimeout = 5000
            val received = ByteArray(64)
            java.io.DataInputStream(local.getInputStream()).readFully(received)
            check(MessageDigest.isEqual(received, token.toByteArray(Charsets.US_ASCII))) { "Invalid local relay token" }
            local.soTimeout = 0
            local.tcpNoDelay = true
            val localInput = local.getInputStream(); val localOutput = local.getOutputStream()
            val remoteInput = remote.input; val remoteOutput = remote.output
            workers.execute { pump(localInput, remoteOutput, opened) }
            pump(remoteInput, localOutput, opened)
        } catch (error: Exception) {
            relay?.close()
            // After endpoint delivery, socket closure tells Dart to reconnect.
            if (!delivered) handler.post {
                result.error((error as? RouteFailure)?.code ?: "NCM_CONNECT_FAILED", error.message, null)
            }
        }
    }

    private fun pump(source: InputStream, target: OutputStream, relay: Relay) {
        try { source.copyTo(target, 32768) }
        catch (_: Exception) { /* Closing either side cancels both pumps. */ }
        finally { relay.close() }
    }

    @Synchronized private fun stop() {
        generation++
        active?.close()
        active = null
    }

    override fun close() {
        disposed = true
        stop()
        channel.setMethodCallHandler(null)
        usbPermission.close()
        workers.shutdownNow()
    }

    private class RouteFailure(val code: String, message: String) : Exception(message)
    private class Relay(val remote: NcmBoardStream) : Closeable {
        private var local: Socket? = null
        private var listener: ServerSocket? = null
        private var closed = false
        @Synchronized fun attachListener(value: ServerSocket) {
            if (closed) { value.close(); error("NCM connection cancelled") }
            listener = value
        }
        @Synchronized fun attachLocal(value: Socket) {
            if (closed) { value.close(); error("NCM connection cancelled") }
            local = value
        }
        @Synchronized override fun close() {
            if (closed) return
            closed = true
            runCatching { listener?.close() }
            runCatching { local?.close() }
            runCatching { remote.close() }
        }
    }
}
