package com.example.gesturehandover

import android.net.ConnectivityManager
import android.net.Network
import android.net.NetworkCapabilities
import java.net.Inet4Address

/** Finds an OS-provided Ethernet/NCM route already configured on 192.168.50.1/30. */
object NcmNetworkSelector {
    fun find(connectivity: ConnectivityManager): Network? = connectivity.allNetworks.firstOrNull { network ->
        val caps = connectivity.getNetworkCapabilities(network)
        val links = connectivity.getLinkProperties(network)
        caps?.hasTransport(NetworkCapabilities.TRANSPORT_ETHERNET) == true &&
            links?.linkAddresses?.any { link ->
                val address = link.address as? Inet4Address
                val bytes = address?.address
                bytes != null && (bytes[0].toInt() and 0xff) == 192 &&
                    (bytes[1].toInt() and 0xff) == 168 &&
                    (bytes[2].toInt() and 0xff) == 50 &&
                    (bytes[3].toInt() and 0xff) == 1 && link.prefixLength == 30
            } == true
    }
}
