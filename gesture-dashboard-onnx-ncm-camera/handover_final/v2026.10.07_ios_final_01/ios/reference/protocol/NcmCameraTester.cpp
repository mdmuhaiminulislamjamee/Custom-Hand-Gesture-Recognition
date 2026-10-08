#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#define _WIN32_WINNT 0x0601

#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <iphlpapi.h>
#include <gdiplus.h>
#include <commdlg.h>
#include <commctrl.h>
#include <objidl.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using namespace Gdiplus;

namespace {

constexpr UINT WM_APP_EVENT = WM_APP + 1;
constexpr uint16_t kDefaultTcpPort = 5000;
constexpr uint16_t kDiscoveryPort = 5001;
constexpr size_t kJlipHeaderSize = 24;
constexpr size_t kMaxJpegSize = 200 * 1024;

constexpr uint8_t JLIP_HELLO_ACK = 0x02;
constexpr uint8_t JLIP_HEARTBEAT = 0x03;
constexpr uint8_t JLIP_HEARTBEAT_ACK = 0x04;
constexpr uint8_t JLIP_JPEG = 0x10;

enum ControlId {
    ID_IP_EDIT = 1001,
    ID_PORT_EDIT,
    ID_CONNECT,
    ID_DISCOVER,
    ID_CHECK_ADAPTER,
    ID_SAVE_JPEG,
    ID_AUTO_RECONNECT,
    ID_ROTATE_180,
};

enum class EventKind {
    Log,
    Connected,
    Disconnected,
    Packet,
    Discovery,
    WorkerStopped,
};

struct UiEvent {
    EventKind kind = EventKind::Log;
    std::wstring text;
    uint8_t packetType = 0;
    uint32_t sequence = 0;
    uint32_t timestampMs = 0;
    std::vector<uint8_t> payload;
};

HINSTANCE g_instance = nullptr;
HWND g_mainWindow = nullptr;
HWND g_ipEdit = nullptr;
HWND g_portEdit = nullptr;
HWND g_connectButton = nullptr;
HWND g_autoReconnectCheck = nullptr;
HWND g_rotate180Check = nullptr;
HWND g_adapterStatus = nullptr;
HWND g_connectionStatus = nullptr;
HWND g_deviceStatus = nullptr;
HWND g_frameStatus = nullptr;
HWND g_fpsStatus = nullptr;
HWND g_jpegStatus = nullptr;
HWND g_crcStatus = nullptr;
HWND g_sequenceStatus = nullptr;
HWND g_byteStatus = nullptr;
HWND g_logEdit = nullptr;

ULONG_PTR g_gdiplusToken = 0;
std::unique_ptr<Bitmap> g_bitmap;
std::vector<uint8_t> g_latestJpeg;
RECT g_imageRect{15, 60, 700, 560};
bool g_rotate180 = false;

std::atomic<bool> g_runNetwork{false};
std::atomic<bool> g_autoReconnect{true};
std::thread g_networkThread;
std::mutex g_socketMutex;
SOCKET g_currentSocket = INVALID_SOCKET;

uint64_t g_frameCount = 0;
uint64_t g_receivedBytes = 0;
uint64_t g_crcErrors = 0;
uint64_t g_sequenceGaps = 0;
uint32_t g_lastSequence = 0;
bool g_haveLastSequence = false;
uint32_t g_lastJpegSize = 0;
uint32_t g_lastWidth = 0;
uint32_t g_lastHeight = 0;
uint64_t g_lastFpsFrames = 0;
double g_currentFps = 0.0;
std::chrono::steady_clock::time_point g_lastFpsTime;

uint16_t ReadBe16(const uint8_t* p) {
    return static_cast<uint16_t>((static_cast<uint16_t>(p[0]) << 8) | p[1]);
}

uint32_t ReadBe32(const uint8_t* p) {
    return (static_cast<uint32_t>(p[0]) << 24) |
           (static_cast<uint32_t>(p[1]) << 16) |
           (static_cast<uint32_t>(p[2]) << 8) |
           static_cast<uint32_t>(p[3]);
}

void WriteBe16(uint8_t* p, uint16_t value) {
    p[0] = static_cast<uint8_t>(value >> 8);
    p[1] = static_cast<uint8_t>(value);
}

void WriteBe32(uint8_t* p, uint32_t value) {
    p[0] = static_cast<uint8_t>(value >> 24);
    p[1] = static_cast<uint8_t>(value >> 16);
    p[2] = static_cast<uint8_t>(value >> 8);
    p[3] = static_cast<uint8_t>(value);
}

uint32_t Crc32(const uint8_t* data, size_t len) {
    uint32_t crc = 0xffffffffu;
    for (size_t i = 0; i < len; ++i) {
        crc ^= data[i];
        for (int bit = 0; bit < 8; ++bit) {
            crc = (crc >> 1) ^ (0xedb88320u & (0u - (crc & 1u)));
        }
    }
    return ~crc;
}

std::vector<uint8_t> BuildJlipPacket(uint8_t type, uint32_t sequence,
                                     const uint8_t* payload, size_t payloadLen) {
    std::vector<uint8_t> packet(kJlipHeaderSize + payloadLen, 0);
    std::memcpy(packet.data(), "JLIP", 4);
    packet[4] = 1;
    packet[5] = type;
    WriteBe16(packet.data() + 6, static_cast<uint16_t>(kJlipHeaderSize));
    WriteBe32(packet.data() + 8, sequence);
    WriteBe32(packet.data() + 12, static_cast<uint32_t>(payloadLen));
    WriteBe32(packet.data() + 16, Crc32(payload, payloadLen));
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    WriteBe32(packet.data() + 20, static_cast<uint32_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(now).count()));
    if (payloadLen) {
        std::memcpy(packet.data() + kJlipHeaderSize, payload, payloadLen);
    }
    return packet;
}

std::wstring Utf8ToWide(const std::string& text) {
    if (text.empty()) {
        return {};
    }
    int count = MultiByteToWideChar(CP_UTF8, 0, text.data(),
                                    static_cast<int>(text.size()), nullptr, 0);
    if (count <= 0) {
        return std::wstring(text.begin(), text.end());
    }
    std::wstring result(static_cast<size_t>(count), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, text.data(), static_cast<int>(text.size()),
                        result.data(), count);
    return result;
}

std::string WideToUtf8(const std::wstring& text) {
    if (text.empty()) {
        return {};
    }
    int count = WideCharToMultiByte(CP_UTF8, 0, text.data(),
                                    static_cast<int>(text.size()), nullptr, 0,
                                    nullptr, nullptr);
    std::string result(static_cast<size_t>(count), '\0');
    WideCharToMultiByte(CP_UTF8, 0, text.data(), static_cast<int>(text.size()),
                        result.data(), count, nullptr, nullptr);
    return result;
}

std::wstring WsaErrorText(int code) {
    wchar_t* buffer = nullptr;
    FormatMessageW(FORMAT_MESSAGE_ALLOCATE_BUFFER |
                       FORMAT_MESSAGE_FROM_SYSTEM |
                       FORMAT_MESSAGE_IGNORE_INSERTS,
                   nullptr, static_cast<DWORD>(code), 0,
                   reinterpret_cast<wchar_t*>(&buffer), 0, nullptr);
    std::wstring result = buffer ? buffer : L"Unknown socket error";
    if (buffer) {
        LocalFree(buffer);
    }
    while (!result.empty() && (result.back() == L'\r' || result.back() == L'\n')) {
        result.pop_back();
    }
    return result;
}

void PostEvent(std::unique_ptr<UiEvent> event) {
    HWND hwnd = g_mainWindow;
    if (hwnd && IsWindow(hwnd)) {
        if (PostMessageW(hwnd, WM_APP_EVENT, 0,
                         reinterpret_cast<LPARAM>(event.get()))) {
            event.release();
        }
    }
}

void PostLog(const std::wstring& text) {
    auto event = std::make_unique<UiEvent>();
    event->kind = EventKind::Log;
    event->text = text;
    PostEvent(std::move(event));
}

void SetCurrentSocket(SOCKET socketValue) {
    std::lock_guard<std::mutex> lock(g_socketMutex);
    g_currentSocket = socketValue;
}

void CloseCurrentSocket() {
    SOCKET socketValue = INVALID_SOCKET;
    {
        std::lock_guard<std::mutex> lock(g_socketMutex);
        socketValue = g_currentSocket;
        g_currentSocket = INVALID_SOCKET;
    }
    if (socketValue != INVALID_SOCKET) {
        shutdown(socketValue, SD_BOTH);
        closesocket(socketValue);
    }
}

bool SendAll(SOCKET socketValue, const uint8_t* data, size_t len) {
    size_t offset = 0;
    while (offset < len && g_runNetwork.load()) {
        int sent = send(socketValue,
                        reinterpret_cast<const char*>(data + offset),
                        static_cast<int>(len - offset), 0);
        if (sent > 0) {
            offset += static_cast<size_t>(sent);
            continue;
        }
        int error = WSAGetLastError();
        if (error == WSAEWOULDBLOCK || error == WSAETIMEDOUT) {
            continue;
        }
        return false;
    }
    return offset == len;
}

bool SleepWhileRunning(DWORD milliseconds) {
    DWORD elapsed = 0;
    while (elapsed < milliseconds && g_runNetwork.load()) {
        Sleep(100);
        elapsed += 100;
    }
    return g_runNetwork.load();
}

SOCKET ConnectToDevice(const std::string& targetIp, uint16_t targetPort) {
    SOCKET socketValue = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (socketValue == INVALID_SOCKET) {
        PostLog(L"socket() failed: " + WsaErrorText(WSAGetLastError()));
        return INVALID_SOCKET;
    }
    SetCurrentSocket(socketValue);

    sockaddr_in local{};
    local.sin_family = AF_INET;
    local.sin_port = 0;
    inet_pton(AF_INET, "192.168.50.1", &local.sin_addr);
    if (bind(socketValue, reinterpret_cast<sockaddr*>(&local), sizeof(local)) == SOCKET_ERROR) {
        PostLog(L"Warning: could not bind to 192.168.50.1. Check the NCM adapter or set its IPv4 address manually.");
    }

    u_long nonBlocking = 1;
    ioctlsocket(socketValue, FIONBIO, &nonBlocking);

    sockaddr_in target{};
    target.sin_family = AF_INET;
    target.sin_port = htons(targetPort);
    if (inet_pton(AF_INET, targetIp.c_str(), &target.sin_addr) != 1) {
        PostLog(L"Invalid device IP address.");
        CloseCurrentSocket();
        return INVALID_SOCKET;
    }

    int result = connect(socketValue, reinterpret_cast<sockaddr*>(&target), sizeof(target));
    if (result == SOCKET_ERROR && WSAGetLastError() != WSAEWOULDBLOCK) {
        int error = WSAGetLastError();
        PostLog(L"Connect failed: " + WsaErrorText(error));
        CloseCurrentSocket();
        return INVALID_SOCKET;
    }

    fd_set writeSet;
    fd_set errorSet;
    FD_ZERO(&writeSet);
    FD_ZERO(&errorSet);
    FD_SET(socketValue, &writeSet);
    FD_SET(socketValue, &errorSet);
    timeval timeout{3, 0};
    result = select(0, nullptr, &writeSet, &errorSet, &timeout);
    if (result <= 0 || FD_ISSET(socketValue, &errorSet)) {
        int socketError = WSAETIMEDOUT;
        int length = sizeof(socketError);
        getsockopt(socketValue, SOL_SOCKET, SO_ERROR,
                   reinterpret_cast<char*>(&socketError), &length);
        if (!socketError) {
            socketError = WSAETIMEDOUT;
        }
        PostLog(L"Connect failed: " + WsaErrorText(socketError));
        CloseCurrentSocket();
        return INVALID_SOCKET;
    }

    nonBlocking = 0;
    ioctlsocket(socketValue, FIONBIO, &nonBlocking);
    DWORD receiveTimeoutMs = 500;
    DWORD sendTimeoutMs = 1000;
    setsockopt(socketValue, SOL_SOCKET, SO_RCVTIMEO,
               reinterpret_cast<const char*>(&receiveTimeoutMs), sizeof(receiveTimeoutMs));
    setsockopt(socketValue, SOL_SOCKET, SO_SNDTIMEO,
               reinterpret_cast<const char*>(&sendTimeoutMs), sizeof(sendTimeoutMs));
    BOOL noDelay = TRUE;
    setsockopt(socketValue, IPPROTO_TCP, TCP_NODELAY,
               reinterpret_cast<const char*>(&noDelay), sizeof(noDelay));
    return socketValue;
}

bool ReceiveJlipStream(SOCKET socketValue) {
    std::vector<uint8_t> receiveBuffer;
    receiveBuffer.reserve(256 * 1024);
    uint32_t clientSequence = 1;
    uint8_t temporary[16384];

    while (g_runNetwork.load()) {
        int received = recv(socketValue, reinterpret_cast<char*>(temporary),
                            sizeof(temporary), 0);
        if (received == 0) {
            PostLog(L"Device closed the TCP connection.");
            return false;
        }
        if (received < 0) {
            int error = WSAGetLastError();
            if (error == WSAETIMEDOUT || error == WSAEWOULDBLOCK) {
                continue;
            }
            if (g_runNetwork.load()) {
                PostLog(L"Receive failed: " + WsaErrorText(error));
            }
            return false;
        }

        receiveBuffer.insert(receiveBuffer.end(), temporary, temporary + received);
        while (receiveBuffer.size() >= kJlipHeaderSize) {
            if (std::memcmp(receiveBuffer.data(), "JLIP", 4) != 0) {
                auto magic = std::search(receiveBuffer.begin() + 1, receiveBuffer.end(),
                                         reinterpret_cast<const uint8_t*>("JLIP"),
                                         reinterpret_cast<const uint8_t*>("JLIP") + 4);
                receiveBuffer.erase(receiveBuffer.begin(), magic);
                continue;
            }

            uint16_t headerLength = ReadBe16(receiveBuffer.data() + 6);
            uint32_t payloadLength = ReadBe32(receiveBuffer.data() + 12);
            if (receiveBuffer[4] != 1 || headerLength != kJlipHeaderSize ||
                payloadLength > kMaxJpegSize) {
                PostLog(L"Invalid JLIP header; searching for the next packet.");
                receiveBuffer.erase(receiveBuffer.begin());
                continue;
            }

            size_t packetLength = static_cast<size_t>(headerLength) + payloadLength;
            if (receiveBuffer.size() < packetLength) {
                break;
            }

            uint8_t type = receiveBuffer[5];
            uint32_t sequence = ReadBe32(receiveBuffer.data() + 8);
            uint32_t expectedCrc = ReadBe32(receiveBuffer.data() + 16);
            uint32_t timestamp = ReadBe32(receiveBuffer.data() + 20);
            const uint8_t* payload = receiveBuffer.data() + headerLength;
            uint32_t actualCrc = Crc32(payload, payloadLength);

            auto event = std::make_unique<UiEvent>();
            event->kind = EventKind::Packet;
            event->packetType = type;
            event->sequence = sequence;
            event->timestampMs = timestamp;
            if (expectedCrc == actualCrc) {
                event->payload.assign(payload, payload + payloadLength);
            } else {
                event->text = L"CRC_ERROR";
            }
            PostEvent(std::move(event));

            if (type == JLIP_HEARTBEAT) {
                auto heartbeat = BuildJlipPacket(JLIP_HEARTBEAT, clientSequence++, nullptr, 0);
                if (!SendAll(socketValue, heartbeat.data(), heartbeat.size())) {
                    PostLog(L"Heartbeat reply failed.");
                    return false;
                }
            }

            receiveBuffer.erase(receiveBuffer.begin(),
                                receiveBuffer.begin() + static_cast<std::ptrdiff_t>(packetLength));
        }

        if (receiveBuffer.size() > (kMaxJpegSize + kJlipHeaderSize) * 2) {
            PostLog(L"Receive buffer overflow; reconnecting.");
            return false;
        }
    }
    return true;
}

void NetworkWorker(std::string targetIp, uint16_t targetPort) {
    while (g_runNetwork.load()) {
        PostLog(L"Connecting to " + Utf8ToWide(targetIp) + L":" +
                std::to_wstring(targetPort) + L"...");
        SOCKET socketValue = ConnectToDevice(targetIp, targetPort);
        if (socketValue != INVALID_SOCKET) {
            auto connected = std::make_unique<UiEvent>();
            connected->kind = EventKind::Connected;
            connected->text = Utf8ToWide(targetIp) + L":" + std::to_wstring(targetPort);
            PostEvent(std::move(connected));
            ReceiveJlipStream(socketValue);
            CloseCurrentSocket();
            auto disconnected = std::make_unique<UiEvent>();
            disconnected->kind = EventKind::Disconnected;
            PostEvent(std::move(disconnected));
        }

        if (!g_runNetwork.load() || !g_autoReconnect.load()) {
            break;
        }
        PostLog(L"Reconnecting in 1 second...");
        if (!SleepWhileRunning(1000)) {
            break;
        }
    }

    g_runNetwork.store(false);
    auto stopped = std::make_unique<UiEvent>();
    stopped->kind = EventKind::WorkerStopped;
    PostEvent(std::move(stopped));
}

void StopNetwork() {
    g_runNetwork.store(false);
    CloseCurrentSocket();
    if (g_networkThread.joinable()) {
        g_networkThread.join();
    }
}

void StartNetwork(const std::wstring& targetIp, uint16_t targetPort) {
    StopNetwork();
    g_runNetwork.store(true);
    g_networkThread = std::thread(NetworkWorker, WideToUtf8(targetIp), targetPort);
}

std::wstring FindNcmAdapter() {
    ULONG size = 16384;
    std::vector<uint8_t> buffer(size);
    auto* addresses = reinterpret_cast<IP_ADAPTER_ADDRESSES*>(buffer.data());
    ULONG result = GetAdaptersAddresses(AF_INET,
                                        GAA_FLAG_SKIP_ANYCAST |
                                            GAA_FLAG_SKIP_MULTICAST |
                                            GAA_FLAG_SKIP_DNS_SERVER,
                                        nullptr, addresses, &size);
    if (result == ERROR_BUFFER_OVERFLOW) {
        buffer.resize(size);
        addresses = reinterpret_cast<IP_ADAPTER_ADDRESSES*>(buffer.data());
        result = GetAdaptersAddresses(AF_INET,
                                     GAA_FLAG_SKIP_ANYCAST |
                                         GAA_FLAG_SKIP_MULTICAST |
                                         GAA_FLAG_SKIP_DNS_SERVER,
                                     nullptr, addresses, &size);
    }
    if (result != NO_ERROR) {
        return L"Adapter query failed: " + std::to_wstring(result);
    }

    for (auto* adapter = addresses; adapter; adapter = adapter->Next) {
        for (auto* unicast = adapter->FirstUnicastAddress; unicast; unicast = unicast->Next) {
            if (!unicast->Address.lpSockaddr ||
                unicast->Address.lpSockaddr->sa_family != AF_INET) {
                continue;
            }
            auto* address = reinterpret_cast<sockaddr_in*>(unicast->Address.lpSockaddr);
            wchar_t ip[INET_ADDRSTRLEN]{};
            InetNtopW(AF_INET, &address->sin_addr, ip, INET_ADDRSTRLEN);
            if (std::wcscmp(ip, L"192.168.50.1") == 0) {
                std::wstring name = adapter->FriendlyName ? adapter->FriendlyName : L"NCM adapter";
                return L"OK: " + name + L" (192.168.50.1)";
            }
        }
    }
    return L"NOT FOUND: Windows needs 192.168.50.1/30";
}

void DiscoverWorker(std::string targetIp) {
    SOCKET socketValue = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (socketValue == INVALID_SOCKET) {
        PostLog(L"UDP discovery socket failed.");
        return;
    }

    DWORD timeoutMs = 1200;
    setsockopt(socketValue, SOL_SOCKET, SO_RCVTIMEO,
               reinterpret_cast<const char*>(&timeoutMs), sizeof(timeoutMs));

    sockaddr_in local{};
    local.sin_family = AF_INET;
    local.sin_port = 0;
    inet_pton(AF_INET, "192.168.50.1", &local.sin_addr);
    bind(socketValue, reinterpret_cast<sockaddr*>(&local), sizeof(local));

    sockaddr_in target{};
    target.sin_family = AF_INET;
    target.sin_port = htons(kDiscoveryPort);
    inet_pton(AF_INET, targetIp.c_str(), &target.sin_addr);
    const char request[] = "JL-CAMERA-DISCOVER";
    sendto(socketValue, request, sizeof(request) - 1, 0,
           reinterpret_cast<sockaddr*>(&target), sizeof(target));

    char response[512]{};
    sockaddr_in peer{};
    int peerLength = sizeof(peer);
    int received = recvfrom(socketValue, response, sizeof(response) - 1, 0,
                            reinterpret_cast<sockaddr*>(&peer), &peerLength);
    auto event = std::make_unique<UiEvent>();
    event->kind = EventKind::Discovery;
    if (received > 0) {
        response[received] = '\0';
        event->text = Utf8ToWide(std::string(response, response + received));
    } else {
        event->text = L"No UDP discovery reply. Check the NCM adapter and firmware link.";
    }
    closesocket(socketValue);
    PostEvent(std::move(event));
}

std::unique_ptr<Bitmap> DecodeJpeg(const std::vector<uint8_t>& jpeg) {
    if (jpeg.size() < 4 || jpeg[0] != 0xff || jpeg[1] != 0xd8) {
        return nullptr;
    }
    HGLOBAL memory = GlobalAlloc(GMEM_MOVEABLE, jpeg.size());
    if (!memory) {
        return nullptr;
    }
    void* destination = GlobalLock(memory);
    if (!destination) {
        GlobalFree(memory);
        return nullptr;
    }
    std::memcpy(destination, jpeg.data(), jpeg.size());
    GlobalUnlock(memory);

    IStream* stream = nullptr;
    if (CreateStreamOnHGlobal(memory, TRUE, &stream) != S_OK) {
        GlobalFree(memory);
        return nullptr;
    }

    std::unique_ptr<Bitmap> result;
    {
        Bitmap source(stream);
        if (source.GetLastStatus() == Ok && source.GetWidth() && source.GetHeight()) {
            Bitmap* clone = source.Clone(0, 0, source.GetWidth(), source.GetHeight(),
                                         PixelFormat32bppPARGB);
            if (clone && clone->GetLastStatus() == Ok) {
                result.reset(clone);
            } else {
                delete clone;
            }
        }
    }
    stream->Release();
    return result;
}

void AppendLog(const std::wstring& text) {
    if (!g_logEdit) {
        return;
    }
    SYSTEMTIME time{};
    GetLocalTime(&time);
    wchar_t prefix[32]{};
    swprintf(prefix, 32, L"%02u:%02u:%02u  ", time.wHour, time.wMinute, time.wSecond);
    std::wstring line = prefix + text + L"\r\n";
    SendMessageW(g_logEdit, EM_SETSEL, static_cast<WPARAM>(-1), static_cast<LPARAM>(-1));
    SendMessageW(g_logEdit, EM_REPLACESEL, FALSE, reinterpret_cast<LPARAM>(line.c_str()));
    SendMessageW(g_logEdit, EM_SCROLLCARET, 0, 0);
}

void UpdateAdapterStatus() {
    std::wstring status = FindNcmAdapter();
    SetWindowTextW(g_adapterStatus, (L"Adapter: " + status).c_str());
    AppendLog(L"Adapter check: " + status);
}

void UpdateStatistics() {
    const auto now = std::chrono::steady_clock::now();
    if (g_lastFpsTime.time_since_epoch().count() == 0) {
        g_lastFpsTime = now;
        g_lastFpsFrames = g_frameCount;
    } else {
        double seconds = std::chrono::duration<double>(now - g_lastFpsTime).count();
        if (seconds >= 0.8) {
            g_currentFps = static_cast<double>(g_frameCount - g_lastFpsFrames) / seconds;
            g_lastFpsFrames = g_frameCount;
            g_lastFpsTime = now;
        }
    }

    SetWindowTextW(g_frameStatus, (L"JPEG frames: " + std::to_wstring(g_frameCount)).c_str());
    wchar_t fps[64]{};
    swprintf(fps, 64, L"Receive FPS: %.1f", g_currentFps);
    SetWindowTextW(g_fpsStatus, fps);
    std::wstring jpeg = L"Last JPEG: " + std::to_wstring(g_lastJpegSize) + L" bytes";
    if (g_lastWidth && g_lastHeight) {
        jpeg += L"  " + std::to_wstring(g_lastWidth) + L"x" + std::to_wstring(g_lastHeight);
    }
    SetWindowTextW(g_jpegStatus, jpeg.c_str());
    SetWindowTextW(g_crcStatus, (L"CRC errors: " + std::to_wstring(g_crcErrors)).c_str());
    SetWindowTextW(g_sequenceStatus,
                   (L"Sequence gaps: " + std::to_wstring(g_sequenceGaps)).c_str());
    SetWindowTextW(g_byteStatus,
                   (L"JLIP payload: " + std::to_wstring(g_receivedBytes) + L" bytes").c_str());
}

void HandlePacket(UiEvent& event) {
    if (g_haveLastSequence && event.sequence != g_lastSequence + 1) {
        ++g_sequenceGaps;
    }
    g_lastSequence = event.sequence;
    g_haveLastSequence = true;

    if (event.text == L"CRC_ERROR") {
        ++g_crcErrors;
        AppendLog(L"JLIP CRC mismatch at sequence " + std::to_wstring(event.sequence));
        return;
    }
    g_receivedBytes += event.payload.size();

    if (event.packetType == JLIP_HELLO_ACK) {
        std::string json(event.payload.begin(), event.payload.end());
        std::wstring hello = Utf8ToWide(json);
        SetWindowTextW(g_deviceStatus, (L"Device: " + hello).c_str());
        AppendLog(L"Device hello: " + hello);
    } else if (event.packetType == JLIP_JPEG) {
        auto bitmap = DecodeJpeg(event.payload);
        if (!bitmap) {
            AppendLog(L"Invalid JPEG payload at sequence " + std::to_wstring(event.sequence));
            return;
        }
        g_lastWidth = bitmap->GetWidth();
        g_lastHeight = bitmap->GetHeight();
        g_lastJpegSize = static_cast<uint32_t>(event.payload.size());
        g_latestJpeg = std::move(event.payload);
        g_bitmap = std::move(bitmap);
        ++g_frameCount;
        InvalidateRect(g_mainWindow, &g_imageRect, FALSE);
    } else if (event.packetType != JLIP_HEARTBEAT &&
               event.packetType != JLIP_HEARTBEAT_ACK) {
        AppendLog(L"Unknown JLIP type 0x" + std::to_wstring(event.packetType));
    }
}

void SaveLatestJpeg(HWND owner) {
    if (g_latestJpeg.empty()) {
        MessageBoxW(owner, L"No JPEG frame has been received yet.",
                    L"Save JPEG", MB_OK | MB_ICONINFORMATION);
        return;
    }
    wchar_t filename[MAX_PATH] = L"ncm_frame.jpg";
    OPENFILENAMEW dialog{};
    dialog.lStructSize = sizeof(dialog);
    dialog.hwndOwner = owner;
    dialog.lpstrFilter = L"JPEG image (*.jpg)\0*.jpg\0All files (*.*)\0*.*\0";
    dialog.lpstrFile = filename;
    dialog.nMaxFile = MAX_PATH;
    dialog.lpstrDefExt = L"jpg";
    dialog.Flags = OFN_OVERWRITEPROMPT | OFN_PATHMUSTEXIST;
    if (!GetSaveFileNameW(&dialog)) {
        return;
    }
    FILE* file = _wfopen(filename, L"wb");
    if (!file) {
        MessageBoxW(owner, L"Could not create the output file.",
                    L"Save JPEG", MB_OK | MB_ICONERROR);
        return;
    }
    fwrite(g_latestJpeg.data(), 1, g_latestJpeg.size(), file);
    fclose(file);
    AppendLog(L"Saved JPEG: " + std::wstring(filename));
}

HWND CreateStatic(HWND parent, const wchar_t* text, int x, int y, int w, int h) {
    return CreateWindowExW(0, L"STATIC", text, WS_CHILD | WS_VISIBLE,
                           x, y, w, h, parent, nullptr, g_instance, nullptr);
}

void LayoutControls(HWND hwnd) {
    RECT client{};
    GetClientRect(hwnd, &client);
    int width = client.right - client.left;
    int height = client.bottom - client.top;
    int rightPanelWidth = 280;
    int logHeight = 105;
    int imageRight = std::max(420, width - rightPanelWidth - 25);
    int imageBottom = std::max(320, height - logHeight - 15);
    g_imageRect = {15, 55, imageRight, imageBottom};

    int rightX = imageRight + 15;
    int rightWidth = std::max(200, width - rightX - 10);
    HWND controls[] = {g_adapterStatus, g_connectionStatus, g_deviceStatus,
                       g_frameStatus, g_fpsStatus, g_jpegStatus, g_crcStatus,
                       g_sequenceStatus, g_byteStatus};
    int y = 65;
    for (HWND control : controls) {
        if (control) {
            MoveWindow(control, rightX, y, rightWidth, control == g_deviceStatus ? 70 : 24, TRUE);
            y += control == g_deviceStatus ? 78 : 30;
        }
    }
    if (g_logEdit) {
        MoveWindow(g_logEdit, 15, imageBottom + 10, width - 30,
                   std::max(70, height - imageBottom - 20), TRUE);
    }
    InvalidateRect(hwnd, nullptr, TRUE);
}

void PaintImage(HWND hwnd) {
    PAINTSTRUCT paint{};
    HDC dc = BeginPaint(hwnd, &paint);
    int boxWidth = g_imageRect.right - g_imageRect.left;
    int boxHeight = g_imageRect.bottom - g_imageRect.top;
    HDC bufferDc = CreateCompatibleDC(dc);
    HBITMAP bufferBitmap = (bufferDc && boxWidth > 0 && boxHeight > 0) ?
                           CreateCompatibleBitmap(dc, boxWidth, boxHeight) : nullptr;
    HGDIOBJ oldBitmap = bufferBitmap ? SelectObject(bufferDc, bufferBitmap) : nullptr;

    if (bufferBitmap) {
        RECT bufferRect{0, 0, boxWidth, boxHeight};
        HBRUSH background = CreateSolidBrush(RGB(20, 20, 20));
        FillRect(bufferDc, &bufferRect, background);
        DeleteObject(background);

        if (g_bitmap && g_bitmap->GetWidth() && g_bitmap->GetHeight()) {
            Graphics graphics(bufferDc);
            graphics.SetCompositingMode(CompositingModeSourceCopy);
            graphics.SetInterpolationMode(InterpolationModeHighQualityBicubic);
            graphics.SetPixelOffsetMode(PixelOffsetModeHighQuality);
        double scale = std::min(static_cast<double>(boxWidth) / g_bitmap->GetWidth(),
                                static_cast<double>(boxHeight) / g_bitmap->GetHeight());
        int drawWidth = static_cast<int>(g_bitmap->GetWidth() * scale);
        int drawHeight = static_cast<int>(g_bitmap->GetHeight() * scale);
            int drawX = (boxWidth - drawWidth) / 2;
            int drawY = (boxHeight - drawHeight) / 2;
            if (g_rotate180) {
                REAL centerX = static_cast<REAL>(drawX + drawWidth / 2.0);
                REAL centerY = static_cast<REAL>(drawY + drawHeight / 2.0);
                graphics.TranslateTransform(centerX, centerY);
                graphics.RotateTransform(180.0f);
                graphics.TranslateTransform(-centerX, -centerY);
            }
            graphics.DrawImage(g_bitmap.get(), drawX, drawY, drawWidth, drawHeight);
        } else {
            SetBkMode(bufferDc, TRANSPARENT);
            SetTextColor(bufferDc, RGB(220, 220, 220));
            DrawTextW(bufferDc, L"Waiting for NCM JPEG stream...", -1, &bufferRect,
                      DT_CENTER | DT_VCENTER | DT_SINGLELINE);
        }

        /* Present the complete frame with one copy.  The former direct path
         * exposed its black clear before DrawImage(), causing visible blink. */
        BitBlt(dc, g_imageRect.left, g_imageRect.top, boxWidth, boxHeight,
               bufferDc, 0, 0, SRCCOPY);
        SelectObject(bufferDc, oldBitmap);
        DeleteObject(bufferBitmap);
    } else {
        HBRUSH background = CreateSolidBrush(RGB(20, 20, 20));
        FillRect(dc, &g_imageRect, background);
        DeleteObject(background);
    }
    if (bufferDc) {
        DeleteDC(bufferDc);
    }
    EndPaint(hwnd, &paint);
}

LRESULT CALLBACK WindowProcedure(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam) {
    switch (message) {
    case WM_CREATE: {
        CreateStatic(hwnd, L"Device IP", 15, 18, 65, 22);
        g_ipEdit = CreateWindowExW(WS_EX_CLIENTEDGE, L"EDIT", L"192.168.50.2",
                                   WS_CHILD | WS_VISIBLE | ES_AUTOHSCROLL,
                                   80, 14, 125, 26, hwnd,
                                   reinterpret_cast<HMENU>(ID_IP_EDIT), g_instance, nullptr);
        CreateStatic(hwnd, L"TCP", 215, 18, 35, 22);
        g_portEdit = CreateWindowExW(WS_EX_CLIENTEDGE, L"EDIT", L"5000",
                                     WS_CHILD | WS_VISIBLE | ES_NUMBER,
                                     250, 14, 65, 26, hwnd,
                                     reinterpret_cast<HMENU>(ID_PORT_EDIT), g_instance, nullptr);
        g_connectButton = CreateWindowExW(0, L"BUTTON", L"Connect",
                                          WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
                                          325, 13, 90, 28, hwnd,
                                          reinterpret_cast<HMENU>(ID_CONNECT), g_instance, nullptr);
        CreateWindowExW(0, L"BUTTON", L"Discover",
                        WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
                        425, 13, 90, 28, hwnd,
                        reinterpret_cast<HMENU>(ID_DISCOVER), g_instance, nullptr);
        CreateWindowExW(0, L"BUTTON", L"Check adapter",
                        WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
                        525, 13, 105, 28, hwnd,
                        reinterpret_cast<HMENU>(ID_CHECK_ADAPTER), g_instance, nullptr);
        CreateWindowExW(0, L"BUTTON", L"Save JPEG",
                        WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
                        640, 13, 95, 28, hwnd,
                        reinterpret_cast<HMENU>(ID_SAVE_JPEG), g_instance, nullptr);
        g_autoReconnectCheck = CreateWindowExW(0, L"BUTTON", L"Auto reconnect",
                                               WS_CHILD | WS_VISIBLE | BS_AUTOCHECKBOX,
                                               750, 16, 130, 24, hwnd,
                                               reinterpret_cast<HMENU>(ID_AUTO_RECONNECT),
                                               g_instance, nullptr);
        SendMessageW(g_autoReconnectCheck, BM_SETCHECK, BST_CHECKED, 0);
        g_rotate180Check = CreateWindowExW(0, L"BUTTON", L"Rotate 180°",
                                           WS_CHILD | WS_VISIBLE | BS_AUTOCHECKBOX,
                                           885, 16, 125, 24, hwnd,
                                           reinterpret_cast<HMENU>(ID_ROTATE_180),
                                           g_instance, nullptr);

        g_adapterStatus = CreateStatic(hwnd, L"Adapter: checking...", 720, 65, 260, 24);
        g_connectionStatus = CreateStatic(hwnd, L"Connection: disconnected", 720, 95, 260, 24);
        g_deviceStatus = CreateStatic(hwnd, L"Device: waiting", 720, 125, 260, 70);
        g_frameStatus = CreateStatic(hwnd, L"JPEG frames: 0", 720, 203, 260, 24);
        g_fpsStatus = CreateStatic(hwnd, L"Receive FPS: 0.0", 720, 233, 260, 24);
        g_jpegStatus = CreateStatic(hwnd, L"Last JPEG: 0 bytes", 720, 263, 260, 24);
        g_crcStatus = CreateStatic(hwnd, L"CRC errors: 0", 720, 293, 260, 24);
        g_sequenceStatus = CreateStatic(hwnd, L"Sequence gaps: 0", 720, 323, 260, 24);
        g_byteStatus = CreateStatic(hwnd, L"JLIP payload: 0 bytes", 720, 353, 260, 24);

        g_logEdit = CreateWindowExW(WS_EX_CLIENTEDGE, L"EDIT", L"",
                                    WS_CHILD | WS_VISIBLE | WS_VSCROLL |
                                        ES_MULTILINE | ES_AUTOVSCROLL | ES_READONLY,
                                    15, 575, 965, 85, hwnd, nullptr, g_instance, nullptr);
        SendMessageW(g_logEdit, WM_SETFONT,
                     reinterpret_cast<WPARAM>(GetStockObject(DEFAULT_GUI_FONT)), TRUE);
        SetTimer(hwnd, 1, 1000, nullptr);
        LayoutControls(hwnd);
        UpdateAdapterStatus();
        AppendLog(L"Tester ready. Firmware target: 192.168.50.2, TCP 5000, UDP 5001.");
        return 0;
    }
    case WM_SIZE:
        LayoutControls(hwnd);
        return 0;
    case WM_TIMER:
        UpdateStatistics();
        return 0;
    case WM_COMMAND: {
        int id = LOWORD(wParam);
        if (id == ID_CONNECT) {
            if (g_runNetwork.load()) {
                AppendLog(L"Disconnect requested.");
                StopNetwork();
                SetWindowTextW(g_connectionStatus, L"Connection: disconnected");
                SetWindowTextW(g_connectButton, L"Connect");
            } else {
                wchar_t ip[64]{};
                wchar_t portText[16]{};
                GetWindowTextW(g_ipEdit, ip, 64);
                GetWindowTextW(g_portEdit, portText, 16);
                unsigned long port = wcstoul(portText, nullptr, 10);
                if (!port || port > 65535) {
                    MessageBoxW(hwnd, L"Enter a valid TCP port.", L"NCM Tester",
                                MB_OK | MB_ICONWARNING);
                    return 0;
                }
                g_autoReconnect.store(
                    SendMessageW(g_autoReconnectCheck, BM_GETCHECK, 0, 0) == BST_CHECKED);
                SetWindowTextW(g_connectButton, L"Disconnect");
                StartNetwork(ip, static_cast<uint16_t>(port));
            }
        } else if (id == ID_DISCOVER) {
            wchar_t ip[64]{};
            GetWindowTextW(g_ipEdit, ip, 64);
            AppendLog(L"Sending UDP discovery to " + std::wstring(ip) + L":5001...");
            std::thread(DiscoverWorker, WideToUtf8(ip)).detach();
        } else if (id == ID_CHECK_ADAPTER) {
            UpdateAdapterStatus();
        } else if (id == ID_SAVE_JPEG) {
            SaveLatestJpeg(hwnd);
        } else if (id == ID_AUTO_RECONNECT) {
            g_autoReconnect.store(
                SendMessageW(g_autoReconnectCheck, BM_GETCHECK, 0, 0) == BST_CHECKED);
        } else if (id == ID_ROTATE_180) {
            g_rotate180 =
                SendMessageW(g_rotate180Check, BM_GETCHECK, 0, 0) == BST_CHECKED;
            InvalidateRect(hwnd, &g_imageRect, FALSE);
        }
        return 0;
    }
    case WM_APP_EVENT: {
        std::unique_ptr<UiEvent> event(reinterpret_cast<UiEvent*>(lParam));
        if (!event) {
            return 0;
        }
        switch (event->kind) {
        case EventKind::Log:
            AppendLog(event->text);
            break;
        case EventKind::Connected:
            SetWindowTextW(g_connectionStatus,
                           (L"Connection: connected " + event->text).c_str());
            AppendLog(L"TCP connected. Waiting for JLIP hello and JPEG frames.");
            g_haveLastSequence = false;
            break;
        case EventKind::Disconnected:
            SetWindowTextW(g_connectionStatus, L"Connection: disconnected");
            AppendLog(L"TCP disconnected.");
            break;
        case EventKind::Packet:
            HandlePacket(*event);
            break;
        case EventKind::Discovery:
            AppendLog(L"Discovery: " + event->text);
            break;
        case EventKind::WorkerStopped:
            SetWindowTextW(g_connectButton, L"Connect");
            SetWindowTextW(g_connectionStatus, L"Connection: disconnected");
            break;
        }
        return 0;
    }
    case WM_PAINT:
        PaintImage(hwnd);
        return 0;
    case WM_GETMINMAXINFO: {
        auto* info = reinterpret_cast<MINMAXINFO*>(lParam);
        info->ptMinTrackSize.x = 1020;
        info->ptMinTrackSize.y = 620;
        return 0;
    }
    case WM_DESTROY:
        KillTimer(hwnd, 1);
        StopNetwork();
        g_bitmap.reset();
        PostQuitMessage(0);
        return 0;
    default:
        return DefWindowProcW(hwnd, message, wParam, lParam);
    }
}

bool ProtocolSelfTest() {
    const uint8_t test[] = {'1','2','3','4','5','6','7','8','9'};
    if (Crc32(test, sizeof(test)) != 0xcbf43926u) {
        return false;
    }
    auto packet = BuildJlipPacket(JLIP_HEARTBEAT, 0x12345678u, nullptr, 0);
    return packet.size() == kJlipHeaderSize &&
           std::memcmp(packet.data(), "JLIP", 4) == 0 &&
           packet[4] == 1 && packet[5] == JLIP_HEARTBEAT &&
           ReadBe16(packet.data() + 6) == kJlipHeaderSize &&
           ReadBe32(packet.data() + 8) == 0x12345678u &&
           ReadBe32(packet.data() + 12) == 0 &&
           ReadBe32(packet.data() + 16) == 0;
}

} // namespace

int WINAPI wWinMain(HINSTANCE instance, HINSTANCE, PWSTR commandLine, int showCommand) {
    if (commandLine && std::wcsstr(commandLine, L"--self-test")) {
        return ProtocolSelfTest() ? 0 : 10;
    }

    g_instance = instance;
    WSADATA winsock{};
    if (WSAStartup(MAKEWORD(2, 2), &winsock) != 0) {
        MessageBoxW(nullptr, L"Winsock initialization failed.", L"NCM Tester",
                    MB_OK | MB_ICONERROR);
        return 1;
    }

    GdiplusStartupInput gdiplusInput;
    if (GdiplusStartup(&g_gdiplusToken, &gdiplusInput, nullptr) != Ok) {
        WSACleanup();
        return 2;
    }

    INITCOMMONCONTROLSEX commonControls{sizeof(commonControls), ICC_STANDARD_CLASSES};
    InitCommonControlsEx(&commonControls);

    WNDCLASSEXW windowClass{};
    windowClass.cbSize = sizeof(windowClass);
    windowClass.lpfnWndProc = WindowProcedure;
    windowClass.hInstance = instance;
    windowClass.hCursor = LoadCursorW(nullptr, IDC_ARROW);
    windowClass.hbrBackground = reinterpret_cast<HBRUSH>(COLOR_WINDOW + 1);
    windowClass.lpszClassName = L"JlNcmCameraTesterWindow";
    windowClass.hIcon = LoadIconW(nullptr, IDI_APPLICATION);
    windowClass.hIconSm = windowClass.hIcon;
    if (!RegisterClassExW(&windowClass)) {
        GdiplusShutdown(g_gdiplusToken);
        WSACleanup();
        return 3;
    }

    g_mainWindow = CreateWindowExW(0, windowClass.lpszClassName,
                                   L"JL NCM Camera Tester - 192.168.50.2",
                                   WS_OVERLAPPEDWINDOW,
                                   CW_USEDEFAULT, CW_USEDEFAULT, 1040, 740,
                                   nullptr, nullptr, instance, nullptr);
    if (!g_mainWindow) {
        GdiplusShutdown(g_gdiplusToken);
        WSACleanup();
        return 4;
    }

    ShowWindow(g_mainWindow, showCommand);
    UpdateWindow(g_mainWindow);

    MSG message{};
    while (GetMessageW(&message, nullptr, 0, 0) > 0) {
        TranslateMessage(&message);
        DispatchMessageW(&message);
    }

    g_mainWindow = nullptr;
    GdiplusShutdown(g_gdiplusToken);
    WSACleanup();
    return static_cast<int>(message.wParam);
}
