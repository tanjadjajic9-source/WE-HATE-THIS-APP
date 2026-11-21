#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include "WebBridge.h"

#include <winsock2.h>
#include <ws2tcpip.h>
#include <iphlpapi.h>
#include <wincodec.h>
#include <wrl/client.h>
#include <winrt/Windows.Storage.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>

#pragma comment(lib, "ws2_32.lib")
#pragma comment(lib, "iphlpapi.lib")
#pragma comment(lib, "windowscodecs.lib")

namespace XboxOTFM {

namespace {

std::string ToLowerAscii(std::string s) {
    for (char& ch : s) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    return s;
}

void SendHttpResponse(SOCKET sock, int statusCode, const char* statusText, const char* contentType, const void* body, size_t bodyLen, const char* extraHeaders = "") {
    std::ostringstream hdr;
    hdr << "HTTP/1.1 " << statusCode << " " << statusText << "\r\n"
        << "Content-Type: " << contentType << "\r\n"
        << "Content-Length: " << bodyLen << "\r\n"
        << "Access-Control-Allow-Origin: *\r\n"
        << "Connection: close\r\n"
        << extraHeaders
        << "\r\n";
    const std::string hdrStr = hdr.str();
    send(sock, hdrStr.data(), static_cast<int>(hdrStr.size()), 0);
    if (body && bodyLen > 0) {
        size_t sent = 0;
        while (sent < bodyLen) {
            const int chunk = send(sock, static_cast<const char*>(body) + sent, static_cast<int>(std::min<size_t>(65536, bodyLen - sent)), 0);
            if (chunk <= 0) break;
            sent += static_cast<size_t>(chunk);
        }
    }
}

void SendTextResponse(SOCKET sock, int code, const char* status, const char* mime, const std::string& text, const char* extra = "") {
    SendHttpResponse(sock, code, status, mime, text.data(), text.size(), extra);
}

std::vector<std::vector<uint8_t>> ExtractStoredZipEntries(const uint8_t* zipData, size_t zipSize) {
    std::vector<std::vector<uint8_t>> files;
    size_t pos = 0;
    while (pos + 30 <= zipSize) {
        if (zipData[pos] != 0x50 || zipData[pos + 1] != 0x4B || zipData[pos + 2] != 0x03 || zipData[pos + 3] != 0x04) {
            break;
        }
        const uint16_t method = static_cast<uint16_t>(zipData[pos + 8] | (zipData[pos + 9] << 8));
        const uint32_t compSize = static_cast<uint32_t>(zipData[pos + 18] | (zipData[pos + 19] << 8) | (zipData[pos + 20] << 16) | (zipData[pos + 21] << 24));
        const uint16_t nameLen = static_cast<uint16_t>(zipData[pos + 26] | (zipData[pos + 27] << 8));
        const uint16_t extraLen = static_cast<uint16_t>(zipData[pos + 28] | (zipData[pos + 29] << 8));
        const size_t dataStart = pos + 30 + nameLen + extraLen;
        if (dataStart + compSize > zipSize) break;
        if (method == 0 && compSize > 0) {
            files.emplace_back(zipData + dataStart, zipData + dataStart + compSize);
        }
        pos = dataStart + compSize;
    }
    return files;
}

} // namespace

WebBridge::WebBridge(FMBrain& brain, Trainer& trainer, std::atomic<uint32_t>& activeResolution, std::atomic<bool>& isTraining)
    : m_brain(brain)
    , m_trainer(trainer)
    , m_activeResolution(activeResolution)
    , m_isTraining(isTraining) {
}

WebBridge::~WebBridge() {
    Stop();
}

std::string WebBridge::CrashLogFilePath() {
    try {
        auto folder = winrt::Windows::Storage::ApplicationData::Current().LocalFolder().Path();
        std::wstring wp(folder);
        int len = WideCharToMultiByte(CP_UTF8, 0, wp.c_str(), static_cast<int>(wp.size()), nullptr, 0, nullptr, nullptr);
        std::string utf8(static_cast<size_t>(len), '\0');
        WideCharToMultiByte(CP_UTF8, 0, wp.c_str(), static_cast<int>(wp.size()), utf8.data(), len, nullptr, nullptr);
        return utf8 + "\\crash_log.txt";
    } catch (...) {
        return "crash_log.txt";
    }
}

void WebBridge::WriteCrashLogToDisk(const std::string& entry) {
    std::ofstream ofs(CrashLogFilePath(), std::ios::app);
    if (ofs.is_open()) {
        ofs << entry << "\n";
        ofs.flush();
    }
}

void WebBridge::AppendLogLine(const std::string& line) {
    std::lock_guard<std::mutex> lock(m_logMutex);
    m_consoleLogs.push_back(line);
    if (m_consoleLogs.size() > 250) {
        m_consoleLogs.erase(m_consoleLogs.begin());
    }
}

std::string WebBridge::ReadCrashAndConsoleLogs() const {
    std::ostringstream oss;
    oss << "=== crash_log.txt ===\n";
    std::ifstream ifs(CrashLogFilePath());
    if (ifs.is_open()) {
        oss << ifs.rdbuf();
    } else {
        oss << "(No crash records logged)\n";
    }
    oss << "\n=== Recent Console Output ===\n";
    std::lock_guard<std::mutex> lock(m_logMutex);
    for (const auto& line : m_consoleLogs) {
        oss << line << "\n";
    }
    return oss.str();
}

std::string WebBridge::DetectLanIPv4() const {
    ULONG bufSize = 16384;
    std::vector<uint8_t> buffer(bufSize);
    auto* addrs = reinterpret_cast<IP_ADAPTER_ADDRESSES*>(buffer.data());
    if (GetAdaptersAddresses(AF_INET, GAA_FLAG_SKIP_ANYCAST | GAA_FLAG_SKIP_MULTICAST, nullptr, addrs, &bufSize) == ERROR_SUCCESS) {
        for (auto* cur = addrs; cur != nullptr; cur = cur->Next) {
            if (cur->OperStatus != IfOperStatusUp || cur->IfType == IF_TYPE_SOFTWARE_LOOPBACK) continue;
            for (auto* ua = cur->FirstUnicastAddress; ua != nullptr; ua = ua->Next) {
                if (ua->Address.lpSockaddr && ua->Address.lpSockaddr->sa_family == AF_INET) {
                    auto* sin = reinterpret_cast<sockaddr_in*>(ua->Address.lpSockaddr);
                    char str[INET_ADDRSTRLEN] = {};
                    if (inet_ntop(AF_INET, &sin->sin_addr, str, sizeof(str))) {
                        return std::string(str);
                    }
                }
            }
        }
    }
    return "127.0.0.1";
}

std::string WebBridge::LocalEndpointUrl() const {
    return "http://" + m_lanIp + ":" + std::to_string(m_port);
}

bool WebBridge::Start(uint16_t port) {
    if (m_running.load()) return true;
    m_port = port;

    WSADATA wsaData = {};
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        AppendLogLine("[WebBridge] WSAStartup failed.");
        return false;
    }

    m_lanIp = DetectLanIPv4();

    SOCKET sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (sock == INVALID_SOCKET) {
        AppendLogLine("[WebBridge] Failed to create listen socket.");
        return false;
    }

    int opt = 1;
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&opt), sizeof(opt));

    sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);

    if (bind(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == SOCKET_ERROR ||
        listen(sock, 16) == SOCKET_ERROR) {
        closesocket(sock);
        AppendLogLine("[WebBridge] Failed to bind port " + std::to_string(port));
        return false;
    }

    m_listenSocket = static_cast<uintptr_t>(sock);
    m_running.store(true);
    m_thread = std::thread(&WebBridge::ServerThreadLoop, this);
    AppendLogLine("[WebBridge] Listening on " + LocalEndpointUrl());
    return true;
}

void WebBridge::Stop() {
    if (!m_running.exchange(false)) return;
    if (m_listenSocket != static_cast<uintptr_t>(-1)) {
        closesocket(static_cast<SOCKET>(m_listenSocket));
        m_listenSocket = static_cast<uintptr_t>(-1);
    }
    if (m_thread.joinable()) {
        m_thread.join();
    }
    WSACleanup();
}

void WebBridge::ServerThreadLoop() {
    CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    SOCKET listenSock = static_cast<SOCKET>(m_listenSocket);
    while (m_running.load()) {
        sockaddr_in clientAddr = {};
        int addrLen = sizeof(clientAddr);
        SOCKET client = accept(listenSock, reinterpret_cast<sockaddr*>(&clientAddr), &addrLen);
        if (client == INVALID_SOCKET) {
            if (!m_running.load()) break;
            continue;
        }
        HandleClient(static_cast<uintptr_t>(client));
        closesocket(client);
    }
    CoUninitialize();
}

bool WebBridge::DecodeImageBytesToCHW(const uint8_t* data, size_t size, std::vector<float>& outCHW, uint32_t& outW, uint32_t& outH) {
    Microsoft::WRL::ComPtr<IWICImagingFactory> wicFactory;
    HRESULT hr = CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&wicFactory));
    if (FAILED(hr)) return false;

    Microsoft::WRL::ComPtr<IWICStream> stream;
    hr = wicFactory->CreateStream(&stream);
    if (FAILED(hr)) return false;

    hr = stream->InitializeFromMemory(const_cast<BYTE*>(data), static_cast<DWORD>(size));
    if (FAILED(hr)) return false;

    Microsoft::WRL::ComPtr<IWICBitmapDecoder> decoder;
    hr = wicFactory->CreateDecoderFromStream(stream.Get(), nullptr, WICDecodeMetadataCacheOnDemand, &decoder);
    if (FAILED(hr)) return false;

    Microsoft::WRL::ComPtr<IWICBitmapFrameDecode> frame;
    hr = decoder->GetFrame(0, &frame);
    if (FAILED(hr)) return false;

    hr = frame->GetSize(&outW, &outH);
    if (FAILED(hr) || outW == 0 || outH == 0) return false;

    Microsoft::WRL::ComPtr<IWICFormatConverter> converter;
    hr = wicFactory->CreateFormatConverter(&converter);
    if (FAILED(hr)) return false;

    hr = converter->Initialize(frame.Get(), GUID_WICPixelFormat24bppRGB, WICBitmapDitherTypeNone, nullptr, 0.0, WICBitmapPaletteTypeCustom);
    if (FAILED(hr)) return false;

    const uint32_t stride = outW * 3;
    std::vector<uint8_t> rgb(static_cast<size_t>(stride) * outH);
    hr = converter->CopyPixels(nullptr, stride, static_cast<UINT>(rgb.size()), rgb.data());
    if (FAILED(hr)) return false;

    const size_t hw = static_cast<size_t>(outW) * outH;
    outCHW.resize(3 * hw);
    for (size_t i = 0; i < hw; ++i) {
        outCHW[0 * hw + i] = (static_cast<float>(rgb[i * 3 + 0]) / 127.5f) - 1.0f;
        outCHW[1 * hw + i] = (static_cast<float>(rgb[i * 3 + 1]) / 127.5f) - 1.0f;
        outCHW[2 * hw + i] = (static_cast<float>(rgb[i * 3 + 2]) / 127.5f) - 1.0f;
    }
    return true;
}

uint32_t WebBridge::ProcessUploadPayload(const std::string& contentType, const std::vector<uint8_t>& body) {
    uint32_t added = 0;
    auto tryDecode = [&](const uint8_t* ptr, size_t len) {
        if (len >= 4 && ptr[0] == 0x50 && ptr[1] == 0x4B && ptr[2] == 0x03 && ptr[3] == 0x04) {
            auto entries = ExtractStoredZipEntries(ptr, len);
            for (const auto& entry : entries) {
                std::vector<float> chw;
                uint32_t w = 0, h = 0;
                if (DecodeImageBytesToCHW(entry.data(), entry.size(), chw, w, h)) {
                    m_trainer.AddCardToDeck(std::move(chw), w, h);
                    added++;
                }
            }
            return;
        }
        std::vector<float> chw;
        uint32_t w = 0, h = 0;
        if (DecodeImageBytesToCHW(ptr, len, chw, w, h)) {
            m_trainer.AddCardToDeck(std::move(chw), w, h);
            added++;
        }
    };

    const std::string ctLower = ToLowerAscii(contentType);
    const size_t bpos = ctLower.find("boundary=");
    if (ctLower.find("multipart/form-data") != std::string::npos && bpos != std::string::npos) {
        std::string boundary = "--" + contentType.substr(bpos + 9);
        if (!boundary.empty() && boundary.back() == '\r') boundary.pop_back();
        const std::string raw(reinterpret_cast<const char*>(body.data()), body.size());
        size_t pos = 0;
        while ((pos = raw.find(boundary, pos)) != std::string::npos) {
            pos += boundary.size();
            const size_t hdrEnd = raw.find("\r\n\r\n", pos);
            if (hdrEnd == std::string::npos) break;
            const size_t partStart = hdrEnd + 4;
            const size_t nextBoundary = raw.find(boundary, partStart);
            if (nextBoundary == std::string::npos) break;
            const size_t partEnd = (nextBoundary >= 2) ? nextBoundary - 2 : nextBoundary;
            if (partEnd > partStart) {
                tryDecode(body.data() + partStart, partEnd - partStart);
            }
            pos = nextBoundary;
        }
    } else {
        tryDecode(body.data(), body.size());
    }
    return added;
}

std::vector<uint8_t> WebBridge::BuildDreamBmp() const {
    uint32_t s = m_activeResolution.load();
    const auto dream = m_trainer.LatestDreamSnapshot(s);
    s = std::max(8u, s);
    const size_t hw = static_cast<size_t>(s) * s;
    const uint32_t rowBytes = ((s * 3 + 3) / 4) * 4;
    const uint32_t imgSize = rowBytes * s;
    const uint32_t fileSize = 54 + imgSize;

    std::vector<uint8_t> bmp(fileSize, 0);
    bmp[0] = 'B'; bmp[1] = 'M';
    std::memcpy(bmp.data() + 2, &fileSize, 4);
    const uint32_t offset = 54;
    std::memcpy(bmp.data() + 10, &offset, 4);
    const uint32_t dibSize = 40;
    std::memcpy(bmp.data() + 14, &dibSize, 4);
    const int32_t w = static_cast<int32_t>(s);
    const int32_t h = -static_cast<int32_t>(s);
    std::memcpy(bmp.data() + 18, &w, 4);
    std::memcpy(bmp.data() + 22, &h, 4);
    const uint16_t planes = 1, bpp = 24;
    std::memcpy(bmp.data() + 26, &planes, 2);
    std::memcpy(bmp.data() + 28, &bpp, 2);

    if (dream.size() >= 3 * hw) {
        for (uint32_t y = 0; y < s; ++y) {
            for (uint32_t x = 0; x < s; ++x) {
                const size_t p = static_cast<size_t>(y) * s + x;
                const auto u8 = [](float v) -> uint8_t {
                    return static_cast<uint8_t>(std::clamp((v + 1.0f) * 0.5f, 0.0f, 1.0f) * 255.0f);
                };
                uint8_t* dst = bmp.data() + 54 + static_cast<size_t>(y) * rowBytes + x * 3;
                dst[0] = u8(dream[2 * hw + p]);
                dst[1] = u8(dream[1 * hw + p]);
                dst[2] = u8(dream[0 * hw + p]);
            }
        }
    }
    return bmp;
}

std::string WebBridge::BuildMobileIndexHtml() const {
    const auto m = m_trainer.SnapshotMetrics();
    std::ostringstream html;
    html << "<!doctype html><html><head><meta charset='utf-8'/>"
         << "<meta name='viewport' content='width=device-width,initial-scale=1'/>"
         << "<title>Xbox OT-FM Phone Bridge</title>"
         << "<style>body{font-family:system-ui,sans-serif;background:#0b0f14;color:#f1f5f9;margin:0;padding:16px;}"
         << ".card{background:#121821;border:1px solid #1e293b;border-radius:10px;padding:16px;margin-bottom:14px;}"
         << "button,a.btn{display:inline-block;background:#16a34a;color:#fff;border:0;border-radius:8px;padding:10px 14px;font-weight:600;text-decoration:none;cursor:pointer;margin:4px 4px 0 0;}"
         << "img{image-rendering:pixelated;width:192px;height:192px;border:1px solid #1e293b;border-radius:8px;}</style></head><body>"
         << "<h2>Xbox OT-FM 2D Companion</h2>"
         << "<div class='card'><div>Step: <b>" << m.stepCount << "</b> | Loss: <b>" << m.lastMseLoss
         << "</b> | Speed: <b>" << m.lessonsPerSec << " lessons/s</b> | Deck: <b>" << m.deckCardCount << "</b></div></div>"
         << "<div class='card'><h3>2D Dream Preview</h3><img src='/api/dream-bmp?t=" << m.stepCount << "'/>"
         << "<br/><a class='btn' href='/api/dream-bmp?download=1'>Download 2D Dream (.bmp)</a></div>"
         << "<div class='card'><h3>Upload Training Images / ZIP (Max 16 MB)</h3>"
         << "<form method='POST' action='/api/upload' enctype='multipart/form-data'>"
         << "<input type='file' name='files' accept='image/*,.zip' multiple/><br/><br/>"
         << "<button type='submit'>Upload to Xbox Memory Deck</button></form></div>"
         << "<div class='card'><h3>Brain Checkpoints & Logs</h3>"
         << "<a class='btn' href='/api/export-brain'>Export Brain (.json)</a>"
         << "<a class='btn' href='/api/export-brain?format=bin'>Export Brain (.bin)</a>"
         << "<a class='btn' href='/api/logs'>View Crash & Console Logs</a></div>"
         << "</body></html>";
    return html.str();
}

void WebBridge::HandleClient(uintptr_t clientSocket) {
    SOCKET sock = static_cast<SOCKET>(clientSocket);
    std::string rawRequest;
    char buf[8192];
    size_t contentLength = 0;
    size_t headerEnd = std::string::npos;

    while (true) {
        const int recvd = recv(sock, buf, sizeof(buf), 0);
        if (recvd <= 0) break;
        rawRequest.append(buf, static_cast<size_t>(recvd));

        if (headerEnd == std::string::npos) {
            headerEnd = rawRequest.find("\r\n\r\n");
            if (headerEnd != std::string::npos) {
                const std::string hdrLower = ToLowerAscii(rawRequest.substr(0, headerEnd));
                const size_t clPos = hdrLower.find("content-length:");
                if (clPos != std::string::npos) {
                    contentLength = static_cast<size_t>(std::strtoull(hdrLower.c_str() + clPos + 15, nullptr, 10));
                    if (contentLength > kMaxPayloadBytes) {
                        SendTextResponse(sock, 413, "Payload Too Large", "application/json",
                            "{\"ok\":false,\"error\":\"Payload exceeds 16 MB console limit\"}");
                        return;
                    }
                }
            } else if (rawRequest.size() > 65536) {
                SendTextResponse(sock, 431, "Request Header Fields Too Large", "text/plain", "Headers too large");
                return;
            }
        }

        if (headerEnd != std::string::npos && rawRequest.size() >= headerEnd + 4 + contentLength) {
            break;
        }
        if (rawRequest.size() > kMaxPayloadBytes + 65536) {
            SendTextResponse(sock, 413, "Payload Too Large", "application/json",
                "{\"ok\":false,\"error\":\"Payload exceeds 16 MB console limit\"}");
            return;
        }
    }

    if (headerEnd == std::string::npos) return;

    const std::string headerBlock = rawRequest.substr(0, headerEnd);
    const size_t firstLineEnd = headerBlock.find("\r\n");
    const std::string reqLine = (firstLineEnd == std::string::npos) ? headerBlock : headerBlock.substr(0, firstLineEnd);

    std::istringstream iss(reqLine);
    std::string method, path;
    iss >> method >> path;

    std::string contentType;
    const std::string hdrLower = ToLowerAscii(headerBlock);
    const size_t ctPos = hdrLower.find("content-type:");
    if (ctPos != std::string::npos) {
        const size_t lineEnd = headerBlock.find("\r\n", ctPos);
        contentType = headerBlock.substr(ctPos + 13, (lineEnd == std::string::npos ? headerBlock.size() : lineEnd) - (ctPos + 13));
    }

    const std::vector<uint8_t> body(
        rawRequest.begin() + static_cast<std::ptrdiff_t>(headerEnd + 4),
        rawRequest.end());

    if (method == "GET" && path == "/") {
        SendTextResponse(sock, 200, "OK", "text/html; charset=utf-8", BuildMobileIndexHtml());
    } else if (method == "GET" && path.rfind("/api/dream-bmp", 0) == 0) {
        const auto bmp = BuildDreamBmp();
        const char* extra = (path.find("download=1") != std::string::npos)
            ? "Content-Disposition: attachment; filename=\"xbox_dream.bmp\"\r\n"
            : "";
        SendHttpResponse(sock, 200, "OK", "image/bmp", bmp.data(), bmp.size(), extra);
    } else if (method == "POST" && path == "/api/upload") {
        const uint32_t added = ProcessUploadPayload(contentType, body);
        AppendLogLine("[WebBridge] Phone upload added " + std::to_string(added) + " training card(s).");
        SendTextResponse(sock, 200, "OK", "application/json", "{\"added\":" + std::to_string(added) + "}");
    } else if (method == "GET" && path.rfind("/api/export-brain", 0) == 0) {
        if (path.find("format=bin") != std::string::npos) {
            const auto bin = m_brain.ExportToBinary();
            SendHttpResponse(sock, 200, "OK", "application/octet-stream", bin.data(), bin.size(),
                "Content-Disposition: attachment; filename=\"xbox_otfm_brain.bin\"\r\n");
        } else {
            const auto json = m_brain.ExportToJson();
            SendTextResponse(sock, 200, "OK", "application/json", json,
                "Content-Disposition: attachment; filename=\"xbox_otfm_brain.json\"\r\n");
        }
    } else if (method == "POST" && path == "/api/import-brain") {
        if (m_isTraining.load()) {
            SendTextResponse(sock, 409, "Conflict", "application/json",
                "{\"ok\":false,\"error\":\"Pause training before importing brain checkpoint\"}");
            return;
        }
        bool ok = false;
        if (body.size() >= 4 && body[0] == 0x4F && body[1] == 0x54 && body[2] == 0x46 && body[3] == 0x4D) {
            ok = m_brain.ImportFromBinary(body);
        } else {
            ok = m_brain.ImportFromJson(std::string(body.begin(), body.end()));
        }
        AppendLogLine(std::string("[WebBridge] Brain import ") + (ok ? "succeeded." : "failed."));
        SendTextResponse(sock, ok ? 200 : 400, ok ? "OK" : "Bad Request", "application/json",
            std::string("{\"ok\":") + (ok ? "true}" : "false}"));
    } else if (method == "GET" && path == "/api/logs") {
        SendTextResponse(sock, 200, "OK", "text/plain; charset=utf-8", ReadCrashAndConsoleLogs());
    } else {
        SendTextResponse(sock, 404, "Not Found", "text/plain", "Not Found");
    }
}

} // namespace XboxOTFM
