#include "GamepadUI.h"

#include <winrt/Windows.Foundation.Collections.h>
#include <winrt/Windows.Gaming.Input.h>
#include <Xinput.h>
#include <dxgi1_4.h>

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace XboxOTFM {

namespace {

std::wstring ToWide(const std::string& utf8) {
    if (utf8.empty()) return {};
    const int len = MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), static_cast<int>(utf8.size()), nullptr, 0);
    std::wstring out(static_cast<size_t>(len), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), static_cast<int>(utf8.size()), out.data(), len);
    return out;
}

const wchar_t* PresetLabel(BrainPreset p) {
    switch (p) {
        case BrainPreset::Low25k: return L"LOW (~25k Dials)";
        case BrainPreset::Medium80k: return L"MEDIUM (~80k Dials)";
        case BrainPreset::Max250k: return L"MAX (~250k Dials)";
    }
    return L"MEDIUM";
}

} // namespace

HRESULT GamepadUI::Initialize(
    ID3D12Device* d3d12Device,
    ID3D12CommandQueue* cmdQueue,
    const std::vector<Microsoft::WRL::ComPtr<ID3D12Resource>>& backBuffers,
    uint32_t width,
    uint32_t height) {
    Release();
    m_width = width;
    m_height = height;

    UINT d3d11Flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
#ifdef _DEBUG
    d3d11Flags |= D3D11_CREATE_DEVICE_DEBUG;
#endif

    IUnknown* queues[] = { cmdQueue };
    HRESULT hr = D3D11On12CreateDevice(
        d3d12Device,
        d3d11Flags,
        nullptr,
        0,
        queues,
        1,
        0,
        &m_d3d11Device,
        &m_d3d11Context,
        nullptr);
    if (FAILED(hr)) return hr;

    hr = m_d3d11Device.As(&m_d3d11On12Device);
    if (FAILED(hr)) return hr;

    D2D1_FACTORY_OPTIONS d2dOpts = {};
    hr = D2D1CreateFactory(D2D1_FACTORY_TYPE_SINGLE_THREADED, __uuidof(ID2D1Factory3), &d2dOpts, &m_d2dFactory);
    if (FAILED(hr)) return hr;

    // Query IDXGIDevice directly from the underlying D3D11 device (NOT ID3D11On12Device).
    Microsoft::WRL::ComPtr<IDXGIDevice> dxgiDevice;
    hr = m_d3d11Device.As(&dxgiDevice);
    if (FAILED(hr)) return hr;

    hr = m_d2dFactory->CreateDevice(dxgiDevice.Get(), &m_d2dDevice);
    if (FAILED(hr)) return hr;

    hr = m_d2dDevice->CreateDeviceContext(D2D1_DEVICE_CONTEXT_OPTIONS_NONE, &m_d2dContext);
    if (FAILED(hr)) return hr;

    hr = DWriteCreateFactory(DWRITE_FACTORY_TYPE_SHARED, __uuidof(IDWriteFactory), reinterpret_cast<IUnknown**>(m_dwriteFactory.GetAddressOf()));
    if (FAILED(hr)) return hr;

    const float scale = static_cast<float>(height) / 1080.0f;
    m_dwriteFactory->CreateTextFormat(L"Segoe UI", nullptr, DWRITE_FONT_WEIGHT_BOLD, DWRITE_FONT_STYLE_NORMAL, DWRITE_FONT_STRETCH_NORMAL, 42.0f * scale, L"en-US", &m_titleFormat);
    m_dwriteFactory->CreateTextFormat(L"Segoe UI", nullptr, DWRITE_FONT_WEIGHT_SEMI_BOLD, DWRITE_FONT_STYLE_NORMAL, DWRITE_FONT_STRETCH_NORMAL, 24.0f * scale, L"en-US", &m_bodyFormat);
    m_dwriteFactory->CreateTextFormat(L"Consolas", nullptr, DWRITE_FONT_WEIGHT_NORMAL, DWRITE_FONT_STYLE_NORMAL, DWRITE_FONT_STRETCH_NORMAL, 21.0f * scale, L"en-US", &m_monoFormat);
    m_dwriteFactory->CreateTextFormat(L"Segoe UI", nullptr, DWRITE_FONT_WEIGHT_BOLD, DWRITE_FONT_STYLE_NORMAL, DWRITE_FONT_STRETCH_NORMAL, 34.0f * scale, L"en-US", &m_crashFormat);

    const float dpi = 96.0f;
    D2D1_BITMAP_PROPERTIES1 bitmapProps = D2D1::BitmapProperties1(
        D2D1_BITMAP_OPTIONS_TARGET | D2D1_BITMAP_OPTIONS_CANNOT_DRAW,
        D2D1::PixelFormat(DXGI_FORMAT_B8G8R8A8_UNORM, D2D1_ALPHA_MODE_PREMULTIPLIED),
        dpi,
        dpi);

    m_wrappedBackBuffers.resize(backBuffers.size());
    m_d2dRenderTargets.resize(backBuffers.size());
    for (size_t i = 0; i < backBuffers.size(); ++i) {
        D3D11_RESOURCE_FLAGS resFlags = { D3D11_BIND_RENDER_TARGET };
        hr = m_d3d11On12Device->CreateWrappedResource(
            backBuffers[i].Get(),
            &resFlags,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            D3D12_RESOURCE_STATE_PRESENT,
            IID_PPV_ARGS(&m_wrappedBackBuffers[i]));
        if (FAILED(hr)) return hr;

        Microsoft::WRL::ComPtr<IDXGISurface> surface;
        hr = m_wrappedBackBuffers[i].As(&surface);
        if (FAILED(hr)) return hr;

        hr = m_d2dContext->CreateBitmapFromDxgiSurface(surface.Get(), &bitmapProps, &m_d2dRenderTargets[i]);
        if (FAILED(hr)) return hr;
    }

    return S_OK;
}

void GamepadUI::Release() {
    m_dreamBitmap.Reset();
    m_dreamBitmapSize = 0;
    m_d2dRenderTargets.clear();
    m_wrappedBackBuffers.clear();
    m_crashFormat.Reset();
    m_monoFormat.Reset();
    m_bodyFormat.Reset();
    m_titleFormat.Reset();
    m_dwriteFactory.Reset();
    m_d2dContext.Reset();
    m_d2dDevice.Reset();
    m_d2dFactory.Reset();
    m_d3d11On12Device.Reset();
    m_d3d11Context.Reset();
    m_d3d11Device.Reset();
}

UIActions GamepadUI::PollInput(UIState& state, double deltaSeconds) {
    UIActions actions;
    m_navCooldown = std::max(0.0, m_navCooldown - deltaSeconds);

    uint32_t currentButtons = 0;
    double leftThumbY = 0.0;
    bool ltPressed = false;
    bool rtPressed = false;

    try {
        auto gamepads = winrt::Windows::Gaming::Input::Gamepad::Gamepads();
        if (gamepads.Size() > 0) {
            auto reading = gamepads.GetAt(0).GetCurrentReading();
            using WGI = winrt::Windows::Gaming::Input::GamepadButtons;
            const auto btns = reading.Buttons;
            if ((btns & WGI::A) == WGI::A) currentButtons |= 0x0001;
            if ((btns & WGI::B) == WGI::B) currentButtons |= 0x0002;
            if ((btns & WGI::X) == WGI::X) currentButtons |= 0x0004;
            if ((btns & WGI::Y) == WGI::Y) currentButtons |= 0x0008;
            if ((btns & WGI::DPadUp) == WGI::DPadUp) currentButtons |= 0x0010;
            if ((btns & WGI::DPadDown) == WGI::DPadDown) currentButtons |= 0x0020;
            if ((btns & WGI::LeftShoulder) == WGI::LeftShoulder) currentButtons |= 0x0040;
            if ((btns & WGI::RightShoulder) == WGI::RightShoulder) currentButtons |= 0x0080;
            if ((btns & WGI::View) == WGI::View) currentButtons |= 0x0100;
            if ((btns & WGI::Menu) == WGI::Menu) currentButtons |= 0x0200;
            leftThumbY = reading.LeftThumbstickY;
            ltPressed = reading.LeftTrigger > 0.5;
            rtPressed = reading.RightTrigger > 0.5;
        }
    } catch (...) {
        // Fallback to XInput below
    }

    XINPUT_STATE xstate = {};
    if (currentButtons == 0 && XInputGetState(0, &xstate) == ERROR_SUCCESS) {
        const WORD b = xstate.Gamepad.wButtons;
        if (b & XINPUT_GAMEPAD_A) currentButtons |= 0x0001;
        if (b & XINPUT_GAMEPAD_B) currentButtons |= 0x0002;
        if (b & XINPUT_GAMEPAD_X) currentButtons |= 0x0004;
        if (b & XINPUT_GAMEPAD_Y) currentButtons |= 0x0008;
        if (b & XINPUT_GAMEPAD_DPAD_UP) currentButtons |= 0x0010;
        if (b & XINPUT_GAMEPAD_DPAD_DOWN) currentButtons |= 0x0020;
        if (b & XINPUT_GAMEPAD_LEFT_SHOULDER) currentButtons |= 0x0040;
        if (b & XINPUT_GAMEPAD_RIGHT_SHOULDER) currentButtons |= 0x0080;
        if (b & XINPUT_GAMEPAD_BACK) currentButtons |= 0x0100;
        if (b & XINPUT_GAMEPAD_START) currentButtons |= 0x0200;
        if (std::abs(xstate.Gamepad.sThumbLY) > XINPUT_GAMEPAD_LEFT_THUMB_DEADZONE) {
            leftThumbY = static_cast<double>(xstate.Gamepad.sThumbLY) / 32767.0;
        }
        ltPressed = xstate.Gamepad.bLeftTrigger > 128;
        rtPressed = xstate.Gamepad.bRightTrigger > 128;
    }

    const uint32_t rising = currentButtons & (~m_prevButtons);
    m_prevButtons = currentButtons;

    constexpr int kMenuCount = 6;
    if ((rising & 0x0010) || (leftThumbY > 0.55 && m_navCooldown <= 0.0)) {
        state.selectedMenuIndex = (state.selectedMenuIndex + (kMenuCount - 1)) % kMenuCount;
        m_navCooldown = 0.22;
    }
    if ((rising & 0x0020) || (leftThumbY < -0.55 && m_navCooldown <= 0.0)) {
        state.selectedMenuIndex = (state.selectedMenuIndex + 1) % kMenuCount;
        m_navCooldown = 0.22;
    }

    if (rising & 0x0001) {
        if (state.emergencyCrashActive) {
            state.emergencyCrashActive = false;
        } else {
            switch (state.selectedMenuIndex) {
                case 0: actions.toggleTraining = true; break;
                case 1: actions.toggleEngine = true; break;
                case 2: actions.cyclePreset = true; break;
                case 3: actions.cycleDreamSteps = true; break;
                case 4: actions.triggerDialScan = true; break;
                case 5: actions.triggerReset = true; break;
            }
        }
    }

    if (rising & 0x0002) {
        if (state.showSystemInfo) {
            state.showSystemInfo = false;
        } else {
            actions.triggerReset = true;
        }
    }

    if (rising & 0x0004) actions.triggerDream = true;
    if (rising & 0x0008) actions.triggerDialScan = true;

    // LB / RB: Cycle resolution (8, 16, 32, 64, 128) + clamp batch at 128x128
    if (rising & 0x0040) {
        if (state.config.resolution == 128) state.config.resolution = 64;
        else if (state.config.resolution == 64) state.config.resolution = 32;
        else if (state.config.resolution == 32) state.config.resolution = 16;
        else if (state.config.resolution == 16) state.config.resolution = 8;
        state.config.batchSize = Trainer::ClampBatchForResolution(state.config.batchSize, state.config.resolution, state.config.engine);
        actions.resolutionChanged = true;
    }
    if (rising & 0x0080) {
        if (state.config.resolution == 8) state.config.resolution = 16;
        else if (state.config.resolution == 16) state.config.resolution = 32;
        else if (state.config.resolution == 32) state.config.resolution = 64;
        else if (state.config.resolution == 64) state.config.resolution = 128;
        state.config.batchSize = Trainer::ClampBatchForResolution(state.config.batchSize, state.config.resolution, state.config.engine);
        actions.resolutionChanged = true;
    }

    // LT / RT: Adjust batch size (clamped to 1..2 GPU / 1..4 CPU at 128x128, 1..32 otherwise)
    if (ltPressed && !m_prevLT) {
        const uint32_t dec = (state.config.batchSize > 4) ? state.config.batchSize - 4 : state.config.batchSize - 1;
        state.config.batchSize = Trainer::ClampBatchForResolution(dec, state.config.resolution, state.config.engine);
        actions.batchChanged = true;
    }
    if (rtPressed && !m_prevRT) {
        const uint32_t inc = (state.config.batchSize >= 4) ? state.config.batchSize + 4 : state.config.batchSize + 1;
        state.config.batchSize = Trainer::ClampBatchForResolution(inc, state.config.resolution, state.config.engine);
        actions.batchChanged = true;
    }
    m_prevLT = ltPressed;
    m_prevRT = rtPressed;

    if (rising & 0x0100) {
        state.config.blindSpeedMode = !state.config.blindSpeedMode;
    }
    if (rising & 0x0200) {
        state.showSystemInfo = !state.showSystemInfo;
    }

    return actions;
}

void GamepadUI::DrawDreamBitmap(
    ID2D1DeviceContext2* ctx,
    const D2D1_RECT_F& destRect,
    const std::vector<float>& dreamRgbCHW,
    uint32_t dreamSize) {
    if (dreamSize == 0 || dreamRgbCHW.size() < static_cast<size_t>(3) * dreamSize * dreamSize) {
        return;
    }
    const size_t hw = static_cast<size_t>(dreamSize) * dreamSize;
    std::vector<uint8_t> bgra(hw * 4);
    for (size_t i = 0; i < hw; ++i) {
        const auto toU8 = [](float v) -> uint8_t {
            const float c = std::clamp((v + 1.0f) * 0.5f, 0.0f, 1.0f);
            return static_cast<uint8_t>(c * 255.0f + 0.5f);
        };
        bgra[i * 4 + 0] = toU8(dreamRgbCHW[2 * hw + i]);
        bgra[i * 4 + 1] = toU8(dreamRgbCHW[1 * hw + i]);
        bgra[i * 4 + 2] = toU8(dreamRgbCHW[0 * hw + i]);
        bgra[i * 4 + 3] = 255;
    }

    if (!m_dreamBitmap || m_dreamBitmapSize != dreamSize) {
        m_dreamBitmap.Reset();
        D2D1_BITMAP_PROPERTIES props = D2D1::BitmapProperties(
            D2D1::PixelFormat(DXGI_FORMAT_B8G8R8A8_UNORM, D2D1_ALPHA_MODE_IGNORE));
        ctx->CreateBitmap(D2D1::SizeU(dreamSize, dreamSize), bgra.data(), dreamSize * 4, &props, &m_dreamBitmap);
        m_dreamBitmapSize = dreamSize;
    } else {
        D2D1_RECT_U copyRect = D2D1::RectU(0, 0, dreamSize, dreamSize);
        m_dreamBitmap->CopyFromMemory(&copyRect, bgra.data(), dreamSize * 4);
    }

    if (m_dreamBitmap) {
        ctx->DrawBitmap(m_dreamBitmap.Get(), destRect, 1.0f, D2D1_BITMAP_INTERPOLATION_MODE_NEAREST_NEIGHBOR);
    }
}

HRESULT GamepadUI::Render(
    uint32_t backBufferIndex,
    const UIState& state,
    const TrainingMetrics& metrics,
    const SystemTelemetry& telemetry,
    const std::vector<float>& dreamRgbCHW,
    uint32_t dreamSize) {
    if (backBufferIndex >= m_wrappedBackBuffers.size()) {
        return E_INVALIDARG;
    }

    ID3D11Resource* wrapped = m_wrappedBackBuffers[backBufferIndex].Get();
    m_d3d11On12Device->AcquireWrappedResources(&wrapped, 1);

    m_d2dContext->SetTarget(m_d2dRenderTargets[backBufferIndex].Get());
    m_d2dContext->BeginDraw();

    const float W = static_cast<float>(m_width);
    const float H = static_cast<float>(m_height);
    const float safeX = W * 0.05f;
    const float safeY = H * 0.05f;

    Microsoft::WRL::ComPtr<ID2D1SolidColorBrush> textBrush;
    Microsoft::WRL::ComPtr<ID2D1SolidColorBrush> mutedBrush;
    Microsoft::WRL::ComPtr<ID2D1SolidColorBrush> accentBrush;
    Microsoft::WRL::ComPtr<ID2D1SolidColorBrush> panelBrush;
    Microsoft::WRL::ComPtr<ID2D1SolidColorBrush> borderBrush;
    Microsoft::WRL::ComPtr<ID2D1SolidColorBrush> redBrush;

    m_d2dContext->CreateSolidColorBrush(D2D1::ColorF(0xF1F5F9), &textBrush);
    m_d2dContext->CreateSolidColorBrush(D2D1::ColorF(0x94A3B8), &mutedBrush);
    m_d2dContext->CreateSolidColorBrush(D2D1::ColorF(0x22C55E), &accentBrush);
    m_d2dContext->CreateSolidColorBrush(D2D1::ColorF(0x121821), &panelBrush);
    m_d2dContext->CreateSolidColorBrush(D2D1::ColorF(0x1E293B), &borderBrush);
    m_d2dContext->CreateSolidColorBrush(D2D1::ColorF(0xDC2626), &redBrush);

    if (state.emergencyCrashActive) {
        m_d2dContext->Clear(D2D1::ColorF(0x2B0909));
        const std::wstring header = L"EMERGENCY RECOVERY SHIELD — GRAPHICS / RUNTIME EXCEPTION";
        const std::wstring body = ToWide(state.crashTitle + "\n\n" + state.crashDetail +
            "\n\nLogged to crash_log.txt. Press [A] to reinitialize pipeline.");
        m_d2dContext->DrawText(header.c_str(), static_cast<UINT32>(header.size()), m_crashFormat.Get(),
            D2D1::RectF(safeX, safeY, W - safeX, safeY + 90.0f), redBrush.Get());
        m_d2dContext->DrawText(body.c_str(), static_cast<UINT32>(body.size()), m_monoFormat.Get(),
            D2D1::RectF(safeX, safeY + 110.0f, W - safeX, H - safeY), textBrush.Get());
    } else {
        m_d2dContext->Clear(D2D1::ColorF(0x0B0F14));

        const std::wstring title = L"XBOX OT-FM 2D NEURAL ENGINE (0% PRE-TRAINED)";
        m_d2dContext->DrawText(title.c_str(), static_cast<UINT32>(title.size()), m_titleFormat.Get(),
            D2D1::RectF(safeX, safeY, W - safeX, safeY + 60.0f), textBrush.Get());

        wchar_t lanBuf[256];
        swprintf_s(lanBuf, L"Phone LAN: %S  |  Lessons: %llu  |  Engine: %s  |  Mode: %s",
            telemetry.lanAddress.c_str(),
            static_cast<unsigned long long>(metrics.stepCount),
            (state.config.engine == ComputeEngine::GPU && telemetry.gpuComputeActive) ? L"GPU" : L"CPU",
            state.config.blindSpeedMode ? L"BLIND SPEED (5s refresh)" : (state.isTraining ? L"TRAINING" : L"PAUSED"));
        m_d2dContext->DrawText(lanBuf, static_cast<UINT32>(wcslen(lanBuf)), m_monoFormat.Get(),
            D2D1::RectF(safeX, safeY + 58.0f, W - safeX, safeY + 96.0f), accentBrush.Get());

        const float colLeftW = (W - 2.0f * safeX) * 0.45f;
        const float topY = safeY + 115.0f;
        const wchar_t* engineLabel = (state.config.engine == ComputeEngine::GPU)
            ? L"[A] Engine: GPU (D3D12 Compute)"
            : L"[A] Engine: CPU (Host Fallback)";

        const wchar_t* menuItems[6] = {
            state.isTraining ? L"[A] Pause Training" : L"[A] Start Training",
            engineLabel,
            L"[A] Cycle Brain Power Preset",
            L"[A] Cycle Dream Euler Steps",
            L"[Y] Scan Dials (Busy vs Sleeping)",
            L"[B] Reset Brain to Pure Noise"
        };

        for (int i = 0; i < 6; ++i) {
            const float y0 = topY + i * 62.0f;
            const D2D1_RECT_F itemRect = D2D1::RectF(safeX, y0, safeX + colLeftW, y0 + 52.0f);
            m_d2dContext->FillRectangle(itemRect, panelBrush.Get());
            m_d2dContext->DrawRectangle(itemRect, (i == state.selectedMenuIndex) ? accentBrush.Get() : borderBrush.Get(), (i == state.selectedMenuIndex) ? 3.0f : 1.0f);
            m_d2dContext->DrawText(menuItems[i], static_cast<UINT32>(wcslen(menuItems[i])), m_bodyFormat.Get(),
                D2D1::RectF(safeX + 18.0f, y0 + 10.0f, safeX + colLeftW - 18.0f, y0 + 48.0f), textBrush.Get());
        }

        wchar_t statsBuf[512];
        if (state.isTraining && state.config.blindSpeedMode) {
            swprintf_s(statsBuf,
                L"Architecture : %s\n"
                L"Resolution   : %ux%u (LB/RB)\n"
                L"Batch Size   : %u (LT/RT)\n"
                L"MSE Loss     : BYPASSED (Blind Mode)\n"
                L"OT Cost      : BYPASSED (Blind Mode)\n"
                L"Speed        : %.1f lessons/sec\n"
                L"Lessons (5s) : %llu",
                PresetLabel(state.preset),
                state.config.resolution, state.config.resolution,
                state.config.batchSize,
                metrics.lessonsPerSec,
                static_cast<unsigned long long>(metrics.stepCount));
        } else {
            swprintf_s(statsBuf,
                L"Architecture : %s\n"
                L"Resolution   : %ux%u (LB/RB)\n"
                L"Batch Size   : %u (LT/RT)\n"
                L"Euler Steps  : %u\n"
                L"MSE Loss     : %.5f\n"
                L"OT Cost      : %.4f (Rand %.4f)\n"
                L"Speed        : %.1f lessons/sec\n"
                L"Busy Dials   : %llu / %llu",
                PresetLabel(state.preset),
                state.config.resolution, state.config.resolution,
                state.config.batchSize,
                state.config.dreamSteps,
                metrics.lastMseLoss,
                metrics.lastOptimalOtCost, metrics.lastRandomOtCost,
                metrics.lessonsPerSec,
                static_cast<unsigned long long>(metrics.lastDialScan.busyDials),
                static_cast<unsigned long long>(metrics.lastDialScan.totalDials));
        }

        const D2D1_RECT_F statsRect = D2D1::RectF(safeX, topY + 388.0f, safeX + colLeftW, H - safeY - 50.0f);
        m_d2dContext->FillRectangle(statsRect, panelBrush.Get());
        m_d2dContext->DrawRectangle(statsRect, borderBrush.Get(), 1.0f);
        m_d2dContext->DrawText(statsBuf, static_cast<UINT32>(wcslen(statsBuf)), m_monoFormat.Get(),
            D2D1::RectF(statsRect.left + 18.0f, statsRect.top + 14.0f, statsRect.right - 18.0f, statsRect.bottom - 14.0f), textBrush.Get());

        // Right column: 2D Dream Canvas (completely skipped while training in Blind Speed mode!)
        const float rightX = safeX + colLeftW + 40.0f;
        const float dreamBoxSize = std::min(W - safeX - rightX, H - safeY - topY - 80.0f);
        const D2D1_RECT_F dreamRect = D2D1::RectF(rightX, topY, rightX + dreamBoxSize, topY + dreamBoxSize);
        m_d2dContext->FillRectangle(dreamRect, panelBrush.Get());

        if (state.isTraining && state.config.blindSpeedMode) {
            const std::wstring blindMsg = L"BLIND SPEED MODE ACTIVE\n\nCanvas Rendering Skipped\nLoss & Dream Bypassed\nWorker Running on Background Thread";
            m_d2dContext->DrawText(blindMsg.c_str(), static_cast<UINT32>(blindMsg.size()), m_monoFormat.Get(),
                D2D1::RectF(dreamRect.left + 32.0f, dreamRect.top + dreamBoxSize * 0.36f, dreamRect.right - 32.0f, dreamRect.bottom - 32.0f), accentBrush.Get());
        } else {
            DrawDreamBitmap(m_d2dContext.Get(), dreamRect, dreamRgbCHW, dreamSize);
        }
        m_d2dContext->DrawRectangle(dreamRect, borderBrush.Get(), 2.0f);

        const std::wstring footer = L"[X] Dream Now   |   [View] Blind Speed   |   [Menu] System Info";
        m_d2dContext->DrawText(footer.c_str(), static_cast<UINT32>(footer.size()), m_monoFormat.Get(),
            D2D1::RectF(safeX, H - safeY - 36.0f, W - safeX, H - safeY), mutedBrush.Get());

        if (state.showSystemInfo) {
            const D2D1_RECT_F modalRect = D2D1::RectF(W * 0.2f, H * 0.22f, W * 0.8f, H * 0.78f);
            m_d2dContext->FillRectangle(modalRect, panelBrush.Get());
            m_d2dContext->DrawRectangle(modalRect, accentBrush.Get(), 2.5f);
            wchar_t sysBuf[512];
            swprintf_s(sysBuf,
                L"XBOX SYSTEM TELEMETRY\n\n"
                L"GPU Adapter      : %s (%s)\n"
                L"Dedicated VRAM   : %llu MB\n"
                L"Compute Engine   : %s\n"
                L"Throughput       : %.1f lessons/sec\n"
                L"Memory Deck      : %u 2D cards\n"
                L"Pipeline Recover : %u\n"
                L"Device Status    : %S\n\n"
                L"Press [Menu] or [B] to close.",
                telemetry.gpuAdapterName.c_str(),
                telemetry.featureLevelLabel.c_str(),
                static_cast<unsigned long long>(telemetry.dedicatedVramMB),
                (state.config.engine == ComputeEngine::GPU && telemetry.gpuComputeActive) ? L"GPU RESIDENT HLSL" : L"CPU FALLBACK",
                metrics.lessonsPerSec,
                metrics.deckCardCount,
                telemetry.deviceLossRecoveries,
                telemetry.deviceStatus.c_str());
            m_d2dContext->DrawText(sysBuf, static_cast<UINT32>(wcslen(sysBuf)), m_monoFormat.Get(),
                D2D1::RectF(modalRect.left + 32.0f, modalRect.top + 32.0f, modalRect.right - 32.0f, modalRect.bottom - 32.0f), textBrush.Get());
        }
    }

    HRESULT hr = m_d2dContext->EndDraw();
    m_d3d11On12Device->ReleaseWrappedResources(&wrapped, 1);
    m_d3d11Context->Flush();
    return hr;
}

} // namespace XboxOTFM
