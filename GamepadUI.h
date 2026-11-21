#pragma once

#include "FMBrain.h"
#include "Trainer.h"

#include <d3d12.h>
#include <d3d11on12.h>
#include <d2d1_3.h>
#include <dwrite_3.h>
#include <wrl/client.h>

#include <cstdint>
#include <string>
#include <vector>

namespace XboxOTFM {

struct SystemTelemetry {
    std::wstring gpuAdapterName = L"Xbox Series S / AMD RDNA2";
    std::wstring featureLevelLabel = L"D3D 12.0";
    uint64_t dedicatedVramMB = 0;
    uint64_t usedVramMB = 0;
    std::string lanAddress = "http://127.0.0.1:8080";
    bool gpuComputeActive = false;
    bool directMLActive = false;
    uint32_t deviceLossRecoveries = 0;
    std::string deviceStatus = "NOMINAL";
};

struct UIState {
    bool isTraining = false;
    bool showSystemInfo = false;
    bool emergencyCrashActive = false;
    // 0: Train/Pause, 1: Engine GPU/CPU, 2: Brain Preset, 3: Dream Steps, 4: Scan Dials, 5: Reset Brain
    int selectedMenuIndex = 0;
    BrainPreset preset = BrainPreset::Medium80k;
    TrainerConfig config;
    std::string crashTitle;
    std::string crashDetail;
};

struct UIActions {
    bool toggleTraining = false;
    bool toggleEngine = false;
    bool triggerDream = false;
    bool triggerDialScan = false;
    bool triggerReset = false;
    bool cyclePreset = false;
    bool cycleDreamSteps = false;
    bool resolutionChanged = false;
    bool batchChanged = false;
};

class GamepadUI {
public:
    HRESULT Initialize(
        ID3D12Device* d3d12Device,
        ID3D12CommandQueue* cmdQueue,
        const std::vector<Microsoft::WRL::ComPtr<ID3D12Resource>>& backBuffers,
        uint32_t width,
        uint32_t height);

    void Release();

    // Polls Windows.Gaming.Input::Gamepad and XInput with edge-triggered buttons and thumbstick repeat.
    UIActions PollInput(UIState& state, double deltaSeconds);

    // Renders the 10-foot TV dashboard (skipping dream canvas in Blind Speed) or emergency crash canvas.
    HRESULT Render(
        uint32_t backBufferIndex,
        const UIState& state,
        const TrainingMetrics& metrics,
        const SystemTelemetry& telemetry,
        const std::vector<float>& dreamRgbCHW,
        uint32_t dreamSize);

private:
    void DrawDreamBitmap(
        ID2D1DeviceContext2* ctx,
        const D2D1_RECT_F& destRect,
        const std::vector<float>& dreamRgbCHW,
        uint32_t dreamSize);

    uint32_t m_width = 1920;
    uint32_t m_height = 1080;

    Microsoft::WRL::ComPtr<ID3D11Device> m_d3d11Device;
    Microsoft::WRL::ComPtr<ID3D11DeviceContext> m_d3d11Context;
    Microsoft::WRL::ComPtr<ID3D11On12Device> m_d3d11On12Device;
    Microsoft::WRL::ComPtr<ID2D1Factory3> m_d2dFactory;
    Microsoft::WRL::ComPtr<ID2D1Device2> m_d2dDevice;
    Microsoft::WRL::ComPtr<ID2D1DeviceContext2> m_d2dContext;
    Microsoft::WRL::ComPtr<IDWriteFactory> m_dwriteFactory;

    Microsoft::WRL::ComPtr<IDWriteTextFormat> m_titleFormat;
    Microsoft::WRL::ComPtr<IDWriteTextFormat> m_bodyFormat;
    Microsoft::WRL::ComPtr<IDWriteTextFormat> m_monoFormat;
    Microsoft::WRL::ComPtr<IDWriteTextFormat> m_crashFormat;

    std::vector<Microsoft::WRL::ComPtr<ID3D11Resource>> m_wrappedBackBuffers;
    std::vector<Microsoft::WRL::ComPtr<ID2D1Bitmap1>> m_d2dRenderTargets;
    Microsoft::WRL::ComPtr<ID2D1Bitmap> m_dreamBitmap;
    uint32_t m_dreamBitmapSize = 0;

    uint32_t m_prevButtons = 0;
    bool m_prevLT = false;
    bool m_prevRT = false;
    double m_navCooldown = 0.0;
};

} // namespace XboxOTFM
