#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <winrt/Windows.ApplicationModel.Core.h>
#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.UI.Core.h>

#include "FMBrain.h"
#include "GamepadUI.h"
#include "Trainer.h"
#include "WebBridge.h"

#include <atomic>
#include <chrono>
#include <exception>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")

using namespace XboxOTFM;
using namespace winrt::Windows::ApplicationModel::Core;
using namespace winrt::Windows::UI::Core;

namespace {

constexpr uint32_t kFrameCount = 2;

const wchar_t* FeatureLevelName(D3D_FEATURE_LEVEL fl) {
    switch (fl) {
        case D3D_FEATURE_LEVEL_12_0: return L"D3D 12.0";
        case D3D_FEATURE_LEVEL_11_1: return L"D3D 11.1";
        case D3D_FEATURE_LEVEL_11_0: return L"D3D 11.0";
        default: return L"D3D";
    }
}

} // namespace

class XboxOTFMApp : public winrt::implements<XboxOTFMApp, IFrameworkViewSource, IFrameworkView> {
public:
    IFrameworkView CreateView() {
        return *this;
    }

    void Initialize(CoreApplicationView const&) {
        std::set_terminate([]() {
            WebBridge::WriteCrashLogToDisk("[std::terminate] Unhandled C++ exception terminated process.");
            std::abort();
        });
    }

    void SetWindow(CoreWindow const& window) {
        m_window = window;
        window.Closed([this](auto&&, auto&&) { m_running = false; });
        window.SizeChanged([this](auto&&, WindowSizeChangedEventArgs const& args) {
            m_width = std::max(640u, static_cast<uint32_t>(args.Size().Width));
            m_height = std::max(480u, static_cast<uint32_t>(args.Size().Height));
            m_resizePending = true;
        });
    }

    void Load(winrt::hstring const&) {}

    void Run() {
        m_window.Activate();

        m_brain.InitializeFromScratch(m_uiState.preset);
        m_trainer.GenerateDream(m_brain, m_uiState.config);
        m_displayedMetrics = m_trainer.SnapshotMetrics();
        m_displayedMetrics.lastDialScan = m_brain.ScanDials(0.01f);

        if (FAILED(InitializeD3D12())) {
            m_uiState.config.engine = ComputeEngine::CPU;
            m_telemetry.gpuComputeActive = false;
            m_uiState.emergencyCrashActive = true;
            m_uiState.crashTitle = "D3D12 INITIALIZATION FAILURE";
            m_uiState.crashDetail = "Unable to acquire Xbox / Direct3D device (12.0 / 11.1 / 11.0 fallback failed).";
            WebBridge::WriteCrashLogToDisk(m_uiState.crashTitle + ": " + m_uiState.crashDetail);
        }

        m_webBridge.Start(8080);
        m_telemetry.lanAddress = m_webBridge.LocalEndpointUrl();

        // Start dedicated background compute worker thread so gamepad UI stays 100% responsive
        m_workerRunning.store(true);
        m_workerThread = std::thread(&XboxOTFMApp::BackgroundTrainingWorker, this);

        auto lastTick = std::chrono::high_resolution_clock::now();
        m_lastBlindDisplayRefresh = lastTick;

        while (m_running) {
            m_window.Dispatcher().ProcessEvents(CoreProcessEventsOption::ProcessAllIfPresent);

            const auto now = std::chrono::high_resolution_clock::now();
            const double dt = std::chrono::duration<double>(now - lastTick).count();
            lastTick = now;

            if (m_resizePending) {
                RebuildSwapChain();
                m_resizePending = false;
            }

            RunGuardedFrame(dt);
        }

        m_trainingActive.store(false);
        m_workerRunning.store(false);
        if (m_workerThread.joinable()) {
            m_workerThread.join();
        }

        m_webBridge.Stop();
        WaitForGpu();
        m_gamepadUI.Release();
        m_brain.ReleaseGPU();
    }

    void Uninitialize() {}

private:
    void BackgroundTrainingWorker() {
        while (m_workerRunning.load()) {
            if (!m_trainingActive.load() || m_uiState.emergencyCrashActive) {
                std::this_thread::sleep_for(std::chrono::milliseconds(15));
                continue;
            }

            try {
                // Synchronize with engine/preset switches so GPU dispatches are never interrupted mid-flight
                std::lock_guard<std::mutex> engineLock(m_engineMutex);
                if (!m_trainingActive.load() || m_uiState.emergencyCrashActive) {
                    continue;
                }

                TrainerConfig cfgSnapshot;
                {
                    std::lock_guard<std::mutex> lock(m_configMutex);
                    cfgSnapshot = m_uiState.config;
                }

                m_trainer.Step(m_brain, cfgSnapshot);
                const auto snap = m_trainer.SnapshotMetrics();

                if (snap.gpuFallbackTriggered) {
                    std::lock_guard<std::mutex> lock(m_configMutex);
                    m_uiState.config.engine = ComputeEngine::CPU;
                    m_telemetry.gpuComputeActive = false;
                    m_webBridge.AppendLogLine("[Engine] GPU compute dispatch failed; gracefully switched to CPU trainer.");
                }

                // Only generate dreams when NOT in Blind Speed mode!
                if (!cfgSnapshot.blindSpeedMode && (snap.stepCount % 12 == 0)) {
                    m_trainer.GenerateDream(m_brain, cfgSnapshot);
                }
            } catch (const std::exception& ex) {
                m_trainingActive.store(false);
                WebBridge::WriteCrashLogToDisk(std::string("[WorkerException] ") + ex.what());
                m_uiState.isTraining = false;
                m_uiState.emergencyCrashActive = true;
                m_uiState.crashTitle = "COMPUTE WORKER EXCEPTION";
                m_uiState.crashDetail = ex.what();
            } catch (...) {
                m_trainingActive.store(false);
                WebBridge::WriteCrashLogToDisk("[WorkerException] Unknown C++ exception in training worker.");
                m_uiState.isTraining = false;
                m_uiState.emergencyCrashActive = true;
                m_uiState.crashTitle = "COMPUTE WORKER EXCEPTION";
                m_uiState.crashDetail = "Unknown exception caught safely in background worker.";
            }
        }
    }

    void RunGuardedFrame(double dt) {
        try {
            StepAndRenderFrame(dt);
        } catch (const std::exception& ex) {
            WebBridge::WriteCrashLogToDisk(std::string("[C++ Exception] ") + ex.what());
            m_trainingActive.store(false);
            m_uiState.isTraining = false;
            m_uiState.emergencyCrashActive = true;
            m_uiState.crashTitle = "RUNTIME EXCEPTION CAUGHT";
            m_uiState.crashDetail = ex.what();
        } catch (...) {
            WebBridge::WriteCrashLogToDisk("[C++ Exception] Unknown runtime exception caught.");
            m_trainingActive.store(false);
            m_uiState.isTraining = false;
            m_uiState.emergencyCrashActive = true;
            m_uiState.crashTitle = "RUNTIME EXCEPTION CAUGHT";
            m_uiState.crashDetail = "Unknown runtime exception caught safely.";
        }
    }

    void StepAndRenderFrame(double dt) {
        UIActions actions;
        {
            std::lock_guard<std::mutex> lock(m_configMutex);
            actions = m_gamepadUI.PollInput(m_uiState, dt);
            m_activeResolution.store(m_uiState.config.resolution);
        }

        if (actions.toggleTraining) {
            const bool nextState = !m_trainingActive.load();
            m_trainingActive.store(nextState);
            m_uiState.isTraining = nextState;
            m_webBridge.AppendLogLine(nextState ? "[Trainer] Background worker started." : "[Trainer] Training paused.");
        }

        if (actions.toggleEngine) {
            // Pause worker and acquire m_engineMutex so active GPU/CPU dispatches complete before switching
            const bool wasTraining = m_trainingActive.exchange(false);
            {
                std::lock_guard<std::mutex> engineLock(m_engineMutex);
                std::lock_guard<std::mutex> cfgLock(m_configMutex);
                if (m_uiState.config.engine == ComputeEngine::GPU) {
                    m_brain.ReadbackWeightsFromGPU();
                    m_uiState.config.engine = ComputeEngine::CPU;
                    m_uiState.config.batchSize = Trainer::ClampBatchForResolution(
                        m_uiState.config.batchSize, m_uiState.config.resolution, ComputeEngine::CPU);
                    m_telemetry.gpuComputeActive = false;
                    m_webBridge.AppendLogLine("[Engine] Switched to CPU trainer.");
                } else {
                    if (m_brain.IsGPUReady() || (m_d3dDevice && SUCCEEDED(m_brain.InitializeGPU(m_d3dDevice.Get(), m_cmdQueue.Get())))) {
                        m_brain.UploadWeightsToGPU();
                        m_uiState.config.engine = ComputeEngine::GPU;
                        m_uiState.config.batchSize = Trainer::ClampBatchForResolution(
                            m_uiState.config.batchSize, m_uiState.config.resolution, ComputeEngine::GPU);
                        m_telemetry.gpuComputeActive = true;
                        m_webBridge.AppendLogLine("[Engine] Switched to GPU Compute Shader trainer.");
                    } else {
                        m_uiState.config.engine = ComputeEngine::CPU;
                        m_telemetry.gpuComputeActive = false;
                        m_webBridge.AppendLogLine("[Engine] GPU shader init failed; remaining on CPU trainer.");
                    }
                }
            }
            if (wasTraining) {
                m_trainingActive.store(true);
            }
        }

        if (actions.cyclePreset) {
            const bool wasTraining = m_trainingActive.exchange(false);
            m_uiState.isTraining = false;
            {
                std::lock_guard<std::mutex> engineLock(m_engineMutex);
                const uint32_t next = (static_cast<uint32_t>(m_uiState.preset) + 1) % 3;
                m_uiState.preset = static_cast<BrainPreset>(next);
                m_brain.InitializeFromScratch(m_uiState.preset);
                if (m_d3dDevice && m_uiState.config.engine == ComputeEngine::GPU) {
                    if (FAILED(m_brain.InitializeGPU(m_d3dDevice.Get(), m_cmdQueue.Get()))) {
                        m_uiState.config.engine = ComputeEngine::CPU;
                        m_telemetry.gpuComputeActive = false;
                    } else {
                        m_telemetry.gpuComputeActive = true;
                    }
                }
                m_trainer.ResetMetrics();
                m_trainer.SetDialScanReport(m_brain.ScanDials(0.01f));
            }
            if (wasTraining) {
                m_trainingActive.store(true);
                m_uiState.isTraining = true;
            }
        }

        if (actions.cycleDreamSteps) {
            std::lock_guard<std::mutex> lock(m_configMutex);
            if (m_uiState.config.dreamSteps == 1) m_uiState.config.dreamSteps = 2;
            else if (m_uiState.config.dreamSteps == 2) m_uiState.config.dreamSteps = 6;
            else m_uiState.config.dreamSteps = 1;
        }

        if (actions.triggerDialScan) {
            m_trainer.SetDialScanReport(m_brain.ScanDials(0.01f));
        }

        if (actions.triggerReset) {
            const bool wasTraining = m_trainingActive.exchange(false);
            {
                std::lock_guard<std::mutex> engineLock(m_engineMutex);
                m_brain.InitializeFromScratch(m_uiState.preset);
                if (m_brain.IsGPUReady()) {
                    m_brain.UploadWeightsToGPU();
                }
                m_trainer.ResetMetrics();
                m_trainer.GenerateDream(m_brain, m_uiState.config);
                m_trainer.SetDialScanReport(m_brain.ScanDials(0.01f));
            }
            m_trainingActive.store(wasTraining);
            m_uiState.isTraining = wasTraining;
        }

        if (actions.triggerDream) {
            m_trainer.GenerateDream(m_brain, m_uiState.config);
        }

        // Blind Speed counter throttling: refresh displayed step count only once every 5 seconds,
        // and sleep 16ms when skipping frame draws so the UI thread never spin-locks the CPU!
        const auto now = std::chrono::high_resolution_clock::now();
        if (m_uiState.isTraining && m_uiState.config.blindSpeedMode) {
            const double secSinceRefresh = std::chrono::duration<double>(now - m_lastBlindDisplayRefresh).count();
            if (secSinceRefresh >= 5.0) {
                m_displayedMetrics = m_trainer.SnapshotMetrics();
                m_lastBlindDisplayRefresh = now;
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(16));
                return;
            }
        } else {
            m_displayedMetrics = m_trainer.SnapshotMetrics();
            m_lastBlindDisplayRefresh = now;
        }

        if (!m_d3dDevice || !m_swapChain) return;

        HRESULT hr = ExecuteFrameCommands();
        if (hr == DXGI_ERROR_DEVICE_REMOVED || hr == DXGI_ERROR_DEVICE_RESET) {
            const HRESULT reason = m_d3dDevice ? m_d3dDevice->GetDeviceRemovedReason() : hr;
            std::ostringstream oss;
            oss << "DXGI_ERROR_DEVICE_REMOVED (Reason: 0x" << std::hex << reason << std::dec << ")";
            WebBridge::WriteCrashLogToDisk("[D3D12] " + oss.str());
            m_webBridge.AppendLogLine("[D3D12] " + oss.str() + " - Recovering pipeline...");
            m_telemetry.deviceLossRecoveries++;
            m_telemetry.deviceStatus = "RECOVERING";

            const bool wasTraining = m_trainingActive.exchange(false);
            HRESULT recHr = E_FAIL;
            {
                std::lock_guard<std::mutex> engineLock(m_engineMutex);
                recHr = RecoverD3D12Pipeline();
            }
            if (FAILED(recHr)) {
                m_uiState.config.engine = ComputeEngine::CPU;
                m_telemetry.gpuComputeActive = false;
                m_uiState.emergencyCrashActive = true;
                m_uiState.crashTitle = "D3D12 DEVICE REMOVAL RECOVERY FAILED";
                m_uiState.crashDetail = oss.str();
            } else {
                m_telemetry.deviceStatus = "NOMINAL (RECOVERED)";
                if (wasTraining) m_trainingActive.store(true);
            }
        }
    }

    HRESULT InitializeD3D12() {
        UINT dxgiFlags = 0;
#ifdef _DEBUG
        Microsoft::WRL::ComPtr<ID3D12Debug> debugCtrl;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debugCtrl)))) {
            debugCtrl->EnableDebugLayer();
            dxgiFlags |= DXGI_CREATE_FACTORY_DEBUG;
        }
#endif
        HRESULT hr = CreateDXGIFactory2(dxgiFlags, IID_PPV_ARGS(&m_dxgiFactory));
        if (FAILED(hr)) return hr;

        constexpr D3D_FEATURE_LEVEL kLevels[] = {
            D3D_FEATURE_LEVEL_12_0,
            D3D_FEATURE_LEVEL_11_1,
            D3D_FEATURE_LEVEL_11_0
        };

        Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter;
        for (UINT idx = 0; m_dxgiFactory->EnumAdapterByGpuPreference(idx, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&adapter)) != DXGI_ERROR_NOT_FOUND; ++idx) {
            DXGI_ADAPTER_DESC1 desc = {};
            adapter->GetDesc1(&desc);
            if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;

            for (D3D_FEATURE_LEVEL fl : kLevels) {
                if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), fl, IID_PPV_ARGS(&m_d3dDevice)))) {
                    m_telemetry.gpuAdapterName = desc.Description;
                    m_telemetry.featureLevelLabel = FeatureLevelName(fl);
                    m_telemetry.dedicatedVramMB = desc.DedicatedVideoMemory / (1024ULL * 1024ULL);
                    break;
                }
            }
            if (m_d3dDevice) break;
        }
        if (!m_d3dDevice) return E_FAIL;

        D3D12_COMMAND_QUEUE_DESC qDesc = {};
        qDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        hr = m_d3dDevice->CreateCommandQueue(&qDesc, IID_PPV_ARGS(&m_cmdQueue));
        if (FAILED(hr)) return hr;

        DXGI_SWAP_CHAIN_DESC1 scDesc = {};
        scDesc.Width = m_width;
        scDesc.Height = m_height;
        scDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        scDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        scDesc.BufferCount = kFrameCount;
        scDesc.SampleDesc.Count = 1;
        scDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;

        Microsoft::WRL::ComPtr<IDXGISwapChain1> sc1;
        hr = m_dxgiFactory->CreateSwapChainForCoreWindow(
            m_cmdQueue.Get(),
            reinterpret_cast<IUnknown*>(winrt::get_abi(m_window)),
            &scDesc,
            nullptr,
            &sc1);
        if (FAILED(hr)) return hr;
        hr = sc1.As(&m_swapChain);
        if (FAILED(hr)) return hr;

        m_frameIndex = m_swapChain->GetCurrentBackBufferIndex();

        for (uint32_t i = 0; i < kFrameCount; ++i) {
            hr = m_d3dDevice->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&m_cmdAllocators[i]));
            if (FAILED(hr)) return hr;
        }
        hr = m_d3dDevice->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, m_cmdAllocators[0].Get(), nullptr, IID_PPV_ARGS(&m_cmdList));
        if (FAILED(hr)) return hr;
        m_cmdList->Close();

        hr = m_d3dDevice->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_fence));
        if (FAILED(hr)) return hr;
        m_fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);

        m_backBuffers.resize(kFrameCount);
        for (uint32_t i = 0; i < kFrameCount; ++i) {
            hr = m_swapChain->GetBuffer(i, IID_PPV_ARGS(&m_backBuffers[i]));
            if (FAILED(hr)) return hr;
        }

        hr = m_gamepadUI.Initialize(m_d3dDevice.Get(), m_cmdQueue.Get(), m_backBuffers, m_width, m_height);
        if (FAILED(hr)) return hr;

        // Initialize GPU Compute + DirectML; fall back gracefully to CPU if shaders/GPU init fail
        if (SUCCEEDED(m_brain.InitializeGPU(m_d3dDevice.Get(), m_cmdQueue.Get()))) {
            m_telemetry.gpuComputeActive = true;
            m_telemetry.directMLActive = m_brain.IsDirectMLReady();
            m_uiState.config.engine = ComputeEngine::GPU;
        } else {
            m_telemetry.gpuComputeActive = false;
            m_telemetry.directMLActive = false;
            m_uiState.config.engine = ComputeEngine::CPU;
            WebBridge::WriteCrashLogToDisk("[Engine] GPU compute shader initialization failed; defaulting gracefully to CPU trainer.");
        }
        return S_OK;
    }

    HRESULT ExecuteFrameCommands() {
        HRESULT hr = m_cmdAllocators[m_frameIndex]->Reset();
        if (FAILED(hr)) return hr;
        hr = m_cmdList->Reset(m_cmdAllocators[m_frameIndex].Get(), nullptr);
        if (FAILED(hr)) return hr;

        D3D12_RESOURCE_BARRIER barrier = {};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition.pResource = m_backBuffers[m_frameIndex].Get();
        barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
        barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
        barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        m_cmdList->ResourceBarrier(1, &barrier);

        hr = m_cmdList->Close();
        if (FAILED(hr)) return hr;

        ID3D12CommandList* lists[] = { m_cmdList.Get() };
        m_cmdQueue->ExecuteCommandLists(1, lists);

        hr = m_d3dDevice->GetDeviceRemovedReason();
        if (FAILED(hr)) return hr;

        uint32_t dreamSize = m_uiState.config.resolution;
        std::vector<float> dreamCopy;
        if (!(m_uiState.isTraining && m_uiState.config.blindSpeedMode)) {
            dreamCopy = m_trainer.LatestDreamSnapshot(dreamSize);
        }

        hr = m_gamepadUI.Render(
            m_frameIndex,
            m_uiState,
            m_displayedMetrics,
            m_telemetry,
            dreamCopy,
            dreamSize);
        if (FAILED(hr)) return hr;

        hr = m_swapChain->Present(m_uiState.config.blindSpeedMode ? 0 : 1, 0);
        if (FAILED(hr)) return hr;

        WaitForGpu();
        m_frameIndex = m_swapChain->GetCurrentBackBufferIndex();
        return S_OK;
    }

    HRESULT RecoverD3D12Pipeline() {
        WaitForGpu();
        m_gamepadUI.Release();
        m_brain.ReleaseGPU();
        m_backBuffers.clear();
        m_cmdList.Reset();
        for (auto& alloc : m_cmdAllocators) alloc.Reset();
        m_fence.Reset();
        if (m_fenceEvent) {
            CloseHandle(m_fenceEvent);
            m_fenceEvent = nullptr;
        }
        m_swapChain.Reset();
        m_cmdQueue.Reset();
        m_d3dDevice.Reset();
        m_dxgiFactory.Reset();
        return InitializeD3D12();
    }

    void RebuildSwapChain() {
        if (!m_swapChain) return;
        WaitForGpu();
        m_gamepadUI.Release();
        m_backBuffers.clear();
        m_swapChain->ResizeBuffers(kFrameCount, m_width, m_height, DXGI_FORMAT_B8G8R8A8_UNORM, 0);
        m_backBuffers.resize(kFrameCount);
        for (uint32_t i = 0; i < kFrameCount; ++i) {
            m_swapChain->GetBuffer(i, IID_PPV_ARGS(&m_backBuffers[i]));
        }
        m_frameIndex = m_swapChain->GetCurrentBackBufferIndex();
        m_gamepadUI.Initialize(m_d3dDevice.Get(), m_cmdQueue.Get(), m_backBuffers, m_width, m_height);
    }

    void WaitForGpu() {
        if (!m_cmdQueue || !m_fence || !m_fenceEvent) return;
        const uint64_t val = ++m_fenceValue;
        if (SUCCEEDED(m_cmdQueue->Signal(m_fence.Get(), val))) {
            if (m_fence->GetCompletedValue() < val) {
                m_fence->SetEventOnCompletion(val, m_fenceEvent);
                WaitForSingleObject(m_fenceEvent, 2000);
            }
        }
    }

    CoreWindow m_window{ nullptr };
    bool m_running = true;
    bool m_resizePending = false;
    uint32_t m_width = 1920;
    uint32_t m_height = 1080;
    uint32_t m_frameIndex = 0;

    std::mutex m_configMutex;
    std::mutex m_engineMutex;
    std::atomic<bool> m_trainingActive{ false };
    std::atomic<bool> m_workerRunning{ false };
    std::atomic<uint32_t> m_activeResolution{ 16 };
    std::thread m_workerThread;
    std::chrono::high_resolution_clock::time_point m_lastBlindDisplayRefresh;

    UIState m_uiState;
    FMBrain m_brain;
    Trainer m_trainer;
    TrainingMetrics m_displayedMetrics;
    SystemTelemetry m_telemetry;
    GamepadUI m_gamepadUI;
    WebBridge m_webBridge{ m_brain, m_trainer, m_activeResolution, m_trainingActive };

    Microsoft::WRL::ComPtr<IDXGIFactory6> m_dxgiFactory;
    Microsoft::WRL::ComPtr<ID3D12Device> m_d3dDevice;
    Microsoft::WRL::ComPtr<ID3D12CommandQueue> m_cmdQueue;
    Microsoft::WRL::ComPtr<IDXGISwapChain3> m_swapChain;
    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> m_cmdAllocators[kFrameCount];
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> m_cmdList;
    std::vector<Microsoft::WRL::ComPtr<ID3D12Resource>> m_backBuffers;
    Microsoft::WRL::ComPtr<ID3D12Fence> m_fence;
    HANDLE m_fenceEvent = nullptr;
    uint64_t m_fenceValue = 0;
};

int WINAPI wWinMain(HINSTANCE, HINSTANCE, PWSTR, int) {
    winrt::init_apartment();
    CoreApplication::Run(winrt::make<XboxOTFMApp>());
    return 0;
}
