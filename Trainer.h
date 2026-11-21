#pragma once

#include "FMBrain.h"
#include "OptimalTransport.h"

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace XboxOTFM {

struct TrainerConfig {
    float learningRate = 0.001f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float adamEpsilon = 1e-8f;
    float gradClipNorm = 1.0f;
    uint32_t resolution = 16;   // 8, 16, 32, 64, or 128
    uint32_t batchSize = 8;     // 1..32 (clamped to 1..4 at 128x128)
    uint32_t dreamSteps = 6;    // 1, 2, or 6 Euler ODE steps
    bool blindSpeedMode = false;
    ComputeEngine engine = ComputeEngine::GPU;
};

struct TrainingMetrics {
    uint64_t stepCount = 0;
    float lastMseLoss = 0.0f;
    float emaMseLoss = 0.0f;
    float lessonsPerSec = 0.0f;
    float lastRandomOtCost = 0.0f;
    float lastOptimalOtCost = 0.0f;
    uint32_t deckCardCount = 0;
    bool usingGpuEngine = false;
    bool gpuFallbackTriggered = false;
    DialScanReport lastDialScan;
};

class Trainer {
public:
    Trainer();

    // Clamps batch size safely (1..2 on GPU / 1..4 on CPU at 128x128 to prevent OOM & TDR timeouts).
    static uint32_t ClampBatchForResolution(
        uint32_t batchSize,
        uint32_t resolution,
        ComputeEngine engine = ComputeEngine::CPU) noexcept;

    // Seeds built-in 2D procedural cards so training works immediately from scratch.
    void SeedStarterMemoryDeck(uint32_t resolution);

    // Appends a 2D RGB image tensor [3, H, W] in [-1, 1] into the memory deck.
    void AddCardToDeck(std::vector<float> rgbTensorCHW, uint32_t srcWidth, uint32_t srcHeight);
    void ClearDeck();

    // Executes one OT-FM training iteration (GPU Compute or CPU fallback).
    // In Blind Speed mode, bypasses MSE loss readback/computation for maximum throughput.
    float Step(FMBrain& brain, TrainerConfig& config);

    // Generates a new 2D Dream image from fresh Gaussian noise using Euler ODE integration.
    std::vector<float> GenerateDream(FMBrain& brain, const TrainerConfig& config);

    void ResetMetrics();
    TrainingMetrics SnapshotMetrics() const;
    std::vector<float> LatestDreamSnapshot(uint32_t& outSize) const;
    uint32_t DeckSize() const;

    void SetDialScanReport(const DialScanReport& report);

private:
    static std::vector<float> BilinearResizeCHW(
        const float* src,
        uint32_t srcW,
        uint32_t srcH,
        uint32_t dstW,
        uint32_t dstH);

    static void Conv2dBackward(
        const float* input,
        const float* gradOutput,
        const ConvLayerWeights& layer,
        uint32_t batch,
        uint32_t height,
        uint32_t width,
        std::vector<float>& gradInput,
        std::vector<float>& gradWeight,
        std::vector<float>& gradBias);

    static void AdamUpdate(
        std::vector<float>& param,
        const std::vector<float>& grad,
        std::vector<float>& m,
        std::vector<float>& v,
        uint64_t step,
        float lr,
        float beta1,
        float beta2,
        float eps,
        float clipScale);

    mutable std::mutex m_deckMutex;
    std::vector<std::vector<float>> m_rawCardsCHW;
    std::vector<uint32_t> m_rawCardSizes;

    mutable std::mutex m_stateMutex;
    TrainingMetrics m_metrics;
    std::vector<float> m_latestDreamCHW;
    uint32_t m_latestDreamSize = 16;

    OptimalTransportMatcher m_otMatcher;
    ForwardCache m_cache;
    std::mt19937_64 m_batchRng;
    std::mt19937_64 m_dreamRng;
};

} // namespace XboxOTFM
