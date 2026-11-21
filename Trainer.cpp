#include "Trainer.h"

#include <algorithm>
#include <chrono>
#include <cmath>

namespace XboxOTFM {

namespace {

inline float SiLUDerivative(float x) noexcept {
    const float clamped = std::clamp(x, -20.0f, 20.0f);
    const float sig = 1.0f / (1.0f + std::exp(-clamped));
    return sig * (1.0f + clamped * (1.0f - sig));
}

} // namespace

Trainer::Trainer()
    : m_otMatcher(0x9E3779B97F4A7C15ULL)
    , m_batchRng(0xA5A55A5A12345678ULL)
    , m_dreamRng(0xCAFEBABE12345678ULL) {
    SeedStarterMemoryDeck(16);
}

uint32_t Trainer::ClampBatchForResolution(
    uint32_t batchSize,
    uint32_t resolution,
    ComputeEngine engine) noexcept {
    if (resolution >= 128u) {
        const uint32_t cap = (engine == ComputeEngine::GPU) ? 2u : 4u;
        return std::clamp(batchSize, 1u, cap);
    }
    return std::clamp(batchSize, 1u, 32u);
}

void Trainer::SeedStarterMemoryDeck(uint32_t resolution) {
    std::lock_guard<std::mutex> lock(m_deckMutex);
    m_rawCardsCHW.clear();
    m_rawCardSizes.clear();

    const uint32_t s = std::max(8u, std::min(128u, resolution));
    const size_t hw = static_cast<size_t>(s) * s;

    // Card 0: Emerald radial ring target
    {
        std::vector<float> card(3 * hw);
        for (uint32_t y = 0; y < s; ++y) {
            const float ny = 2.0f * static_cast<float>(y) / static_cast<float>(s - 1) - 1.0f;
            for (uint32_t x = 0; x < s; ++x) {
                const float nx = 2.0f * static_cast<float>(x) / static_cast<float>(s - 1) - 1.0f;
                const float r = std::sqrt(nx * nx + ny * ny);
                const float ring = std::exp(-14.0f * (r - 0.55f) * (r - 0.55f));
                const size_t p = static_cast<size_t>(y) * s + x;
                card[0 * hw + p] = -0.8f + 0.6f * ring;
                card[1 * hw + p] = -0.7f + 1.6f * ring;
                card[2 * hw + p] = -0.7f + 0.9f * ring;
            }
        }
        m_rawCardsCHW.push_back(std::move(card));
        m_rawCardSizes.push_back(s);
    }

    // Card 1: Warm amber-crimson diagonal wave
    {
        std::vector<float> card(3 * hw);
        for (uint32_t y = 0; y < s; ++y) {
            const float ny = 2.0f * static_cast<float>(y) / static_cast<float>(s - 1) - 1.0f;
            for (uint32_t x = 0; x < s; ++x) {
                const float nx = 2.0f * static_cast<float>(x) / static_cast<float>(s - 1) - 1.0f;
                const float wave = 0.5f + 0.5f * std::sin(3.14159f * (nx + ny));
                const size_t p = static_cast<size_t>(y) * s + x;
                card[0 * hw + p] = -0.5f + 1.4f * wave;
                card[1 * hw + p] = -0.6f + 0.9f * wave;
                card[2 * hw + p] = -0.8f + 0.3f * (1.0f - wave);
            }
        }
        m_rawCardsCHW.push_back(std::move(card));
        m_rawCardSizes.push_back(s);
    }

    // Card 2: Geometric diamond cross
    {
        std::vector<float> card(3 * hw);
        for (uint32_t y = 0; y < s; ++y) {
            const float ny = 2.0f * static_cast<float>(y) / static_cast<float>(s - 1) - 1.0f;
            for (uint32_t x = 0; x < s; ++x) {
                const float nx = 2.0f * static_cast<float>(x) / static_cast<float>(s - 1) - 1.0f;
                const float d = std::abs(nx) + std::abs(ny);
                const float mask = (d < 0.72f) ? 1.0f : 0.0f;
                const size_t p = static_cast<size_t>(y) * s + x;
                card[0 * hw + p] = -0.7f + 1.3f * mask;
                card[1 * hw + p] = -0.4f + 1.2f * mask;
                card[2 * hw + p] = 0.4f + 0.5f * mask;
            }
        }
        m_rawCardsCHW.push_back(std::move(card));
        m_rawCardSizes.push_back(s);
    }

    // Card 3: Dual quadrant harmonic checker
    {
        std::vector<float> card(3 * hw);
        for (uint32_t y = 0; y < s; ++y) {
            const float ny = 2.0f * static_cast<float>(y) / static_cast<float>(s - 1) - 1.0f;
            for (uint32_t x = 0; x < s; ++x) {
                const float nx = 2.0f * static_cast<float>(x) / static_cast<float>(s - 1) - 1.0f;
                const float sgn = (nx * ny >= 0.0f) ? 0.75f : -0.75f;
                const size_t p = static_cast<size_t>(y) * s + x;
                card[0 * hw + p] = 0.2f * sgn;
                card[1 * hw + p] = 0.8f * sgn;
                card[2 * hw + p] = -0.5f * sgn;
            }
        }
        m_rawCardsCHW.push_back(std::move(card));
        m_rawCardSizes.push_back(s);
    }

    std::lock_guard<std::mutex> stateLock(m_stateMutex);
    m_metrics.deckCardCount = static_cast<uint32_t>(m_rawCardsCHW.size());
}

void Trainer::AddCardToDeck(std::vector<float> rgbTensorCHW, uint32_t srcWidth, uint32_t srcHeight) {
    if (srcWidth == 0 || srcHeight == 0 || rgbTensorCHW.size() < static_cast<size_t>(3) * srcWidth * srcHeight) {
        return;
    }
    // Store at canonical 128x128 so scaling to 8/16/32/64/128 preserves detail.
    constexpr uint32_t kCanonical = 128;
    std::vector<float> canonical = BilinearResizeCHW(rgbTensorCHW.data(), srcWidth, srcHeight, kCanonical, kCanonical);
    {
        std::lock_guard<std::mutex> lock(m_deckMutex);
        m_rawCardsCHW.push_back(std::move(canonical));
        m_rawCardSizes.push_back(kCanonical);
    }
    std::lock_guard<std::mutex> stateLock(m_stateMutex);
    m_metrics.deckCardCount = DeckSize();
}

void Trainer::ClearDeck() {
    {
        std::lock_guard<std::mutex> lock(m_deckMutex);
        m_rawCardsCHW.clear();
        m_rawCardSizes.clear();
    }
    std::lock_guard<std::mutex> stateLock(m_stateMutex);
    m_metrics.deckCardCount = 0;
}

uint32_t Trainer::DeckSize() const {
    std::lock_guard<std::mutex> lock(m_deckMutex);
    return static_cast<uint32_t>(m_rawCardsCHW.size());
}

std::vector<float> Trainer::BilinearResizeCHW(
    const float* src,
    uint32_t srcW,
    uint32_t srcH,
    uint32_t dstW,
    uint32_t dstH) {
    if (srcW == dstW && srcH == dstH) {
        return std::vector<float>(src, src + static_cast<size_t>(3) * srcW * srcH);
    }

    std::vector<float> dst(static_cast<size_t>(3) * dstW * dstH, 0.0f);
    const size_t srcHW = static_cast<size_t>(srcW) * srcH;
    const size_t dstHW = static_cast<size_t>(dstW) * dstH;

    for (uint32_t c = 0; c < 3; ++c) {
        const float* sPlane = src + static_cast<size_t>(c) * srcHW;
        float* dPlane = dst.data() + static_cast<size_t>(c) * dstHW;

        for (uint32_t y = 0; y < dstH; ++y) {
            const float fy = (dstH > 1) ? (static_cast<float>(y) * (srcH - 1) / static_cast<float>(dstH - 1)) : 0.0f;
            const uint32_t y0 = static_cast<uint32_t>(fy);
            const uint32_t y1 = std::min(y0 + 1, srcH - 1);
            const float wy = fy - static_cast<float>(y0);

            for (uint32_t x = 0; x < dstW; ++x) {
                const float fx = (dstW > 1) ? (static_cast<float>(x) * (srcW - 1) / static_cast<float>(dstW - 1)) : 0.0f;
                const uint32_t x0 = static_cast<uint32_t>(fx);
                const uint32_t x1 = std::min(x0 + 1, srcW - 1);
                const float wx = fx - static_cast<float>(x0);

                const float v00 = sPlane[static_cast<size_t>(y0) * srcW + x0];
                const float v01 = sPlane[static_cast<size_t>(y0) * srcW + x1];
                const float v10 = sPlane[static_cast<size_t>(y1) * srcW + x0];
                const float v11 = sPlane[static_cast<size_t>(y1) * srcW + x1];

                const float top = v00 * (1.0f - wx) + v01 * wx;
                const float bot = v10 * (1.0f - wx) + v11 * wx;
                dPlane[static_cast<size_t>(y) * dstW + x] = top * (1.0f - wy) + bot * wy;
            }
        }
    }
    return dst;
}

void Trainer::Conv2dBackward(
    const float* input,
    const float* gradOutput,
    const ConvLayerWeights& layer,
    uint32_t batch,
    uint32_t height,
    uint32_t width,
    std::vector<float>& gradInput,
    std::vector<float>& gradWeight,
    std::vector<float>& gradBias) {
    const uint32_t inC = layer.inChannels;
    const uint32_t outC = layer.outChannels;
    const uint32_t k = layer.kernelSize;
    const int pad = static_cast<int>(k / 2);
    const size_t hw = static_cast<size_t>(height) * width;

    gradInput.assign(static_cast<size_t>(batch) * inC * hw, 0.0f);
    gradWeight.assign(layer.weight.size(), 0.0f);
    gradBias.assign(outC, 0.0f);

    for (uint32_t n = 0; n < batch; ++n) {
        const float* inBatch = input + static_cast<size_t>(n) * inC * hw;
        const float* gOutBatch = gradOutput + static_cast<size_t>(n) * outC * hw;
        float* gInBatch = gradInput.data() + static_cast<size_t>(n) * inC * hw;

        for (uint32_t oc = 0; oc < outC; ++oc) {
            const float* gOutPlane = gOutBatch + static_cast<size_t>(oc) * hw;

            double bSum = 0.0;
            for (size_t idx = 0; idx < hw; ++idx) {
                bSum += gOutPlane[idx];
            }
            gradBias[oc] += static_cast<float>(bSum);

            for (uint32_t ic = 0; ic < inC; ++ic) {
                const float* inPlane = inBatch + static_cast<size_t>(ic) * hw;
                float* gInPlane = gInBatch + static_cast<size_t>(ic) * hw;
                const float* wKernel = layer.weight.data() + ((static_cast<size_t>(oc) * inC + ic) * k * k);
                float* gWKernel = gradWeight.data() + ((static_cast<size_t>(oc) * inC + ic) * k * k);

                if (k == 1) {
                    const float w = wKernel[0];
                    double gwAcc = 0.0;
                    for (size_t idx = 0; idx < hw; ++idx) {
                        const float go = gOutPlane[idx];
                        gwAcc += static_cast<double>(go) * static_cast<double>(inPlane[idx]);
                        gInPlane[idx] += go * w;
                    }
                    gWKernel[0] += static_cast<float>(gwAcc);
                } else {
                    for (uint32_t y = 0; y < height; ++y) {
                        for (uint32_t x = 0; x < width; ++x) {
                            const float go = gOutPlane[static_cast<size_t>(y) * width + x];
                            for (int ky = 0; ky < static_cast<int>(k); ++ky) {
                                const int iy = static_cast<int>(y) + ky - pad;
                                if (iy < 0 || iy >= static_cast<int>(height)) continue;
                                for (int kx = 0; kx < static_cast<int>(k); ++kx) {
                                    const int ix = static_cast<int>(x) + kx - pad;
                                    if (ix < 0 || ix >= static_cast<int>(width)) continue;
                                    const size_t inIdx = static_cast<size_t>(iy) * width + ix;
                                    const size_t kIdx = static_cast<size_t>(ky) * k + kx;
                                    gWKernel[kIdx] += go * inPlane[inIdx];
                                    gInPlane[inIdx] += go * wKernel[kIdx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void Trainer::AdamUpdate(
    std::vector<float>& param,
    const std::vector<float>& grad,
    std::vector<float>& m,
    std::vector<float>& v,
    uint64_t step,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float clipScale) {
    const float bc1 = 1.0f - std::pow(beta1, static_cast<float>(step));
    const float bc2 = 1.0f - std::pow(beta2, static_cast<float>(step));

    for (size_t i = 0; i < param.size(); ++i) {
        const float g = grad[i] * clipScale;
        m[i] = beta1 * m[i] + (1.0f - beta1) * g;
        v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;
        const float mHat = m[i] / std::max(1e-8f, bc1);
        const float vHat = v[i] / std::max(1e-8f, bc2);
        param[i] -= lr * mHat / (std::sqrt(vHat) + eps);
    }
}

float Trainer::Step(FMBrain& brain, TrainerConfig& config) {
    const auto t0 = std::chrono::high_resolution_clock::now();

    const uint32_t res = std::clamp(config.resolution, 8u, 128u);
    const uint32_t batch = ClampBatchForResolution(config.batchSize, res, config.engine);
    config.batchSize = batch;

    // Sample ONLY `batch` mini-batch cards under lock, then resize ONLY those sampled cards!
    std::vector<std::vector<float>> sampledCards;
    uint32_t totalDeckCount = 0;
    {
        std::lock_guard<std::mutex> lock(m_deckMutex);
        totalDeckCount = static_cast<uint32_t>(m_rawCardsCHW.size());
        if (totalDeckCount > 0) {
            sampledCards.reserve(batch);
            std::uniform_int_distribution<size_t> dist(0, totalDeckCount - 1);
            for (uint32_t i = 0; i < batch; ++i) {
                const size_t idx = dist(m_batchRng);
                const uint32_t srcSize = m_rawCardSizes[idx];
                sampledCards.push_back(BilinearResizeCHW(m_rawCardsCHW[idx].data(), srcSize, srcSize, res, res));
            }
        }
    }

    TensorShape shape;
    shape.batch = batch;
    shape.channels = 3;
    shape.height = res;
    shape.width = res;

    OTBatchResult otBatch = m_otMatcher.BuildTrainingBatch(sampledCards, shape);

    uint64_t nextStep = 0;
    {
        std::lock_guard<std::mutex> stateLock(m_stateMutex);
        nextStep = ++m_metrics.stepCount;
    }

    float mseLoss = 0.0f;
    bool usedGpu = false;
    bool fallbackTriggered = false;

    {
        std::lock_guard<std::recursive_mutex> brainLock(brain.Mutex());

        // Attempt real GPU training when [Engine: GPU] is selected
        if (config.engine == ComputeEngine::GPU && brain.IsGPUReady()) {
            const HRESULT hr = brain.TrainStepGPU(
                otBatch.xt.data(),
                otBatch.vTarget.data(),
                otBatch.tValues.data(),
                batch,
                res,
                res,
                nextStep,
                config.learningRate,
                config.beta1,
                config.beta2,
                config.adamEpsilon,
                !config.blindSpeedMode, // Completely skip loss readback in Blind Speed mode!
                mseLoss);

            if (SUCCEEDED(hr)) {
                usedGpu = true;
            } else {
                // Graceful fallback to CPU trainer without crashing
                config.engine = ComputeEngine::CPU;
                fallbackTriggered = true;
                brain.ReadbackWeightsFromGPU();
            }
        }

        if (!usedGpu) {
            // Sync any GPU-trained weights before CPU step if switching from GPU -> CPU
            brain.ReadbackWeightsFromGPU();
            brain.Forward(otBatch.xt.data(), otBatch.tValues.data(), batch, res, res, m_cache);

            const size_t outCount = otBatch.vTarget.size();
            const float invCount = 1.0f / static_cast<float>(std::max<size_t>(1, outCount));
            std::vector<float> gradPred(outCount);

            if (config.blindSpeedMode) {
                // Blind Speed: skip MSE reduction sum; compute gradient only
                for (size_t i = 0; i < outCount; ++i) {
                    gradPred[i] = 2.0f * invCount * (m_cache.velocityPred[i] - otBatch.vTarget[i]);
                }
            } else {
                double mseSum = 0.0;
                for (size_t i = 0; i < outCount; ++i) {
                    const float diff = m_cache.velocityPred[i] - otBatch.vTarget[i];
                    mseSum += static_cast<double>(diff) * static_cast<double>(diff);
                    gradPred[i] = 2.0f * invCount * diff;
                }
                mseLoss = static_cast<float>(mseSum * invCount);
            }

            std::vector<float> gradHidden;
            std::vector<float> gradHeadW, gradHeadB;
            Conv2dBackward(
                m_cache.headInput.data(),
                gradPred.data(),
                brain.Head(),
                batch,
                res,
                res,
                gradHidden,
                gradHeadW,
                gradHeadB);

            auto& blocks = brain.Blocks();
            const size_t numBlocks = blocks.size();
            std::vector<std::vector<float>> gradBlkW1(numBlocks), gradBlkB1(numBlocks);
            std::vector<std::vector<float>> gradBlkW2(numBlocks), gradBlkB2(numBlocks);
            std::vector<float> gradActMid;
            std::vector<float> gradBlockIn;

            for (size_t b = numBlocks; b-- > 0;) {
                Conv2dBackward(
                    m_cache.blockMidAct[b].data(),
                    gradHidden.data(),
                    blocks[b].conv2,
                    batch,
                    res,
                    res,
                    gradActMid,
                    gradBlkW2[b],
                    gradBlkB2[b]);

                for (size_t i = 0; i < gradActMid.size(); ++i) {
                    gradActMid[i] *= SiLUDerivative(m_cache.blockMidPre[b][i]);
                }

                Conv2dBackward(
                    m_cache.blockIn[b].data(),
                    gradActMid.data(),
                    blocks[b].conv1,
                    batch,
                    res,
                    res,
                    gradBlockIn,
                    gradBlkW1[b],
                    gradBlkB1[b]);

                for (size_t i = 0; i < gradHidden.size(); ++i) {
                    gradHidden[i] = gradHidden[i] + gradBlockIn[i];
                }
            }

            for (size_t i = 0; i < gradHidden.size(); ++i) {
                gradHidden[i] *= SiLUDerivative(m_cache.stemPreAct[i]);
            }

            std::vector<float> gradStemIn, gradStemW, gradStemB;
            Conv2dBackward(
                m_cache.stemInput.data(),
                gradHidden.data(),
                brain.Stem(),
                batch,
                res,
                res,
                gradStemIn,
                gradStemW,
                gradStemB);

            double gradNormSq = 0.0;
            auto accumNorm = [&](const std::vector<float>& g) {
                for (float val : g) gradNormSq += static_cast<double>(val) * static_cast<double>(val);
            };
            accumNorm(gradStemW);
            accumNorm(gradStemB);
            for (size_t b = 0; b < numBlocks; ++b) {
                accumNorm(gradBlkW1[b]);
                accumNorm(gradBlkB1[b]);
                accumNorm(gradBlkW2[b]);
                accumNorm(gradBlkB2[b]);
            }
            accumNorm(gradHeadW);
            accumNorm(gradHeadB);

            const float gradNorm = static_cast<float>(std::sqrt(gradNormSq));
            const float clipScale = (gradNorm > config.gradClipNorm && gradNorm > 1e-6f)
                ? (config.gradClipNorm / gradNorm)
                : 1.0f;

            AdamUpdate(brain.Stem().weight, gradStemW, brain.StemMW(), brain.StemVW(), nextStep, config.learningRate, config.beta1, config.beta2, config.adamEpsilon, clipScale);
            AdamUpdate(brain.Stem().bias, gradStemB, brain.StemMB(), brain.StemVB(), nextStep, config.learningRate, config.beta1, config.beta2, config.adamEpsilon, clipScale);

            for (size_t b = 0; b < numBlocks; ++b) {
                AdamUpdate(blocks[b].conv1.weight, gradBlkW1[b], blocks[b].mW1, blocks[b].vW1, nextStep, config.learningRate, config.beta1, config.beta2, config.adamEpsilon, clipScale);
                AdamUpdate(blocks[b].conv1.bias, gradBlkB1[b], blocks[b].mB1, blocks[b].vB1, nextStep, config.learningRate, config.beta1, config.beta2, config.adamEpsilon, clipScale);
                AdamUpdate(blocks[b].conv2.weight, gradBlkW2[b], blocks[b].mW2, blocks[b].vW2, nextStep, config.learningRate, config.beta1, config.beta2, config.adamEpsilon, clipScale);
                AdamUpdate(blocks[b].conv2.bias, gradBlkB2[b], blocks[b].mB2, blocks[b].vB2, nextStep, config.learningRate, config.beta1, config.beta2, config.adamEpsilon, clipScale);
            }

            AdamUpdate(brain.Head().weight, gradHeadW, brain.HeadMW(), brain.HeadVW(), nextStep, config.learningRate, config.beta1, config.beta2, config.adamEpsilon, clipScale);
            AdamUpdate(brain.Head().bias, gradHeadB, brain.HeadMB(), brain.HeadVB(), nextStep, config.learningRate, config.beta1, config.beta2, config.adamEpsilon, clipScale);
        }
    }

    const auto t1 = std::chrono::high_resolution_clock::now();
    const float elapsedSec = std::max(1e-5f, std::chrono::duration<float>(t1 - t0).count());

    {
        std::lock_guard<std::mutex> lock(m_stateMutex);
        m_metrics.usingGpuEngine = usedGpu;
        if (fallbackTriggered) {
            m_metrics.gpuFallbackTriggered = true;
        }
        m_metrics.deckCardCount = totalDeckCount;
        m_metrics.lessonsPerSec = static_cast<float>(batch) / elapsedSec;

        if (!config.blindSpeedMode) {
            m_metrics.lastMseLoss = mseLoss;
            m_metrics.emaMseLoss = (nextStep == 1) ? mseLoss : (0.9f * m_metrics.emaMseLoss + 0.1f * mseLoss);
            m_metrics.lastRandomOtCost = otBatch.randomPairingCost;
            m_metrics.lastOptimalOtCost = otBatch.optimalPairingCost;
        }
    }

    return mseLoss;
}

std::vector<float> Trainer::GenerateDream(FMBrain& brain, const TrainerConfig& config) {
    const uint32_t res = std::clamp(config.resolution, 8u, 128u);
    const size_t sampleSize = static_cast<size_t>(3) * res * res;
    std::vector<float> z0(sampleSize);
    std::normal_distribution<float> norm(0.0f, 1.0f);
    for (size_t i = 0; i < sampleSize; ++i) {
        z0[i] = norm(m_dreamRng);
    }

    std::vector<float> dream;
    {
        std::lock_guard<std::recursive_mutex> brainLock(brain.Mutex());
        brain.ReadbackWeightsFromGPU();
        dream = brain.DreamEulerODE(z0.data(), res, res, config.dreamSteps);
    }
    for (float& v : dream) {
        v = std::clamp(v, -1.0f, 1.0f);
    }

    std::lock_guard<std::mutex> lock(m_stateMutex);
    m_latestDreamCHW = dream;
    m_latestDreamSize = res;
    return dream;
}

void Trainer::ResetMetrics() {
    std::lock_guard<std::mutex> lock(m_stateMutex);
    m_metrics.stepCount = 0;
    m_metrics.lastMseLoss = 0.0f;
    m_metrics.emaMseLoss = 0.0f;
    m_metrics.lessonsPerSec = 0.0f;
    m_metrics.lastRandomOtCost = 0.0f;
    m_metrics.lastOptimalOtCost = 0.0f;
}

TrainingMetrics Trainer::SnapshotMetrics() const {
    std::lock_guard<std::mutex> lock(m_stateMutex);
    return m_metrics;
}

std::vector<float> Trainer::LatestDreamSnapshot(uint32_t& outSize) const {
    std::lock_guard<std::mutex> lock(m_stateMutex);
    outSize = m_latestDreamSize;
    return m_latestDreamCHW;
}

void Trainer::SetDialScanReport(const DialScanReport& report) {
    std::lock_guard<std::mutex> lock(m_stateMutex);
    m_metrics.lastDialScan = report;
}

} // namespace XboxOTFM
