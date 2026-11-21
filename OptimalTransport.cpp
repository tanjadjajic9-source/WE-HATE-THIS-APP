#include "OptimalTransport.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace XboxOTFM {

OptimalTransportMatcher::OptimalTransportMatcher(uint64_t seed)
    : m_rng(seed)
    , m_normalDist(0.0f, 1.0f)
    , m_uniformDist(0.001f, 0.999f) {
}

void OptimalTransportMatcher::Reseed(uint64_t seed) {
    m_rng.seed(seed);
}

void OptimalTransportMatcher::SampleGaussianNoise(
    uint32_t batch,
    uint32_t sampleStride,
    std::vector<float>& outNoise) {
    const size_t total = static_cast<size_t>(batch) * sampleStride;
    outNoise.resize(total);
    for (size_t idx = 0; idx < total; ++idx) {
        outNoise[idx] = m_normalDist(m_rng);
    }
}

void OptimalTransportMatcher::ComputePairwiseCostMatrix(
    const float* x0,
    const float* x1,
    uint32_t batch,
    uint32_t sampleStride,
    std::vector<float>& outCostMatrix) {
    outCostMatrix.assign(static_cast<size_t>(batch) * batch, 0.0f);
    const float invStride = 1.0f / static_cast<float>(std::max(1u, sampleStride));

    for (uint32_t i = 0; i < batch; ++i) {
        const float* a = x0 + static_cast<size_t>(i) * sampleStride;
        for (uint32_t j = 0; j < batch; ++j) {
            const float* b = x1 + static_cast<size_t>(j) * sampleStride;
            double sumSq = 0.0;
            for (uint32_t k = 0; k < sampleStride; ++k) {
                const double diff = static_cast<double>(a[k]) - static_cast<double>(b[k]);
                sumSq += diff * diff;
            }
            // Normalized squared L2 per dimension keeps Sinkhorn temperature scale-invariant across resolutions.
            outCostMatrix[static_cast<size_t>(i) * batch + j] = static_cast<float>(sumSq * invStride);
        }
    }
}

void OptimalTransportMatcher::SolveSinkhornKnopp(
    const float* costMatrix,
    uint32_t batch,
    float epsilon,
    uint32_t maxIterations,
    std::vector<float>& outPlan) {
    const uint32_t n = batch;
    outPlan.assign(static_cast<size_t>(n) * n, 0.0f);
    if (n == 0) {
        return;
    }

    const float eps = std::max(1e-3f, epsilon);
    const float targetMarginal = 1.0f / static_cast<float>(n);
    const float logTarget = std::log(targetMarginal);

    // Log-domain stabilized Sinkhorn-Knopp iterations prevent underflow on GPU/CPU.
    std::vector<float> f(n, 0.0f);
    std::vector<float> g(n, 0.0f);

    for (uint32_t iter = 0; iter < maxIterations; ++iter) {
        // Row update: f_i = logTarget - logsumexp_j((g_j - C_{i,j}) / eps)
        for (uint32_t i = 0; i < n; ++i) {
            float maxVal = -std::numeric_limits<float>::infinity();
            for (uint32_t j = 0; j < n; ++j) {
                const float v = (g[j] - costMatrix[static_cast<size_t>(i) * n + j]) / eps;
                if (v > maxVal) maxVal = v;
            }
            double sumExp = 0.0;
            for (uint32_t j = 0; j < n; ++j) {
                const float v = (g[j] - costMatrix[static_cast<size_t>(i) * n + j]) / eps;
                sumExp += std::exp(static_cast<double>(v - maxVal));
            }
            const float lse = maxVal + static_cast<float>(std::log(sumExp));
            f[i] = eps * (logTarget - lse);
        }

        // Column update: g_j = logTarget - logsumexp_i((f_i - C_{i,j}) / eps)
        for (uint32_t j = 0; j < n; ++j) {
            float maxVal = -std::numeric_limits<float>::infinity();
            for (uint32_t i = 0; i < n; ++i) {
                const float v = (f[i] - costMatrix[static_cast<size_t>(i) * n + j]) / eps;
                if (v > maxVal) maxVal = v;
            }
            double sumExp = 0.0;
            for (uint32_t i = 0; i < n; ++i) {
                const float v = (f[i] - costMatrix[static_cast<size_t>(i) * n + j]) / eps;
                sumExp += std::exp(static_cast<double>(v - maxVal));
            }
            const float lse = maxVal + static_cast<float>(std::log(sumExp));
            g[j] = eps * (logTarget - lse);
        }
    }

    for (uint32_t i = 0; i < n; ++i) {
        for (uint32_t j = 0; j < n; ++j) {
            const float exponent = (f[i] + g[j] - costMatrix[static_cast<size_t>(i) * n + j]) / eps;
            outPlan[static_cast<size_t>(i) * n + j] = std::exp(std::clamp(exponent, -40.0f, 10.0f));
        }
    }
}

void OptimalTransportMatcher::SolveHungarianAssignment(
    const float* costMatrix,
    uint32_t batch,
    std::vector<uint32_t>& outPermutation) {
    const int n = static_cast<int>(batch);
    outPermutation.assign(batch, 0u);
    if (n <= 0) {
        return;
    }

    // Classic O(N^3) Hungarian / Jonker-Volgenant augmenting-path algorithm (1-indexed).
    constexpr double kInf = 1e18;
    std::vector<double> u(n + 1, 0.0);
    std::vector<double> v(n + 1, 0.0);
    std::vector<int> p(n + 1, 0);
    std::vector<int> way(n + 1, 0);

    for (int i = 1; i <= n; ++i) {
        p[0] = i;
        int j0 = 0;
        std::vector<double> minv(n + 1, kInf);
        std::vector<uint8_t> used(n + 1, 0);

        do {
            used[j0] = 1;
            const int i0 = p[j0];
            double delta = kInf;
            int j1 = 0;
            for (int j = 1; j <= n; ++j) {
                if (!used[j]) {
                    const double cur =
                        static_cast<double>(costMatrix[static_cast<size_t>(i0 - 1) * n + (j - 1)]) - u[i0] - v[j];
                    if (cur < minv[j]) {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if (minv[j] < delta) {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            for (int j = 0; j <= n; ++j) {
                if (used[j]) {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
        } while (p[j0] != 0);

        do {
            const int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0 != 0);
    }

    for (int j = 1; j <= n; ++j) {
        if (p[j] >= 1 && p[j] <= n) {
            outPermutation[static_cast<size_t>(p[j] - 1)] = static_cast<uint32_t>(j - 1);
        }
    }
}

OTBatchResult OptimalTransportMatcher::BuildTrainingBatch(
    const std::vector<std::vector<float>>& memoryDeck,
    const TensorShape& shape,
    bool useSinkhornGuidance,
    float sinkhornEpsilon,
    uint32_t sinkhornIters) {
    OTBatchResult result;
    const uint32_t n = std::max(1u, shape.batch);
    const uint32_t stride = shape.SampleStride();
    const size_t total = static_cast<size_t>(n) * stride;

    // 1. Sample Gaussian noise x0 ~ N(0, I)
    SampleGaussianNoise(n, stride, result.x0Matched);

    // 2. Sample N cards from memory deck (with replacement if deck.size() < N)
    std::vector<float> x1Raw(total, 0.0f);
    if (!memoryDeck.empty()) {
        std::uniform_int_distribution<size_t> cardDist(0, memoryDeck.size() - 1);
        for (uint32_t i = 0; i < n; ++i) {
            const auto& card = memoryDeck[cardDist(m_rng)];
            const size_t copyCount = std::min<size_t>(stride, card.size());
            std::copy_n(card.data(), copyCount, x1Raw.data() + static_cast<size_t>(i) * stride);
        }
    }

    // 3. Pairwise cost matrix C_{i,j}
    ComputePairwiseCostMatrix(result.x0Matched.data(), x1Raw.data(), n, stride, result.costMatrix);

    // 4. Solve Sinkhorn-Knopp soft transport plan and exact Hungarian permutation
    if (useSinkhornGuidance) {
        SolveSinkhornKnopp(result.costMatrix.data(), n, sinkhornEpsilon, sinkhornIters, result.transportPlan);
    }
    SolveHungarianAssignment(result.costMatrix.data(), n, result.permutation);

    // 5. Compute cost reduction diagnostics and assemble straight-line OT-FM trajectories
    result.x1Matched.resize(total);
    result.xt.resize(total);
    result.vTarget.resize(total);
    result.tValues.resize(n);

    double randomCostSum = 0.0;
    double optimalCostSum = 0.0;

    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t pi = result.permutation[i];
        randomCostSum += result.costMatrix[static_cast<size_t>(i) * n + i];
        optimalCostSum += result.costMatrix[static_cast<size_t>(i) * n + pi];

        const float t = m_uniformDist(m_rng);
        result.tValues[i] = t;
        const float oneMinusT = 1.0f - t;

        const float* x0Ptr = result.x0Matched.data() + static_cast<size_t>(i) * stride;
        const float* x1Src = x1Raw.data() + static_cast<size_t>(pi) * stride;
        float* x1Dst = result.x1Matched.data() + static_cast<size_t>(i) * stride;
        float* xtDst = result.xt.data() + static_cast<size_t>(i) * stride;
        float* vDst = result.vTarget.data() + static_cast<size_t>(i) * stride;

        for (uint32_t k = 0; k < stride; ++k) {
            const float targetVal = x1Src[k];
            const float noiseVal = x0Ptr[k];
            x1Dst[k] = targetVal;
            xtDst[k] = oneMinusT * noiseVal + t * targetVal;
            vDst[k] = targetVal - noiseVal;
        }
    }

    result.randomPairingCost = static_cast<float>(randomCostSum / static_cast<double>(n));
    result.optimalPairingCost = static_cast<float>(optimalCostSum / static_cast<double>(n));
    return result;
}

} // namespace XboxOTFM
