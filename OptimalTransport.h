#pragma once

#include <cstdint>
#include <vector>
#include <random>

namespace XboxOTFM {

struct TensorShape {
    uint32_t batch = 1;
    uint32_t channels = 3;
    uint32_t height = 16;
    uint32_t width = 16;

    uint32_t SampleStride() const noexcept {
        return channels * height * width;
    }

    uint32_t TotalElements() const noexcept {
        return batch * SampleStride();
    }
};

struct OTBatchResult {
    std::vector<float> costMatrix;        // [N * N] row-major C_{i,j} = ||x0[i] - x1[j]||_2^2
    std::vector<float> transportPlan;     // [N * N] doubly-stochastic Sinkhorn plan P_{i,j}
    std::vector<uint32_t> permutation;    // [N] bijection pi(i) minimizing transport cost
    std::vector<float> x0Matched;         // [N, C, H, W] Gaussian noise
    std::vector<float> x1Matched;         // [N, C, H, W] target cards reordered by pi
    std::vector<float> xt;                // [N, C, H, W] (1 - t) * x0 + t * x1
    std::vector<float> vTarget;           // [N, C, H, W] x1 - x0
    std::vector<float> tValues;           // [N] Uniform(0, 1)
    float randomPairingCost = 0.0f;
    float optimalPairingCost = 0.0f;
};

class OptimalTransportMatcher {
public:
    explicit OptimalTransportMatcher(uint64_t seed = 0x58424F58ULL);

    void Reseed(uint64_t seed);

    // Generates N Gaussian noise tensors x0 ~ N(0, I) in NCHW layout.
    void SampleGaussianNoise(uint32_t batch, uint32_t sampleStride, std::vector<float>& outNoise);

    // Computes exact pairwise squared L2 cost matrix C_{i,j} = ||x0[i] - x1[j]||_2^2.
    static void ComputePairwiseCostMatrix(
        const float* x0,
        const float* x1,
        uint32_t batch,
        uint32_t sampleStride,
        std::vector<float>& outCostMatrix);

    // Log-domain stabilized Sinkhorn-Knopp optimal transport solver.
    static void SolveSinkhornKnopp(
        const float* costMatrix,
        uint32_t batch,
        float epsilon,
        uint32_t maxIterations,
        std::vector<float>& outPlan);

    // Exact Jonker-Volgenant / Hungarian O(N^3) assignment solver on the cost matrix
    // (or Sinkhorn-guided extraction when softPlan != nullptr) guaranteeing a valid bijection.
    static void SolveHungarianAssignment(
        const float* costMatrix,
        uint32_t batch,
        std::vector<uint32_t>& outPermutation);

    // Full OT-FM mini-batch construction step:
    // 1. Sample x0 ~ N(0, I) and draw N cards from deck.
    // 2. Compute C_{i,j}, run Sinkhorn + Hungarian assignment pi.
    // 3. Interpolate straight-line flow xt = (1 - t) * x0[i] + t * x1[pi(i)] and v_target = x1[pi(i)] - x0[i].
    OTBatchResult BuildTrainingBatch(
        const std::vector<std::vector<float>>& memoryDeck,
        const TensorShape& shape,
        bool useSinkhornGuidance = true,
        float sinkhornEpsilon = 0.08f,
        uint32_t sinkhornIters = 24);

private:
    std::mt19937_64 m_rng;
    std::normal_distribution<float> m_normalDist;
    std::uniform_real_distribution<float> m_uniformDist;
};

} // namespace XboxOTFM
