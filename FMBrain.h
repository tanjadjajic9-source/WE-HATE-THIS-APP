#pragma once

#include <d3d12.h>
#include <DirectML.h>
#include <wrl/client.h>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace XboxOTFM {

enum class BrainPreset : uint32_t {
    Low25k = 0,    // Width 26, 2 residual blocks (~25k dials)
    Medium80k = 1, // Width 38, 3 residual blocks (~80k dials)
    Max250k = 2    // Width 58, 4 residual blocks (~250k dials)
};

enum class ComputeEngine : uint32_t {
    GPU = 0,
    CPU = 1
};

struct BrainArchitecture {
    BrainPreset preset = BrainPreset::Medium80k;
    uint32_t inputImageChannels = 3;
    uint32_t coordChannels = 2;       // (x, y) in [-1, 1]
    uint32_t harmonicBands = 4;       // k = 0..3 -> sin/cos for x and y = 16 channels
    uint32_t timeChannels = 4;        // sinusoidal embedding of scalar t
    uint32_t width = 38;
    uint32_t numResBlocks = 3;

    uint32_t TotalInputChannels() const noexcept {
        return inputImageChannels + coordChannels + (harmonicBands * 4) + timeChannels;
    }
};

struct DialScanReport {
    uint64_t totalDials = 0;
    uint64_t busyDials = 0;     // |weight| > 0.01f
    uint64_t sleepingDials = 0; // |weight| <= 0.01f
    float maxMagnitude = 0.0f;
    float meanMagnitude = 0.0f;
};

struct ConvLayerWeights {
    uint32_t inChannels = 0;
    uint32_t outChannels = 0;
    uint32_t kernelSize = 3; // 1 or 3
    std::vector<float> weight; // [outC, inC, K, K]
    std::vector<float> bias;   // [outC]
};

struct GPULayerBuffers {
    Microsoft::WRL::ComPtr<ID3D12Resource> weight;
    Microsoft::WRL::ComPtr<ID3D12Resource> bias;
    Microsoft::WRL::ComPtr<ID3D12Resource> mW;
    Microsoft::WRL::ComPtr<ID3D12Resource> vW;
    Microsoft::WRL::ComPtr<ID3D12Resource> mB;
    Microsoft::WRL::ComPtr<ID3D12Resource> vB;
    Microsoft::WRL::ComPtr<ID3D12Resource> gradW;
    Microsoft::WRL::ComPtr<ID3D12Resource> gradB;
};

struct ResidualBlockWeights {
    ConvLayerWeights conv1; // 3x3
    ConvLayerWeights conv2; // 3x3
    std::vector<float> mW1, vW1, mB1, vB1;
    std::vector<float> mW2, vW2, mB2, vB2;
};

struct ForwardCache {
    uint32_t batch = 0;
    uint32_t height = 0;
    uint32_t width = 0;
    std::vector<float> stemInput;              // [N, Cin, H, W]
    std::vector<float> stemPreAct;             // [N, W, H, W]
    std::vector<float> stemOut;                // [N, W, H, W]
    std::vector<std::vector<float>> blockIn;   // [B][N, W, H, W]
    std::vector<std::vector<float>> blockMidPre;
    std::vector<std::vector<float>> blockMidAct;
    std::vector<std::vector<float>> blockConv2Out;
    std::vector<float> headInput;              // [N, W, H, W]
    std::vector<float> velocityPred;           // [N, 3, H, W]
};

struct GPUTrainConstants {
    uint32_t gBatch;
    uint32_t gWidth;
    uint32_t gHeight;
    uint32_t gInC;
    uint32_t gOutC;
    uint32_t gKernelSize;
    uint32_t gApplySiLU;
    uint32_t gUnusedFlag;
    uint32_t gParamCount;
    uint32_t gSampleStride;
    float gTimeT;
    float gDt;
    float gLearningRate;
    float gBeta1;
    float gBeta2;
    float gEpsilon;
    float gBiasCorr1;
    float gBiasCorr2;
    float gInvOutCount;
    float gGradClipScale;
};

class FMBrain {
public:
    explicit FMBrain(BrainPreset preset = BrainPreset::Medium80k);

    // Rebuilds all layers from 0% pre-trained weights (pure random Kaiming/He noise).
    void InitializeFromScratch(BrainPreset preset, uint64_t seed = 0x1337BEEFULL);

    // Initializes DirectML device, Root Signature, Compute PSOs, and GPU-resident weight/moment buffers.
    HRESULT InitializeGPU(ID3D12Device* d3dDevice, ID3D12CommandQueue* cmdQueue);
    void ReleaseGPU();

    // Uploads CPU weights -> GPU-resident DEFAULT buffers (only on init/import, never per-lesson).
    HRESULT UploadWeightsToGPU();
    // Syncs GPU-resident weights -> CPU vectors (only on Scan Dials / Export Brain / Switch to CPU).
    HRESULT ReadbackWeightsFromGPU();

    // Runs an entire forward + MSE grad + backward + Adam weight update purely on GPU without per-step CPU weight copies.
    HRESULT TrainStepGPU(
        const float* xt,
        const float* vTarget,
        const float* tValues,
        uint32_t batch,
        uint32_t height,
        uint32_t width,
        uint64_t stepIndex,
        float lr,
        float beta1,
        float beta2,
        float eps,
        bool computeLossReadback,
        float& outMseLoss);

    const BrainArchitecture& Architecture() const noexcept { return m_arch; }
    uint64_t ParameterCount() const noexcept;
    static uint64_t ExpectedParameterCount(BrainPreset preset) noexcept;
    bool IsGPUReady() const noexcept { return m_gpuReady; }
    bool IsDirectMLReady() const noexcept { return m_dmlReady; }

    // Encodes x_t [N, 3, H, W] + [-1,1] spatial grid + sin/cos(2^k * pi * xy) k=0..3 + time embedding t.
    void BuildSineOperatorInput(
        const float* xt,
        const float* tValues,
        uint32_t batch,
        uint32_t height,
        uint32_t width,
        std::vector<float>& outEncoded) const;

    // CPU Forward pass producing predicted velocity field v_pred [N, 3, H, W].
    void Forward(
        const float* xt,
        const float* tValues,
        uint32_t batch,
        uint32_t height,
        uint32_t width,
        ForwardCache& cache) const;

    // Euler ODE integration from t=0 -> t=1 in 1, 2, or 6 steps.
    std::vector<float> DreamEulerODE(
        const float* initialNoise,
        uint32_t height,
        uint32_t width,
        uint32_t steps) const;

    // Scans all model dials counting busy (|w| > 0.01) vs sleeping weights.
    DialScanReport ScanDials(float threshold = 0.01f);

    // Serialization for Phone LAN Bridge export/import (validates before mutating brain).
    std::string ExportToJson();
    bool ImportFromJson(const std::string& json);
    std::vector<uint8_t> ExportToBinary();
    bool ImportFromBinary(const std::vector<uint8_t>& bytes);

    std::recursive_mutex& Mutex() noexcept { return m_brainMutex; }

    // Direct accessors for CPU Trainer backpropagation & Adam updates.
    ConvLayerWeights& Stem() noexcept { return m_stem; }
    std::vector<float>& StemMW() noexcept { return m_stemMW; }
    std::vector<float>& StemVW() noexcept { return m_stemVW; }
    std::vector<float>& StemMB() noexcept { return m_stemMB; }
    std::vector<float>& StemVB() noexcept { return m_stemVB; }

    std::vector<ResidualBlockWeights>& Blocks() noexcept { return m_blocks; }

    ConvLayerWeights& Head() noexcept { return m_head; }
    std::vector<float>& HeadMW() noexcept { return m_headMW; }
    std::vector<float>& HeadVW() noexcept { return m_headVW; }
    std::vector<float>& HeadMB() noexcept { return m_headMB; }
    std::vector<float>& HeadVB() noexcept { return m_headVB; }

private:
    static BrainArchitecture PresetConfig(BrainPreset preset);
    static void InitConvLayer(ConvLayerWeights& layer, uint32_t inC, uint32_t outC, uint32_t k, uint64_t& rngState);

    void InitializeFromScratchLocked(BrainPreset preset, uint64_t seed);
    void ReleaseGPULocked();
    HRESULT UploadWeightsToGPULocked();
    HRESULT ReadbackWeightsFromGPULocked();

    HRESULT CreateDefaultBuffer(size_t byteSize, Microsoft::WRL::ComPtr<ID3D12Resource>& outBuf);
    HRESULT CreateUploadBuffer(size_t byteSize, Microsoft::WRL::ComPtr<ID3D12Resource>& outBuf);
    HRESULT CreateReadbackBuffer(size_t byteSize, Microsoft::WRL::ComPtr<ID3D12Resource>& outBuf);
    HRESULT EnsureActivationBuffers(uint32_t batch, uint32_t height, uint32_t width);
    HRESULT AllocateGPULayer(const ConvLayerWeights& layer, GPULayerBuffers& gpuBufs);
    HRESULT WaitComputeQueue();

    void BindLayerForward(
        ID3D12GraphicsCommandList* cmdList,
        ID3D12Resource* inTensor,
        const GPULayerBuffers& layer,
        ID3D12Resource* outTensor,
        ID3D12Resource* outPreAct,
        const GPUTrainConstants& cb);

    void BindAddResidual(
        ID3D12GraphicsCommandList* cmdList,
        ID3D12Resource* inA,
        ID3D12Resource* inB,
        ID3D12Resource* outSum,
        uint32_t count,
        GPUTrainConstants cb);

    void BindSiLUBackward(
        ID3D12GraphicsCommandList* cmdList,
        ID3D12Resource* gradPostAct,
        ID3D12Resource* preAct,
        ID3D12Resource* gradPreAct,
        uint32_t count,
        GPUTrainConstants cb);

    void BindAdamDispatch(
        ID3D12GraphicsCommandList* cmdList,
        ID3D12Resource* gradBuf,
        ID3D12Resource* paramBuf,
        ID3D12Resource* mBuf,
        ID3D12Resource* vBuf,
        const GPUTrainConstants& cb);

    mutable std::recursive_mutex m_brainMutex;
    BrainArchitecture m_arch;

    ConvLayerWeights m_stem;
    std::vector<float> m_stemMW, m_stemVW, m_stemMB, m_stemVB;

    std::vector<ResidualBlockWeights> m_blocks;

    ConvLayerWeights m_head;
    std::vector<float> m_headMW, m_headVW, m_headMB, m_headVB;

    // DirectML & D3D12 GPU-resident training resources
    bool m_dmlReady = false;
    bool m_gpuReady = false;
    bool m_gpuWeightsDirtyOnDevice = false;

    Microsoft::WRL::ComPtr<ID3D12Device> m_d3dDevice;
    Microsoft::WRL::ComPtr<ID3D12CommandQueue> m_cmdQueue;
    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> m_computeAlloc;
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> m_computeCmdList;
    Microsoft::WRL::ComPtr<ID3D12Fence> m_computeFence;
    HANDLE m_computeFenceEvent = nullptr;
    uint64_t m_computeFenceValue = 0;

    Microsoft::WRL::ComPtr<IDMLDevice> m_dmlDevice;
    Microsoft::WRL::ComPtr<IDMLCommandRecorder> m_dmlRecorder;

    Microsoft::WRL::ComPtr<ID3D12RootSignature> m_rootSig;
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_uavSrvHeap;
    uint32_t m_descSize = 0;

    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_psoEncodeSine;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_psoConvFwd;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_psoAddResidual;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_psoMseGrad;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_psoConvBwdIn;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_psoSiLUBwd;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_psoConvBwdW;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_psoAdam;

    GPULayerBuffers m_gpuStem;
    std::vector<std::pair<GPULayerBuffers, GPULayerBuffers>> m_gpuBlocks;
    GPULayerBuffers m_gpuHead;

    // GPU-resident activation & scratch buffers sized dynamically for current batch/resolution (up to 128x128)
    uint32_t m_allocBatch = 0;
    uint32_t m_allocHW = 0;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_gpuXt;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_gpuVTarget;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_gpuTValues;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_gpuStemIn;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_gpuStemPre;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_gpuStemOut;
    std::vector<Microsoft::WRL::ComPtr<ID3D12Resource>> m_gpuBlockIn;
    std::vector<Microsoft::WRL::ComPtr<ID3D12Resource>> m_gpuBlockMidPre;
    std::vector<Microsoft::WRL::ComPtr<ID3D12Resource>> m_gpuBlockMidAct;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_gpuConvScratch;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_gpuHeadIn;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_gpuVPred;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_gpuGradPred;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_gpuGradHiddenA;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_gpuGradHiddenB;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_gpuGradMid;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_gpuGradMidPre;

    Microsoft::WRL::ComPtr<ID3D12Resource> m_uploadStaging;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_readbackStaging;
};

} // namespace XboxOTFM
