#include "FMBrain.h"

#include <winrt/Windows.ApplicationModel.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>

namespace XboxOTFM {

namespace {

constexpr float kPi = 3.14159265358979323846f;

inline float NextUniform(uint64_t& state) {
    state += 0x9E3779B97F4A7C15ULL;
    uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z = z ^ (z >> 31);
    return static_cast<float>((z >> 40) & 0xFFFFFFULL) / static_cast<float>(0x1000000ULL);
}

inline float NextGaussian(uint64_t& state) {
    const float u1 = std::max(1e-6f, NextUniform(state));
    const float u2 = NextUniform(state);
    return std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * kPi * u2);
}

inline float SiLU(float x) noexcept {
    return x / (1.0f + std::exp(-std::clamp(x, -20.0f, 20.0f)));
}

void Conv2dNCHW(
    const float* input,
    const ConvLayerWeights& layer,
    uint32_t batch,
    uint32_t height,
    uint32_t width,
    float* output) {
    const uint32_t inC = layer.inChannels;
    const uint32_t outC = layer.outChannels;
    const uint32_t k = layer.kernelSize;
    const int pad = static_cast<int>(k / 2);
    const size_t hw = static_cast<size_t>(height) * width;

    for (uint32_t n = 0; n < batch; ++n) {
        const float* inBatch = input + static_cast<size_t>(n) * inC * hw;
        float* outBatch = output + static_cast<size_t>(n) * outC * hw;

        for (uint32_t oc = 0; oc < outC; ++oc) {
            const float bias = layer.bias[oc];
            float* outPlane = outBatch + static_cast<size_t>(oc) * hw;
            std::fill_n(outPlane, hw, bias);

            for (uint32_t ic = 0; ic < inC; ++ic) {
                const float* inPlane = inBatch + static_cast<size_t>(ic) * hw;
                const float* wKernel = layer.weight.data() + ((static_cast<size_t>(oc) * inC + ic) * k * k);

                if (k == 1) {
                    const float w = wKernel[0];
                    for (size_t idx = 0; idx < hw; ++idx) {
                        outPlane[idx] += inPlane[idx] * w;
                    }
                } else {
                    for (uint32_t y = 0; y < height; ++y) {
                        for (uint32_t x = 0; x < width; ++x) {
                            float acc = 0.0f;
                            for (int ky = 0; ky < static_cast<int>(k); ++ky) {
                                const int iy = static_cast<int>(y) + ky - pad;
                                if (iy < 0 || iy >= static_cast<int>(height)) continue;
                                for (int kx = 0; kx < static_cast<int>(k); ++kx) {
                                    const int ix = static_cast<int>(x) + kx - pad;
                                    if (ix < 0 || ix >= static_cast<int>(width)) continue;
                                    acc += inPlane[static_cast<size_t>(iy) * width + ix] *
                                           wKernel[static_cast<size_t>(ky) * k + kx];
                                }
                            }
                            outPlane[static_cast<size_t>(y) * width + x] += acc;
                        }
                    }
                }
            }
        }
    }
}

bool LoadShaderBytecode(const wchar_t* filename, std::vector<uint8_t>& outBytes) {
    try {
        auto installPath = winrt::Windows::ApplicationModel::Package::Current().InstalledLocation().Path();
        std::wstring fullPath = std::wstring(installPath) + L"\\Shaders\\" + filename;
        std::ifstream ifs(fullPath, std::ios::binary | std::ios::ate);
        if (!ifs.is_open()) return false;
        const auto sz = ifs.tellg();
        if (sz <= 0) return false;
        outBytes.resize(static_cast<size_t>(sz));
        ifs.seekg(0, std::ios::beg);
        ifs.read(reinterpret_cast<char*>(outBytes.data()), sz);
        return true;
    } catch (...) {
        return false;
    }
}

void InsertUAVBarrier(ID3D12GraphicsCommandList* cmdList, ID3D12Resource* res) {
    D3D12_RESOURCE_BARRIER b = {};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    b.UAV.pResource = res;
    cmdList->ResourceBarrier(1, &b);
}

void TransitionBuffer(
    ID3D12GraphicsCommandList* cmdList,
    ID3D12Resource* res,
    D3D12_RESOURCE_STATES before,
    D3D12_RESOURCE_STATES after) {
    if (before == after) return;
    D3D12_RESOURCE_BARRIER b = {};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = res;
    b.Transition.StateBefore = before;
    b.Transition.StateAfter = after;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    cmdList->ResourceBarrier(1, &b);
}

} // namespace

FMBrain::FMBrain(BrainPreset preset) {
    InitializeFromScratch(preset);
}

BrainArchitecture FMBrain::PresetConfig(BrainPreset preset) {
    BrainArchitecture arch;
    arch.preset = preset;
    arch.inputImageChannels = 3;
    arch.coordChannels = 2;
    arch.harmonicBands = 4;
    arch.timeChannels = 4;

    switch (preset) {
        case BrainPreset::Low25k:
            arch.width = 26;
            arch.numResBlocks = 2;
            break;
        case BrainPreset::Medium80k:
            arch.width = 38;
            arch.numResBlocks = 3;
            break;
        case BrainPreset::Max250k:
            arch.width = 58;
            arch.numResBlocks = 4;
            break;
    }
    return arch;
}

uint64_t FMBrain::ExpectedParameterCount(BrainPreset preset) noexcept {
    const BrainArchitecture arch = PresetConfig(preset);
    const uint64_t cIn = arch.TotalInputChannels();
    const uint64_t w = arch.width;
    const uint64_t stem = (w * cIn * 1ULL * 1ULL) + w;
    const uint64_t blk = 2ULL * ((w * w * 3ULL * 3ULL) + w);
    const uint64_t head = (arch.inputImageChannels * w * 1ULL * 1ULL) + arch.inputImageChannels;
    return stem + static_cast<uint64_t>(arch.numResBlocks) * blk + head;
}

void FMBrain::InitConvLayer(ConvLayerWeights& layer, uint32_t inC, uint32_t outC, uint32_t k, uint64_t& rngState) {
    layer.inChannels = inC;
    layer.outChannels = outC;
    layer.kernelSize = k;
    const size_t weightCount = static_cast<size_t>(outC) * inC * k * k;
    layer.weight.resize(weightCount);
    layer.bias.assign(outC, 0.0f);

    const float fanIn = static_cast<float>(inC * k * k);
    const float stdDev = std::sqrt(2.0f / std::max(1.0f, fanIn));
    for (size_t i = 0; i < weightCount; ++i) {
        layer.weight[i] = NextGaussian(rngState) * stdDev;
    }
}

void FMBrain::InitializeFromScratch(BrainPreset preset, uint64_t seed) {
    std::lock_guard<std::recursive_mutex> lock(m_brainMutex);
    InitializeFromScratchLocked(preset, seed);
}

void FMBrain::InitializeFromScratchLocked(BrainPreset preset, uint64_t seed) {
    m_arch = PresetConfig(preset);
    uint64_t rngState = seed;

    const uint32_t cIn = m_arch.TotalInputChannels();
    const uint32_t w = m_arch.width;

    InitConvLayer(m_stem, cIn, w, 1, rngState);
    m_stemMW.assign(m_stem.weight.size(), 0.0f);
    m_stemVW.assign(m_stem.weight.size(), 0.0f);
    m_stemMB.assign(m_stem.bias.size(), 0.0f);
    m_stemVB.assign(m_stem.bias.size(), 0.0f);

    m_blocks.resize(m_arch.numResBlocks);
    for (auto& blk : m_blocks) {
        InitConvLayer(blk.conv1, w, w, 3, rngState);
        InitConvLayer(blk.conv2, w, w, 3, rngState);
        blk.mW1.assign(blk.conv1.weight.size(), 0.0f);
        blk.vW1.assign(blk.conv1.weight.size(), 0.0f);
        blk.mB1.assign(blk.conv1.bias.size(), 0.0f);
        blk.vB1.assign(blk.conv1.bias.size(), 0.0f);

        blk.mW2.assign(blk.conv2.weight.size(), 0.0f);
        blk.vW2.assign(blk.conv2.weight.size(), 0.0f);
        blk.mB2.assign(blk.conv2.bias.size(), 0.0f);
        blk.vB2.assign(blk.conv2.bias.size(), 0.0f);
    }

    InitConvLayer(m_head, w, m_arch.inputImageChannels, 1, rngState);
    for (float& val : m_head.weight) {
        val *= 0.1f;
    }
    m_headMW.assign(m_head.weight.size(), 0.0f);
    m_headVW.assign(m_head.weight.size(), 0.0f);
    m_headMB.assign(m_head.bias.size(), 0.0f);
    m_headVB.assign(m_head.bias.size(), 0.0f);
    m_gpuWeightsDirtyOnDevice = false;
}

uint64_t FMBrain::ParameterCount() const noexcept {
    uint64_t total = m_stem.weight.size() + m_stem.bias.size();
    for (const auto& blk : m_blocks) {
        total += blk.conv1.weight.size() + blk.conv1.bias.size();
        total += blk.conv2.weight.size() + blk.conv2.bias.size();
    }
    total += m_head.weight.size() + m_head.bias.size();
    return total;
}

HRESULT FMBrain::CreateDefaultBuffer(size_t byteSize, Microsoft::WRL::ComPtr<ID3D12Resource>& outBuf) {
    D3D12_HEAP_PROPERTIES heap = {};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = std::max<size_t>(256, (byteSize + 255) & ~size_t(255));
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    return m_d3dDevice->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&outBuf));
}

HRESULT FMBrain::CreateUploadBuffer(size_t byteSize, Microsoft::WRL::ComPtr<ID3D12Resource>& outBuf) {
    D3D12_HEAP_PROPERTIES heap = {};
    heap.Type = D3D12_HEAP_TYPE_UPLOAD;
    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = std::max<size_t>(256, (byteSize + 255) & ~size_t(255));
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    return m_d3dDevice->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&outBuf));
}

HRESULT FMBrain::CreateReadbackBuffer(size_t byteSize, Microsoft::WRL::ComPtr<ID3D12Resource>& outBuf) {
    D3D12_HEAP_PROPERTIES heap = {};
    heap.Type = D3D12_HEAP_TYPE_READBACK;
    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = std::max<size_t>(256, (byteSize + 255) & ~size_t(255));
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    return m_d3dDevice->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&outBuf));
}

HRESULT FMBrain::AllocateGPULayer(const ConvLayerWeights& layer, GPULayerBuffers& gpuBufs) {
    const size_t wBytes = layer.weight.size() * sizeof(float);
    const size_t bBytes = layer.bias.size() * sizeof(float);
    HRESULT hr = CreateDefaultBuffer(wBytes, gpuBufs.weight);
    if (FAILED(hr)) return hr;
    if (FAILED(hr = CreateDefaultBuffer(bBytes, gpuBufs.bias))) return hr;
    if (FAILED(hr = CreateDefaultBuffer(wBytes, gpuBufs.mW))) return hr;
    if (FAILED(hr = CreateDefaultBuffer(wBytes, gpuBufs.vW))) return hr;
    if (FAILED(hr = CreateDefaultBuffer(bBytes, gpuBufs.mB))) return hr;
    if (FAILED(hr = CreateDefaultBuffer(bBytes, gpuBufs.vB))) return hr;
    if (FAILED(hr = CreateDefaultBuffer(wBytes, gpuBufs.gradW))) return hr;
    if (FAILED(hr = CreateDefaultBuffer(bBytes, gpuBufs.gradB))) return hr;
    return S_OK;
}

HRESULT FMBrain::EnsureActivationBuffers(uint32_t batch, uint32_t height, uint32_t width) {
    const uint32_t hw = height * width;
    if (m_allocBatch >= batch && m_allocHW >= hw && m_gpuStemIn) {
        return S_OK;
    }

    const uint32_t cIn = m_arch.TotalInputChannels();
    const uint32_t w = m_arch.width;
    const size_t rgbBytes = static_cast<size_t>(batch) * 3 * hw * sizeof(float);
    const size_t encBytes = static_cast<size_t>(batch) * cIn * hw * sizeof(float);
    const size_t hidBytes = static_cast<size_t>(batch) * w * hw * sizeof(float);

    HRESULT hr = S_OK;
    if (FAILED(hr = CreateDefaultBuffer(rgbBytes, m_gpuXt))) return hr;
    if (FAILED(hr = CreateDefaultBuffer(rgbBytes, m_gpuVTarget))) return hr;
    if (FAILED(hr = CreateDefaultBuffer(static_cast<size_t>(batch) * sizeof(float), m_gpuTValues))) return hr;
    if (FAILED(hr = CreateDefaultBuffer(encBytes, m_gpuStemIn))) return hr;
    if (FAILED(hr = CreateDefaultBuffer(hidBytes, m_gpuStemPre))) return hr;
    if (FAILED(hr = CreateDefaultBuffer(hidBytes, m_gpuStemOut))) return hr;

    m_gpuBlockIn.resize(m_blocks.size());
    m_gpuBlockMidPre.resize(m_blocks.size());
    m_gpuBlockMidAct.resize(m_blocks.size());
    for (size_t b = 0; b < m_blocks.size(); ++b) {
        if (FAILED(hr = CreateDefaultBuffer(hidBytes, m_gpuBlockIn[b]))) return hr;
        if (FAILED(hr = CreateDefaultBuffer(hidBytes, m_gpuBlockMidPre[b]))) return hr;
        if (FAILED(hr = CreateDefaultBuffer(hidBytes, m_gpuBlockMidAct[b]))) return hr;
    }

    if (FAILED(hr = CreateDefaultBuffer(hidBytes, m_gpuConvScratch))) return hr;
    if (FAILED(hr = CreateDefaultBuffer(hidBytes, m_gpuHeadIn))) return hr;
    if (FAILED(hr = CreateDefaultBuffer(rgbBytes, m_gpuVPred))) return hr;
    if (FAILED(hr = CreateDefaultBuffer(rgbBytes, m_gpuGradPred))) return hr;
    if (FAILED(hr = CreateDefaultBuffer(hidBytes, m_gpuGradHiddenA))) return hr;
    if (FAILED(hr = CreateDefaultBuffer(hidBytes, m_gpuGradHiddenB))) return hr;
    if (FAILED(hr = CreateDefaultBuffer(hidBytes, m_gpuGradMid))) return hr;
    if (FAILED(hr = CreateDefaultBuffer(hidBytes, m_gpuGradMidPre))) return hr;

    const size_t maxStaging = std::max<size_t>(rgbBytes * 2 + 4096, ParameterCount() * sizeof(float) + 4096);
    if (FAILED(hr = CreateUploadBuffer(maxStaging, m_uploadStaging))) return hr;
    if (FAILED(hr = CreateReadbackBuffer(maxStaging, m_readbackStaging))) return hr;

    m_allocBatch = batch;
    m_allocHW = hw;
    return S_OK;
}

HRESULT FMBrain::InitializeGPU(ID3D12Device* d3dDevice, ID3D12CommandQueue* cmdQueue) {
    std::lock_guard<std::recursive_mutex> lock(m_brainMutex);
    ReleaseGPULocked();
    if (!d3dDevice || !cmdQueue) {
        return E_INVALIDARG;
    }

    m_d3dDevice = d3dDevice;
    m_cmdQueue = cmdQueue;

    // Optional DirectML device validation (gracefully proceeds with D3D12 compute if DML is unavailable)
    if (SUCCEEDED(DMLCreateDevice(d3dDevice, DML_CREATE_DEVICE_FLAG_NONE, IID_PPV_ARGS(&m_dmlDevice)))) {
        if (SUCCEEDED(m_dmlDevice->CreateCommandRecorder(IID_PPV_ARGS(&m_dmlRecorder)))) {
            m_dmlReady = true;
        }
    }

    HRESULT hr = d3dDevice->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&m_computeAlloc));
    if (FAILED(hr)) { ReleaseGPULocked(); return hr; }
    hr = d3dDevice->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, m_computeAlloc.Get(), nullptr, IID_PPV_ARGS(&m_computeCmdList));
    if (FAILED(hr)) { ReleaseGPULocked(); return hr; }
    m_computeCmdList->Close();

    hr = d3dDevice->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_computeFence));
    if (FAILED(hr)) { ReleaseGPULocked(); return hr; }
    m_computeFenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);

    // Root Signature: [0] 32-bit constants (20 DWORDs), [1..3] 3 SRVs (t0..t2), [4..6] 3 UAVs (u0..u2)
    D3D12_ROOT_PARAMETER params[7] = {};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[0].Constants.ShaderRegister = 0;
    params[0].Constants.Num32BitValues = sizeof(GPUTrainConstants) / sizeof(uint32_t);
    for (UINT i = 0; i < 3; ++i) {
        params[1 + i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
        params[1 + i].Descriptor.ShaderRegister = i;
        params[4 + i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        params[4 + i].Descriptor.ShaderRegister = i;
    }

    D3D12_ROOT_SIGNATURE_DESC rsDesc = {};
    rsDesc.NumParameters = 7;
    rsDesc.pParameters = params;
    Microsoft::WRL::ComPtr<ID3DBlob> sigBlob, errBlob;
    hr = D3D12SerializeRootSignature(&rsDesc, D3D_ROOT_SIGNATURE_VERSION_1, &sigBlob, &errBlob);
    if (FAILED(hr)) { ReleaseGPULocked(); return hr; }
    hr = d3dDevice->CreateRootSignature(0, sigBlob->GetBufferPointer(), sigBlob->GetBufferSize(), IID_PPV_ARGS(&m_rootSig));
    if (FAILED(hr)) { ReleaseGPULocked(); return hr; }

    auto createComputePso = [&](const wchar_t* csoName, Microsoft::WRL::ComPtr<ID3D12PipelineState>& outPso) -> HRESULT {
        std::vector<uint8_t> bytecode;
        if (!LoadShaderBytecode(csoName, bytecode)) {
            return HRESULT_FROM_WIN32(ERROR_FILE_NOT_FOUND);
        }
        D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
        psoDesc.pRootSignature = m_rootSig.Get();
        psoDesc.CS = { bytecode.data(), bytecode.size() };
        return d3dDevice->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&outPso));
    };

    if (FAILED(hr = createComputePso(L"CSEncodeSineOperators.cso", m_psoEncodeSine)) ||
        FAILED(hr = createComputePso(L"CSConvForward.cso", m_psoConvFwd)) ||
        FAILED(hr = createComputePso(L"CSAddResidual.cso", m_psoAddResidual)) ||
        FAILED(hr = createComputePso(L"CSMSELossGrad.cso", m_psoMseGrad)) ||
        FAILED(hr = createComputePso(L"CSConvBackwardInput.cso", m_psoConvBwdIn)) ||
        FAILED(hr = createComputePso(L"CSSiLUBackward.cso", m_psoSiLUBwd)) ||
        FAILED(hr = createComputePso(L"CSConvBackwardWeights.cso", m_psoConvBwdW)) ||
        FAILED(hr = createComputePso(L"CSAdamUpdate.cso", m_psoAdam))) {
        ReleaseGPULocked();
        return hr;
    }

    if (FAILED(hr = AllocateGPULayer(m_stem, m_gpuStem))) { ReleaseGPULocked(); return hr; }
    m_gpuBlocks.resize(m_blocks.size());
    for (size_t b = 0; b < m_blocks.size(); ++b) {
        if (FAILED(hr = AllocateGPULayer(m_blocks[b].conv1, m_gpuBlocks[b].first)) ||
            FAILED(hr = AllocateGPULayer(m_blocks[b].conv2, m_gpuBlocks[b].second))) {
            ReleaseGPULocked();
            return hr;
        }
    }
    if (FAILED(hr = AllocateGPULayer(m_head, m_gpuHead))) { ReleaseGPULocked(); return hr; }

    if (FAILED(hr = EnsureActivationBuffers(4, 32, 32))) { ReleaseGPULocked(); return hr; }
    if (FAILED(hr = UploadWeightsToGPULocked())) { ReleaseGPULocked(); return hr; }

    m_gpuReady = true;
    return S_OK;
}

void FMBrain::ReleaseGPU() {
    std::lock_guard<std::recursive_mutex> lock(m_brainMutex);
    ReleaseGPULocked();
}

void FMBrain::ReleaseGPULocked() {
    m_gpuReady = false;
    m_dmlReady = false;
    m_allocBatch = 0;
    m_allocHW = 0;

    m_uploadStaging.Reset();
    m_readbackStaging.Reset();
    m_gpuGradMidPre.Reset();
    m_gpuGradMid.Reset();
    m_gpuGradHiddenB.Reset();
    m_gpuGradHiddenA.Reset();
    m_gpuGradPred.Reset();
    m_gpuVPred.Reset();
    m_gpuHeadIn.Reset();
    m_gpuConvScratch.Reset();
    m_gpuBlockMidAct.clear();
    m_gpuBlockMidPre.clear();
    m_gpuBlockIn.clear();
    m_gpuStemOut.Reset();
    m_gpuStemPre.Reset();
    m_gpuStemIn.Reset();
    m_gpuTValues.Reset();
    m_gpuVTarget.Reset();
    m_gpuXt.Reset();

    m_gpuHead = {};
    m_gpuBlocks.clear();
    m_gpuStem = {};

    m_psoAdam.Reset();
    m_psoConvBwdW.Reset();
    m_psoSiLUBwd.Reset();
    m_psoConvBwdIn.Reset();
    m_psoMseGrad.Reset();
    m_psoAddResidual.Reset();
    m_psoConvFwd.Reset();
    m_psoEncodeSine.Reset();
    m_uavSrvHeap.Reset();
    m_rootSig.Reset();

    m_dmlRecorder.Reset();
    m_dmlDevice.Reset();

    if (m_computeFenceEvent) {
        CloseHandle(m_computeFenceEvent);
        m_computeFenceEvent = nullptr;
    }
    m_computeFence.Reset();
    m_computeCmdList.Reset();
    m_computeAlloc.Reset();
    m_cmdQueue.Reset();
    m_d3dDevice.Reset();
}

HRESULT FMBrain::WaitComputeQueue() {
    if (!m_cmdQueue || !m_computeFence || !m_computeFenceEvent) return E_FAIL;
    const uint64_t target = ++m_computeFenceValue;
    HRESULT hr = m_cmdQueue->Signal(m_computeFence.Get(), target);
    if (FAILED(hr)) return hr;
    if (m_computeFence->GetCompletedValue() < target) {
        m_computeFence->SetEventOnCompletion(target, m_computeFenceEvent);
        WaitForSingleObject(m_computeFenceEvent, 3000);
    }
    return S_OK;
}

HRESULT FMBrain::UploadWeightsToGPU() {
    std::lock_guard<std::recursive_mutex> lock(m_brainMutex);
    return UploadWeightsToGPULocked();
}

HRESULT FMBrain::UploadWeightsToGPULocked() {
    if (!m_d3dDevice || !m_uploadStaging) return E_NOT_VALID_STATE;
    HRESULT hr = m_computeAlloc->Reset();
    if (FAILED(hr)) return hr;
    hr = m_computeCmdList->Reset(m_computeAlloc.Get(), nullptr);
    if (FAILED(hr)) return hr;

    uint8_t* mapped = nullptr;
    hr = m_uploadStaging->Map(0, nullptr, reinterpret_cast<void**>(&mapped));
    if (FAILED(hr)) return hr;

    size_t stagingOffset = 0;
    auto copyVec = [&](const std::vector<float>& vec, ID3D12Resource* dst) {
        const size_t bytes = vec.size() * sizeof(float);
        std::memcpy(mapped + stagingOffset, vec.data(), bytes);
        TransitionBuffer(m_computeCmdList.Get(), dst, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
        m_computeCmdList->CopyBufferRegion(dst, 0, m_uploadStaging.Get(), stagingOffset, bytes);
        TransitionBuffer(m_computeCmdList.Get(), dst, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
        stagingOffset = (stagingOffset + bytes + 255) & ~size_t(255);
    };

    auto zeroLayerMoments = [&](const ConvLayerWeights& layer, const GPULayerBuffers& gpuL) {
        auto zeroBuf = [&](size_t floatCount, ID3D12Resource* dst) {
            const size_t bytes = floatCount * sizeof(float);
            std::memset(mapped + stagingOffset, 0, bytes);
            TransitionBuffer(m_computeCmdList.Get(), dst, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
            m_computeCmdList->CopyBufferRegion(dst, 0, m_uploadStaging.Get(), stagingOffset, bytes);
            TransitionBuffer(m_computeCmdList.Get(), dst, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
            stagingOffset = (stagingOffset + bytes + 255) & ~size_t(255);
        };
        zeroBuf(layer.weight.size(), gpuL.mW.Get());
        zeroBuf(layer.weight.size(), gpuL.vW.Get());
        zeroBuf(layer.bias.size(), gpuL.mB.Get());
        zeroBuf(layer.bias.size(), gpuL.vB.Get());
    };

    copyVec(m_stem.weight, m_gpuStem.weight.Get());
    copyVec(m_stem.bias, m_gpuStem.bias.Get());
    zeroLayerMoments(m_stem, m_gpuStem);

    for (size_t b = 0; b < m_blocks.size(); ++b) {
        copyVec(m_blocks[b].conv1.weight, m_gpuBlocks[b].first.weight.Get());
        copyVec(m_blocks[b].conv1.bias, m_gpuBlocks[b].first.bias.Get());
        zeroLayerMoments(m_blocks[b].conv1, m_gpuBlocks[b].first);

        copyVec(m_blocks[b].conv2.weight, m_gpuBlocks[b].second.weight.Get());
        copyVec(m_blocks[b].conv2.bias, m_gpuBlocks[b].second.bias.Get());
        zeroLayerMoments(m_blocks[b].conv2, m_gpuBlocks[b].second);
    }

    copyVec(m_head.weight, m_gpuHead.weight.Get());
    copyVec(m_head.bias, m_gpuHead.bias.Get());
    zeroLayerMoments(m_head, m_gpuHead);

    m_uploadStaging->Unmap(0, nullptr);
    m_computeCmdList->Close();
    ID3D12CommandList* lists[] = { m_computeCmdList.Get() };
    m_cmdQueue->ExecuteCommandLists(1, lists);
    m_gpuWeightsDirtyOnDevice = false;
    return WaitComputeQueue();
}

HRESULT FMBrain::ReadbackWeightsFromGPU() {
    std::lock_guard<std::recursive_mutex> lock(m_brainMutex);
    return ReadbackWeightsFromGPULocked();
}

HRESULT FMBrain::ReadbackWeightsFromGPULocked() {
    if (!m_gpuReady || !m_gpuWeightsDirtyOnDevice) return S_OK;
    HRESULT hr = m_computeAlloc->Reset();
    if (FAILED(hr)) return hr;
    hr = m_computeCmdList->Reset(m_computeAlloc.Get(), nullptr);
    if (FAILED(hr)) return hr;

    size_t stagingOffset = 0;
    struct CopySpan { std::vector<float>* dst; size_t offset; size_t bytes; };
    std::vector<CopySpan> spans;

    auto queueRead = [&](std::vector<float>& vec, ID3D12Resource* src) {
        const size_t bytes = vec.size() * sizeof(float);
        TransitionBuffer(m_computeCmdList.Get(), src, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
        m_computeCmdList->CopyBufferRegion(m_readbackStaging.Get(), stagingOffset, src, 0, bytes);
        TransitionBuffer(m_computeCmdList.Get(), src, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
        spans.push_back({ &vec, stagingOffset, bytes });
        stagingOffset = (stagingOffset + bytes + 255) & ~size_t(255);
    };

    queueRead(m_stem.weight, m_gpuStem.weight.Get());
    queueRead(m_stem.bias, m_gpuStem.bias.Get());
    for (size_t b = 0; b < m_blocks.size(); ++b) {
        queueRead(m_blocks[b].conv1.weight, m_gpuBlocks[b].first.weight.Get());
        queueRead(m_blocks[b].conv1.bias, m_gpuBlocks[b].first.bias.Get());
        queueRead(m_blocks[b].conv2.weight, m_gpuBlocks[b].second.weight.Get());
        queueRead(m_blocks[b].conv2.bias, m_gpuBlocks[b].second.bias.Get());
    }
    queueRead(m_head.weight, m_gpuHead.weight.Get());
    queueRead(m_head.bias, m_gpuHead.bias.Get());

    m_computeCmdList->Close();
    ID3D12CommandList* lists[] = { m_computeCmdList.Get() };
    m_cmdQueue->ExecuteCommandLists(1, lists);
    hr = WaitComputeQueue();
    if (FAILED(hr)) return hr;

    uint8_t* mapped = nullptr;
    D3D12_RANGE range = { 0, stagingOffset };
    hr = m_readbackStaging->Map(0, &range, reinterpret_cast<void**>(&mapped));
    if (FAILED(hr)) return hr;
    for (const auto& sp : spans) {
        std::memcpy(sp.dst->data(), mapped + sp.offset, sp.bytes);
    }
    D3D12_RANGE emptyRange = { 0, 0 };
    m_readbackStaging->Unmap(0, &emptyRange);
    m_gpuWeightsDirtyOnDevice = false;
    return S_OK;
}

void FMBrain::BindLayerForward(
    ID3D12GraphicsCommandList* cmdList,
    ID3D12Resource* inTensor,
    const GPULayerBuffers& layer,
    ID3D12Resource* outTensor,
    ID3D12Resource* outPreAct,
    const GPUTrainConstants& cb) {
    cmdList->SetPipelineState(m_psoConvFwd.Get());
    cmdList->SetComputeRoot32BitConstants(0, sizeof(GPUTrainConstants) / sizeof(uint32_t), &cb, 0);
    cmdList->SetComputeRootShaderResourceView(1, inTensor->GetGPUVirtualAddress());
    cmdList->SetComputeRootShaderResourceView(2, layer.weight->GetGPUVirtualAddress());
    cmdList->SetComputeRootShaderResourceView(3, layer.bias->GetGPUVirtualAddress());
    cmdList->SetComputeRootUnorderedAccessView(4, outTensor->GetGPUVirtualAddress());
    cmdList->SetComputeRootUnorderedAccessView(5, outPreAct->GetGPUVirtualAddress());
    cmdList->Dispatch((cb.gWidth + 7) / 8, (cb.gHeight + 7) / 8, cb.gBatch * cb.gOutC);
    InsertUAVBarrier(cmdList, outTensor);
}

void FMBrain::BindAddResidual(
    ID3D12GraphicsCommandList* cmdList,
    ID3D12Resource* inA,
    ID3D12Resource* inB,
    ID3D12Resource* outSum,
    uint32_t count,
    GPUTrainConstants cb) {
    cb.gParamCount = count;
    cmdList->SetPipelineState(m_psoAddResidual.Get());
    cmdList->SetComputeRoot32BitConstants(0, sizeof(GPUTrainConstants) / sizeof(uint32_t), &cb, 0);
    cmdList->SetComputeRootShaderResourceView(1, inA->GetGPUVirtualAddress());
    cmdList->SetComputeRootShaderResourceView(2, inB->GetGPUVirtualAddress());
    cmdList->SetComputeRootUnorderedAccessView(4, outSum->GetGPUVirtualAddress());
    cmdList->Dispatch((count + 63) / 64, 1, 1);
    InsertUAVBarrier(cmdList, outSum);
}

void FMBrain::BindSiLUBackward(
    ID3D12GraphicsCommandList* cmdList,
    ID3D12Resource* gradPostAct,
    ID3D12Resource* preAct,
    ID3D12Resource* gradPreAct,
    uint32_t count,
    GPUTrainConstants cb) {
    cb.gParamCount = count;
    cmdList->SetPipelineState(m_psoSiLUBwd.Get());
    cmdList->SetComputeRoot32BitConstants(0, sizeof(GPUTrainConstants) / sizeof(uint32_t), &cb, 0);
    cmdList->SetComputeRootShaderResourceView(1, gradPostAct->GetGPUVirtualAddress());
    cmdList->SetComputeRootShaderResourceView(2, preAct->GetGPUVirtualAddress());
    cmdList->SetComputeRootUnorderedAccessView(4, gradPreAct->GetGPUVirtualAddress());
    cmdList->Dispatch((count + 63) / 64, 1, 1);
    InsertUAVBarrier(cmdList, gradPreAct);
}

void FMBrain::BindAdamDispatch(
    ID3D12GraphicsCommandList* cmdList,
    ID3D12Resource* gradBuf,
    ID3D12Resource* paramBuf,
    ID3D12Resource* mBuf,
    ID3D12Resource* vBuf,
    const GPUTrainConstants& cb) {
    cmdList->SetPipelineState(m_psoAdam.Get());
    cmdList->SetComputeRoot32BitConstants(0, sizeof(GPUTrainConstants) / sizeof(uint32_t), &cb, 0);
    cmdList->SetComputeRootShaderResourceView(1, gradBuf->GetGPUVirtualAddress());
    cmdList->SetComputeRootUnorderedAccessView(4, paramBuf->GetGPUVirtualAddress());
    cmdList->SetComputeRootUnorderedAccessView(5, mBuf->GetGPUVirtualAddress());
    cmdList->SetComputeRootUnorderedAccessView(6, vBuf->GetGPUVirtualAddress());
    cmdList->Dispatch((cb.gParamCount + 63) / 64, 1, 1);
    InsertUAVBarrier(cmdList, paramBuf);
}

HRESULT FMBrain::TrainStepGPU(
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
    float& outMseLoss) {
    std::lock_guard<std::recursive_mutex> lock(m_brainMutex);
    if (!m_gpuReady) return E_NOT_VALID_STATE;

    // Guard against GPU TDR watchdog timeout at 128x128 (max batch 2 on GPU)
    if (width >= 128u || height >= 128u) {
        batch = std::min(batch, 2u);
    }

    HRESULT hr = EnsureActivationBuffers(batch, height, width);
    if (FAILED(hr)) return hr;

    const uint32_t hw = height * width;
    const uint32_t rgbCount = batch * 3 * hw;
    const uint32_t hiddenCount = batch * m_arch.width * hw;
    const size_t rgbBytes = static_cast<size_t>(rgbCount) * sizeof(float);
    const size_t tBytes = static_cast<size_t>(batch) * sizeof(float);

    hr = m_computeAlloc->Reset();
    if (FAILED(hr)) return hr;
    hr = m_computeCmdList->Reset(m_computeAlloc.Get(), nullptr);
    if (FAILED(hr)) return hr;

    // Upload only this lesson's small batch inputs (weights stay 100% resident in GPU VRAM)
    uint8_t* mapped = nullptr;
    hr = m_uploadStaging->Map(0, nullptr, reinterpret_cast<void**>(&mapped));
    if (FAILED(hr)) return hr;
    const size_t offXt = 0;
    const size_t offVt = (rgbBytes + 255) & ~size_t(255);
    const size_t offT = (offVt + rgbBytes + 255) & ~size_t(255);
    std::memcpy(mapped + offXt, xt, rgbBytes);
    std::memcpy(mapped + offVt, vTarget, rgbBytes);
    std::memcpy(mapped + offT, tValues, tBytes);
    m_uploadStaging->Unmap(0, nullptr);

    m_computeCmdList->CopyBufferRegion(m_gpuXt.Get(), 0, m_uploadStaging.Get(), offXt, rgbBytes);
    m_computeCmdList->CopyBufferRegion(m_gpuVTarget.Get(), 0, m_uploadStaging.Get(), offVt, rgbBytes);
    m_computeCmdList->CopyBufferRegion(m_gpuTValues.Get(), 0, m_uploadStaging.Get(), offT, tBytes);
    InsertUAVBarrier(m_computeCmdList.Get(), m_gpuXt.Get());

    m_computeCmdList->SetComputeRootSignature(m_rootSig.Get());

    GPUTrainConstants cb = {};
    cb.gBatch = batch;
    cb.gWidth = width;
    cb.gHeight = height;
    cb.gSampleStride = 3 * hw;
    cb.gLearningRate = lr;
    cb.gBeta1 = beta1;
    cb.gBeta2 = beta2;
    cb.gEpsilon = eps;
    cb.gBiasCorr1 = 1.0f - std::pow(beta1, static_cast<float>(stepIndex));
    cb.gBiasCorr2 = 1.0f - std::pow(beta2, static_cast<float>(stepIndex));
    cb.gInvOutCount = 1.0f / static_cast<float>(std::max(1u, rgbCount));
    cb.gGradClipScale = 1.0f;

    // 1. Encode Sine-Operator Input on GPU
    m_computeCmdList->SetPipelineState(m_psoEncodeSine.Get());
    m_computeCmdList->SetComputeRoot32BitConstants(0, sizeof(GPUTrainConstants) / sizeof(uint32_t), &cb, 0);
    m_computeCmdList->SetComputeRootShaderResourceView(1, m_gpuXt->GetGPUVirtualAddress());
    m_computeCmdList->SetComputeRootShaderResourceView(3, m_gpuTValues->GetGPUVirtualAddress());
    m_computeCmdList->SetComputeRootUnorderedAccessView(4, m_gpuStemIn->GetGPUVirtualAddress());
    m_computeCmdList->Dispatch((width + 7) / 8, (height + 7) / 8, batch);
    InsertUAVBarrier(m_computeCmdList.Get(), m_gpuStemIn.Get());

    // 2. Forward Stem (1x1 Conv + SiLU)
    cb.gInC = m_stem.inChannels;
    cb.gOutC = m_stem.outChannels;
    cb.gKernelSize = 1;
    cb.gApplySiLU = 1;
    BindLayerForward(m_computeCmdList.Get(), m_gpuStemIn.Get(), m_gpuStem, m_gpuStemOut.Get(), m_gpuStemPre.Get(), cb);

    // 3. Forward Residual Blocks: BlockOutput = Conv2(SiLU(Conv1(BlockIn))) + BlockIn
    ID3D12Resource* curAct = m_gpuStemOut.Get();
    for (size_t b = 0; b < m_blocks.size(); ++b) {
        m_computeCmdList->CopyResource(m_gpuBlockIn[b].Get(), curAct);
        InsertUAVBarrier(m_computeCmdList.Get(), m_gpuBlockIn[b].Get());

        cb.gInC = m_arch.width;
        cb.gOutC = m_arch.width;
        cb.gKernelSize = 3;
        cb.gApplySiLU = 1;
        BindLayerForward(m_computeCmdList.Get(), m_gpuBlockIn[b].Get(), m_gpuBlocks[b].first, m_gpuBlockMidAct[b].Get(), m_gpuBlockMidPre[b].Get(), cb);

        // Conv2 is strictly linear; add true BlockIn tensor via explicit CSAddResidual pass
        cb.gApplySiLU = 0;
        BindLayerForward(m_computeCmdList.Get(), m_gpuBlockMidAct[b].Get(), m_gpuBlocks[b].second, m_gpuConvScratch.Get(), m_gpuGradHiddenB.Get(), cb);

        ID3D12Resource* blkOut = (b + 1 == m_blocks.size()) ? m_gpuHeadIn.Get() : m_gpuBlockIn[b + 1].Get();
        BindAddResidual(m_computeCmdList.Get(), m_gpuConvScratch.Get(), m_gpuBlockIn[b].Get(), blkOut, hiddenCount, cb);
        curAct = blkOut;
    }

    // 4. Forward Velocity Head (1x1 Conv, linear)
    cb.gInC = m_head.inChannels;
    cb.gOutC = 3;
    cb.gKernelSize = 1;
    cb.gApplySiLU = 0;
    BindLayerForward(m_computeCmdList.Get(), m_gpuHeadIn.Get(), m_gpuHead, m_gpuVPred.Get(), m_gpuGradPred.Get(), cb);

    // 5. GPU MSE Loss Gradient
    cb.gParamCount = rgbCount;
    m_computeCmdList->SetPipelineState(m_psoMseGrad.Get());
    m_computeCmdList->SetComputeRoot32BitConstants(0, sizeof(GPUTrainConstants) / sizeof(uint32_t), &cb, 0);
    m_computeCmdList->SetComputeRootShaderResourceView(1, m_gpuVPred->GetGPUVirtualAddress());
    m_computeCmdList->SetComputeRootShaderResourceView(2, m_gpuVTarget->GetGPUVirtualAddress());
    m_computeCmdList->SetComputeRootUnorderedAccessView(4, m_gpuGradPred->GetGPUVirtualAddress());
    m_computeCmdList->Dispatch((rgbCount + 63) / 64, 1, 1);
    InsertUAVBarrier(m_computeCmdList.Get(), m_gpuGradPred.Get());

    auto dispatchBwdWeights = [&](ID3D12Resource* inAct, ID3D12Resource* gradOut, const ConvLayerWeights& layer, const GPULayerBuffers& gpuL) {
        cb.gInC = layer.inChannels;
        cb.gOutC = layer.outChannels;
        cb.gKernelSize = layer.kernelSize;
        cb.gParamCount = static_cast<uint32_t>(layer.weight.size());
        m_computeCmdList->SetPipelineState(m_psoConvBwdW.Get());
        m_computeCmdList->SetComputeRoot32BitConstants(0, sizeof(GPUTrainConstants) / sizeof(uint32_t), &cb, 0);
        m_computeCmdList->SetComputeRootShaderResourceView(1, inAct->GetGPUVirtualAddress());
        m_computeCmdList->SetComputeRootShaderResourceView(2, gradOut->GetGPUVirtualAddress());
        m_computeCmdList->SetComputeRootUnorderedAccessView(5, gpuL.gradW->GetGPUVirtualAddress());
        m_computeCmdList->SetComputeRootUnorderedAccessView(6, gpuL.gradB->GetGPUVirtualAddress());
        // 1 threadgroup (256 threads) per weight scalar for fast parallel reduction
        m_computeCmdList->Dispatch(cb.gParamCount, 1, 1);
        InsertUAVBarrier(m_computeCmdList.Get(), gpuL.gradW.Get());
        InsertUAVBarrier(m_computeCmdList.Get(), gpuL.gradB.Get());
    };

    auto dispatchBwdInput = [&](ID3D12Resource* gradOut, ID3D12Resource* weightBuf, ID3D12Resource* gradInOut, uint32_t inC, uint32_t outC, uint32_t k) {
        cb.gInC = inC;
        cb.gOutC = outC;
        cb.gKernelSize = k;
        m_computeCmdList->SetPipelineState(m_psoConvBwdIn.Get());
        m_computeCmdList->SetComputeRoot32BitConstants(0, sizeof(GPUTrainConstants) / sizeof(uint32_t), &cb, 0);
        m_computeCmdList->SetComputeRootShaderResourceView(1, gradOut->GetGPUVirtualAddress());
        m_computeCmdList->SetComputeRootShaderResourceView(2, weightBuf->GetGPUVirtualAddress());
        m_computeCmdList->SetComputeRootUnorderedAccessView(4, gradInOut->GetGPUVirtualAddress());
        m_computeCmdList->Dispatch((width + 7) / 8, (height + 7) / 8, batch * inC);
        InsertUAVBarrier(m_computeCmdList.Get(), gradInOut);
    };

    // 6. Backward Head
    dispatchBwdWeights(m_gpuHeadIn.Get(), m_gpuGradPred.Get(), m_head, m_gpuHead);
    dispatchBwdInput(m_gpuGradPred.Get(), m_gpuHead.weight.Get(), m_gpuGradHiddenA.Get(), m_head.inChannels, 3, 1);

    // 7. Backward Residual Blocks
    // Forward: blkOut = Conv2(SiLU(Conv1(blkIn))) + blkIn
    // Backward:
    //   1) gradW2, gradB2 from (blockMidAct, gradCur)
    //   2) gradMid = Conv2_BwdIn(gradCur, W2)           [Conv2 is linear -> no SiLU here]
    //   3) gradMidPre = gradMid * SiLU'(blockMidPre)    [SiLU belongs to Conv1 output!]
    //   4) gradW1, gradB1 from (blockIn, gradMidPre)
    //   5) gradConv1In = Conv1_BwdIn(gradMidPre, W1)
    //   6) gradAlt = gradCur + gradConv1In              [exact residual skip gradient]
    ID3D12Resource* gradCur = m_gpuGradHiddenA.Get();
    ID3D12Resource* gradAlt = m_gpuGradHiddenB.Get();
    for (size_t b = m_blocks.size(); b-- > 0;) {
        dispatchBwdWeights(m_gpuBlockMidAct[b].Get(), gradCur, m_blocks[b].conv2, m_gpuBlocks[b].second);
        dispatchBwdInput(gradCur, m_gpuBlocks[b].second.weight.Get(), m_gpuGradMid.Get(), m_arch.width, m_arch.width, 3);
        BindSiLUBackward(m_computeCmdList.Get(), m_gpuGradMid.Get(), m_gpuBlockMidPre[b].Get(), m_gpuGradMidPre.Get(), hiddenCount, cb);
        dispatchBwdWeights(m_gpuBlockIn[b].Get(), m_gpuGradMidPre.Get(), m_blocks[b].conv1, m_gpuBlocks[b].first);
        dispatchBwdInput(m_gpuGradMidPre.Get(), m_gpuBlocks[b].first.weight.Get(), m_gpuConvScratch.Get(), m_arch.width, m_arch.width, 3);
        BindAddResidual(m_computeCmdList.Get(), gradCur, m_gpuConvScratch.Get(), gradAlt, hiddenCount, cb);
        std::swap(gradCur, gradAlt);
    }

    // 8. Backward Stem (stem had SiLU activation)
    BindSiLUBackward(m_computeCmdList.Get(), gradCur, m_gpuStemPre.Get(), m_gpuGradMidPre.Get(), hiddenCount, cb);
    dispatchBwdWeights(m_gpuStemIn.Get(), m_gpuGradMidPre.Get(), m_stem, m_gpuStem);

    // 9. GPU-Resident Adam Weight Updates
    auto updateLayer = [&](const ConvLayerWeights& layer, const GPULayerBuffers& gpuL) {
        cb.gParamCount = static_cast<uint32_t>(layer.weight.size());
        BindAdamDispatch(m_computeCmdList.Get(), gpuL.gradW.Get(), gpuL.weight.Get(), gpuL.mW.Get(), gpuL.vW.Get(), cb);
        cb.gParamCount = static_cast<uint32_t>(layer.bias.size());
        BindAdamDispatch(m_computeCmdList.Get(), gpuL.gradB.Get(), gpuL.bias.Get(), gpuL.mB.Get(), gpuL.vB.Get(), cb);
    };

    updateLayer(m_stem, m_gpuStem);
    for (size_t b = 0; b < m_blocks.size(); ++b) {
        updateLayer(m_blocks[b].conv1, m_gpuBlocks[b].first);
        updateLayer(m_blocks[b].conv2, m_gpuBlocks[b].second);
    }
    updateLayer(m_head, m_gpuHead);

    if (computeLossReadback) {
        m_computeCmdList->CopyBufferRegion(m_readbackStaging.Get(), 0, m_gpuGradPred.Get(), 0, rgbBytes);
    }

    m_computeCmdList->Close();
    ID3D12CommandList* lists[] = { m_computeCmdList.Get() };
    m_cmdQueue->ExecuteCommandLists(1, lists);
    m_gpuWeightsDirtyOnDevice = true;

    hr = WaitComputeQueue();
    if (FAILED(hr)) return hr;

    if (computeLossReadback) {
        uint8_t* rb = nullptr;
        D3D12_RANGE range = { 0, rgbBytes };
        if (SUCCEEDED(m_readbackStaging->Map(0, &range, reinterpret_cast<void**>(&rb)))) {
            const float* gPtr = reinterpret_cast<const float*>(rb);
            const float scale = static_cast<float>(rgbCount) * 0.5f;
            double sumSq = 0.0;
            for (uint32_t i = 0; i < rgbCount; ++i) {
                const double diff = static_cast<double>(gPtr[i]) * scale;
                sumSq += diff * diff;
            }
            outMseLoss = static_cast<float>(sumSq / static_cast<double>(std::max(1u, rgbCount)));
            D3D12_RANGE empty = { 0, 0 };
            m_readbackStaging->Unmap(0, &empty);
        }
    }
    return S_OK;
}

void FMBrain::BuildSineOperatorInput(
    const float* xt,
    const float* tValues,
    uint32_t batch,
    uint32_t height,
    uint32_t width,
    std::vector<float>& outEncoded) const {
    const uint32_t cIn = m_arch.TotalInputChannels();
    const size_t hw = static_cast<size_t>(height) * width;
    outEncoded.resize(static_cast<size_t>(batch) * cIn * hw);

    for (uint32_t n = 0; n < batch; ++n) {
        const float* xBatch = xt + static_cast<size_t>(n) * 3 * hw;
        float* dstBatch = outEncoded.data() + static_cast<size_t>(n) * cIn * hw;
        std::copy_n(xBatch, 3 * hw, dstBatch);

        const float t = tValues[n];
        const float tSin1 = std::sin(kPi * t);
        const float tCos1 = std::cos(kPi * t);
        const float tSin2 = std::sin(2.0f * kPi * t);
        const float tCos2 = std::cos(2.0f * kPi * t);

        for (uint32_t y = 0; y < height; ++y) {
            const float ny = (height > 1) ? (2.0f * static_cast<float>(y) / static_cast<float>(height - 1) - 1.0f) : 0.0f;
            for (uint32_t x = 0; x < width; ++x) {
                const float nx = (width > 1) ? (2.0f * static_cast<float>(x) / static_cast<float>(width - 1) - 1.0f) : 0.0f;
                const size_t p = static_cast<size_t>(y) * width + x;

                dstBatch[3 * hw + p] = nx;
                dstBatch[4 * hw + p] = ny;

                uint32_t ch = 5;
                for (uint32_t k = 0; k < m_arch.harmonicBands; ++k) {
                    const float freq = static_cast<float>(1u << k) * kPi;
                    dstBatch[(ch + 0) * hw + p] = std::sin(freq * nx);
                    dstBatch[(ch + 1) * hw + p] = std::cos(freq * nx);
                    dstBatch[(ch + 2) * hw + p] = std::sin(freq * ny);
                    dstBatch[(ch + 3) * hw + p] = std::cos(freq * ny);
                    ch += 4;
                }

                dstBatch[(ch + 0) * hw + p] = tSin1;
                dstBatch[(ch + 1) * hw + p] = tCos1;
                dstBatch[(ch + 2) * hw + p] = tSin2;
                dstBatch[(ch + 3) * hw + p] = tCos2;
            }
        }
    }
}

void FMBrain::Forward(
    const float* xt,
    const float* tValues,
    uint32_t batch,
    uint32_t height,
    uint32_t width,
    ForwardCache& cache) const {
    cache.batch = batch;
    cache.height = height;
    cache.width = width;

    const uint32_t w = m_arch.width;
    const size_t hw = static_cast<size_t>(height) * width;
    const size_t hiddenSize = static_cast<size_t>(batch) * w * hw;

    BuildSineOperatorInput(xt, tValues, batch, height, width, cache.stemInput);

    cache.stemPreAct.resize(hiddenSize);
    cache.stemOut.resize(hiddenSize);
    Conv2dNCHW(cache.stemInput.data(), m_stem, batch, height, width, cache.stemPreAct.data());
    for (size_t i = 0; i < hiddenSize; ++i) {
        cache.stemOut[i] = SiLU(cache.stemPreAct[i]);
    }

    const size_t numBlocks = m_blocks.size();
    cache.blockIn.resize(numBlocks);
    cache.blockMidPre.resize(numBlocks);
    cache.blockMidAct.resize(numBlocks);
    cache.blockConv2Out.resize(numBlocks);

    const float* cur = cache.stemOut.data();
    std::vector<float> nextHidden(hiddenSize);

    for (size_t b = 0; b < numBlocks; ++b) {
        cache.blockIn[b].assign(cur, cur + hiddenSize);
        cache.blockMidPre[b].resize(hiddenSize);
        cache.blockMidAct[b].resize(hiddenSize);
        cache.blockConv2Out[b].resize(hiddenSize);

        Conv2dNCHW(cache.blockIn[b].data(), m_blocks[b].conv1, batch, height, width, cache.blockMidPre[b].data());
        for (size_t i = 0; i < hiddenSize; ++i) {
            cache.blockMidAct[b][i] = SiLU(cache.blockMidPre[b][i]);
        }

        Conv2dNCHW(cache.blockMidAct[b].data(), m_blocks[b].conv2, batch, height, width, cache.blockConv2Out[b].data());
        for (size_t i = 0; i < hiddenSize; ++i) {
            nextHidden[i] = cache.blockIn[b][i] + cache.blockConv2Out[b][i];
        }
        cur = nextHidden.data();
    }

    cache.headInput.assign(cur, cur + hiddenSize);
    cache.velocityPred.resize(static_cast<size_t>(batch) * 3 * hw);
    Conv2dNCHW(cache.headInput.data(), m_head, batch, height, width, cache.velocityPred.data());
}

std::vector<float> FMBrain::DreamEulerODE(
    const float* initialNoise,
    uint32_t height,
    uint32_t width,
    uint32_t steps) const {
    const uint32_t stepCount = std::max(1u, steps);
    const size_t sampleSize = static_cast<size_t>(3) * height * width;
    std::vector<float> x(initialNoise, initialNoise + sampleSize);
    const float dt = 1.0f / static_cast<float>(stepCount);

    ForwardCache cache;
    for (uint32_t s = 0; s < stepCount; ++s) {
        const float t = (static_cast<float>(s) + 0.5f) * dt;
        Forward(x.data(), &t, 1, height, width, cache);
        for (size_t i = 0; i < sampleSize; ++i) {
            x[i] += dt * cache.velocityPred[i];
        }
    }
    return x;
}

DialScanReport FMBrain::ScanDials(float threshold) {
    std::lock_guard<std::recursive_mutex> lock(m_brainMutex);
    ReadbackWeightsFromGPULocked();

    DialScanReport report;
    double sumAbs = 0.0;

    auto inspectVec = [&](const std::vector<float>& vec) {
        for (float w : vec) {
            const float mag = std::abs(w);
            report.totalDials++;
            if (mag > threshold) {
                report.busyDials++;
            } else {
                report.sleepingDials++;
            }
            if (mag > report.maxMagnitude) {
                report.maxMagnitude = mag;
            }
            sumAbs += mag;
        }
    };

    inspectVec(m_stem.weight);
    inspectVec(m_stem.bias);
    for (const auto& blk : m_blocks) {
        inspectVec(blk.conv1.weight);
        inspectVec(blk.conv1.bias);
        inspectVec(blk.conv2.weight);
        inspectVec(blk.conv2.bias);
    }
    inspectVec(m_head.weight);
    inspectVec(m_head.bias);

    if (report.totalDials > 0) {
        report.meanMagnitude = static_cast<float>(sumAbs / static_cast<double>(report.totalDials));
    }
    return report;
}

std::string FMBrain::ExportToJson() {
    std::lock_guard<std::recursive_mutex> lock(m_brainMutex);
    ReadbackWeightsFromGPULocked();

    std::ostringstream oss;
    oss << "{\"preset\":" << static_cast<uint32_t>(m_arch.preset)
        << ",\"width\":" << m_arch.width
        << ",\"blocks\":" << m_arch.numResBlocks
        << ",\"params\":" << ParameterCount()
        << ",\"weights\":[";

    bool first = true;
    auto emitVec = [&](const std::vector<float>& vec) {
        for (float v : vec) {
            if (!first) oss << ',';
            first = false;
            oss << v;
        }
    };
    emitVec(m_stem.weight);
    emitVec(m_stem.bias);
    for (const auto& blk : m_blocks) {
        emitVec(blk.conv1.weight);
        emitVec(blk.conv1.bias);
        emitVec(blk.conv2.weight);
        emitVec(blk.conv2.bias);
    }
    emitVec(m_head.weight);
    emitVec(m_head.bias);
    oss << "]}";
    return oss.str();
}

bool FMBrain::ImportFromJson(const std::string& json) {
    // 1. Parse and validate completely into temporary buffers BEFORE mutating existing brain!
    auto parseInt = [&](const char* key, uint32_t& out) -> bool {
        const size_t pos = json.find(key);
        if (pos == std::string::npos) return false;
        const size_t colon = json.find(':', pos);
        if (colon == std::string::npos) return false;
        try {
            out = static_cast<uint32_t>(std::stoul(json.substr(colon + 1)));
            return true;
        } catch (...) {
            return false;
        }
    };

    uint32_t presetRaw = 1;
    if (!parseInt("\"preset\"", presetRaw) || presetRaw > 2) {
        return false;
    }

    const size_t arrPos = json.find("\"weights\"");
    const size_t openBracket = (arrPos == std::string::npos) ? std::string::npos : json.find('[', arrPos);
    const size_t closeBracket = (openBracket == std::string::npos) ? std::string::npos : json.find(']', openBracket);
    if (openBracket == std::string::npos || closeBracket == std::string::npos) {
        return false;
    }

    const BrainPreset targetPreset = static_cast<BrainPreset>(presetRaw);
    const uint64_t expectedCount = ExpectedParameterCount(targetPreset);

    std::vector<float> parsed;
    parsed.reserve(static_cast<size_t>(expectedCount));
    const char* ptr = json.c_str() + openBracket + 1;
    const char* end = json.c_str() + closeBracket;
    while (ptr < end) {
        while (ptr < end && (*ptr == ' ' || *ptr == ',' || *ptr == '\n' || *ptr == '\r' || *ptr == '\t')) ++ptr;
        if (ptr >= end) break;
        char* next = nullptr;
        const float val = std::strtof(ptr, &next);
        if (next == ptr || !std::isfinite(val)) {
            return false;
        }
        parsed.push_back(val);
        ptr = next;
    }

    if (parsed.size() != expectedCount) {
        return false;
    }

    // 2. Validation passed: acquire lock, rebuild preset if needed, apply weights, and upload to GPU
    std::lock_guard<std::recursive_mutex> lock(m_brainMutex);
    InitializeFromScratchLocked(targetPreset, 0x1337BEEFULL);

    size_t offset = 0;
    auto loadVec = [&](std::vector<float>& vec) {
        std::copy_n(parsed.data() + offset, vec.size(), vec.data());
        offset += vec.size();
    };
    loadVec(m_stem.weight);
    loadVec(m_stem.bias);
    for (auto& blk : m_blocks) {
        loadVec(blk.conv1.weight);
        loadVec(blk.conv1.bias);
        loadVec(blk.conv2.weight);
        loadVec(blk.conv2.bias);
    }
    loadVec(m_head.weight);
    loadVec(m_head.bias);
    if (m_gpuReady) {
        UploadWeightsToGPULocked();
    }
    return true;
}

std::vector<uint8_t> FMBrain::ExportToBinary() {
    std::lock_guard<std::recursive_mutex> lock(m_brainMutex);
    ReadbackWeightsFromGPULocked();

    const uint32_t magic = 0x4D46544Fu; // "OTFM"
    const uint32_t presetVal = static_cast<uint32_t>(m_arch.preset);
    const uint64_t count = ParameterCount();

    std::vector<uint8_t> out(16 + count * sizeof(float));
    std::memcpy(out.data() + 0, &magic, 4);
    std::memcpy(out.data() + 4, &presetVal, 4);
    std::memcpy(out.data() + 8, &count, 8);

    size_t byteOffset = 16;
    auto appendVec = [&](const std::vector<float>& vec) {
        const size_t bytes = vec.size() * sizeof(float);
        std::memcpy(out.data() + byteOffset, vec.data(), bytes);
        byteOffset += bytes;
    };
    appendVec(m_stem.weight);
    appendVec(m_stem.bias);
    for (const auto& blk : m_blocks) {
        appendVec(blk.conv1.weight);
        appendVec(blk.conv1.bias);
        appendVec(blk.conv2.weight);
        appendVec(blk.conv2.bias);
    }
    appendVec(m_head.weight);
    appendVec(m_head.bias);
    return out;
}

bool FMBrain::ImportFromBinary(const std::vector<uint8_t>& bytes) {
    // 1. Validate header, expected parameter count, and payload length BEFORE mutating active brain!
    if (bytes.size() < 16) return false;
    uint32_t magic = 0;
    uint32_t presetVal = 0;
    uint64_t count = 0;
    std::memcpy(&magic, bytes.data() + 0, 4);
    std::memcpy(&presetVal, bytes.data() + 4, 4);
    std::memcpy(&count, bytes.data() + 8, 8);
    if (magic != 0x4D46544Fu || presetVal > 2) return false;

    const BrainPreset targetPreset = static_cast<BrainPreset>(presetVal);
    const uint64_t expectedCount = ExpectedParameterCount(targetPreset);
    if (count != expectedCount || bytes.size() != 16 + count * sizeof(float)) {
        return false;
    }

    std::vector<float> parsed(static_cast<size_t>(count));
    std::memcpy(parsed.data(), bytes.data() + 16, static_cast<size_t>(count) * sizeof(float));
    for (float v : parsed) {
        if (!std::isfinite(v)) return false;
    }

    // 2. Validation passed: commit atomically under mutex
    std::lock_guard<std::recursive_mutex> lock(m_brainMutex);
    InitializeFromScratchLocked(targetPreset, 0x1337BEEFULL);

    size_t offset = 0;
    auto readVec = [&](std::vector<float>& vec) {
        std::copy_n(parsed.data() + offset, vec.size(), vec.data());
        offset += vec.size();
    };
    readVec(m_stem.weight);
    readVec(m_stem.bias);
    for (auto& blk : m_blocks) {
        readVec(blk.conv1.weight);
        readVec(blk.conv1.bias);
        readVec(blk.conv2.weight);
        readVec(blk.conv2.bias);
    }
    readVec(m_head.weight);
    readVec(m_head.bias);
    if (m_gpuReady) {
        UploadWeightsToGPULocked();
    }
    return true;
}

} // namespace XboxOTFM
