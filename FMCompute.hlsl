// FMCompute.hlsl - Xbox Series S/X D3D12 Compute Shaders for GPU-Resident OT-FM Neural Training
// Compiled offline into CSO entry points and dispatched against GPU-resident weight/moment/activation buffers.

cbuffer TrainConstants : register(b0) {
    uint  gBatch;
    uint  gWidth;
    uint  gHeight;
    uint  gInC;
    uint  gOutC;
    uint  gKernelSize;   // 1 or 3
    uint  gApplySiLU;    // 1 = apply SiLU activation on forward, 0 = linear
    uint  gUnusedFlag;
    uint  gParamCount;   // number of scalar floats for elementwise/Adam/WeightGrad kernels
    uint  gSampleStride; // 3 * W * H
    float gTimeT;
    float gDt;
    float gLearningRate;
    float gBeta1;
    float gBeta2;
    float gEpsilon;
    float gBiasCorr1;    // 1 - beta1^step
    float gBiasCorr2;    // 1 - beta2^step
    float gInvOutCount;  // 1 / (N * 3 * W * H)
    float gGradClipScale;
};

// Generic SRVs / UAVs bound via root descriptor table
StructuredBuffer<float>   gInA       : register(t0);
StructuredBuffer<float>   gInB       : register(t1);
StructuredBuffer<float>   gInC_Buf   : register(t2);
RWStructuredBuffer<float> gOutA      : register(u0);
RWStructuredBuffer<float> gOutB      : register(u1);
RWStructuredBuffer<float> gOutBufC   : register(u2);

static const float PI = 3.14159265359f;

inline float EvalSiLU(float x) {
    float c = clamp(x, -20.0f, 20.0f);
    return c / (1.0f + exp(-c));
}

inline float EvalSiLUGrad(float x) {
    float c = clamp(x, -20.0f, 20.0f);
    float s = 1.0f / (1.0f + exp(-c));
    return s * (1.0f + c * (1.0f - s));
}

// 1. Pairwise squared L2 cost matrix kernel: C_{i,j} = ||x0[i] - x1[j]||_2^2 / stride
[numthreads(8, 8, 1)]
void CSPairwiseCost(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    uint j = tid.y;
    if (i >= gBatch || j >= gBatch) return;

    uint baseA = i * gSampleStride;
    uint baseB = j * gSampleStride;
    float sumSq = 0.0f;
    for (uint k = 0; k < gSampleStride; ++k) {
        float diff = gInA[baseA + k] - gInB[baseB + k];
        sumSq += diff * diff;
    }
    gOutA[i * gBatch + j] = sumSq / max(1.0f, (float)gSampleStride);
}

// 2. Spatial FM Sine-Operator Bank encoder: x_t RGB + [-1,1] grid + harmonic waves k=0..3 + time t
[numthreads(8, 8, 1)]
void CSEncodeSineOperators(uint3 tid : SV_DispatchThreadID) {
    uint x = tid.x;
    uint y = tid.y;
    uint n = tid.z;
    if (x >= gWidth || y >= gHeight || n >= gBatch) return;

    uint hw = gWidth * gHeight;
    uint p = y * gWidth + x;
    uint cIn = 25;
    uint dstBase = n * cIn * hw;
    uint srcBase = n * 3 * hw;

    gOutA[dstBase + 0 * hw + p] = gInA[srcBase + 0 * hw + p];
    gOutA[dstBase + 1 * hw + p] = gInA[srcBase + 1 * hw + p];
    gOutA[dstBase + 2 * hw + p] = gInA[srcBase + 2 * hw + p];

    float nx = (gWidth > 1) ? (2.0f * (float)x / (float)(gWidth - 1) - 1.0f) : 0.0f;
    float ny = (gHeight > 1) ? (2.0f * (float)y / (float)(gHeight - 1) - 1.0f) : 0.0f;
    gOutA[dstBase + 3 * hw + p] = nx;
    gOutA[dstBase + 4 * hw + p] = ny;

    [unroll]
    for (uint k = 0; k < 4; ++k) {
        float freq = (float)(1u << k) * PI;
        uint ch = 5 + k * 4;
        gOutA[dstBase + (ch + 0) * hw + p] = sin(freq * nx);
        gOutA[dstBase + (ch + 1) * hw + p] = cos(freq * nx);
        gOutA[dstBase + (ch + 2) * hw + p] = sin(freq * ny);
        gOutA[dstBase + (ch + 3) * hw + p] = cos(freq * ny);
    }

    float t = gInC_Buf[n];
    gOutA[dstBase + 21 * hw + p] = sin(PI * t);
    gOutA[dstBase + 22 * hw + p] = cos(PI * t);
    gOutA[dstBase + 23 * hw + p] = sin(2.0f * PI * t);
    gOutA[dstBase + 24 * hw + p] = cos(2.0f * PI * t);
}

// 3. GPU 2D Convolution Forward (1x1 or 3x3) + optional SiLU
// gInA = input [N, inC, H, W], gInB = weight [outC, inC, K, K], gInC_Buf = bias [outC]
// gOutA = output [N, outC, H, W], gOutB = preActivation [N, outC, H, W]
[numthreads(8, 8, 1)]
void CSConvForward(uint3 tid : SV_DispatchThreadID) {
    uint x = tid.x;
    uint y = tid.y;
    uint noc = tid.z;
    uint n = noc / gOutC;
    uint oc = noc % gOutC;
    if (x >= gWidth || y >= gHeight || n >= gBatch) return;

    uint hw = gWidth * gHeight;
    uint inBatchBase = n * gInC * hw;
    uint outIdx = (n * gOutC + oc) * hw + (y * gWidth + x);
    float acc = gInC_Buf[oc];

    if (gKernelSize == 1) {
        uint p = y * gWidth + x;
        for (uint ic = 0; ic < gInC; ++ic) {
            acc += gInA[inBatchBase + ic * hw + p] * gInB[oc * gInC + ic];
        }
    } else {
        int pad = 1;
        for (uint ic = 0; ic < gInC; ++ic) {
            uint inPlane = inBatchBase + ic * hw;
            uint wBase = (oc * gInC + ic) * 9;
            [unroll]
            for (int ky = 0; ky < 3; ++ky) {
                int iy = (int)y + ky - pad;
                if (iy < 0 || iy >= (int)gHeight) continue;
                [unroll]
                for (int kx = 0; kx < 3; ++kx) {
                    int ix = (int)x + kx - pad;
                    if (ix < 0 || ix >= (int)gWidth) continue;
                    acc += gInA[inPlane + (uint)iy * gWidth + (uint)ix] * gInB[wBase + (uint)ky * 3 + (uint)kx];
                }
            }
        }
    }

    gOutB[outIdx] = acc;
    gOutA[outIdx] = (gApplySiLU != 0) ? EvalSiLU(acc) : acc;
}

// 3b. Dedicated Elementwise Residual Addition:
// Forward: BlockOutput[i] = Conv2Out[i] + BlockIn[i]
// Backward: BlockInGrad[i] = gradCur[i] + dL/d(Conv1_Input)[i]
[numthreads(64, 1, 1)]
void CSAddResidual(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= gParamCount) return;
    gOutA[idx] = gInA[idx] + gInB[idx];
}

// 4. GPU MSE Loss Gradient: dL/dv_pred = 2 * (v_pred - v_target) / (N * 3 * H * W)
// gInA = v_pred, gInB = v_target, gOutA = gradPred
[numthreads(64, 1, 1)]
void CSMSELossGrad(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= gParamCount) return;
    float diff = gInA[idx] - gInB[idx];
    gOutA[idx] = 2.0f * gInvOutCount * diff;
}

// 5. GPU Conv2D Backward w.r.t. Input (pure transposed convolution)
// gInA = gradOutput [N, outC, H, W], gInB = weights [outC, inC, K, K]
// gOutA = gradInput [N, inC, H, W]
[numthreads(8, 8, 1)]
void CSConvBackwardInput(uint3 tid : SV_DispatchThreadID) {
    uint x = tid.x;
    uint y = tid.y;
    uint nic = tid.z;
    uint n = nic / gInC;
    uint ic = nic % gInC;
    if (x >= gWidth || y >= gHeight || n >= gBatch) return;

    uint hw = gWidth * gHeight;
    uint outBatchBase = n * gOutC * hw;
    uint inIdx = (n * gInC + ic) * hw + (y * gWidth + x);
    float acc = 0.0f;

    if (gKernelSize == 1) {
        uint p = y * gWidth + x;
        for (uint oc = 0; oc < gOutC; ++oc) {
            acc += gInA[outBatchBase + oc * hw + p] * gInB[oc * gInC + ic];
        }
    } else {
        int pad = 1;
        for (uint oc = 0; oc < gOutC; ++oc) {
            uint outPlane = outBatchBase + oc * hw;
            uint wBase = (oc * gInC + ic) * 9;
            [unroll]
            for (int ky = 0; ky < 3; ++ky) {
                int oy = (int)y - ky + pad;
                if (oy < 0 || oy >= (int)gHeight) continue;
                [unroll]
                for (int kx = 0; kx < 3; ++kx) {
                    int ox = (int)x - kx + pad;
                    if (ox < 0 || ox >= (int)gWidth) continue;
                    acc += gInA[outPlane + (uint)oy * gWidth + (uint)ox] * gInB[wBase + (uint)ky * 3 + (uint)kx];
                }
            }
        }
    }

    gOutA[inIdx] = acc;
}

// 5b. Elementwise SiLU Backward: gOutA[i] = gInA[i] * EvalSiLUGrad(gInB[i])
// gInA = gradPostAct, gInB = preActivation, gOutA = gradPreAct
[numthreads(64, 1, 1)]
void CSSiLUBackward(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= gParamCount) return;
    gOutA[idx] = gInA[idx] * EvalSiLUGrad(gInB[idx]);
}

// 6. Parallel-Reduced GPU Conv2D Backward w.r.t. Weights and Biases
// Dispatched as (totalWeights, 1, 1) groups of 256 threads each so even 128x128 completes
// in ~128 iterations per thread with zero risk of WDDM TDR watchdog timeout.
// gInA = input [N, inC, H, W], gInB = gradOutput [N, outC, H, W]
// gOutB = gradWeights [outC * inC * K * K], gOutC = gradBias [outC]
groupshared float sPartialW[256];
groupshared float sPartialB[256];

[numthreads(256, 1, 1)]
void CSConvBackwardWeights(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint wIdx = gid.x;
    uint t = gtid.x;
    uint kk = gKernelSize * gKernelSize;
    uint totalWeights = gOutC * gInC * kk;
    if (wIdx >= totalWeights) return;

    uint oc = wIdx / (gInC * kk);
    uint rem = wIdx % (gInC * kk);
    uint ic = rem / kk;
    uint kOff = rem % kk;
    int ky = (int)(kOff / gKernelSize);
    int kx = (int)(kOff % gKernelSize);
    int pad = (int)(gKernelSize / 2);
    uint hw = gWidth * gHeight;
    uint totalElems = gBatch * hw;

    float wGrad = 0.0f;
    float bGrad = 0.0f;
    bool computeBias = (ic == 0 && kOff == 0);

    for (uint p = t; p < totalElems; p += 256) {
        uint n = p / hw;
        uint remP = p % hw;
        uint y = remP / gWidth;
        uint x = remP % gWidth;

        float go = gInB[(n * gOutC + oc) * hw + remP];
        if (computeBias) {
            bGrad += go;
        }

        int iy = (int)y + ky - pad;
        int ix = (int)x + kx - pad;
        if (iy >= 0 && iy < (int)gHeight && ix >= 0 && ix < (int)gWidth) {
            wGrad += go * gInA[(n * gInC + ic) * hw + (uint)iy * gWidth + (uint)ix];
        }
    }

    sPartialW[t] = wGrad;
    sPartialB[t] = bGrad;
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (t < stride) {
            sPartialW[t] += sPartialW[t + stride];
            if (computeBias) {
                sPartialB[t] += sPartialB[t + stride];
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (t == 0) {
        gOutB[wIdx] = sPartialW[0];
        if (computeBias) {
            gOutBufC[oc] = sPartialB[0];
        }
    }
}

// 7. GPU-Resident Adam Optimizer Update
// gInA = grad, gOutA = param (in-place), gOutB = m (in-place), gOutC = v (in-place)
[numthreads(64, 1, 1)]
void CSAdamUpdate(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= gParamCount) return;

    float g = gInA[idx] * gGradClipScale;
    float m = gBeta1 * gOutB[idx] + (1.0f - gBeta1) * g;
    float v = gBeta2 * gOutBufC[idx] + (1.0f - gBeta2) * g * g;
    gOutB[idx] = m;
    gOutBufC[idx] = v;

    float mHat = m / max(1e-8f, gBiasCorr1);
    float vHat = v / max(1e-8f, gBiasCorr2);
    gOutA[idx] -= gLearningRate * mHat / (sqrt(vHat) + gEpsilon);
}

// 8. GPU Euler ODE Integration Step: x_{t+dt} = x_t + dt * v_pred
[numthreads(64, 1, 1)]
void CSEulerStep(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= gSampleStride) return;
    gOutA[idx] = gOutA[idx] + gDt * gInB[idx];
}
