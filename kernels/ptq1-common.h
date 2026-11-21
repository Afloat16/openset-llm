// ptq1-common.h: architecture-neutral parts of the PTQ1_0 CPU path -- trit codec, the
// panel re-layout and the Q8_0-identical activation quantizer. The host must define
// block_ptq1_0, ggml_half, GGML_CPU_FP16_TO_FP32 and GGML_CPU_FP32_TO_FP16 first.
// One source of truth for the layout: the AVX-512 kernel (ptq1-panel.inc) and the NEON
// kernel (ptq1-panel-neon.h) both consume what this file produces.
#pragma once
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static inline uint8_t ptq1f_trit(uint8_t b, int n) {          // reference decode, n-th trit in {0,1,2}
    static const uint8_t pow3[5] = {1, 3, 9, 27, 81};
    const uint8_t v = (uint8_t) (b * pow3[n]);
    return (uint8_t) (((uint16_t) v * 3) >> 8);
}

// original block -> 128 trit codes {0,1,2} in element order
static void ptq1f_unpack_block(const block_ptq1_0 * b, uint8_t * c) {
    int o = 0;
    for (int n = 0; n < 5; ++n) for (int m = 0;  m < 16; ++m) c[o++] = ptq1f_trit(b->qs[m], n);
    for (int n = 0; n < 5; ++n) for (int m = 16; m < 24; ++m) c[o++] = ptq1f_trit(b->qs[m], n);
    for (int n = 0; n < 4; ++n) for (int h = 0; h < 2;  ++h) c[o++] = ptq1f_trit(b->qh[h], n);
}

// five codes -> one byte whose successive decode steps reproduce them (TQ1_0 encoding)
static inline uint8_t ptq1f_pack5(const uint8_t * c5) {
    int q = 0;
    for (int n = 0; n < 5; ++n) q = q * 3 + c5[n];
    return (uint8_t) ((q * 256 + 242) / 243);
}

#define PTQ1P_ROWS 16
#define PTQ1P_KC   20            // k per chunk
#define PTQ1P_KPAD 160           // lcm(20, 32): chunks and Q8 blocks both end on this
#define PTQ1P_TB   8             // tokens per decode pass

typedef struct ptq1p_entry {
    const void * src;
    uint8_t    * packed;
    int64_t      rows, k, kpad, npanel, nck, ngp, panel_stride;
    int64_t      nsb, n36;       // GGML_PTQ1_FAST=5 (ptq1-vbmi.inc): 2 KB (608-k) and 1 KB (288-k) superblocks per panel; 0 for the chunk layout
    struct ptq1p_entry * next;
    struct ptq1p_entry * succ;   // K7: the entry of the next M = 1 call in graph order (learned, prefetch hint only)
    int          ndone;          // threads finished with the first-use repack (the last one drops the source pages)
} ptq1p_entry;

static void ptq1p_init_entry(ptq1p_entry * e, const void * src, int64_t rows, int64_t k) {
    e->src = src; e->rows = rows; e->k = k; e->nsb = 0; e->n36 = 0;
    e->kpad = (k + PTQ1P_KPAD - 1) / PTQ1P_KPAD * PTQ1P_KPAD;
    e->npanel = (rows + PTQ1P_ROWS - 1) / PTQ1P_ROWS;
    e->nck = e->kpad / PTQ1P_KC;
    e->ngp = (e->kpad + 127) / 128;
    e->panel_stride = (e->nck * 64 + e->ngp * PTQ1P_ROWS * 2 + 63) / 64 * 64;
}

static void ptq1p_repack_panels(ptq1p_entry * e, int64_t p0, int64_t p1) {
    const int64_t nb = e->k / 128;
    uint8_t * codes = malloc((size_t) (PTQ1P_ROWS * e->kpad));
    for (int64_t p = p0; p < p1; ++p) {
        memset(codes, 1, (size_t) (PTQ1P_ROWS * e->kpad));             // trit 1 == weight 0
        uint8_t * dst = e->packed + p * e->panel_stride;
        ggml_half * sc = (ggml_half *) (dst + e->nck * 64);
        memset(sc, 0, (size_t) (e->ngp * PTQ1P_ROWS * 2));
        for (int r = 0; r < PTQ1P_ROWS; ++r) {
            const int64_t row = p * PTQ1P_ROWS + r;
            if (row >= e->rows) continue;
            const block_ptq1_0 * src = (const block_ptq1_0 *) e->src + row * nb;
            for (int64_t b = 0; b < nb; ++b) {
                ptq1f_unpack_block(src + b, codes + r * e->kpad + b * 128);
                sc[b * PTQ1P_ROWS + r] = src[b].d;
            }
        }
        for (int64_t c = 0; c < e->nck; ++c) {
            for (int j = 0; j < 64; ++j) {
                uint8_t c5[5];
                for (int n = 0; n < 5; ++n) c5[n] = codes[(j / 4) * e->kpad + c * PTQ1P_KC + 4 * n + (j % 4)];
                dst[c * 64 + j] = ptq1f_pack5(c5);
            }
        }
    }
    free(codes);
}

// per token: int8 q[kpad] | f32 dy[kpad/32] | i32 asum[kpad/32]
static size_t ptq1p_col_bytes(int64_t k) {
    const int64_t kpad = (k + PTQ1P_KPAD - 1) / PTQ1P_KPAD * PTQ1P_KPAD;
    return (size_t) (kpad + (kpad / 32) * 8 + 64);
}

// Blocks [b0, b1) of one token's column; the caller with b0 == 0 also zeroes the padding past k. Lets all threads
// share the quantization of a single (decode) token; the whole-column version below is the b0 = 0, b1 = k/32 case.
// kpad is the layout's padded k (a multiple of 32): PTQ1P_KPAD-rounded for the chunk layout, ptq1v_kpad for ptq1-vbmi.
// neg_asum stores -sum(q) per block (ptq1-vbmi starts each block's int32 accumulator there instead of subtracting).
static void ptq1p_quantize_col_part_kp(const float * x, int64_t k, int64_t kpad, uint8_t * out, int64_t b0, int64_t b1, int neg_asum) {
    int8_t * q = (int8_t *) out;
    float * dy = (float *) (out + kpad);
    int32_t * asum = (int32_t *) (dy + kpad / 32);
    if (b0 == 0) {
        memset(q + k, 0, (size_t) (kpad - k));
        for (int64_t b = k / 32; b < kpad / 32; ++b) { dy[b] = 0.0f; asum[b] = 0; }
    }
    for (int64_t b = b0; b < b1; ++b) {
        float amax = 0.0f;
        for (int j = 0; j < 32; ++j) amax = fmaxf(amax, fabsf(x[b * 32 + j]));
        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f / d : 0.0f;
        int s = 0;
        for (int j = 0; j < 32; ++j) { const int v = (int) roundf(x[b * 32 + j] * id); q[b * 32 + j] = (int8_t) v; s += v; }
        dy[b] = GGML_CPU_FP16_TO_FP32(GGML_CPU_FP32_TO_FP16(d));
        asum[b] = neg_asum ? -s : s;
    }
}

static void ptq1p_quantize_col_part(const float * x, int64_t k, uint8_t * out, int64_t b0, int64_t b1) {
    const int64_t kpad = (k + PTQ1P_KPAD - 1) / PTQ1P_KPAD * PTQ1P_KPAD;
    ptq1p_quantize_col_part_kp(x, k, kpad, out, b0, b1, 0);
}

static void ptq1p_quantize_col(const float * x, int64_t k, uint8_t * out) {
    const int64_t kpad = (k + PTQ1P_KPAD - 1) / PTQ1P_KPAD * PTQ1P_KPAD;
    int8_t * q = (int8_t *) out;
    float * dy = (float *) (out + kpad);
    int32_t * asum = (int32_t *) (dy + kpad / 32);
    memset(out, 0, (size_t) (kpad + (kpad / 32) * 8));
    for (int64_t b = 0; b < k / 32; ++b) {
        float amax = 0.0f;
        for (int j = 0; j < 32; ++j) amax = fmaxf(amax, fabsf(x[b * 32 + j]));
        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f / d : 0.0f;
        int s = 0;
        for (int j = 0; j < 32; ++j) { const int v = (int) roundf(x[b * 32 + j] * id); q[b * 32 + j] = (int8_t) v; s += v; }
        dy[b] = GGML_CPU_FP16_TO_FP32(GGML_CPU_FP32_TO_FP16(d));
        asum[b] = s;
    }
}

