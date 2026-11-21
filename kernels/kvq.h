// kvq.h: low-bit KV cache rows: GGML_TYPE_KV4 / KV5 / KV6 (`-ctk kv4|kv5|kv6`, same for -ctv). `./serve --kv6 | --kv5` use two of them.
// One block = one (token, KV head) row of 256 values: an f32 gain, then 256 b-bit codes as a little-endian bit stream
// (value i in bits [i*b, (i+1)*b)): 132 / 164 / 196 B for b = 4 / 5 / 6 (f16 512 B, q8_K 292 B).
// code = sign << (b-1) | level index; value = d * LEVEL[index]: Lloyd-Max levels of a unit Gaussian, rounded to int8
// (relative MSE on a Gaussian row 0.0095 / 0.0025 / 0.00068). The KV cache Hadamard-rotates K and V rows for every
// quantized cache type before they are stored, which makes the coordinates near-Gaussian with one scale; d is the
// least-squares gain for the chosen codes.
// Included by ggml-cpu.c (the SET_ROWS quantizers) and by attn-fast.inc (the reader).
#pragma once
#include <stdint.h>
#include <string.h>
#include <math.h>

static const int8_t KVQ_LV4[8]  = { 6, 18, 30, 43, 58, 74, 95, 126 };
static const int8_t KVQ_LV5[16] = { 3, 8, 13, 18, 23, 28, 34, 40, 46, 53, 60, 68, 77, 88, 102, 124 };
static const int8_t KVQ_LV6[32] = { 1, 3, 6, 8, 10, 12, 15, 17, 19, 22, 24, 27, 29, 32, 34, 37, 40, 43, 46, 49, 52, 55, 59, 63, 67, 72, 77, 83,
                                    89, 97, 108, 125 };
static const float KVQ_UNIT[7] = { 0, 0, 0, 0, 46.0f, 37.81f, 33.68f };   // level units per sigma
static inline const int8_t * kvq_levels(int bits) { return bits == 4 ? KVQ_LV4 : bits == 5 ? KVQ_LV5 : KVQ_LV6; }
static inline size_t kvq_block_size(int bits) { return 4 + 32 * (size_t) bits; }

static void kvq_quantize(const float * x, void * vy, int64_t k, int bits) {
    const int nl = 1 << (bits - 1);
    const int8_t * lv = kvq_levels(bits);
    const size_t bs = kvq_block_size(bits);
    float thr[32];
    for (int t = 0; t + 1 < nl; ++t) thr[t] = 0.5f * (float) (lv[t] + lv[t + 1]);   // midpoints of adjacent levels
    uint8_t * y = (uint8_t *) vy;
    for (int64_t b = 0; b < k / 256; ++b, x += 256, y += bs) {
        float ss = 0.0f;
        for (int i = 0; i < 256; ++i) ss += x[i] * x[i];
        const float rms = sqrtf(ss / 256.0f), is = rms > 0.0f ? KVQ_UNIT[bits] / rms : 0.0f;
        float a[256]; int32_t idx[256];
        for (int i = 0; i < 256; ++i) { a[i] = fabsf(x[i]) * is; idx[i] = 0; }
        for (int t = 0; t + 1 < nl; ++t) for (int i = 0; i < 256; ++i) idx[i] += a[i] > thr[t];
        float xc = 0.0f, cc = 0.0f;
        for (int i = 0; i < 256; ++i) { const float l = (float) lv[idx[i]]; xc += fabsf(x[i]) * l; cc += l * l; }
        const float d = cc > 0.0f ? xc / cc : 0.0f;
        memcpy(y, &d, 4);
        uint8_t * qs = y + 4;
        memset(qs, 0, bs - 4);
        for (int i = 0; i < 256; ++i) {
            const uint32_t code = (uint32_t) idx[i] | (x[i] < 0.0f ? (uint32_t) nl : 0u);
            const int bit = i * bits, byte = bit >> 3, sh = bit & 7;
            const uint32_t v = code << sh;
            qs[byte] |= (uint8_t) v;
            if (v >> 8) qs[byte + 1] |= (uint8_t) (v >> 8);
        }
    }
}
static void quantize_row_kv4(const float * x, void * y, int64_t k) { kvq_quantize(x, y, k, 4); }
static void quantize_row_kv5(const float * x, void * y, int64_t k) { kvq_quantize(x, y, k, 5); }
static void quantize_row_kv6(const float * x, void * y, int64_t k) { kvq_quantize(x, y, k, 6); }
