// attn-fast.inc: fused CPU attention for GGML_OP_FLASH_ATTN_EXT (GGML_ATTN, default 1; 0 = the fork's paths).
// #included into ggml-cpu/ops.cpp by patches/0008 (copied there by patches/apply.sh).
//
// One kernel for every query count M (1 decode, 4 MTP verify, up to the ubatch in prefill), GQA-grouped: the G query
// heads of a KV head and all M query positions form R = M*G rows that share every K/V row read. Streaming (online)
// softmax, nothing materialised. KV types: F16 (exact mode: f32 math on converted tiles), Q8_K (int8 mode, opt-in via
// `-ctk q8_K -ctv q8_K`: one f32 scale per (token, head) row of 256; Q·K as u8 x s8 VNNI dot products in int32, P·V in f32)
// and KV4 / KV5 / KV6 (kvq.h: 4 / 5 / 6-bit rows, `-ctk kv6 -ctv kv6`; unpacked to int8 tiles, then the int8 paths).
//
// Canonical order. Every query row is computed by the same sequence of float operations whatever M is, however the
// rows are blocked and however the work is split across threads: keys are cut into absolute chunks of ATTNF_CHUNK
// (chunk c = [c*CH, (c+1)*CH)), each chunk into absolute tiles of ATTNF_T keys; a chunk's online softmax runs tile by
// tile from (-inf, 0, 0); chunk partials (max, sum, unnormalised V sum) are merged in ascending chunk order by
// attnf_merge(). Splitting decode work over (KV head x chunk) units and running prefill rows chunk after chunk in one
// thread therefore give bit-identical rows, and a verify row equals the plain-decode row at the same position.
//
// Interface for other kernels (e.g. retrieval attention over selected KV chunks): attnf_ws_get / attnf_q_prepare /
// attnf_range / attnf_merge below. attnf_range returns partial softmax stats over a key range, so ranges computed
// anywhere can be merged in any agreed order.
//
// Requires AVX-512 F/BW/VNNI + F16C for the fast path. Without them F16 falls back to the fork and Q8_K runs a scalar
// reference path (correct, slow).

#include <stdlib.h>
#include <string.h>
#include <math.h>

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__) && defined(__F16C__)
#define ATTNF_SIMD 1
#include <immintrin.h>
#else
#define ATTNF_SIMD 0
#endif

#define ATTNF_T     32     // keys per tile
#define ATTNF_CHUNK 1024   // keys per canonical chunk (multiple of ATTNF_T)
#define ATTNF_RMAX  96     // max rows (positions x heads of a group) per work unit
#define ATTNF_RSPLIT 48    // M*G up to this: decode-style split over KV chunks (else rows mode)
#define ATTNF_DMAX  512    // max head size (DK, DV), multiple of 64

typedef struct { float d; int8_t qs[256]; int16_t bsums[16]; } attnf_q8k;   // = block_q8_K (checked at run time)
#include "kvq.h"   // GGML_TYPE_KV4 / KV5 / KV6: b-bit rows, unpacked to int8 tiles here, then the int8 paths
static inline int attnf_kvq_bits(enum ggml_type t) { return t == GGML_TYPE_KV4 ? 4 : t == GGML_TYPE_KV5 ? 5 : t == GGML_TYPE_KV6 ? 6 : 0; }
static inline bool attnf_is_i8(enum ggml_type t) { return t == GGML_TYPE_Q8_K || attnf_kvq_bits(t) != 0; }
static inline size_t attnf_blk(enum ggml_type t) { const int b = attnf_kvq_bits(t); return b ? kvq_block_size(b) : sizeof(attnf_q8k); }

typedef struct {
    const char * k; size_t nbk;   // key row t of this KV head: k + t*nbk
    const char * v; size_t nbv;
    enum ggml_type type;          // GGML_TYPE_F16, GGML_TYPE_Q8_K or GGML_TYPE_KV4/5/6, same for K and V
    int dk, dv;
} attnf_kv;

typedef struct {
    float   * Q;      // [RMAX][DMAX]   q * scale (f32 rows)
    uint8_t * Qu;     // [RMAX][DMAX]   int8 mode: round(q*scale/dQ) + 128
    uint8_t * Qu2;    // [RMAX][DMAX]   int8 mode, GGML_ATTN_Q16: round(254 * residual) + 128 (q = dQ*(q1 + q2/254))
    float   * dQ;     // [RMAX]
    float   * Kt;     // [T][DMAX]      f16 K tile as f32
    float   * Vt;     // [T][DMAX]      V tile as f32 (int8 mode: the int8 values)
    float   * S;      // [RMAX][T]      scores, then probabilities
    float   * Mt;     // [RMAX][T]      mask tile per position
    float   * O;      // [RMAX][DMAX]   chunk accumulator
    float   * OF;     // [RMAX][DMAX]   final accumulator (rows mode)
    float   m[ATTNF_RMAX], l[ATTNF_RMAX], MF[ATTNF_RMAX], SF[ATTNF_RMAX], al[ATTNF_RMAX];
    float   kd[ATTNF_T], vd[ATTNF_T];
    int32_t kc[ATTNF_T];
    const int8_t * kq[ATTNF_T];
    int8_t * K8;      // [T][DMAX]      kv4/5/6: K tile unpacked to int8 levels
} attnf_ws;

static attnf_ws * attnf_ws_get(void) {
    static thread_local attnf_ws * ws = NULL;
    if (ws) return ws;
    const size_t RD = (size_t) ATTNF_RMAX * ATTNF_DMAX, TD = (size_t) ATTNF_T * ATTNF_DMAX, RT = (size_t) ATTNF_RMAX * ATTNF_T;
    const size_t bytes = sizeof(float) * (RD + ATTNF_RMAX + TD + TD + RT + RT + RD + RD) + 2 * RD + TD + 64 * 16;
    char * p = (char *) aligned_alloc(64, (bytes + 63) / 64 * 64);
    if (!p) return NULL;
    attnf_ws * w = (attnf_ws *) aligned_alloc(64, (sizeof(attnf_ws) + 63) / 64 * 64);
    if (!w) { free(p); return NULL; }
    memset(w, 0, sizeof(*w));
#define ATTNF_TAKE(dst, T_, n) do { dst = (T_ *) p; p += ((n) * sizeof(T_) + 63) / 64 * 64; } while (0)
    ATTNF_TAKE(w->Q, float, RD);  ATTNF_TAKE(w->Qu, uint8_t, RD); ATTNF_TAKE(w->Qu2, uint8_t, RD); ATTNF_TAKE(w->dQ, float, ATTNF_RMAX);
    ATTNF_TAKE(w->Kt, float, TD); ATTNF_TAKE(w->Vt, float, TD);   ATTNF_TAKE(w->S, float, RT);
    ATTNF_TAKE(w->Mt, float, RT); ATTNF_TAKE(w->O, float, RD);    ATTNF_TAKE(w->OF, float, RD);
    ATTNF_TAKE(w->K8, int8_t, TD);
#undef ATTNF_TAKE
    ws = w;
    return ws;
}

static const int8_t attnf_zero_row[ATTNF_DMAX] = {0};

static int attnf_q16(void) {   // GGML_ATTN_Q16 (int8 KV): 1 = query as two int8 planes (~15-bit), 0 = one plane
    static int v = -1;
    if (v < 0) { const char * e = getenv("GGML_ATTN_Q16"); v = e ? atoi(e) : 1; }
    return v;
}

static int attnf_pf(void) {   // GGML_ATTN_PF=0: no software prefetch of the next K/V tile
    static int pf = -1;
    if (pf < 0) { const char * e = getenv("GGML_ATTN_PF"); pf = e ? atoi(e) : 1; }
    return pf;
}

// q[r] -> ws rows (f32 q*scale; int8 mode also the u8 quantization). R <= ATTNF_RMAX.
static void attnf_q_prepare(attnf_ws * ws, const attnf_kv * kv, int R, const float * const * q, float scale) {
    const int DK = kv->dk;
    static int ulp = -1;   // GGML_ATTN_ULP=1 (experiments only): q*scale rounded 1 ulp up on even d, a float-noise yardstick
    if (ulp < 0) { const char * e = getenv("GGML_ATTN_ULP"); ulp = e ? atoi(e) : 0; }
    for (int r = 0; r < R; ++r) {
        float * Qr = ws->Q + (size_t) r * ATTNF_DMAX;
        for (int d = 0; d < DK; ++d) Qr[d] = q[r][d] * scale;
        if (ulp) for (int d = 0; d < DK; d += 2) Qr[d] = nextafterf(Qr[d], INFINITY);
        if (attnf_is_i8(kv->type)) {
            float amax = 0.0f;
            for (int d = 0; d < DK; ++d) amax = fmaxf(amax, fabsf(Qr[d]));
            const float dq = amax / 127.0f, iq = amax > 0.0f ? 127.0f / amax : 0.0f;
            ws->dQ[r] = dq;
            uint8_t * u = ws->Qu + (size_t) r * ATTNF_DMAX, * u2 = ws->Qu2 + (size_t) r * ATTNF_DMAX;
            for (int d = 0; d < DK; ++d) {
                const float x = Qr[d] * iq;
                int v = (int) nearbyintf(x);
                v = v < -127 ? -127 : v > 127 ? 127 : v;
                u[d] = (uint8_t) (v + 128);
                int v2 = (int) nearbyintf((x - (float) v) * 254.0f);
                v2 = v2 < -127 ? -127 : v2 > 127 ? 127 : v2;
                u2[d] = (uint8_t) (v2 + 128);
            }
        }
    }
}

// merge partial (m2, l2, O2) into (M, S, O). The one merge used everywhere (canonical order: ascending chunks).
static inline void attnf_merge(int dv, float * M, float * S, float * O, float m2, float l2, const float * O2) {
    if (l2 == 0.0f) return;
    if (*S == 0.0f) { *M = m2; *S = l2; memcpy(O, O2, dv * sizeof(float)); return; }
    const float Mn = fmaxf(*M, m2), a = expf(*M - Mn), b = expf(m2 - Mn);
#if ATTNF_SIMD
    const __m512 va = _mm512_set1_ps(a), vb = _mm512_set1_ps(b);
    for (int d = 0; d < dv; d += 16)
        _mm512_storeu_ps(O + d, _mm512_fmadd_ps(_mm512_loadu_ps(O2 + d), vb, _mm512_mul_ps(_mm512_loadu_ps(O + d), va)));
#else
    for (int d = 0; d < dv; ++d) O[d] = O2[d] * b + O[d] * a;
#endif
    *S = *S * a + l2 * b;
    *M = Mn;
}

#if ATTNF_SIMD
// lane n of the result = sum of the 16 lanes of v[n]; the same addition tree for every n
static inline __m512 attnf_hsum16(const __m512 * v) {
    __m512 a[8], b[4];
    for (int i = 0; i < 8; ++i) a[i] = _mm512_add_ps(_mm512_unpacklo_ps(v[2*i], v[2*i+1]), _mm512_unpackhi_ps(v[2*i], v[2*i+1]));
    for (int j = 0; j < 4; ++j) {
        const __m512d x = _mm512_castps_pd(a[2*j]), y = _mm512_castps_pd(a[2*j+1]);
        b[j] = _mm512_add_ps(_mm512_castpd_ps(_mm512_unpacklo_pd(x, y)), _mm512_castpd_ps(_mm512_unpackhi_pd(x, y)));
    }
    const __m512 c0 = _mm512_add_ps(_mm512_shuffle_f32x4(b[0], b[1], 0x88), _mm512_shuffle_f32x4(b[0], b[1], 0xDD));
    const __m512 c1 = _mm512_add_ps(_mm512_shuffle_f32x4(b[2], b[3], 0x88), _mm512_shuffle_f32x4(b[2], b[3], 0xDD));
    return _mm512_add_ps(_mm512_shuffle_f32x4(c0, c1, 0x88), _mm512_shuffle_f32x4(c0, c1, 0xDD));
}
static inline __m512i attnf_hsum16i(const __m512i * v) {
    __m512i a[8], b[4];
    for (int i = 0; i < 8; ++i) a[i] = _mm512_add_epi32(_mm512_unpacklo_epi32(v[2*i], v[2*i+1]), _mm512_unpackhi_epi32(v[2*i], v[2*i+1]));
    for (int j = 0; j < 4; ++j) b[j] = _mm512_add_epi32(_mm512_unpacklo_epi64(a[2*j], a[2*j+1]), _mm512_unpackhi_epi64(a[2*j], a[2*j+1]));
    const __m512i c0 = _mm512_add_epi32(_mm512_shuffle_i32x4(b[0], b[1], 0x88), _mm512_shuffle_i32x4(b[0], b[1], 0xDD));
    const __m512i c1 = _mm512_add_epi32(_mm512_shuffle_i32x4(b[2], b[3], 0x88), _mm512_shuffle_i32x4(b[2], b[3], 0xDD));
    return _mm512_add_epi32(_mm512_shuffle_i32x4(c0, c1, 0x88), _mm512_shuffle_i32x4(c0, c1, 0xDD));
}

// S[r][t0+t] += Q[r] . K[t0+t] for NR rows starting at r0, 4 keys starting at t0 (f32 tile)
template <int NR>
static inline void attnf_qk_f32(const attnf_ws * ws, int DK, int r0, int t0) {
    __m512 acc[16];
    for (int i = 0; i < 16; ++i) acc[i] = _mm512_setzero_ps();
    const float * q = ws->Q + (size_t) r0 * ATTNF_DMAX;
    const float * k = ws->Kt + (size_t) t0 * ATTNF_DMAX;
    for (int d = 0; d < DK; d += 16) {
        const __m512 k0 = _mm512_load_ps(k + d), k1 = _mm512_load_ps(k + ATTNF_DMAX + d),
                     k2 = _mm512_load_ps(k + 2*ATTNF_DMAX + d), k3 = _mm512_load_ps(k + 3*ATTNF_DMAX + d);
        for (int r = 0; r < NR; ++r) {
            const __m512 qv = _mm512_load_ps(q + (size_t) r * ATTNF_DMAX + d);
            acc[4*r+0] = _mm512_fmadd_ps(qv, k0, acc[4*r+0]);
            acc[4*r+1] = _mm512_fmadd_ps(qv, k1, acc[4*r+1]);
            acc[4*r+2] = _mm512_fmadd_ps(qv, k2, acc[4*r+2]);
            acc[4*r+3] = _mm512_fmadd_ps(qv, k3, acc[4*r+3]);
        }
    }
    alignas(64) float s[16];
    _mm512_store_ps(s, attnf_hsum16(acc));
    for (int r = 0; r < NR; ++r) {
        float * Sr = ws->S + (size_t) (r0 + r) * ATTNF_T + t0;
        for (int t = 0; t < 4; ++t) Sr[t] = s[4*r+t] + Sr[t];
    }
}

// int8: S[r][t] += dQ[r]*kd[t] * (sum_d qu*k - 128*sum_d k)
template <int NR>
static inline void attnf_qk_i8_pass(const attnf_ws * ws, const uint8_t * Qbase, int DK, int r0, int t0, int32_t * s) {
    __m512i acc[16];
    for (int i = 0; i < 16; ++i) acc[i] = _mm512_setzero_si512();
    const uint8_t * q = Qbase + (size_t) r0 * ATTNF_DMAX;
    const int8_t * k0p = ws->kq[t0], * k1p = ws->kq[t0+1], * k2p = ws->kq[t0+2], * k3p = ws->kq[t0+3];
    for (int d = 0; d < DK; d += 64) {
        const __m512i k0 = _mm512_loadu_si512(k0p + d), k1 = _mm512_loadu_si512(k1p + d),
                      k2 = _mm512_loadu_si512(k2p + d), k3 = _mm512_loadu_si512(k3p + d);
        for (int r = 0; r < NR; ++r) {
            const __m512i qv = _mm512_loadu_si512(q + (size_t) r * ATTNF_DMAX + d);
            acc[4*r+0] = _mm512_dpbusd_epi32(acc[4*r+0], qv, k0);
            acc[4*r+1] = _mm512_dpbusd_epi32(acc[4*r+1], qv, k1);
            acc[4*r+2] = _mm512_dpbusd_epi32(acc[4*r+2], qv, k2);
            acc[4*r+3] = _mm512_dpbusd_epi32(acc[4*r+3], qv, k3);
        }
    }
    _mm512_store_si512((__m512i *) s, attnf_hsum16i(acc));
}
template <int NR>
static inline void attnf_qk_i8(const attnf_ws * ws, int DK, int r0, int t0) {
    alignas(64) int32_t s[16], s2[16];
    attnf_qk_i8_pass<NR>(ws, ws->Qu, DK, r0, t0, s);
    const bool q16 = attnf_q16() != 0;
    if (q16) attnf_qk_i8_pass<NR>(ws, ws->Qu2, DK, r0, t0, s2);
    for (int r = 0; r < NR; ++r) {
        float * Sr = ws->S + (size_t) (r0 + r) * ATTNF_T + t0;
        const float dq = ws->dQ[r0 + r];
        for (int t = 0; t < 4; ++t) {
            float x = (float) (s[4*r+t] - ws->kc[t0+t]);
            if (q16) x += (float) (s2[4*r+t] - ws->kc[t0+t]) * (1.0f / 254.0f);
            Sr[t] = x * (dq * ws->kd[t0+t]) + Sr[t];
        }
    }
}

// O[r] = O[r]*al[r] + sum_{t<nt} S[r][t] * V[t] for NR rows from r0, 64 columns from d0
template <int NR>
static inline void attnf_pv(attnf_ws * ws, int r0, int d0, int nt) {
    __m512 acc[NR][4];
    for (int r = 0; r < NR; ++r) {
        float * o = ws->O + (size_t) (r0 + r) * ATTNF_DMAX + d0;
        const float a = ws->al[r0 + r];
        for (int j = 0; j < 4; ++j) {
            acc[r][j] = _mm512_load_ps(o + 16*j);
            if (a != 1.0f) acc[r][j] = _mm512_mul_ps(acc[r][j], _mm512_set1_ps(a));
        }
    }
    const float * P = ws->S + (size_t) r0 * ATTNF_T;
    for (int t = 0; t < nt; ++t) {
        const float * vt = ws->Vt + (size_t) t * ATTNF_DMAX + d0;
        const __m512 v0 = _mm512_load_ps(vt), v1 = _mm512_load_ps(vt + 16), v2 = _mm512_load_ps(vt + 32), v3 = _mm512_load_ps(vt + 48);
        for (int r = 0; r < NR; ++r) {
            const __m512 p = _mm512_set1_ps(P[(size_t) r * ATTNF_T + t]);
            acc[r][0] = _mm512_fmadd_ps(p, v0, acc[r][0]);
            acc[r][1] = _mm512_fmadd_ps(p, v1, acc[r][1]);
            acc[r][2] = _mm512_fmadd_ps(p, v2, acc[r][2]);
            acc[r][3] = _mm512_fmadd_ps(p, v3, acc[r][3]);
        }
    }
    for (int r = 0; r < NR; ++r) {
        float * o = ws->O + (size_t) (r0 + r) * ATTNF_DMAX + d0;
        for (int j = 0; j < 4; ++j) _mm512_store_ps(o + 16*j, acc[r][j]);
    }
}
#endif // ATTNF_SIMD

// kv4/5/6 row (256 b-bit codes) -> 256 int8 levels; returns their sum
#if ATTNF_SIMD && defined(__AVX512VBMI__)
#define ATTNF_KVQ_SIMD 1
struct attnf_kvq_tab {
    __m512i perm, shift, mask, lut;
    attnf_kvq_tab(int bits) {
        alignas(64) uint8_t p[64], sh[64], l[64];
        const int nl = 1 << (bits - 1);
        const int8_t * lv = kvq_levels(bits);
        for (int q = 0; q < 8; ++q) for (int k2 = 0; k2 < 8; ++k2) {
            p[8 * q + k2]  = (uint8_t) (bits * q + (k2 < bits ? k2 : 0));   // the `bits` bytes that hold 8 codes -> one qword
            sh[8 * q + k2] = (uint8_t) (k2 * bits);
        }
        for (int c = 0; c < 64; ++c) l[c] = (uint8_t) (c < nl ? lv[c] : c < 2 * nl ? -lv[c - nl] : 0);
        perm = _mm512_load_si512(p); shift = _mm512_load_si512(sh); lut = _mm512_load_si512(l);
        mask = _mm512_set1_epi8((char) ((1 << bits) - 1));
    }
};
static inline const attnf_kvq_tab * attnf_kvq_tabs(int bits) {
    static const attnf_kvq_tab t4(4), t5(5), t6(6);
    return bits == 4 ? &t4 : bits == 5 ? &t5 : &t6;
}
static inline int32_t attnf_kvq_unpack(const uint8_t * qs, int bits, int8_t * out) {
    const attnf_kvq_tab * tb = attnf_kvq_tabs(bits);
    const int nb = 8 * bits;                               // bytes per 64 codes
    const __mmask64 lm = (((__mmask64) 1) << nb) - 1;      // masked load: never reads past the row
    const __m512i one = _mm512_set1_epi8(1);
    __m512i sum = _mm512_setzero_si512();
    for (int j = 0; j < 4; ++j) {
        const __m512i v = _mm512_permutexvar_epi8(tb->perm, _mm512_maskz_loadu_epi8(lm, qs + nb * j));
        const __m512i idx = _mm512_and_si512(_mm512_multishift_epi64_epi8(tb->shift, v), tb->mask);
        const __m512i lv = _mm512_permutexvar_epi8(idx, tb->lut);
        _mm512_storeu_si512(out + 64 * j, lv);
        sum = _mm512_dpbusd_epi32(sum, one, lv);
    }
    return _mm512_reduce_add_epi32(sum);
}
#else
#define ATTNF_KVQ_SIMD 0
static inline int32_t attnf_kvq_unpack(const uint8_t * qs, int bits, int8_t * out) {
    const int nl = 1 << (bits - 1);
    const int8_t * lv = kvq_levels(bits);
    int32_t s = 0;
    for (int i = 0; i < 256; ++i) {
        const int bit = i * bits, byte = bit >> 3, sh = bit & 7;
        const uint32_t w = (uint32_t) qs[byte] | (byte + 1 < 32 * bits ? (uint32_t) qs[byte + 1] << 8 : 0u);
        const int c = (int) ((w >> sh) & ((1u << bits) - 1));
        out[i] = (int8_t) (c < nl ? lv[c] : -lv[c - nl]);
        s += out[i];
    }
    return s;
}
#endif

// Online softmax of the R prepared rows over keys [k0, k1): returns m[r], l[r] and the unnormalised O[r] (row stride
// ldo floats). Tiles start at k0 and every ATTNF_T keys after it (callers keep k0 on the canonical grid).
// pos[r] indexes mrow[] (several rows can share one mask row); mrow[i] = f16 mask indexed by absolute key, or NULL.
static void attnf_range(attnf_ws * ws, const attnf_kv * kv, int R, const int * pos, int npos,
                        const ggml_fp16_t * const * mrow, int64_t k0, int64_t k1, float * m, float * l, float * O, size_t ldo) {
    const int DK = kv->dk, DV = kv->dv, T = ATTNF_T;
    const bool i8 = attnf_is_i8(kv->type);
    const int kqb = attnf_kvq_bits(kv->type);   // 0, or the bits of a kvq.h row
    for (int r = 0; r < R; ++r) { ws->m[r] = -INFINITY; ws->l[r] = 0.0f; memset(ws->O + (size_t) r * ATTNF_DMAX, 0, DV * sizeof(float)); }
    for (int64_t t0 = k0; t0 < k1; t0 += T) {
        const int nt = (int) (k1 - t0 < T ? k1 - t0 : T);
        const int nt4 = (nt + 3) & ~3;
        // mask tile per position; skip the tile if every position is fully masked
        bool any = false;
        for (int i = 0; i < npos; ++i) {
            float * mt = ws->Mt + (size_t) i * T;
            if (mrow[i]) {
                for (int t = 0; t < nt; ++t) { mt[t] = GGML_CPU_FP16_TO_FP32(mrow[i][t0 + t]); any |= mt[t] != -INFINITY; }
            } else {
                for (int t = 0; t < nt; ++t) mt[t] = 0.0f;
                any = true;
            }
            for (int t = nt; t < T; ++t) mt[t] = -INFINITY;
        }
        if (!any) continue;
#if ATTNF_SIMD
        if (attnf_pf() && t0 + T < k1) {   // next tile's K and V rows into L2 (decode streams them from DRAM)
            const int64_t n2 = k1 - (t0 + T) < T ? k1 - (t0 + T) : T;
            const size_t kb = i8 ? attnf_blk(kv->type) * (DK / 256) : (size_t) DK * 2, vb = i8 ? attnf_blk(kv->type) * (DV / 256) : (size_t) DV * 2;
            for (int64_t t = 0; t < n2; ++t) {
                const char * kr = kv->k + (t0 + T + t) * kv->nbk, * vr = kv->v + (t0 + T + t) * kv->nbv;
                for (size_t o = 0; o < kb; o += 64) _mm_prefetch(kr + o, _MM_HINT_T1);
                for (size_t o = 0; o < vb; o += 64) _mm_prefetch(vr + o, _MM_HINT_T1);
            }
        }
#endif
        for (int r = 0; r < R; ++r) memcpy(ws->S + (size_t) r * T, ws->Mt + (size_t) pos[r] * T, T * sizeof(float));
        // K tile
        if (i8) {
            for (int t = 0; t < nt4; ++t) {
                if (t < nt && kqb) {
                    const uint8_t * b = (const uint8_t *) (kv->k + (t0 + t) * kv->nbk);
                    int8_t * k8 = ws->K8 + (size_t) t * ATTNF_DMAX;
                    ws->kc[t] = 128 * attnf_kvq_unpack(b + 4, kqb, k8);
                    ws->kq[t] = k8; memcpy(&ws->kd[t], b, 4);
                } else if (t < nt) {
                    const attnf_q8k * b = (const attnf_q8k *) (kv->k + (t0 + t) * kv->nbk);
                    int32_t sk = 0;
                    for (int j = 0; j < DK / 16; ++j) sk += b[j / 16].bsums[j % 16];   // DK/256 blocks of 16 sums
                    ws->kq[t] = b->qs; ws->kd[t] = b->d; ws->kc[t] = 128 * sk;
                    if (DK > 256) { ws->kq[t] = NULL; }   // multi-block rows: handled below (scalar)
                } else { ws->kq[t] = attnf_zero_row; ws->kd[t] = 0.0f; ws->kc[t] = 0; }
            }
        } else {
#if ATTNF_SIMD
            for (int t = 0; t < nt4; ++t) {
                float * kt = ws->Kt + (size_t) t * ATTNF_DMAX;
                if (t < nt) {
                    const ggml_fp16_t * kr = (const ggml_fp16_t *) (kv->k + (t0 + t) * kv->nbk);
                    for (int d = 0; d < DK; d += 16) _mm512_store_ps(kt + d, _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) (kr + d))));
                } else {
                    memset(kt, 0, DK * sizeof(float));
                }
            }
#endif
        }
        // scores
#if ATTNF_SIMD
        if (!i8 || DK == 256) {
            for (int t = 0; t < nt4; t += 4) {
                int r = 0;
                for (; r + 4 <= R; r += 4) { if (i8) attnf_qk_i8<4>(ws, DK, r, t); else attnf_qk_f32<4>(ws, DK, r, t); }
                switch (R - r) {
                    case 3: if (i8) attnf_qk_i8<3>(ws, DK, r, t); else attnf_qk_f32<3>(ws, DK, r, t); break;
                    case 2: if (i8) attnf_qk_i8<2>(ws, DK, r, t); else attnf_qk_f32<2>(ws, DK, r, t); break;
                    case 1: if (i8) attnf_qk_i8<1>(ws, DK, r, t); else attnf_qk_f32<1>(ws, DK, r, t); break;
                    default: break;
                }
            }
        } else
#endif
        {   // scalar reference (int8 rows of more than one block, or no SIMD)
            for (int t = 0; t < nt; ++t) {
                const attnf_q8k * b = i8 ? (const attnf_q8k *) (kv->k + (t0 + t) * kv->nbk) : NULL;
                const ggml_fp16_t * kr = i8 ? NULL : (const ggml_fp16_t *) (kv->k + (t0 + t) * kv->nbk);
                for (int r = 0; r < R; ++r) {
                    float s = 0.0f;
                    if (i8) {
                        const uint8_t * u = ws->Qu + (size_t) r * ATTNF_DMAX, * u2 = ws->Qu2 + (size_t) r * ATTNF_DMAX;
                        for (int j = 0; j < DK / 256; ++j) {
                            int32_t acc = 0, acc2 = 0;
                            for (int d = 0; d < 256; ++d) {
                                acc  += ((int32_t) u[256*j + d] - 128) * b[j].qs[d];
                                acc2 += ((int32_t) u2[256*j + d] - 128) * b[j].qs[d];
                            }
                            float x = (float) acc;
                            if (attnf_q16()) x += (float) acc2 * (1.0f / 254.0f);
                            s += x * (ws->dQ[r] * b[j].d);
                        }
                    } else {
                        const float * q = ws->Q + (size_t) r * ATTNF_DMAX;
                        for (int d = 0; d < DK; ++d) s += q[d] * GGML_CPU_FP16_TO_FP32(kr[d]);
                    }
                    ws->S[(size_t) r * T + t] += s;
                }
            }
        }
        // softmax update per row
        for (int r = 0; r < R; ++r) {
            float * Sr = ws->S + (size_t) r * T;
#if ATTNF_SIMD
            const __m512 s0 = _mm512_load_ps(Sr), s1 = _mm512_load_ps(Sr + 16);
            const float tmax = _mm512_reduce_max_ps(_mm512_max_ps(s0, s1));
#else
            float tmax = -INFINITY;
            for (int t = 0; t < T; ++t) tmax = fmaxf(tmax, Sr[t]);
#endif
            if (tmax == -INFINITY) {   // fully masked here: contributes nothing
                ws->al[r] = 1.0f;
                memset(Sr, 0, T * sizeof(float));
                continue;
            }
            const float mn = fmaxf(ws->m[r], tmax);
            const float a  = expf(ws->m[r] - mn);
#if ATTNF_SIMD
            const __m512 vm = _mm512_set1_ps(mn);
            const __m512 p0 = ggml_v_expf(_mm512_sub_ps(s0, vm)), p1 = ggml_v_expf(_mm512_sub_ps(s1, vm));
            _mm512_store_ps(Sr, p0); _mm512_store_ps(Sr + 16, p1);
            const float sum = _mm512_reduce_add_ps(_mm512_add_ps(p0, p1));
#else
            float sum = 0.0f;
            for (int t = 0; t < T; ++t) { Sr[t] = Sr[t] == -INFINITY ? 0.0f : expf(Sr[t] - mn); }
            for (int t = 0; t < T; ++t) sum += Sr[t];
#endif
            ws->l[r] = ws->l[r] * a + sum;
            ws->m[r] = mn;
            ws->al[r] = a;
        }
        // V tile
        if (i8 && kqb) {
            for (int t = 0; t < nt; ++t) {
                const uint8_t * b = (const uint8_t *) (kv->v + (t0 + t) * kv->nbv);
                float * vt = ws->Vt + (size_t) t * ATTNF_DMAX;
                alignas(64) int8_t v8[256];
                attnf_kvq_unpack(b + 4, kqb, v8);
#if ATTNF_SIMD
                for (int d = 0; d < 256; d += 16)
                    _mm512_store_ps(vt + d, _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_load_si128((const __m128i *) (v8 + d)))));
#else
                for (int d = 0; d < 256; ++d) vt[d] = (float) v8[d];
#endif
                memcpy(&ws->vd[t], b, 4);
            }
            for (int r = 0; r < R; ++r) { float * Sr = ws->S + (size_t) r * T; for (int t = 0; t < nt; ++t) Sr[t] *= ws->vd[t]; }
        } else if (i8) {
            for (int t = 0; t < nt; ++t) {
                const attnf_q8k * b = (const attnf_q8k *) (kv->v + (t0 + t) * kv->nbv);
                float * vt = ws->Vt + (size_t) t * ATTNF_DMAX;
                for (int j = 0; j < DV / 256; ++j) {
#if ATTNF_SIMD
                    for (int d = 0; d < 256; d += 16)
                        _mm512_store_ps(vt + 256*j + d, _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *) (b[j].qs + d)))));
#else
                    for (int d = 0; d < 256; ++d) vt[256*j + d] = (float) b[j].qs[d];
#endif
                }
                ws->vd[t] = b->d;   // one scale per (token, head) row when DV == 256
            }
            if (DV == 256) {
                for (int r = 0; r < R; ++r) { float * Sr = ws->S + (size_t) r * T; for (int t = 0; t < nt; ++t) Sr[t] *= ws->vd[t]; }
            } else {   // several blocks per row: dequantize the tile instead
                for (int t = 0; t < nt; ++t) {
                    const attnf_q8k * b = (const attnf_q8k *) (kv->v + (t0 + t) * kv->nbv);
                    float * vt = ws->Vt + (size_t) t * ATTNF_DMAX;
                    for (int d = 0; d < DV; ++d) vt[d] *= b[d / 256].d;
                }
            }
        } else {
            for (int t = 0; t < nt; ++t) {
                const ggml_fp16_t * vr = (const ggml_fp16_t *) (kv->v + (t0 + t) * kv->nbv);
                float * vt = ws->Vt + (size_t) t * ATTNF_DMAX;
#if ATTNF_SIMD
                for (int d = 0; d < DV; d += 16) _mm512_store_ps(vt + d, _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) (vr + d))));
#else
                for (int d = 0; d < DV; ++d) vt[d] = GGML_CPU_FP16_TO_FP32(vr[d]);
#endif
            }
        }
        // O = O*al + P V
#if ATTNF_SIMD
        for (int d0 = 0; d0 < DV; d0 += 64) {
            int r = 0;
            for (; r + 6 <= R; r += 6) attnf_pv<6>(ws, r, d0, nt);
            switch (R - r) {
                case 5: attnf_pv<5>(ws, r, d0, nt); break;
                case 4: attnf_pv<4>(ws, r, d0, nt); break;
                case 3: attnf_pv<3>(ws, r, d0, nt); break;
                case 2: attnf_pv<2>(ws, r, d0, nt); break;
                case 1: attnf_pv<1>(ws, r, d0, nt); break;
                default: break;
            }
        }
#else
        for (int r = 0; r < R; ++r) {
            float * o = ws->O + (size_t) r * ATTNF_DMAX;
            const float a = ws->al[r];
            if (a != 1.0f) for (int d = 0; d < DV; ++d) o[d] *= a;
            for (int t = 0; t < nt; ++t) {
                const float p = ws->S[(size_t) r * T + t];
                const float * vt = ws->Vt + (size_t) t * ATTNF_DMAX;
                for (int d = 0; d < DV; ++d) o[d] += p * vt[d];
            }
        }
#endif
    }
    for (int r = 0; r < R; ++r) {
        m[r] = ws->m[r]; l[r] = ws->l[r];
        if (O != ws->O || ldo != ATTNF_DMAX) memcpy(O + (size_t) r * ldo, ws->O + (size_t) r * ATTNF_DMAX, DV * sizeof(float));
    }
}

// ---------------------------------------------------------------- the op
static int attnf_mode(void) {
    static int mode = -1;
    if (mode < 0) { const char * e = getenv("GGML_ATTN"); mode = e ? atoi(e) : 1; }
    return mode;
}

// first / last+1 key a mask row leaves visible, within [0, nkv)
static void attnf_visible(const ggml_fp16_t * mr, int64_t nkv, int64_t * lo, int64_t * hi) {
    if (!mr) { *lo = 0; *hi = nkv; return; }
    int64_t h = nkv;
    while (h > 0 && GGML_CPU_FP16_TO_FP32(mr[h - 1]) == -INFINITY) --h;
    int64_t a = 0;
    while (a < h && GGML_CPU_FP16_TO_FP32(mr[a]) == -INFINITY) ++a;
    *lo = a; *hi = h;
}

// returns true when the op was computed here
static bool attnf_forward(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0], * k = dst->src[1], * v = dst->src[2], * mask = dst->src[3], * sinks = dst->src[4];
    const bool i8 = attnf_is_i8(k->type);
    if (!i8 && (attnf_mode() == 0 || !ATTNF_SIMD)) return false;
    float scale = 1.0f, max_bias = 0.0f, softcap = 0.0f;
    memcpy(&scale,    (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&softcap,  (const float *) dst->op_params + 2, sizeof(float));
    const int64_t DK = k->ne[0], DV = v->ne[0], M = q->ne[1], NH = q->ne[2], HKV = k->ne[2], NKV = k->ne[1];
    bool ok = q->type == GGML_TYPE_F32 && (k->type == GGML_TYPE_F16 || i8) && v->type == k->type && !sinks &&
              max_bias == 0.0f && softcap == 0.0f && q->ne[3] == 1 && k->ne[3] == 1 && v->ne[3] == 1 &&
              DK % 64 == 0 && DV % 64 == 0 && DK <= ATTNF_DMAX && DV <= ATTNF_DMAX && q->ne[0] == DK && v->ne[1] == NKV &&
              v->ne[2] == HKV && HKV > 0 && NH % HKV == 0 && NH / HKV <= ATTNF_RMAX && q->nb[0] == sizeof(float) &&
              dst->nb[0] == sizeof(float) && dst->ne[0] == DV && dst->ne[1] == NH && dst->ne[2] == M &&
              (!mask || (mask->type == GGML_TYPE_F16 && mask->ne[2] == 1 && mask->ne[3] == 1 && mask->ne[0] >= NKV && mask->ne[1] >= M));
    if (i8) ok = ok && DK % 256 == 0 && DV % 256 == 0 && ggml_type_size(k->type) == attnf_blk(k->type);
    if (attnf_kvq_bits(k->type)) ok = ok && DK == 256 && DV == 256 && ATTNF_SIMD;
    if (!ok) {
        if (i8) GGML_ABORT("attn-fast: q8_K / kv4-6 KV cache needs plain GQA attention (no sinks/alibi/softcap, head size multiple of 256; kv4-6: head size 256, AVX-512)");
        return false;
    }
    const int ith = params->ith, nth = params->nth;
    const int G = (int) (NH / HKV);
    attnf_ws * ws = attnf_ws_get();
    GGML_ASSERT(ws && "attn-fast: out of memory");
    const ggml_fp16_t * mrows[ATTNF_RMAX];
    int pos[ATTNF_RMAX];
    const float * qp[ATTNF_RMAX];

    if ((int64_t) M * G <= ATTNF_RSPLIT) {
        // split mode: units = KV head x chunk, partials merged after a barrier
        const int R = (int) M * G;
        int64_t lo = NKV, hi = 0;
        for (int64_t i = 0; i < M; ++i) {
            mrows[i] = mask ? (const ggml_fp16_t *) ((const char *) mask->data + i * mask->nb[1]) : NULL;
            int64_t a, b; attnf_visible(mrows[i], NKV, &a, &b);
            if (b > a) { lo = a < lo ? a : lo; hi = b > hi ? b : hi; }
        }
        const int64_t c0 = hi > lo ? lo / ATTNF_CHUNK : 0, c1 = hi > lo ? (hi + ATTNF_CHUNK - 1) / ATTNF_CHUNK : 0;
        const int64_t nch = c1 - c0, units = HKV * nch;
        const size_t ps = (size_t) DV + 2;   // [m, l, O[DV]] per row
        const size_t need = (size_t) units * R * ps;
        // The graph owns this scratch buffer; another context may run attention concurrently.
        GGML_ASSERT(params->wsize >= need * sizeof(float) + 63);
        float * attnf_part = (float *) (((uintptr_t) params->wdata + 63) & ~(uintptr_t) 63);
        ggml_barrier(params->threadpool);
        float ml[2 * ATTNF_RMAX];
        // each thread takes a contiguous run of (chunk, head) units, chunk-major: it streams one token range of the
        // cache with all its heads (every head of a token sits in one contiguous row of the cache)
        const int64_t ua = units * ith / nth, ub = units * (ith + 1) / nth;
        for (int64_t u = ua; u < ub; ++u) {
            const int64_t h = u % HKV, cc = u / HKV, c = c0 + cc;
            attnf_kv kv = { (const char *) k->data + h * k->nb[2], k->nb[1], (const char *) v->data + h * v->nb[2], v->nb[1], k->type, (int) DK, (int) DV };
            for (int r = 0; r < R; ++r) {
                const int64_t i = r / G, hq = h * G + r % G;
                qp[r] = (const float *) ((const char *) q->data + i * q->nb[1] + hq * q->nb[2]);
                pos[r] = (int) i;
            }
            attnf_q_prepare(ws, &kv, R, qp, scale);
            const int64_t k0 = c * ATTNF_CHUNK, k1 = k0 + ATTNF_CHUNK < NKV ? k0 + ATTNF_CHUNK : NKV;
            float * P = attnf_part + (size_t) (h * nch + cc) * R * ps;
            attnf_range(ws, &kv, R, pos, (int) M, mrows, k0, k1, ml, ml + ATTNF_RMAX, P + 2, ps);
            for (int r = 0; r < R; ++r) { P[(size_t) r * ps] = ml[r]; P[(size_t) r * ps + 1] = ml[ATTNF_RMAX + r]; }
        }
        ggml_barrier(params->threadpool);
        for (int64_t w = ith; w < HKV * R; w += nth) {
            const int64_t h = w / R; const int r = (int) (w % R);
            float Mx = -INFINITY, Sx = 0.0f;
            float * o = ws->OF;
            memset(o, 0, DV * sizeof(float));
            for (int64_t c = 0; c < nch; ++c) {
                const float * P = attnf_part + ((size_t) (h * nch + c) * R + r) * ps;
                attnf_merge((int) DV, &Mx, &Sx, o, P[0], P[1], P + 2);
            }
            const float si = Sx == 0.0f ? 0.0f : 1.0f / Sx;
            const int64_t i = r / G, hq = h * G + r % G;
            float * out = (float *) ((char *) dst->data + i * dst->nb[2] + hq * dst->nb[1]);
            for (int64_t d = 0; d < DV; ++d) out[d] = o[d] * si;
        }
        return true;
    }

    // rows mode: units = KV head x block of BQ positions, every chunk of a row in one thread, dynamic scheduling
    static int bq_env = -1;
    if (bq_env < 0) { const char * e = getenv("GGML_ATTN_BQ"); bq_env = e ? atoi(e) : 0; }
    int BQ = bq_env > 0 ? bq_env : 16;   // positions per unit (rows = BQ*G <= ATTNF_RMAX)
    if (BQ * G > ATTNF_RMAX) BQ = ATTNF_RMAX / G;
    const int64_t nqb = (M + BQ - 1) / BQ, units = HKV * nqb;
    if (ith == 0) ggml_threadpool_chunk_set(params->threadpool, nth);
    ggml_barrier(params->threadpool);
    float ml[2 * ATTNF_RMAX];
    for (int64_t u = ith; u < units; u = ggml_threadpool_chunk_add(params->threadpool, 1)) {
        const int64_t qb = nqb - 1 - u / HKV, h = u % HKV;   // last (longest) positions first
        const int64_t i0 = qb * BQ, np = i0 + BQ < M ? BQ : M - i0;
        const int R = (int) np * G;
        attnf_kv kv = { (const char *) k->data + h * k->nb[2], k->nb[1], (const char *) v->data + h * v->nb[2], v->nb[1], k->type, (int) DK, (int) DV };
        int64_t lo = NKV, hi = 0;
        for (int64_t i = 0; i < np; ++i) {
            mrows[i] = mask ? (const ggml_fp16_t *) ((const char *) mask->data + (i0 + i) * mask->nb[1]) : NULL;
            int64_t a, b; attnf_visible(mrows[i], NKV, &a, &b);
            if (b > a) { lo = a < lo ? a : lo; hi = b > hi ? b : hi; }
        }
        for (int r = 0; r < R; ++r) {
            const int64_t i = r / G, hq = h * G + r % G;
            qp[r] = (const float *) ((const char *) q->data + (i0 + i) * q->nb[1] + hq * q->nb[2]);
            pos[r] = (int) i;
            ws->MF[r] = -INFINITY; ws->SF[r] = 0.0f;
            memset(ws->OF + (size_t) r * ATTNF_DMAX, 0, DV * sizeof(float));
        }
        attnf_q_prepare(ws, &kv, R, qp, scale);
        if (hi > lo) {
            for (int64_t c = lo / ATTNF_CHUNK; c * ATTNF_CHUNK < hi; ++c) {
                const int64_t k0 = c * ATTNF_CHUNK;
                int64_t k1 = k0 + ATTNF_CHUNK < NKV ? k0 + ATTNF_CHUNK : NKV;
                k1 = k1 < hi ? k1 : hi;   // keys >= hi are masked for every row here (tile grid unchanged)
                attnf_range(ws, &kv, R, pos, (int) np, mrows, k0, k1, ml, ml + ATTNF_RMAX, ws->O, ATTNF_DMAX);
                for (int r = 0; r < R; ++r)
                    attnf_merge((int) DV, &ws->MF[r], &ws->SF[r], ws->OF + (size_t) r * ATTNF_DMAX, ml[r], ml[ATTNF_RMAX + r], ws->O + (size_t) r * ATTNF_DMAX);
            }
        }
        for (int r = 0; r < R; ++r) {
            const float si = ws->SF[r] == 0.0f ? 0.0f : 1.0f / ws->SF[r];
            const int64_t i = r / G, hq = h * G + r % G;
            float * out = (float *) ((char *) dst->data + (i0 + i) * dst->nb[2] + hq * dst->nb[1]);
            const float * o = ws->OF + (size_t) r * ATTNF_DMAX;
            for (int64_t d = 0; d < DV; ++d) out[d] = o[d] * si;
        }
    }
    return true;
}
