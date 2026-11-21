// X1 (2026-10-29), GGML_X1 bit 64: the MTP draft layer's Q4_K matmuls (M = 1..8) on a 16-row panel layout with
// AVX-512 VNNI. Stock (CPU_REPACK q4_K 8x8 GEMV) streams the draft layer at ~37 GB/s; here one 64-byte load carries
// 16 rows x 8 weights (both nibbles), two vpdpbusd per load per token, integer sub-block scales/mins (vpmulld), and one
// float flush per 256-weight super-block. Same q4_K / q8_K integer math as ggml's vec_dot_q4_K_q8_K; the float combine
// order differs, so draft logits can differ from stock in the last bits. Only the DRAFT uses it (the target is PTQ1_0):
// verify decides every output token, so outputs cannot change; only draft acceptance could, and only on near-ties.
// The weights must be plain q4_K (run with LLAMA_ARG_REPACK=0 so ggml's CPU_REPACK does not own them); the panels are
// built once per tensor on first use (~0.2 s for the 27B's MTP layer).
#include <mutex>
#include <vector>
#include <cstring>
struct x1q4k_blk { uint16_t d, dmin; uint8_t scales[12]; uint8_t qs[128]; };           // = block_q4_K (144 B)
struct x1q8k_blk { float d; int8_t qs[256]; int16_t bsums[16]; };                        // = block_q8_K (292 B)
static_assert(sizeof(x1q4k_blk) == 144, "q4_K layout");
static_assert(sizeof(x1q8k_blk) == 292, "q8_K layout");
#define X1Q4K_SBB 2368   // per panel per super-block: 2048 nibble bytes, 128 sc, 128 mins, 32 d, 32 dmin
struct x1q4k_e { const void * src; int64_t rows, k, nsb, npanel; uint8_t * packed; };
static std::mutex x1q4k_mu;
static std::vector<x1q4k_e *> x1q4k_list;
static inline void x1q4k_scale_min(int j, const uint8_t * q, uint8_t * d, uint8_t * m) {   // = get_scale_min_k4
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else { *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4); }
}
static x1q4k_e * x1q4k_build(const void * key, const void * data, int64_t k, int64_t rows, size_t nb1);
static x1q4k_e * x1q4k_get(const ggml_tensor * w) {
    std::lock_guard<std::mutex> lk(x1q4k_mu);
    for (auto * e : x1q4k_list) if (e->src == w->data) return e;
    return x1q4k_build(w->data, w->data, w->ne[0], w->ne[1], w->nb[1]);
}
// panels from plain q4_K rows at data (row stride nb1); key identifies the source tensor (caller holds x1q4k_mu)
static x1q4k_e * x1q4k_build(const void * key, const void * data, int64_t k, int64_t rows, size_t nb1) {
    struct { const void * data; size_t nb[2]; } wv = { data, { 0, nb1 } };
    const auto * w = &wv;
    auto * e = new x1q4k_e();
    e->src = key; e->k = k; e->rows = rows; e->nsb = e->k / 256; e->npanel = e->rows / 16;
    e->packed = (uint8_t *) aligned_alloc(64, (size_t) e->npanel * e->nsb * X1Q4K_SBB);
    for (int64_t p = 0; p < e->npanel; ++p) for (int64_t sb = 0; sb < e->nsb; ++sb) {
        uint8_t * o = e->packed + ((size_t) p * e->nsb + sb) * X1Q4K_SBB;
        for (int r = 0; r < 16; ++r) {
            const x1q4k_blk * b = (const x1q4k_blk *) ((const char *) w->data + (p * 16 + r) * w->nb[1]) + sb;
            for (int i = 0; i < 4; ++i) for (int l4 = 0; l4 < 8; ++l4) memcpy(o + i * 512 + l4 * 64 + r * 4, b->qs + 32 * i + 4 * l4, 4);
            for (int j = 0; j < 8; ++j) x1q4k_scale_min(j, b->scales, o + 2048 + j * 16 + r, o + 2176 + j * 16 + r);
            memcpy(o + 2304 + r * 2, &b->d, 2); memcpy(o + 2336 + r * 2, &b->dmin, 2);
        }
    }
    x1q4k_list.push_back(e);
    return e;
}

// NP consecutive panels x NT tokens, interleaved per 64-byte load (NP weight streams in flight per core, as K5 for M = 1)
template <int NT, int NP>
static inline void x1q4k_panel(const x1q4k_e * e, int64_t p, const x1q8k_blk * const * y, float * dst, int64_t ld) {
    const __m512i m4 = _mm512_set1_epi8(0x0F);
    __m512 accf[NP][NT];
    for (int q = 0; q < NP; ++q) for (int t = 0; t < NT; ++t) accf[q][t] = _mm512_setzero_ps();
    const size_t pst = (size_t) e->nsb * X1Q4K_SBB;
    const uint8_t * P = e->packed + (size_t) p * pst;
    for (int64_t sb = 0; sb < e->nsb; ++sb, P += X1Q4K_SBB) {
        __m512i isum[NP][NT], msum[NP][NT];
        for (int q = 0; q < NP; ++q) for (int t = 0; t < NT; ++t) { isum[q][t] = _mm512_setzero_si512(); msum[q][t] = _mm512_setzero_si512(); }
        for (int i = 0; i < 4; ++i) {
            __m512i alo[NP][NT], ahi[NP][NT];
            for (int q = 0; q < NP; ++q) for (int t = 0; t < NT; ++t) { alo[q][t] = _mm512_setzero_si512(); ahi[q][t] = _mm512_setzero_si512(); }
            for (int l4 = 0; l4 < 8; ++l4) {
                for (int q = 0; q < NP; ++q) {
                    const uint8_t * Pq = P + q * pst;
                    _mm_prefetch((const char *) Pq + i * 512 + l4 * 64 + 2048, _MM_HINT_T0);
                    const __m512i V = _mm512_loadu_si512((const void *) (Pq + i * 512 + l4 * 64));
                    const __m512i lo = _mm512_and_si512(V, m4), hi = _mm512_and_si512(_mm512_srli_epi16(V, 4), m4);
                    for (int t = 0; t < NT; ++t) {
                        const int8_t * yq = y[t][sb].qs + 64 * i + 4 * l4;
                        alo[q][t] = _mm512_dpbusd_epi32(alo[q][t], lo, _mm512_set1_epi32(*(const int32_t *) yq));
                        ahi[q][t] = _mm512_dpbusd_epi32(ahi[q][t], hi, _mm512_set1_epi32(*(const int32_t *) (yq + 32)));
                    }
                }
            }
            for (int q = 0; q < NP; ++q) {
                const uint8_t * Pq = P + q * pst;
                const __m512i sclo = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *) (Pq + 2048 + (2 * i) * 16)));
                const __m512i schi = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *) (Pq + 2048 + (2 * i + 1) * 16)));
                const __m512i mlo = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *) (Pq + 2176 + (2 * i) * 16)));
                const __m512i mhi = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *) (Pq + 2176 + (2 * i + 1) * 16)));
                for (int t = 0; t < NT; ++t) {
                    isum[q][t] = _mm512_add_epi32(isum[q][t], _mm512_add_epi32(_mm512_mullo_epi32(alo[q][t], sclo), _mm512_mullo_epi32(ahi[q][t], schi)));
                    const int16_t * bs = y[t][sb].bsums + 4 * i;
                    msum[q][t] = _mm512_add_epi32(msum[q][t], _mm512_add_epi32(_mm512_mullo_epi32(mlo, _mm512_set1_epi32(bs[0] + bs[1])),
                                                                                _mm512_mullo_epi32(mhi, _mm512_set1_epi32(bs[2] + bs[3]))));
                }
            }
        }
        for (int q = 0; q < NP; ++q) {
            const uint8_t * Pq = P + q * pst;
            const __m512 dv = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) (Pq + 2304)));
            const __m512 dmv = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) (Pq + 2336)));
            for (int t = 0; t < NT; ++t) {
                const __m512 yd = _mm512_set1_ps(y[t][sb].d);
                accf[q][t] = _mm512_fmadd_ps(_mm512_mul_ps(yd, dv), _mm512_cvtepi32_ps(isum[q][t]), accf[q][t]);
                accf[q][t] = _mm512_fnmadd_ps(_mm512_mul_ps(yd, dmv), _mm512_cvtepi32_ps(msum[q][t]), accf[q][t]);
            }
        }
    }
    for (int q = 0; q < NP; ++q) for (int t = 0; t < NT; ++t) _mm512_storeu_ps(dst + t * ld + (p + q) * 16, accf[q][t]);
}

static bool x1q4k_applies(const ggml_tensor * n) {
    const ggml_tensor * w = n->src[0], * x = n->src[1];
    if (n->op != GGML_OP_MUL_MAT || w->type != GGML_TYPE_Q4_K || x->type != GGML_TYPE_F32 || n->type != GGML_TYPE_F32) return false;
    if (w->ne[0] % 256 || w->ne[1] % 16 || w->ne[2] != 1 || w->ne[3] != 1 || x->ne[1] < 1 || x->ne[1] > 8 || x->ne[2] != 1 || x->ne[3] != 1) return false;
    if (!ggml_is_contiguous(w) || !ggml_is_contiguous(x) || !ggml_is_contiguous(n) || !w->data) return false;
    if (w->buffer && strstr(ggml_backend_buffer_name(w->buffer), "REPACK")) return false;   // ggml's repack owns the layout
    return true;
}
static void x1q4k_compute(const ggml_compute_params * params, const ggml_tensor * n, x1q4k_e * e);
static void x1q4k_mul_mat(const ggml_compute_params * params, const ggml_tensor * n) { x1q4k_compute(params, n, x1q4k_get(n->src[0])); }
static void x1q4k_compute(const ggml_compute_params * params, const ggml_tensor * n, x1q4k_e * e) {
    const ggml_tensor * w = n->src[0], * x = n->src[1];
    const int64_t m = x->ne[1], nsb = e->nsb;
    thread_local std::vector<x1q8k_blk> qb;
    qb.resize((size_t) m * nsb);
    const auto from_float = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K)->from_float;
    for (int64_t t = 0; t < m; ++t) from_float((const float *) ((const char *) x->data + t * x->nb[1]), qb.data() + t * nsb, e->k);
    const x1q8k_blk * y[8];
    for (int64_t t = 0; t < m; ++t) y[t] = qb.data() + t * nsb;
    const int64_t per = (e->npanel + params->nth - 1) / params->nth;
    const int64_t p0 = std::min<int64_t>(e->npanel, per * params->ith), p1 = std::min<int64_t>(e->npanel, p0 + per);
    float * dst = (float *) n->data; const int64_t ld = n->nb[1] / sizeof(float);
    int64_t p = p0;
    if (m == 1) for (; p + 2 <= p1; p += 2) x1q4k_panel<1, 2>(e, p, y, dst, ld);
    for (; p < p1; ++p) {
        int64_t t0 = 0;
        for (; t0 + 4 <= m; t0 += 4) x1q4k_panel<4, 1>(e, p, y + t0, dst + t0 * ld, ld);
        switch (m - t0) {
            case 1: x1q4k_panel<1, 1>(e, p, y + t0, dst + t0 * ld, ld); break;
            case 2: x1q4k_panel<2, 1>(e, p, y + t0, dst + t0 * ld, ld); break;
            case 3: x1q4k_panel<3, 1>(e, p, y + t0, dst + t0 * ld, ld); break;
            default: break;
        }
    }
    static int chk = getenv("X1_Q4K_CHECK") ? 1 : 0;   // dev: one row per tensor vs a scalar dequant dot on the float input
    if (chk && params->ith == 0 && w->type == GGML_TYPE_Q4_K) {
        static std::vector<const void *> seen;
        if (std::find(seen.begin(), seen.end(), w->data) == seen.end()) {
            seen.push_back(w->data);
            const int64_t r = p0 * 16 + 5;   // a row thread 0 computed
            double ref = 0;
            const x1q4k_blk * b = (const x1q4k_blk *) ((const char *) w->data + r * w->nb[1]);
            const float * xv = (const float *) x->data;
            for (int64_t sb = 0; sb < nsb; ++sb) {
                const float d = GGML_CPU_FP16_TO_FP32(b[sb].d), dmin = GGML_CPU_FP16_TO_FP32(b[sb].dmin);
                for (int j = 0; j < 8; ++j) {
                    uint8_t sc, mn; x1q4k_scale_min(j, b[sb].scales, &sc, &mn);
                    for (int l = 0; l < 32; ++l) {
                        const int q = (j & 1) ? (b[sb].qs[32 * (j / 2) + l] >> 4) : (b[sb].qs[32 * (j / 2) + l] & 0xF);
                        ref += (double) (d * sc * q - dmin * mn) * xv[sb * 256 + j * 32 + l];
                    }
                }
            }
            fprintf(stderr, "X1 q4k check %s [%lld x %lld] m=%lld: row %lld kernel %.6g ref %.6g\n", n->name, (long long) e->k,
                    (long long) e->rows, (long long) m, (long long) r, dst[r], ref);
        }
    }
}

// ---- Q6_K (the draft layer's Q/K/V and attention output): same panel scheme. A super-block of 256 weights per row is
// repacked into u6 values split as a nibble plane (the q4_K layout above) + a 2-bit plane (4 fields per byte), 16 int8
// group scales and d; 3360 B per panel per super-block = exactly q6_K's bytes. dot = sum_g sc_g * (sum u*y - 32 * bsum_g).
struct x1q6k_blk { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; uint16_t d; };       // = block_q6_K (210 B)
static_assert(sizeof(x1q6k_blk) == 210, "q6_K layout");
#define X1Q6K_SBB 3360
static std::vector<x1q4k_e *> x1q6k_list;
static x1q4k_e * x1q6k_get(const ggml_tensor * w) {
    std::lock_guard<std::mutex> lk(x1q4k_mu);
    for (auto * e : x1q6k_list) if (e->src == w->data) return e;
    auto * e = new x1q4k_e();
    e->src = w->data; e->k = w->ne[0]; e->rows = w->ne[1]; e->nsb = e->k / 256; e->npanel = e->rows / 16;
    e->packed = (uint8_t *) aligned_alloc(64, (size_t) e->npanel * e->nsb * X1Q6K_SBB);
    for (int64_t p = 0; p < e->npanel; ++p) for (int64_t sb = 0; sb < e->nsb; ++sb) {
        uint8_t * o = e->packed + ((size_t) p * e->nsb + sb) * X1Q6K_SBB;
        for (int r = 0; r < 16; ++r) {
            const x1q6k_blk * b = (const x1q6k_blk *) ((const char *) w->data + (p * 16 + r) * w->nb[1]) + sb;
            uint8_t v[256];
            for (int h = 0; h < 2; ++h) for (int l = 0; l < 32; ++l) {
                const uint8_t * ql = b->ql + 64 * h, * qh = b->qh + 32 * h;
                v[128 * h + l]      = (ql[l] & 0xF)      | (((qh[l] >> 0) & 3) << 4);
                v[128 * h + l + 32] = (ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4);
                v[128 * h + l + 64] = (ql[l] >> 4)       | (((qh[l] >> 4) & 3) << 4);
                v[128 * h + l + 96] = (ql[l + 32] >> 4)  | (((qh[l] >> 6) & 3) << 4);
            }
            for (int i = 0; i < 4; ++i) for (int l4 = 0; l4 < 8; ++l4) for (int bb = 0; bb < 4; ++bb) {
                const int wl = 64 * i + 4 * l4 + bb, wh = wl + 32;
                o[i * 512 + l4 * 64 + r * 4 + bb] = (uint8_t) ((v[wl] & 0xF) | ((v[wh] & 0xF) << 4));
                uint8_t & hb = o[2048 + i * 256 + (l4 / 2) * 64 + r * 4 + bb];
                const int sh = (l4 & 1) * 4;
                hb = (uint8_t) ((hb & ~(0xF << sh)) | (((v[wl] >> 4) | ((v[wh] >> 4) << 2)) << sh));
            }
            for (int g = 0; g < 16; ++g) o[3072 + g * 16 + r] = (uint8_t) b->scales[g];
            memcpy(o + 3328 + r * 2, &b->d, 2);
        }
    }
    x1q6k_list.push_back(e);
    return e;
}
template <int NT, int NP>
static inline void x1q6k_panel(const x1q4k_e * e, int64_t p, const x1q8k_blk * const * y, float * dst, int64_t ld) {
    const __m512i m4 = _mm512_set1_epi8(0x0F), m30 = _mm512_set1_epi8(0x30);
    __m512 accf[NP][NT];
    for (int q = 0; q < NP; ++q) for (int t = 0; t < NT; ++t) accf[q][t] = _mm512_setzero_ps();
    const size_t pst = (size_t) e->nsb * X1Q6K_SBB;
    const uint8_t * P = e->packed + (size_t) p * pst;
    for (int64_t sb = 0; sb < e->nsb; ++sb, P += X1Q6K_SBB) {
        __m512i isum[NP][NT];
        for (int q = 0; q < NP; ++q) for (int t = 0; t < NT; ++t) isum[q][t] = _mm512_setzero_si512();
        for (int i = 0; i < 4; ++i) {
            __m512i a[NP][NT][4];   // groups 4i, 4i+1 (low sub-block 2i) and 4i+2, 4i+3 (high sub-block 2i+1)
            for (int q = 0; q < NP; ++q) for (int t = 0; t < NT; ++t) for (int g = 0; g < 4; ++g) a[q][t][g] = _mm512_setzero_si512();
            for (int l4 = 0; l4 < 8; ++l4) {
                for (int q = 0; q < NP; ++q) {
                    const uint8_t * Pq = P + q * pst;
                    _mm_prefetch((const char *) Pq + i * 512 + l4 * 64 + 2048, _MM_HINT_T0);
                    if ((l4 & 1) == 0) _mm_prefetch((const char *) Pq + 2048 + i * 256 + (l4 / 2) * 64 + 1024, _MM_HINT_T0);
                    const __m512i V = _mm512_loadu_si512((const void *) (Pq + i * 512 + l4 * 64));
                    const __m512i H = _mm512_loadu_si512((const void *) (Pq + 2048 + i * 256 + (l4 / 2) * 64));
                    const __m512i Hs = (l4 & 1) ? _mm512_srli_epi16(H, 4) : H;      // low 4 bits of each byte: lo hi2 | hi hi2
                    const __m512i lo = _mm512_or_si512(_mm512_and_si512(V, m4), _mm512_and_si512(_mm512_slli_epi16(Hs, 4), m30));
                    const __m512i hi = _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(V, 4), m4), _mm512_and_si512(_mm512_slli_epi16(Hs, 2), m30));
                    const int gl = l4 / 4;   // 0: first 16 of the sub-block, 1: second 16
                    for (int t = 0; t < NT; ++t) {
                        const int8_t * yq = y[t][sb].qs + 64 * i + 4 * l4;
                        a[q][t][gl]     = _mm512_dpbusd_epi32(a[q][t][gl], lo, _mm512_set1_epi32(*(const int32_t *) yq));
                        a[q][t][2 + gl] = _mm512_dpbusd_epi32(a[q][t][2 + gl], hi, _mm512_set1_epi32(*(const int32_t *) (yq + 32)));
                    }
                }
            }
            for (int q = 0; q < NP; ++q) for (int g = 0; g < 4; ++g) {
                const int gi = 4 * i + g;
                const __m512i sc = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *) (P + q * pst + 3072 + gi * 16)));
                for (int t = 0; t < NT; ++t) {
                    const __m512i biased = _mm512_sub_epi32(a[q][t][g], _mm512_set1_epi32(32 * y[t][sb].bsums[gi]));
                    isum[q][t] = _mm512_add_epi32(isum[q][t], _mm512_mullo_epi32(biased, sc));
                }
            }
        }
        for (int q = 0; q < NP; ++q) {
            const __m512 dv = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) (P + q * pst + 3328)));
            for (int t = 0; t < NT; ++t) accf[q][t] = _mm512_fmadd_ps(_mm512_mul_ps(_mm512_set1_ps(y[t][sb].d), dv), _mm512_cvtepi32_ps(isum[q][t]), accf[q][t]);
        }
    }
    for (int q = 0; q < NP; ++q) for (int t = 0; t < NT; ++t) _mm512_storeu_ps(dst + t * ld + (p + q) * 16, accf[q][t]);
}
static bool x1q6k_applies(const ggml_tensor * n) {
    const ggml_tensor * w = n->src[0], * x = n->src[1];
    if (n->op != GGML_OP_MUL_MAT || w->type != GGML_TYPE_Q6_K || x->type != GGML_TYPE_F32 || n->type != GGML_TYPE_F32) return false;
    if (w->ne[0] % 256 || w->ne[1] % 16 || w->ne[2] != 1 || w->ne[3] != 1 || x->ne[1] < 1 || x->ne[1] > 8 || x->ne[2] != 1 || x->ne[3] != 1) return false;
    if (!ggml_is_contiguous(w) || !ggml_is_contiguous(x) || !ggml_is_contiguous(n) || !w->data) return false;
    if (w->buffer && strstr(ggml_backend_buffer_name(w->buffer), "REPACK")) return false;
    return true;
}
// X1_Q6_AS_Q4=1 (draft only; in the recommended set): requantize the Q6_K weights to Q4_K once and run them through the Q4_K kernel
// (31 % fewer bytes; changes draft logits, so it may move acceptance -- measured, not assumed)
#include "ggml-quants.h"
static std::vector<std::pair<const void *, x1q4k_e *>> x1q64_list;
static x1q4k_e * x1q64_get(const ggml_tensor * w) {
    std::lock_guard<std::mutex> lk(x1q4k_mu);
    for (auto & pr : x1q64_list) if (pr.first == w->data) return pr.second;
    const int64_t k = w->ne[0], rows = w->ne[1];
    std::vector<float> f(k);
    std::vector<uint8_t> q4((size_t) rows * (k / 256) * 144);
    for (int64_t r = 0; r < rows; ++r) {
        dequantize_row_q6_K((const block_q6_K *) ((const char *) w->data + r * w->nb[1]), f.data(), k);
        quantize_row_q4_K_ref(f.data(), (block_q4_K *) (q4.data() + (size_t) r * (k / 256) * 144), k);
    }
    x1q4k_e * e = x1q4k_build(w->data, q4.data(), k, rows, (k / 256) * 144);
    x1q64_list.push_back({ w->data, e });
    return e;
}
static void x1q6k_mul_mat(const ggml_compute_params * params, const ggml_tensor * n) {
    const ggml_tensor * w = n->src[0], * x = n->src[1];
    static const bool as_q4 = getenv("X1_Q6_AS_Q4") && atoi(getenv("X1_Q6_AS_Q4"));
    if (as_q4) { x1q4k_compute(params, n, x1q64_get(w)); return; }
    x1q4k_e * e = x1q6k_get(w);
    const int64_t m = x->ne[1], nsb = e->nsb;
    thread_local std::vector<x1q8k_blk> qb;
    qb.resize((size_t) m * nsb);
    const auto from_float = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K)->from_float;
    for (int64_t t = 0; t < m; ++t) from_float((const float *) ((const char *) x->data + t * x->nb[1]), qb.data() + t * nsb, e->k);
    const x1q8k_blk * y[8];
    for (int64_t t = 0; t < m; ++t) y[t] = qb.data() + t * nsb;
    const int64_t per = (e->npanel + params->nth - 1) / params->nth;
    const int64_t p0 = std::min<int64_t>(e->npanel, per * params->ith), p1 = std::min<int64_t>(e->npanel, p0 + per);
    float * dst = (float *) n->data; const int64_t ld = n->nb[1] / sizeof(float);
    int64_t p = p0;
    if (m == 1) for (; p + 2 <= p1; p += 2) x1q6k_panel<1, 2>(e, p, y, dst, ld);
    for (; p < p1; ++p) {
        int64_t t0 = 0;
        for (; t0 + 2 <= m; t0 += 2) x1q6k_panel<2, 1>(e, p, y + t0, dst + t0 * ld, ld);
        if (t0 < m) x1q6k_panel<1, 1>(e, p, y + t0, dst + t0 * ld, ld);
    }
    static int chk = getenv("X1_Q4K_CHECK") ? 1 : 0;
    if (chk && params->ith == 0) {
        static std::vector<const void *> seen;
        if (std::find(seen.begin(), seen.end(), w->data) == seen.end()) {
            seen.push_back(w->data);
            const int64_t r = p0 * 16 + 5;   // a row thread 0 computed
            double ref = 0;
            const x1q6k_blk * b = (const x1q6k_blk *) ((const char *) w->data + r * w->nb[1]);
            const float * xv = (const float *) x->data;
            for (int64_t sb = 0; sb < nsb; ++sb) {
                const float d = GGML_CPU_FP16_TO_FP32(b[sb].d);
                for (int h = 0; h < 2; ++h) for (int l = 0; l < 32; ++l) {
                    const uint8_t * ql = b[sb].ql + 64 * h, * qh = b[sb].qh + 32 * h;
                    const int q[4] = { ((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32, ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32,
                                       ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32, ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32 };
                    for (int u = 0; u < 4; ++u) { const int wi = 128 * h + l + 32 * u; ref += (double) d * b[sb].scales[wi / 16] * q[u] * xv[sb * 256 + wi]; }
                }
            }
            fprintf(stderr, "X1 q6k check %s [%lld x %lld] m=%lld: row %lld kernel %.6g ref %.6g\n", n->name, (long long) e->k,
                    (long long) e->rows, (long long) m, (long long) r, dst[r], ref);
        }
    }
}
