// ptq1-panel: row-panel VNNI GEMM straight over base-3 ternary weights (GGML_PTQ1_FAST=3).
//
// Layout ("made CPU-friendly" at load): weights are grouped in panels of 16 rows. A panel is
// a run of 64-byte chunks; chunk c covers k in [20c, 20c+20). Byte j holds 5 trits for row j/4
// at k = 20c + 4n + (j%4), n = 0..4, so decode step n yields one zmm laid out exactly as
// vpdpbusd wants it: lane l = row l, 4 consecutive k per lane. Each token then costs one
// vpdpbusd per step against a 4-byte activation broadcast, the 16 lanes are 16 output rows
// (no horizontal reductions), and int32 sums run across a full Q8_0 block (8 steps) before
// one float flush -- numerics identical to Q8_0 x PTQ1_0 up to float summation order.
// Bytes: 64 per 320 trits (1.6 bpw) + 16 fp16 scales per panel per 128-k group.
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)

static ptq1p_entry * ptq1p_list = NULL;


static inline __attribute__((always_inline)) void ptq1p_panel(
        const ptq1p_entry * e, const uint8_t * panel, const uint8_t * acts, size_t cb, const int NT,
        float * dst, int64_t ld, int64_t row0, int64_t t0) {
    const int64_t kpad = e->kpad;
    const ggml_half * sc = (const ggml_half *) (panel + e->nck * 64);
    const int8_t * q[PTQ1P_TB]; const float * dy[PTQ1P_TB]; const int32_t * as[PTQ1P_TB];
    for (int t = 0; t < NT; ++t) {
        const uint8_t * base = acts + (size_t) (t0 + t) * cb;
        q[t] = (const int8_t *) base; dy[t] = (const float *) (base + kpad); as[t] = (const int32_t *) (dy[t] + kpad / 32);
    }
    __m512 accf[PTQ1P_TB]; __m512i acci[PTQ1P_TB];
    for (int t = 0; t < NT; ++t) { accf[t] = _mm512_setzero_ps(); acci[t] = _mm512_setzero_si512(); }
    __m512 dxv = _mm512_setzero_ps();
    int64_t s = 0;
    for (int64_t c = 0; c < e->nck; ++c) {
        __m512i x = _mm512_loadu_si512((const void *) (panel + c * 64));
        for (int n = 0; n < 5; ++n, ++s) {
            const __m512i codes = ptq1f_decode(&x);
            const int64_t k0 = 4 * s;
            for (int t = 0; t < NT; ++t)
                acci[t] = _mm512_dpbusd_epi32(acci[t], codes, _mm512_set1_epi32(*(const int32_t *) (q[t] + k0)));
            if ((s & 7) == 7) {                                   // end of a 32-k Q8_0 block
                const int64_t b = s >> 3;
                if ((b & 3) == 0) dxv = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) (sc + (b >> 2) * PTQ1P_ROWS)));
                for (int t = 0; t < NT; ++t) {
                    const __m512 v = _mm512_cvtepi32_ps(_mm512_sub_epi32(acci[t], _mm512_set1_epi32(as[t][b])));
                    accf[t] = _mm512_fmadd_ps(v, _mm512_mul_ps(dxv, _mm512_set1_ps(dy[t][b])), accf[t]);
                    acci[t] = _mm512_setzero_si512();
                }
            }
        }
    }
    const int64_t nr = (e->rows - row0) < PTQ1P_ROWS ? (e->rows - row0) : PTQ1P_ROWS;
    const __mmask16 m = (__mmask16) ((1u << nr) - 1);
    for (int t = 0; t < NT; ++t) _mm512_mask_storeu_ps(dst + (t0 + t) * ld + row0, m, accf[t]);
}

// M = 1 (decode) path for GGML_PTQ1_FAST=4: same math as ptq1p_panel(NT = 1), bit-identical, but walks the panel
// in 160-k super-blocks (8 chunks x 5 decode steps = 5 Q8_0 blocks), fully unrolled so every flush point is fixed
// at compile time. Measured 1.2-1.85x the M = 1 GB/s of ptq1p_gemm (our microbenchmark, 2026-10-26).
static void ptq1p_panel1(const ptq1p_entry * e, const uint8_t * panel, const uint8_t * acts, float * dst, int64_t row0) {
    const int64_t kpad = e->kpad;
    const ggml_half * sc = (const ggml_half *) (panel + e->nck * 64);
    const int8_t * q = (const int8_t *) acts;
    const float * dy = (const float *) (acts + kpad);
    const int32_t * as = (const int32_t *) (dy + kpad / 32);
    __m512 accf = _mm512_setzero_ps(), dxv = _mm512_setzero_ps();
    for (int64_t sb = 0; sb < e->nck / 8; ++sb) {
        __m512i acc = _mm512_setzero_si512(), x = _mm512_setzero_si512();
#pragma GCC unroll 40
        for (int s = 0; s < 40; ++s) {
            if (s % 5 == 0) x = _mm512_loadu_si512((const void *) (panel + (sb * 8 + s / 5) * 64));
            const __m512i codes = ptq1f_decode(&x);
            acc = _mm512_dpbusd_epi32(acc, codes, _mm512_set1_epi32(*(const int32_t *) (q + 4 * (sb * 40 + s))));
            if (s % 8 == 7) {
                const int64_t b = sb * 5 + s / 8;
                if ((b & 3) == 0) dxv = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) (sc + (b >> 2) * PTQ1P_ROWS)));
                const __m512 v = _mm512_cvtepi32_ps(_mm512_sub_epi32(acc, _mm512_set1_epi32(as[b])));
                accf = _mm512_fmadd_ps(v, _mm512_mul_ps(dxv, _mm512_set1_ps(dy[b])), accf);
                acc = _mm512_setzero_si512();
            }
        }
    }
    const int64_t nr = (e->rows - row0) < PTQ1P_ROWS ? (e->rows - row0) : PTQ1P_ROWS;
    _mm512_mask_storeu_ps(dst + row0, (__mmask16) ((1u << nr) - 1), accf);
}

static void ptq1p_gemm(const ptq1p_entry * e, const uint8_t * acts, size_t cb, int64_t m,
                       int64_t p0, int64_t p1, float * dst, int64_t ld) {
    for (int64_t p = p0; p < p1; ++p) {
        const uint8_t * panel = e->packed + p * e->panel_stride;
        for (int64_t t0 = 0; t0 < m; t0 += PTQ1P_TB) {
            const int nt = (int) (m - t0 < PTQ1P_TB ? m - t0 : PTQ1P_TB);
            switch (nt) {   // compile-time NT so accumulators live in registers
                case 1: ptq1p_panel(e, panel, acts, cb, 1, dst, ld, p * PTQ1P_ROWS, t0); break;
                case 2: ptq1p_panel(e, panel, acts, cb, 2, dst, ld, p * PTQ1P_ROWS, t0); break;
                case 3: ptq1p_panel(e, panel, acts, cb, 3, dst, ld, p * PTQ1P_ROWS, t0); break;
                case 4: ptq1p_panel(e, panel, acts, cb, 4, dst, ld, p * PTQ1P_ROWS, t0); break;
                case 5: ptq1p_panel(e, panel, acts, cb, 5, dst, ld, p * PTQ1P_ROWS, t0); break;
                case 6: ptq1p_panel(e, panel, acts, cb, 6, dst, ld, p * PTQ1P_ROWS, t0); break;
                case 7: ptq1p_panel(e, panel, acts, cb, 7, dst, ld, p * PTQ1P_ROWS, t0); break;
                default: ptq1p_panel(e, panel, acts, cb, 8, dst, ld, p * PTQ1P_ROWS, t0); break;
            }
        }
    }
}

#include "ptq1-vbmi.inc"   // GGML_PTQ1_FAST=5: bit-aligned trit fields, vpmultishiftqb + vpermb decode (needs VBMI)
static inline int ptq1p_mode5(void) { return PTQ1V_OK && ptq1f_mode() >= 5; }   // the VBMI field layout (modes 5 and 6)
static inline int ptq1p_mode6(void) { return PTQ1V_OK && ptq1f_mode() >= 6; }   // + the shared-quantize, dynamic-chunk M = 2..8 path (modes 6 and 7)
static inline int ptq1p_mode7(void) { return PTQ1V_OK && ptq1f_mode() == 7; }   // + the two-panel M = 1 decode (ptq1v_panels7)
// FL128 (GGML_PTQ1_FL128=1, default off): the M = 2..8 path with per-128 activation scales and one flush per 128 k (K4=2 numerics)
static int ptq1p_fl128_cache = -1;
static inline int ptq1p_fl128(void) {
    if (ptq1p_fl128_cache < 0) { const char * s = getenv("GGML_PTQ1_FL128"); ptq1p_fl128_cache = s ? atoi(s) : 0; }
    return PTQ1V_OK && ptq1f_mode() >= 6 ? ptq1p_fl128_cache : 0;
}
// GGML_PTQ1_K4=1 (default; on top of mode 6): the chunked prefill GEMM (ptq1k4_gemm) and the AVX-512 quantizer for M > 8
static int ptq1k4_on_cache = -1;
static inline int ptq1k4_on(void) {
    if (ptq1k4_on_cache < 0) { const char * s = getenv("GGML_PTQ1_K4"); ptq1k4_on_cache = s ? atoi(s) : 1; }   // default on (bit-exact); GGML_PTQ1_K4=0 turns it off
    return PTQ1V_OK && ptq1f_mode() >= 6 ? ptq1k4_on_cache : 0;   // 1 = bit-exact prefill GEMM, 2 = + per-128 activation scales (not bit-exact)
}

#ifndef PTQ1F_NO_GGML
// GGML_PTQ1_PROF=1: thread 0 times the phases of every ptq1p_mul_mat call and prints totals at exit (dev only).
static int ptq1p_prof_on = -1;
static double ptq1p_prof_t[5];   // lookup (hit), quantize, barrier wait, gemm, first-use repack
static long ptq1p_prof_n;
static double ptq1p_prof_now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static void ptq1p_prof_dump(void) {
    const char * nm[5] = {"lookup", "quantize", "barrier", "gemm", "repack"};
    double tot = 0; for (int i = 0; i < 4; ++i) tot += ptq1p_prof_t[i];   // steady state; repack shown apart
    fprintf(stderr, "\n== ptq1p_mul_mat phases, thread 0, %ld calls, %.3f s ==\n", ptq1p_prof_n, tot);
    for (int i = 0; i < 5; ++i) fprintf(stderr, "%-9s %8.3f s %5.1f%%  %7.1f us/call\n", nm[i], ptq1p_prof_t[i],
                                        100 * ptq1p_prof_t[i] / tot, 1e6 * ptq1p_prof_t[i] / ptq1p_prof_n);
}

// GGML_PTQ1_THP=1: back the repacked weights with 2 MB pages (madvise before first touch), so streaming ~6 GB per
// token takes ~3k TLB fills instead of ~1.5M. Off by default, for A/B.
// GGML_PTQ1_THP=2 (K7): THP is not granted on a nearly full box, so take explicit hugetlbfs pages instead: one
// MAP_HUGETLB arena sized from HugePages_Free (reserve them first: sudo sysctl vm.nr_hugepages=N), bump-allocated
// 4 KB-aligned; a tensor that doesn't fit falls back to the normal allocation.
static char * ptq1p_huge_base; static size_t ptq1p_huge_size, ptq1p_huge_used; static int ptq1p_huge_init;
static pthread_mutex_t ptq1p_huge_mu = PTHREAD_MUTEX_INITIALIZER;
static void * ptq1p_alloc_huge(size_t bytes) {
    pthread_mutex_lock(&ptq1p_huge_mu);
    if (!ptq1p_huge_init) {
        ptq1p_huge_init = 1;
        long nfree = 0; char line[256];
        FILE * f = fopen("/proc/meminfo", "r");
        if (f) { while (fgets(line, sizeof line, f)) if (sscanf(line, "HugePages_Free: %ld", &nfree) == 1) break; fclose(f); }
        const size_t sz = (size_t) nfree << 21;
        if (sz) {
            void * p = mmap(NULL, sz, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
            if (p != MAP_FAILED) { ptq1p_huge_base = (char *) p; ptq1p_huge_size = sz; }
        }
        fprintf(stderr, "ptq1: hugetlb arena %zu MB\n", ptq1p_huge_size >> 20);
    }
    void * r = NULL;
    const size_t need = (bytes + 4095) / 4096 * 4096;
    if (ptq1p_huge_base && ptq1p_huge_used + need <= ptq1p_huge_size) { r = ptq1p_huge_base + ptq1p_huge_used; ptq1p_huge_used += need; }
    pthread_mutex_unlock(&ptq1p_huge_mu);
    return r;
}
static void * ptq1p_alloc_packed(size_t bytes) {
    const char * thp = getenv("GGML_PTQ1_THP");
    if (thp && atoi(thp) == 2) { void * p = ptq1p_alloc_huge(bytes); if (p) return p; }
    if (!(thp && atoi(thp) == 1)) return aligned_alloc(64, (bytes + 63) / 64 * 64);
    const size_t huge = (size_t) 2 << 20, sz = (bytes + huge - 1) / huge * huge;
    void * p = aligned_alloc(huge, sz);
    if (p) madvise(p, sz, MADV_HUGEPAGE);
    return p;
}

// GGML_PTQ1_TRACE=1 (dev only, K6): every thread times the phases of every M = 1 call and the dump at exit shows
// where the "barrier" phase goes: late exit from the previous graph barrier (enter lateness), a slow quantize, or
// late arrival at the inner barrier; all in us per call, per thread.
static int ptq1p_tr_on = -1;
static double ptq1p_tr_enter[64], ptq1p_tr_arrive[64];
static double ptq1p_tr_acc[64][6];   // enter lateness, quantize, arrive lateness, barrier wait, gemm, whole call
static double ptq1p_tr_max[64][2];   // max enter lateness, max barrier wait
static long ptq1p_tr_n[64], ptq1p_tr_seen[64]; static int ptq1p_tr_nth;
static void ptq1p_tr_dump(void) {
    const char * nm[6] = {"enter_late", "quantize", "arrive_late", "bar_wait", "gemm", "call"};
    fprintf(stderr, "\n== ptq1p_mul_mat M = 1 per-thread phases (us/call, mean) ==\nthr  calls");
    for (int i = 0; i < 6; ++i) fprintf(stderr, " %11s", nm[i]);
    fprintf(stderr, "  max_enter_late  max_bar_wait\n");
    for (int t = 0; t < ptq1p_tr_nth; ++t) {
        fprintf(stderr, "%3d %6ld", t, ptq1p_tr_n[t]);
        for (int i = 0; i < 6; ++i) fprintf(stderr, " %11.1f", 1e6 * ptq1p_tr_acc[t][i] / (ptq1p_tr_n[t] ? ptq1p_tr_n[t] : 1));
        fprintf(stderr, " %15.0f %13.0f\n", 1e6 * ptq1p_tr_max[t][0], 1e6 * ptq1p_tr_max[t][1]);
    }
}

// GGML_K6 & 8 (K6, on by default in this release): the decode token's block-range quantizer in AVX-512 -- the same q, -sum(q) and fp16-rounded
// d per block as ptq1p_quantize_col_part_kp (roundf reproduced as trunc + half-away-from-zero, as in ptq1k4_quantize_col_bs),
// so the acts bytes are identical; it just takes ~1/5 of the time, which is on every call's critical path.
static int ptq1p_k6_cache = -1;
static inline int ptq1p_k6(void) {
    if (ptq1p_k6_cache < 0) { const char * s = getenv("GGML_K6"); ptq1p_k6_cache = s ? atoi(s) : 123; }   // default 123 (same as ops.cpp K6); bit 8 = this quantizer
    return ptq1p_k6_cache;
}
#if PTQ1V_OK
static void ptq1k6_quantize_part(const float * x, int64_t k, int64_t kpad, uint8_t * out, int64_t b0, int64_t b1, int neg_asum) {
    int8_t * q = (int8_t *) out;
    float * dy = (float *) (out + kpad);
    int32_t * asum = (int32_t *) (dy + kpad / 32);
    if (b0 == 0) {
        memset(q + k, 0, (size_t) (kpad - k));
        for (int64_t b = k / 32; b < kpad / 32; ++b) { dy[b] = 0.0f; asum[b] = 0; }
    }
    const __m512 half = _mm512_set1_ps(0.5f), one = _mm512_set1_ps(1.0f), mone = _mm512_set1_ps(-1.0f), zero = _mm512_setzero_ps();
    for (int64_t b = b0; b < b1; ++b) {
        const __m512 x0 = _mm512_loadu_ps(x + b * 32), x1 = _mm512_loadu_ps(x + b * 32 + 16);
        const float amax = _mm512_reduce_max_ps(_mm512_max_ps(_mm512_abs_ps(x0), _mm512_abs_ps(x1)));
        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f / d : 0.0f;
        const __m512 idv = _mm512_set1_ps(id);
        int s = 0;
        const __m512 xv[2] = { x0, x1 };
        for (int h = 0; h < 2; ++h) {
            const __m512 v = _mm512_mul_ps(xv[h], idv);
            const __m512 r = _mm512_roundscale_ps(v, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
            const __mmask16 away = _mm512_cmp_ps_mask(_mm512_abs_ps(_mm512_sub_ps(v, r)), half, _CMP_GE_OQ);
            const __m512 sg = _mm512_mask_blend_ps(_mm512_cmp_ps_mask(v, zero, _CMP_LT_OQ), one, mone);
            const __m512i qi = _mm512_cvttps_epi32(_mm512_mask_add_ps(r, away, r, sg));
            s += _mm512_reduce_add_epi32(qi);
            _mm_storeu_si128((__m128i *) (q + b * 32 + 16 * h), _mm512_cvtepi32_epi8(qi));
        }
        dy[b] = GGML_CPU_FP16_TO_FP32(GGML_CPU_FP32_TO_FP16(d));
        asum[b] = neg_asum ? -s : s;
    }
}
#endif

// GGML_K7 (K7, opt-in bitmask): overlap the non-matmul ops and the barrier waits with the next matmul's weight stream.
// A prefetch is only a hint and the panel schedule only moves work between threads: the numerics are untouched.
//   & 1  a decode token calls the ternary matmuls in the same order every token, so each entry learns its successor
//        (thread 0, M = 1 calls only). Every M = 1 call starts with GGML_K7_S static chunks per thread (chunks ith,
//        ith + nth, ...; the rest stays dynamic), and a thread leaving a call queues ITS static chunks of the successor;
//        the compute loop drains the queue with prefetcht1 (into L2), GGML_K7_PFQ KB after each non-matmul node.
//   & 2  thread 0 doesn't drain after nodes (it runs the one-row ops)
//   & 4  drain after a node only within GGML_K7_LA nodes of the next ternary matmul (hook in apply_k7.py)
//   & 8  a thread waiting in ggml_barrier prefetches two queued lines per spin instead of pausing (needs the
//        non-OpenMP threadpool, -DGGML_OPENMP=OFF: the OpenMP barrier can't be hooked)
static int ptq1p_k7_cache = -1, ptq1p_k7_pfq = 32 << 10, ptq1p_k7_la = 4, ptq1p_k7_S = 1;
static inline int ptq1p_k7(void) {
    if (ptq1p_k7_cache < 0) {
        const char * q = getenv("GGML_K7_PFQ"); if (q) ptq1p_k7_pfq = atoi(q) << 10;
        const char * l = getenv("GGML_K7_LA"); if (l) ptq1p_k7_la = atoi(l);
        const char * n = getenv("GGML_K7_S"); if (n && atoi(n) > 0) ptq1p_k7_S = atoi(n);
        const char * s = getenv("GGML_K7"); ptq1p_k7_cache = s ? atoi(s) : 0;
    }
    return ptq1p_k7_cache;
}
static inline int64_t ptq1p_m1_ch(const ptq1p_entry * e, int nth) {   // the M = 1 chunk size (panels)
    int64_t ch = PTQ1F_MIN(8, (e->npanel + 8 * nth - 1) / (8 * nth));
    if (ptq1p_mode7()) ch = (ch + 1) / 2 * 2;   // whole panel pairs per chunk
    return ch;
}
static ptq1p_entry * ptq1p_k7_last;
static __thread const ptq1p_entry * ptq1p_k7_e;          // the queued entry
static __thread int64_t ptq1p_k7_c, ptq1p_k7_clast, ptq1p_k7_ch;
static __thread int ptq1p_k7_nth;
static __thread const uint8_t * ptq1p_k7_p, * ptq1p_k7_end;
static void ptq1p_k7_set(int64_t c) {
    const ptq1p_entry * e = ptq1p_k7_e;
    ptq1p_k7_c = c;
    ptq1p_k7_p = e->packed + PTQ1F_MIN(e->npanel, c * ptq1p_k7_ch) * e->panel_stride;
    ptq1p_k7_end = e->packed + PTQ1F_MIN(e->npanel, (c + 1) * ptq1p_k7_ch) * e->panel_stride;
}
static void ptq1p_k7_queue(const ptq1p_entry * s, int ith, int nth) {
    ptq1p_k7_e = NULL; ptq1p_k7_p = ptq1p_k7_end = NULL;
    if (!s || !s->packed) return;
    const int64_t ch = ptq1p_m1_ch(s, nth), nch = (s->npanel + ch - 1) / ch;
    if (ith >= nch) return;
    ptq1p_k7_e = s; ptq1p_k7_ch = ch; ptq1p_k7_nth = nth;
    ptq1p_k7_clast = PTQ1F_MIN(nch - 1, ith + (int64_t) (ptq1p_k7_S - 1) * nth);
    ptq1p_k7_set(ith);
}
static inline void ptq1p_k7_pf(int64_t lines) {
    while (lines > 0) {
        if (ptq1p_k7_p >= ptq1p_k7_end) {
            if (!ptq1p_k7_e || ptq1p_k7_c + ptq1p_k7_nth > ptq1p_k7_clast) { ptq1p_k7_e = NULL; return; }
            ptq1p_k7_set(ptq1p_k7_c + ptq1p_k7_nth);
            continue;
        }
        _mm_prefetch((const char *) ptq1p_k7_p, _MM_HINT_T1);
        ptq1p_k7_p += 64; --lines;
    }
}
static void ptq1p_k7_drain(int ith) {
    if (!ptq1p_k7_e || ((ptq1p_k7_cache & 2) && ith == 0)) return;
    ptq1p_k7_pf(ptq1p_k7_pfq / 64);
}
static void ptq1p_k7_spin(void) {   // one iteration of a barrier wait
    if (ptq1p_k7_cache > 0 && (ptq1p_k7_cache & 8) && ptq1p_k7_e) ptq1p_k7_pf(2); else _mm_pause();
}

static bool ptq1p_mul_mat(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    const int ith = params->ith, nth = params->nth;
    const int64_t k = src0->ne[0], rows = src0->ne[1], m = src1->ne[1];
    if (ptq1p_prof_on < 0 && ith == 0) {
        const char * pe = getenv("GGML_PTQ1_PROF");
        ptq1p_prof_on = pe && atoi(pe);
        if (ptq1p_prof_on) atexit(ptq1p_prof_dump);
    }
    const bool prof = ith == 0 && ptq1p_prof_on > 0;
    double pt0 = prof ? ptq1p_prof_now() : 0, pt1 = 0;
    if (ptq1p_tr_on < 0 && ith == 0) {
        const char * te = getenv("GGML_PTQ1_TRACE");
        ptq1p_tr_nth = nth; __atomic_store_n(&ptq1p_tr_on, te && atoi(te) ? 1 : 0, __ATOMIC_RELEASE);
        if (ptq1p_tr_on) atexit(ptq1p_tr_dump);
    }
    bool tr = m == 1 && nth <= 64 && __atomic_load_n(&ptq1p_tr_on, __ATOMIC_ACQUIRE) > 0 && ptq1p_tr_seen[ith]++ > 0;   // from the 2nd call: every thread has seen the flag
    double tr0 = 0, tr1 = 0, tr2 = 0, tr3 = 0;
    if (tr) { tr0 = ptq1p_prof_now(); ptq1p_tr_enter[ith] = tr0; }

    ptq1p_entry * e = NULL;
    // entries are only ever prepended (release store, under the mutex) and never freed, so a reader can walk
    // the list from an acquire load of the head without taking the lock
    for (ptq1p_entry * it = __atomic_load_n(&ptq1p_list, __ATOMIC_ACQUIRE); it; it = it->next) if (it->src == src0->data) { e = it; break; }
    bool missed = false;
    if (!e) {
        missed = true;
        ggml_barrier(params->threadpool);
        if (ith == 0) {
            ptq1p_entry * ne = calloc(1, sizeof(*ne));
#if PTQ1V_OK
            if (ptq1p_mode5()) ptq1v_init_entry(ne, src0->data, rows, k); else
#endif
            ptq1p_init_entry(ne, src0->data, rows, k);
            ne->packed = ptq1p_alloc_packed((size_t) (ne->panel_stride * ne->npanel));
            pthread_mutex_lock(&ptq1f_mu); ne->next = ptq1p_list; __atomic_store_n(&ptq1p_list, ne, __ATOMIC_RELEASE); pthread_mutex_unlock(&ptq1f_mu);
        }
        ggml_barrier(params->threadpool);
        pthread_mutex_lock(&ptq1f_mu);
        for (ptq1p_entry * it = ptq1p_list; it; it = it->next) if (it->src == src0->data) { e = it; break; }
        pthread_mutex_unlock(&ptq1f_mu);
        const int64_t per = (e->npanel + nth - 1) / nth;
#if PTQ1V_OK
        if (ptq1p_mode5()) ptq1v_repack_panels(e, PTQ1F_MIN(e->npanel, ith * per), PTQ1F_MIN(e->npanel, (ith + 1) * per)); else
#endif
        ptq1p_repack_panels(e, PTQ1F_MIN(e->npanel, ith * per), PTQ1F_MIN(e->npanel, (ith + 1) * per));
        // GGML_PTQ1_DROP (ptq1-drop.h): the last thread to finish its panels gives the GGUF pages back
        if (__atomic_add_fetch(&e->ndone, 1, __ATOMIC_ACQ_REL) == nth)
            ptq1d_drop(src0->data, ggml_nbytes(src0), strcmp(src0->name, "x1_draft_head") == 0);
    }
    if (missed) tr = false;   // first-use repack: not steady state
#if PTQ1V_OK
    const size_t cb = ptq1p_mode5() ? ptq1v_col_bytes(k) : ptq1p_col_bytes(k);
#else
    const size_t cb = ptq1p_col_bytes(k);   // AVX-512 VNNI without VBMI (e.g. Cascade Lake): no VBMI layout
#endif
    uint8_t * acts = (uint8_t *) (((uintptr_t) params->wdata + 63) & ~(uintptr_t) 63);
    if (prof) { pt1 = ptq1p_prof_now(); ptq1p_prof_t[missed ? 4 : 0] += pt1 - pt0; pt0 = pt1; }
    const bool one = ptq1f_mode() >= 4 && m == 1;   // modes 4..7 share the decode token's work (a build without VBMI runs mode 4 here)
    const bool small = ptq1p_mode6() && m > 1 && m <= PTQ1P_TB;         // mode 6: a verify pass of 2..8 tokens gets the same treatment
    const bool big = ptq1k4_on() && m > PTQ1P_TB;                        // GGML_PTQ1_K4=1: prefill, the chunked GEMM + AVX-512 quantizer
    if (one || small || big) ggml_threadpool_chunk_set(params->threadpool, nth);
    const bool k7 = one && (ptq1p_k7() & 1) && (ptq1p_mode5() || e->nck % 8 == 0);
    if (k7) {   // K7: learn the call order; a queue still left for this call keeps prefetching (inner barrier spin)
        if (ith == 0) { if (ptq1p_k7_last && ptq1p_k7_last != e) __atomic_store_n(&ptq1p_k7_last->succ, e, __ATOMIC_RELAXED); ptq1p_k7_last = e; }
        if (ptq1p_k7_e != e) ptq1p_k7_e = NULL;
        ggml_threadpool_chunk_set(params->threadpool, ptq1p_k7_S * nth);
    }
#if PTQ1V_OK
    const bool fl128 = small && ptq1p_fl128();
    if (fl128) {          // FL128: every thread quantizes a slice of the 128-wide groups of every token
        const int64_t ng = (k + 127) / 128, per = (ng + nth - 1) / nth;
        for (int64_t t = 0; t < m; ++t)
            ptq1k4_quantize_part128((const float *) ((const char *) src1->data + t * src1->nb[1]), k, e->kpad, acts + t * cb,
                                    PTQ1F_MIN(ng, ith * per), PTQ1F_MIN(ng, (ith + 1) * per));
    } else
#endif
    if (one || small) {   // every thread quantizes a slice of the 32-wide blocks of every token (1 token: 7 threads no longer idle)
        const int64_t nb = k / 32, per = (nb + nth - 1) / nth;
        for (int64_t t = 0; t < m; ++t)
#if PTQ1V_OK
            if (ptq1p_mode5() && (ptq1p_k6() & 8))
                ptq1k6_quantize_part((const float *) ((const char *) src1->data + t * src1->nb[1]), k, e->kpad, acts + t * cb,
                                     PTQ1F_MIN(nb, ith * per), PTQ1F_MIN(nb, (ith + 1) * per), 1);
            else
#endif
            ptq1p_quantize_col_part_kp((const float *) ((const char *) src1->data + t * src1->nb[1]), k, e->kpad, acts + t * cb,
                                       PTQ1F_MIN(nb, ith * per), PTQ1F_MIN(nb, (ith + 1) * per), ptq1p_mode5());
#if PTQ1V_OK
    } else if (big) {
        for (int64_t t = ith; t < m; t += nth)
            (ptq1k4_on() >= 2 ? ptq1k4_quantize_col128 : ptq1k4_quantize_col)((const float *) ((const char *) src1->data + t * src1->nb[1]), k, e->kpad, acts + t * cb);
#endif
    } else {
        for (int64_t t = ith; t < m; t += nth)
            ptq1p_quantize_col_part_kp((const float *) ((const char *) src1->data + t * src1->nb[1]), k, e->kpad, acts + t * cb, 0, k / 32, ptq1p_mode5());
    }
    if (prof) { pt1 = ptq1p_prof_now(); ptq1p_prof_t[1] += pt1 - pt0; pt0 = pt1; }
    if (tr) { tr1 = ptq1p_prof_now(); ptq1p_tr_arrive[ith] = tr1; }
    ggml_barrier(params->threadpool);
    if (prof) { pt1 = ptq1p_prof_now(); ptq1p_prof_t[2] += pt1 - pt0; pt0 = pt1; }
    if (tr) {
        tr2 = ptq1p_prof_now();
        double emin = 1e300, amin = 1e300;
        for (int t = 0; t < nth; ++t) { if (ptq1p_tr_enter[t] < emin) emin = ptq1p_tr_enter[t]; if (ptq1p_tr_arrive[t] < amin) amin = ptq1p_tr_arrive[t]; }
        ptq1p_tr_acc[ith][0] += tr0 - emin; ptq1p_tr_acc[ith][1] += tr1 - tr0; ptq1p_tr_acc[ith][2] += tr1 - amin; ptq1p_tr_acc[ith][3] += tr2 - tr1;
        if (tr0 - emin > ptq1p_tr_max[ith][0]) ptq1p_tr_max[ith][0] = tr0 - emin;
        if (tr2 - tr1 > ptq1p_tr_max[ith][1]) ptq1p_tr_max[ith][1] = tr2 - tr1;
    }
    if (one && (ptq1p_mode5() || e->nck % 8 == 0)) {
        // dynamic chunks: a thread that finishes early takes more panels instead of idling at the next barrier
        const int64_t ch = ptq1p_m1_ch(e, nth);
        const int64_t nch = (e->npanel + ch - 1) / ch;
        const int nst = k7 ? ptq1p_k7_S : 1;   // K7: the first nst chunks of every thread are static (ith, ith + nth, ...)
        int64_t c = ith;
        for (int st = 0; c < nch; ++st) {
#if PTQ1V_OK
            if (ptq1p_mode7()) ptq1v_panels7(e, acts, (float *) dst->data, c * ch, PTQ1F_MIN(e->npanel, (c + 1) * ch)); else
#endif
            for (int64_t p = c * ch; p < PTQ1F_MIN(e->npanel, (c + 1) * ch); ++p) {
#if PTQ1V_OK
                if (ptq1p_mode5()) ptq1v_panel1(e, e->packed + p * e->panel_stride, acts, (float *) dst->data, p * PTQ1P_ROWS); else
#endif
                ptq1p_panel1(e, e->packed + p * e->panel_stride, acts, (float *) dst->data, p * PTQ1P_ROWS);
            }
            c = st + 1 < nst ? c + nth : ggml_threadpool_chunk_add(params->threadpool, 1);
        }
#if PTQ1V_OK
    } else if (small) {   // mode 6, M = 2..8: dynamic panel chunks, token tiles of PTQ1V_TBM, one chain per token
        const int64_t ch = PTQ1F_MIN(8, (e->npanel + 8 * nth - 1) / (8 * nth));
        const int64_t nch = (e->npanel + ch - 1) / ch;
        for (int64_t c = ith; c < nch; c = ggml_threadpool_chunk_add(params->threadpool, 1))
            (fl128 ? ptq1v_gemm_m_fl4 : ptq1v_gemm_m)(e, acts, cb, m, c * ch, PTQ1F_MIN(e->npanel, (c + 1) * ch), (float *) dst->data, dst->nb[1] / sizeof(float));
    } else if (big) {     // K4: dynamic groups of PTQ1K4_G panels (even, so the 2-panel register tiles pair up), all m tokens per group
        const int64_t ch = PTQ1F_MIN(PTQ1K4_G, ((e->npanel + 8 * nth - 1) / (8 * nth) + 1) / 2 * 2);
        const int64_t nch = (e->npanel + ch - 1) / ch;
        for (int64_t c = ith; c < nch; c = ggml_threadpool_chunk_add(params->threadpool, 1))
            (ptq1k4_on() >= 2 ? ptq1k4_gemm2 : ptq1k4_gemm)(e, acts, cb, m, c * ch, PTQ1F_MIN(e->npanel, (c + 1) * ch), (float *) dst->data, dst->nb[1] / sizeof(float));
#endif
    } else
        for (int64_t p = ith; p < e->npanel; p += nth) {
#if PTQ1V_OK
            if (ptq1p_mode5()) ptq1v_gemm(e, acts, cb, m, p, p + 1, (float *) dst->data, dst->nb[1] / sizeof(float)); else
#endif
            ptq1p_gemm(e, acts, cb, m, p, p + 1, (float *) dst->data, dst->nb[1] / sizeof(float));
        }
    if (k7) ptq1p_k7_queue(__atomic_load_n(&e->succ, __ATOMIC_RELAXED), ith, nth);
    if (prof) { ptq1p_prof_t[3] += ptq1p_prof_now() - pt0; ++ptq1p_prof_n; }
    if (tr) { tr3 = ptq1p_prof_now(); ptq1p_tr_acc[ith][4] += tr3 - tr2; ptq1p_tr_acc[ith][5] += tr3 - tr0; ++ptq1p_tr_n[ith]; }
#if PTQ1V_OK
    static int k4dbg = -1;
    if (big && ith == 0 && (k4dbg >= 0 ? k4dbg : (k4dbg = getenv("GGML_PTQ1_K4DBG") != NULL))) {   // dev: thread 0's first panel group vs column 0, per call
        static int ncall = 0;
        if (ncall++ < 12) {
            const float * x = (const float *) src1->data; float xmax = 0; for (int64_t i = 0; i < k; ++i) xmax = fmaxf(xmax, fabsf(x[i]));
            const int64_t ngq = ptq1k4_on() >= 2 ? (e->kpad + 127) / 128 : e->kpad / 32;
            const float * dy = (const float *) (acts + e->kpad); const int32_t * as = (const int32_t *) (dy + ngq);
            const float * d = (const float *) dst->data; double dmax = 0; int nnan = 0;
            for (int64_t t = 0; t < m; ++t) for (int64_t r = 0; r < 16 && r < rows; ++r) { const float v = d[t * (dst->nb[1] / 4) + r]; if (v != v) ++nnan; else dmax = fmax(dmax, fabs(v)); }
            fprintf(stderr, "K4DBG call %d k=%lld rows=%lld m=%lld kpad=%lld ng=%lld xmax=%g dy0=%g as0=%d q0..3=%d,%d,%d,%d dst(16 rows x m): max=%g nan=%d\n",
                    ncall, (long long) k, (long long) rows, (long long) m, (long long) e->kpad, (long long) ngq, xmax, dy[0], as[0],
                    ((const int8_t *) acts)[0], ((const int8_t *) acts)[1], ((const int8_t *) acts)[2], ((const int8_t *) acts)[3], dmax, nnan);
        }
    }
#endif
    return true;
}
#endif
#endif
