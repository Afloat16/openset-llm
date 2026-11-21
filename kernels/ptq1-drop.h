// ptq1-drop.h: once a PTQ1_0 weight has been repacked into panels, the ternary path never reads its GGUF bytes again.
// GGML_PTQ1_DROP (default 1; 0 = off) then gives the source pages back with madvise(MADV_DONTNEED), but only when they
// lie in a read-only, shared, file-backed mapping (llama.cpp's model mmap). There, DONTNEED only unmaps clean page-cache
// pages; any later read refaults the same bytes from the file, so no reader can ever see different data. Anonymous
// memory (--no-mmap, copied tensors) is left alone, with one exception the caller vouches for (anon_ok): X1's draft
// head, a private copy of 65k rows of the output head that only ever feeds this matmul (if anything else read it, it
// would see zeros, which could only change draft proposals, never the verified output). Pages shared with a
// neighbouring tensor are kept (the range is rounded inward). Outputs unchanged by construction.
// GGML_PTQ1_DROP=2 logs each drop to stderr.
#pragma once
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#if defined(__linux__)
#include <unistd.h>
#include <sys/mman.h>
#endif

static int ptq1d_on_cache = -1;
static inline int ptq1d_on(void) {
    if (ptq1d_on_cache < 0) { const char * s = getenv("GGML_PTQ1_DROP"); ptq1d_on_cache = s ? atoi(s) : 1; }
    return ptq1d_on_cache;
}

#if defined(__linux__)
// 1 if [a, b) lies inside one mapping that is readable, not writable, shared and backed by a file (inode != 0);
// 2 if it lies inside one private anonymous mapping (no path); 0 otherwise
static inline int ptq1d_kind(uintptr_t a, uintptr_t b) {
    FILE * f = fopen("/proc/self/maps", "r");
    if (!f) return 0;
    char line[512]; int ok = 0;
    while (fgets(line, sizeof line, f)) {
        unsigned long lo, hi, off, ino; char perm[8], dev[32]; int n = 0;
        if (sscanf(line, "%lx-%lx %7s %lx %31s %lu %n", &lo, &hi, perm, &off, dev, &ino, &n) != 6) continue;
        if (a >= lo && a < hi) {
            if (b > hi) break;
            if (ino != 0 && perm[0] == 'r' && perm[1] == '-' && perm[3] == 's') ok = 1;
            else if (ino == 0 && perm[3] == 'p' && (line[n] == '\n' || line[n] == '\0')) ok = 2;
            break;
        }
    }
    fclose(f);
    return ok;
}
#endif

static inline void ptq1d_drop(const void * p, size_t n, int anon_ok) {
#if defined(__linux__)
    if (!p || !n || !ptq1d_on()) return;
    const uintptr_t pg = (uintptr_t) sysconf(_SC_PAGESIZE);
    const uintptr_t a = ((uintptr_t) p + pg - 1) & ~(pg - 1), b = ((uintptr_t) p + n) & ~(pg - 1);
    if (b <= a) return;
    const int kind = ptq1d_kind((uintptr_t) p, (uintptr_t) p + n);
    if (kind != 1 && !(kind == 2 && anon_ok)) return;
    const int rc = madvise((void *) a, b - a, MADV_DONTNEED);
    if (ptq1d_on() >= 2) fprintf(stderr, "ptq1 drop: %zu KiB %s source pages at %p (rc %d)\n", (size_t) (b - a) >> 10,
                                 kind == 1 ? "file" : "anon", p, rc);
#else
    (void) p; (void) n; (void) anon_ok;
#endif
}
