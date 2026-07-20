/* numa.h — NUMA-aware thread pinning and memory policy (f10).
 *
 * Uses only glibc-provided syscalls (no libnuma dependency), so it builds on
 * any Linux with libc-dev. Three facilities:
 *
 *   numa_setup()           — discover NUMA topology (cpu <-> node map).
 *   numa_bind_thread(cpu)  — pin the calling thread to a single CPU (and thus
 *                            a single NUMA node). Uses sched_setaffinity.
 *   numa_interleave_memory(addr, len, nodeset)
 *                          — interleave the given region's pages across the
 *                            specified NUMA nodes via mbind(MPOL_INTERLEAVE).
 *                            Falls back to a no-op on non-Linux.
 *
 *   numa_disable_balancing() — write 0 to /proc/sys/kernel/numa_balancing.
 *                            Best-effort; silently ignored if not root.
 *
 * The engine calls these from main() before load_model() so model pages are
 * interleaved across both sockets and worker threads are bound per-NUMA-node.
 * This is the documented Cascade Lake footgun (RESEARCH.md §5.7).
 */
#ifndef COLIBRI_M3_NUMA_H
#define COLIBRI_M3_NUMA_H

#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/syscall.h>

#define MAX_CPUS 1024

/* Linux MPOL constants (from linux/mempolicy.h); defined here so we don't
 * need libnuma-dev headers. */
#ifndef MPOL_INTERLEAVE
#define MPOL_INTERLEAVE 2
#endif
#ifndef MPOL_MF_MOVE
#define MPOL_MF_MOVE (1 << 1)
#endif

typedef struct {
    int n_cpus;             /* total logical CPUs online */
    int n_nodes;            /* NUMA nodes (best-effort: from /sys/devices/system/node/normal) */
    int cpu_node[MAX_CPUS]; /* cpu -> NUMA node (or -1 if unknown) */
    int n_workers;          /* configured worker thread count (OMP_NUM_THREADS or n_cpus) */
    int pin_threads;        /* 1 = bind OMP threads to specific CPUs (default if NUMA aware) */
    int interleave;         /* 1 = interleave model memory across all nodes (default) */
} NumaTopo;

/* Returns 1 on success, 0 on failure (topology stays zeroed). */
static inline int numa_discover(NumaTopo *t) {
    memset(t, 0, sizeof(*t));
    t->n_cpus = sysconf(_SC_NPROCESSORS_ONLN);
    if (t->n_cpus <= 0 || t->n_cpus > MAX_CPUS) t->n_cpus = 1;
    /* Default: assume interleave + pin both on (caller can disable via env). */
    t->interleave = 1;
    t->pin_threads = 1;
    /* Map cpu -> node by walking /sys/devices/system/cpu/cpuN/topology/physical_package_id.
     * Note: physical_package_id is the socket index, not the NUMA node index.
     * On Cascade Lake dual-socket these coincide (one node per socket). For
     * multi-node-per-socket hosts (Sapphire Rapids with sub-NUMA clustering),
     * we'd want /sys/devices/system/node/nodeN/cpulist instead. */
    for (int c = 0; c < t->n_cpus; c++) {
        char path[256];
        snprintf(path, sizeof(path),
                 "/sys/devices/system/cpu/cpu%d/topology/physical_package_id", c);
        FILE *f = fopen(path, "r");
        if (!f) { t->cpu_node[c] = -1; continue; }
        int node = -1;
        if (fscanf(f, "%d", &node) != 1) node = -1;
        fclose(f);
        t->cpu_node[c] = node;
        if (node >= 0 && node + 1 > t->n_nodes) t->n_nodes = node + 1;
    }
    if (t->n_nodes <= 0) t->n_nodes = 1;
    return 1;
}

/* Pin the calling thread to a single CPU. Returns 0 on success, -1 on error. */
static inline int numa_bind_thread(int cpu) {
#if defined(__linux__)
    if (cpu < 0) return -1;
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(cpu, &mask);
    return sched_setaffinity(0, sizeof(mask), &mask);
#else
    (void)cpu;
    return 0;
#endif
}

/* Interleave a memory region across all NUMA nodes. Best-effort: returns 0 on
 * success, -1 if mbind is unavailable or fails. The region must be page-aligned
 * for full coverage; unaligned tails are silently skipped. */
static inline int numa_interleave_memory(void *addr, size_t len, int n_nodes) {
#if defined(__linux__) && defined(__NR_mbind)
    if (n_nodes <= 0 || !addr || len == 0) return 0;
    unsigned long nodemask = 0;
    for (int n = 0; n < n_nodes && n < 8 * (int)sizeof(unsigned long); n++)
        nodemask |= (1UL << n);
    /* mbind with MPOL_INTERLEAVE on the full range, plus MPOL_MF_MOVE so
     * existing pages are migrated (used after a large malloc pre-touch). */
    long r = syscall(__NR_mbind, addr, len, MPOL_INTERLEAVE, &nodemask,
                     8 * sizeof(unsigned long), MPOL_MF_MOVE);
    return r == 0 ? 0 : -1;
#else
    (void)addr; (void)len; (void)n_nodes;
    return 0;
#endif
}

/* Best-effort: disable kernel NUMA balancing (writes /proc/sys/...).
 * Silently ignored if we don't have permission. Returns 0 on success. */
static inline int numa_disable_balancing(void) {
#if defined(__linux__)
    FILE *f = fopen("/proc/sys/kernel/numa_balancing", "w");
    if (!f) return -1;
    int rc = (fputs("0\n", f) >= 0) ? 0 : -1;
    fclose(f);
    return rc;
#else
    return 0;
#endif
}

/* Per-thread OMP callback: called from inside a parallel region with the
 * thread's logical rank (0..n-1). Pins to the corresponding CPU round-robin
 * across the topology. Returns the CPU index bound to. */
static inline int numa_pin_omp_thread(NumaTopo *t, int rank) {
    if (!t->pin_threads || t->n_cpus <= 0) return -1;
    int cpu = rank % t->n_cpus;
    numa_bind_thread(cpu);
    return cpu;
}

#endif /* COLIBRI_M3_NUMA_H */
