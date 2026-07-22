/* test_numa.c — f10: NUMA topology discovery + thread pinning smoke test.
 *
 * Build: gcc -DTESTING -O3 -march=native -fopenmp -Wall -Wextra \
 *          -Wno-unused-function -Isrc c/tests/test_numa.c -o c/tests/test_numa -lm
 */
#ifndef TESTING
#define TESTING
#endif
#include "../src/engine.c"

#include <stdio.h>
#include <sched.h>

int main(void) {
    NumaTopo t;
    int rc = 0;
    if (!numa_discover(&t)) {
        printf("[test_numa] FAIL: numa_discover returned 0\n");
        return 1;
    }
    printf("[test_numa] n_cpus=%d n_nodes=%d pin=%d interleave=%d\n",
           t.n_cpus, t.n_nodes, t.pin_threads, t.interleave);
    int nodes_seen[8] = {0};
    for (int c = 0; c < t.n_cpus && c < MAX_CPUS; c++) {
        if (t.cpu_node[c] >= 0 && t.cpu_node[c] < 8)
            nodes_seen[t.cpu_node[c]]++;
    }
    int n_active_nodes = 0;
    for (int i = 0; i < 8; i++) if (nodes_seen[i] > 0) n_active_nodes++;
    printf("[test_numa] active NUMA nodes: %d (per-node cpu counts: %d, %d, %d, %d)\n",
           n_active_nodes, nodes_seen[0], nodes_seen[1], nodes_seen[2], nodes_seen[3]);

    int r = numa_bind_thread(0);
    printf("[test_numa] bind_thread(0) -> %d %s\n", r, r == 0 ? "OK" : "FAIL");
    if (r != 0) rc = 1;

    int cpu = sched_getcpu();
    printf("[test_numa] now running on CPU %d\n", cpu);

    int br = numa_disable_balancing();
    printf("[test_numa] disable_balancing -> %d %s\n", br,
           br == 0 ? "OK" : (br == -1 ? "NO_PERM (ok as non-root)" : "FAIL"));

    if (t.n_cpus <= 0) { printf("[test_numa] FAIL: n_cpus <= 0\n"); rc = 1; }
    if (n_active_nodes < 1) { printf("[test_numa] FAIL: no active NUMA nodes\n"); rc = 1; }
    printf("[test_numa] overall %s\n", rc ? "FAIL" : "PASS");
    return rc;
}
