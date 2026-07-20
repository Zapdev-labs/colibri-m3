/* observability.h — engine-side telemetry hooks (f15).
 *
 * Emits a structured JSON-per-event log to a file path set via M3_TELEMETRY_PATH.
 * Events:
 *   load_complete    — model load finished (RSS, hits, miss, msa_layers)
 *   prefill_done     — prefill phase wall clock, prompt length
 *   decode_step      — per-token decode time, expert cache hit/miss
 *   throughput        — sustained tok/s rolling average over last 32 tokens
 *   nan_check         — NaN/Inf detection events
 *   run_complete     — totals: tokens, wall clock, tok/s, expert hit rate
 *
 * Schema is intentionally a flat, line-delimited JSON object per event so
 * downstream tools (tools/observability.py) can ingest without a JSON parser.
 * Each line: {"ts": <float epoch>, "event": "<name>", "kv": {...}}.
 *
 * When M3_TELEMETRY_PATH is unset, observability is a no-op (zero overhead).
 * This is the cross-engine observability surface; the llama.cpp fork's
 * msa-runtime.cpp emits the same schema so a single dashboard can compare
 * both engines side by side.
 */
#ifndef COLIBRI_M3_OBSERVABILITY_H
#define COLIBRI_M3_OBSERVABILITY_H

#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

typedef struct {
    FILE *fp;
    pthread_mutex_t mu;
    int enabled;
    double t0;
} Telemetry;

static Telemetry g_telem = { NULL, PTHREAD_MUTEX_INITIALIZER, 0, 0.0 };

static inline double telem_now(void) {
    struct timespec t;
    clock_gettime(CLOCK_REALTIME, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static inline double telem_uptime(void) {
    return telem_now() - g_telem.t0;
}

/* Initialise telemetry from M3_TELEMETRY_PATH env var. Safe to call multiple
 * times; the first call opens the file, subsequent calls are no-ops. */
static inline void telem_init(void) {
    if (g_telem.enabled) return;
    const char *p = getenv("M3_TELEMETRY_PATH");
    if (!p || !p[0]) return;
    g_telem.fp = fopen(p, "w");
    if (!g_telem.fp) return;
    g_telem.t0 = telem_now();
    g_telem.enabled = 1;
}

static inline void telem_event(const char *event, const char *kv_json) {
    if (!g_telem.enabled) return;
    pthread_mutex_lock(&g_telem.mu);
    fprintf(g_telem.fp, "{\"ts\": %.6f, \"event\": \"%s\", %s}\n",
            telem_uptime(), event, kv_json ? kv_json : "\"_\":{}");
    fflush(g_telem.fp);
    pthread_mutex_unlock(&g_telem.mu);
}

/* Convenience helpers for the common event shapes used in engine.c. */
static inline void telem_load_complete(int layers, int experts, long rss_kb,
                                        int msa_layers, int kv_i8, int planar) {
    if (!g_telem.enabled) return;
    char buf[256];
    snprintf(buf, sizeof(buf),
             "\"event\":\"load_complete\",\"layers\":%d,\"experts\":%d,\"rss_kb\":%ld,\"msa_layers\":%d,\"kv_i8\":%d,\"planar\":%d",
             layers, experts, rss_kb, msa_layers, kv_i8, planar);
    telem_event("load_complete", buf);
}

static inline void telem_decode_step(int pos, double step_ms, int hit, int miss,
                                      double rolling_tokps) {
    if (!g_telem.enabled) return;
    char buf[256];
    snprintf(buf, sizeof(buf),
             "\"pos\":%d,\"step_ms\":%.3f,\"hit\":%d,\"miss\":%d,\"tokps\":%.3f",
             pos, step_ms, hit, miss, rolling_tokps);
    telem_event("decode_step", buf);
}

static inline void telem_run_complete(int tokens, double wall_s, double tokps,
                                       double hit_rate, const char *engine_id) {
    if (!g_telem.enabled) return;
    char buf[256];
    snprintf(buf, sizeof(buf),
             "\"tokens\":%d,\"wall_s\":%.3f,\"tokps\":%.3f,\"hit_rate\":%.4f,\"engine\":\"%s\"",
             tokens, wall_s, tokps, hit_rate, engine_id ? engine_id : "colibri-m3");
    telem_event("run_complete", buf);
}

#endif /* COLIBRI_M3_OBSERVABILITY_H */
