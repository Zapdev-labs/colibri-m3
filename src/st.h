/* Safetensors shard index — pread + DONTNEED (no mmap RSS trap) */
#ifndef COLI_ST_H
#define COLI_ST_H
#define _GNU_SOURCE
#include <dirent.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include "json.h"

typedef struct {
    char *name;
    int fd;
    int64_t off, nbytes, numel;
    int dtype; /* 0 bf16 1 f16 2 f32 3 u8/i8 */
} st_tensor;

typedef struct {
    st_tensor *t;
    int n, cap;
    int fds[512];
    char *paths[512];
    int nfd;
    int *hidx, hcap;
    /* mmap regions per shard file */
    void *mmaps[512];      /* mmap base address (NULL = not mapped) */
    size_t mmap_sizes[512]; /* size of each mmap region */
} shards;

static uint64_t st_hash(const char *s) {
    uint64_t h = 1469598103934665603ULL;
    while (*s) {
        h ^= (unsigned char)*s++;
        h *= 1099511628211ULL;
    }
    return h;
}
static int st_dtype(const char *s) {
    if (!strcmp(s, "BF16")) return 0;
    if (!strcmp(s, "F16")) return 1;
    if (!strcmp(s, "F32")) return 2;
    if (!strcmp(s, "U8") || !strcmp(s, "I8")) return 3;
    fprintf(stderr, "dtype %s\n", s);
    exit(1);
}
static int st_open(shards *S, const char *path) {
    for (int i = 0; i < S->nfd; i++)
        if (!strcmp(S->paths[i], path)) return S->fds[i];
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        perror(path);
        exit(1);
    }
    if (S->nfd >= 512) {
        fprintf(stderr, "too many shards\n");
        exit(1);
    }
    S->fds[S->nfd] = fd;
    S->paths[S->nfd] = strdup(path);
    S->mmaps[S->nfd] = NULL;
    S->mmap_sizes[S->nfd] = 0;
    return S->fds[S->nfd++];
}

/* Forward declarations (defined below, but needed by st_mmap_ptr). */
static int st_find(shards *S, const char *name);
static int64_t st_nbytes(shards *S, const char *name);

/* mmap a shard file for zero-copy access. Returns base pointer or NULL. */
static void *st_mmap_file(shards *S, int fd_idx) {
    if (fd_idx < 0 || fd_idx >= S->nfd) return NULL;
    if (S->mmaps[fd_idx]) return S->mmaps[fd_idx];
    /* Get file size */
    off_t sz = lseek(S->fds[fd_idx], 0, SEEK_END);
    if (sz < 0) return NULL;
    void *base = mmap(NULL, (size_t)sz, PROT_READ, MAP_PRIVATE | MAP_NORESERVE,
                      S->fds[fd_idx], 0);
    if (base == MAP_FAILED) return NULL;
    S->mmaps[fd_idx] = base;
    S->mmap_sizes[fd_idx] = (size_t)sz;
    /* Advise sequential access for initial population, then let OS manage */
    posix_fadvise(S->fds[fd_idx], 0, 0, POSIX_FADV_WILLNEED);
    /* Interleave pages across NUMA nodes for balanced memory bandwidth */
#if defined(__linux__) && defined(__NR_mbind)
    {
        unsigned long nodemask = 3;  /* nodes 0 and 1 */
        syscall(__NR_mbind, base, (size_t)sz, 1 /* MPOL_INTERLEAVE */,
               &nodemask, 64, 0);
    }
#endif
    return base;
}

/* Get a pointer to a tensor's data via mmap (zero-copy). */
static void *st_mmap_ptr(shards *S, const char *name) {
    int i = st_find(S, name);
    if (i < 0) return NULL;
    st_tensor *t = &S->t[i];
    /* Find which mmap this fd belongs to */
    for (int f = 0; f < S->nfd; f++) {
        if (S->fds[f] == t->fd) {
            void *base = st_mmap_file(S, f);
            if (!base) return NULL;
            return (char *)base + t->off;
        }
    }
    return NULL;
}
static void st_add(shards *S, const char *name, int fd, int64_t off, int64_t nb, int dt, int64_t n) {
    if (S->n >= S->cap) {
        S->cap = S->cap ? S->cap * 2 : 4096;
        S->t = (st_tensor *)realloc(S->t, (size_t)S->cap * sizeof(st_tensor));
    }
    st_tensor *x = &S->t[S->n++];
    x->name = strdup(name);
    x->fd = fd;
    x->off = off;
    x->nbytes = nb;
    x->dtype = dt;
    x->numel = n;
}
static void st_rehash(shards *S) {
    int cap = 1;
    while (cap < S->n * 2) cap <<= 1;
    int *h = (int *)malloc((size_t)cap * sizeof(int));
    for (int i = 0; i < cap; i++) h[i] = -1;
    for (int i = 0; i < S->n; i++) {
        uint64_t k = st_hash(S->t[i].name) & (uint64_t)(cap - 1);
        while (h[k] >= 0) k = (k + 1) & (uint64_t)(cap - 1);
        h[k] = i;
    }
    free(S->hidx);
    S->hidx = h;
    S->hcap = cap;
}
static int st_find(shards *S, const char *name) {
    if (!S->hidx) return -1;
    uint64_t k = st_hash(name) & (uint64_t)(S->hcap - 1);
    for (;;) {
        int i = S->hidx[k];
        if (i < 0) return -1;
        if (!strcmp(S->t[i].name, name)) return i;
        k = (k + 1) & (uint64_t)(S->hcap - 1);
    }
}
static int st_has(shards *S, const char *name) { return st_find(S, name) >= 0; }
static int64_t st_nbytes(shards *S, const char *name) {
    int i = st_find(S, name);
    return i < 0 ? -1 : S->t[i].nbytes;
}
static void st_index_file(shards *S, const char *path) {
    int fd = st_open(S, path);
    uint64_t hlen = 0;
    if (read(fd, &hlen, 8) != 8) {
        perror("hdr");
        exit(1);
    }
    char *buf = (char *)malloc(hlen + 1);
    if (read(fd, buf, hlen) != (ssize_t)hlen) {
        perror("hdr body");
        exit(1);
    }
    buf[hlen] = 0;
    char *ar = NULL;
    jval *root = json_parse(buf, &ar);
    free(buf);
    int64_t data0 = 8 + (int64_t)hlen;
    for (int i = 0; i < root->len; i++) {
        if (!strcmp(root->keys[i], "__metadata__")) continue;
        jval *t = root->kids[i];
        jval *dt = json_get(t, "dtype");
        jval *sh = json_get(t, "shape");
        jval *off = json_get(t, "data_offsets");
        if (!dt || !sh || !off || off->len < 2) continue;
        int64_t n = 1;
        for (int k = 0; k < sh->len; k++) n *= (int64_t)sh->kids[k]->num;
        int64_t a = (int64_t)off->kids[0]->num, b = (int64_t)off->kids[1]->num;
        st_add(S, root->keys[i], fd, data0 + a, b - a, st_dtype(dt->str), n);
    }
}
static int st_is_out(const char *n) {
    return !strncmp(n, "out-", 4) && strstr(n, ".safetensors") != NULL;
}
static void st_init(shards *S, const char *dir) {
    memset(S, 0, sizeof(*S));
    DIR *d = opendir(dir);
    if (!d) {
        perror(dir);
        exit(1);
    }
    struct dirent *e;
    char path[4096];
    while ((e = readdir(d))) {
        if (!st_is_out(e->d_name)) continue;
        snprintf(path, sizeof(path), "%s/%s", dir, e->d_name);
        st_index_file(S, path);
    }
    closedir(d);
    st_rehash(S);
    fprintf(stderr, "[st] indexed %d tensors in %d shards\n", S->n, S->nfd);
}
static void st_read(shards *S, const char *name, void *dst, int64_t n) {
    int i = st_find(S, name);
    if (i < 0) {
        fprintf(stderr, "missing tensor: %s\n", name);
        exit(1);
    }
    st_tensor *t = &S->t[i];
    if (n > t->nbytes) n = t->nbytes;
    /* Prefetch: tell kernel we need these pages soon (readahead). */
#if defined(__linux__)
    posix_fadvise(t->fd, t->off, n, POSIX_FADV_WILLNEED);
#endif
    int64_t got = 0;
    while (got < n) {
        ssize_t r = pread(t->fd, (char *)dst + got, (size_t)(n - got), t->off + got);
        if (r <= 0) {
            perror("pread");
            exit(1);
        }
        got += r;
    }
    /* DO NOT call POSIX_FADV_DONTNEED — keep pages in OS page cache so
     * repeated expert loads hit cache instead of disk. With 368GB RAM and
     * a 212GB model, the full model fits in page cache. */
}
static st_tensor *st_get(shards *S, const char *name) {
    int i = st_find(S, name);
    return i < 0 ? NULL : &S->t[i];
}
#endif
