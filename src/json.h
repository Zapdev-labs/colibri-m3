/* Minimal JSON parser for safetensors headers + config.json */
#ifndef COLI_JSON_H
#define COLI_JSON_H
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef enum { J_NULL, J_BOOL, J_NUM, J_STR, J_ARR, J_OBJ } jtype;
typedef struct jval {
    jtype t;
    double num;
    int boolean;
    char *str;
    struct jval **kids;
    char **keys;
    int len;
} jval;

typedef struct { const char *s; } jparser;

static jval *j_new(jtype t) {
    jval *v = (jval *)calloc(1, sizeof(jval));
    v->t = t;
    return v;
}
static void j_ws(jparser *p) {
    while (*p->s && isspace((unsigned char)*p->s)) p->s++;
}
static char *j_str(jparser *p) {
    p->s++;
    char buf[65536];
    int n = 0;
    while (*p->s && *p->s != '"') {
        char c = *p->s++;
        if (c == '\\' && *p->s) {
            char e = *p->s++;
            if (e == 'n') c = '\n';
            else if (e == 't') c = '\t';
            else if (e == 'r') c = '\r';
            else if (e == '"' || e == '\\' || e == '/') c = e;
            else c = e;
        }
        if (n < (int)sizeof(buf) - 1) buf[n++] = c;
    }
    if (*p->s == '"') p->s++;
    char *o = (char *)malloc((size_t)n + 1);
    memcpy(o, buf, (size_t)n);
    o[n] = 0;
    return o;
}
static jval *j_parse(jparser *p);
static jval *j_parse_val(jparser *p) {
    j_ws(p);
    if (*p->s == '"') {
        jval *v = j_new(J_STR);
        v->str = j_str(p);
        return v;
    }
    if (*p->s == '{') {
        p->s++;
        jval *v = j_new(J_OBJ);
        int cap = 8;
        v->keys = (char **)malloc((size_t)cap * sizeof(char *));
        v->kids = (jval **)malloc((size_t)cap * sizeof(jval *));
        for (;;) {
            j_ws(p);
            if (*p->s == '}') {
                p->s++;
                break;
            }
            if (*p->s == ',') {
                p->s++;
                j_ws(p);
            }
            if (*p->s != '"') break;
            char *k = j_str(p);
            j_ws(p);
            if (*p->s == ':') p->s++;
            jval *child = j_parse_val(p);
            if (v->len >= cap) {
                cap *= 2;
                v->keys = (char **)realloc(v->keys, (size_t)cap * sizeof(char *));
                v->kids = (jval **)realloc(v->kids, (size_t)cap * sizeof(jval *));
            }
            v->keys[v->len] = k;
            v->kids[v->len++] = child;
        }
        return v;
    }
    if (*p->s == '[') {
        p->s++;
        jval *v = j_new(J_ARR);
        int cap = 8;
        v->kids = (jval **)malloc((size_t)cap * sizeof(jval *));
        for (;;) {
            j_ws(p);
            if (*p->s == ']') {
                p->s++;
                break;
            }
            if (*p->s == ',') {
                p->s++;
                j_ws(p);
            }
            jval *child = j_parse_val(p);
            if (v->len >= cap) {
                cap *= 2;
                v->kids = (jval **)realloc(v->kids, (size_t)cap * sizeof(jval *));
            }
            v->kids[v->len++] = child;
        }
        return v;
    }
    if (!strncmp(p->s, "true", 4)) {
        p->s += 4;
        jval *v = j_new(J_BOOL);
        v->boolean = 1;
        return v;
    }
    if (!strncmp(p->s, "false", 5)) {
        p->s += 5;
        jval *v = j_new(J_BOOL);
        v->boolean = 0;
        return v;
    }
    if (!strncmp(p->s, "null", 4)) {
        p->s += 4;
        return j_new(J_NULL);
    }
    char *end = NULL;
    double num = strtod(p->s, &end);
    if (end == p->s) return j_new(J_NULL);
    p->s = end;
    jval *v = j_new(J_NUM);
    v->num = num;
    return v;
}
static jval *json_parse(const char *s, char **arena_out) {
    (void)arena_out;
    jparser p = {s};
    return j_parse_val(&p);
}
static jval *json_get(jval *o, const char *key) {
    if (!o || o->t != J_OBJ) return NULL;
    for (int i = 0; i < o->len; i++)
        if (!strcmp(o->keys[i], key)) return o->kids[i];
    return NULL;
}
#endif
