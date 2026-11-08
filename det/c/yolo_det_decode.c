/* yolo_det_decode.c -- mirrors det/yolo/decode.py (decode with pre_nms=256). */
#include <math.h>
#include "yolo_det_decode.h"

#define IN_W 640
#define IN_H 384
#define REG 16
#define PRE_NMS 256
#define MAXK 64
#define FALLBACK_TRIES 64   /* strongest anchors tried when nothing valid survives (decode.py) */

typedef struct { float s; int idx; } cand_t;

static const int GH[3] = {48, 24, 12}, GW[3] = {80, 40, 20}, STRIDE[3] = {8, 16, 32};

static int better(const cand_t *a, const cand_t *b) {           /* score desc, index asc */
    if (a->s != b->s) return a->s > b->s;
    return a->idx < b->idx;
}

static float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

/* global anchor index -> level, row, col */
static void locate(int idx, int *lv, int *i, int *j) {
    int l = 0;
    while (idx >= GH[l] * GW[l]) { idx -= GH[l] * GW[l]; l++; }
    *lv = l; *i = idx / GW[l]; *j = idx % GW[l];
}

static float score_of(const float *const outs[6], int idx) {
    int l, i, j; locate(idx, &l, &i, &j);
    return 1.f / (1.f + expf(-outs[2 * l + 1][i * GW[l] + j]));
}

/* box in network-input pixels */
static void box_of(const float *const outs[6], int idx, float b[4]) {
    int l, i, j; locate(idx, &l, &i, &j);
    const float *bx = outs[2 * l];
    int hw = GH[l] * GW[l], p = i * GW[l] + j;
    float d[4];
    for (int side = 0; side < 4; side++) {
        float mx = -1e30f, sum = 0.f, ex = 0.f;
        for (int k = 0; k < REG; k++) { float v = bx[(side * REG + k) * hw + p]; if (v > mx) mx = v; }
        for (int k = 0; k < REG; k++) { float e = expf(bx[(side * REG + k) * hw + p] - mx); sum += e; ex += e * k; }
        d[side] = ex / sum;
    }
    float cx = j + 0.5f, cy = i + 0.5f, s = (float)STRIDE[l];
    b[0] = (cx - d[0]) * s; b[1] = (cy - d[1]) * s; b[2] = (cx + d[2]) * s; b[3] = (cy + d[3]) * s;
}

int yolo_det_decode(const float *const outs[6], int src_w, int src_h,
                    float thr, int max_k, float nms_iou, int fallback, bear_box_t *out) {
    cand_t c[PRE_NMS], fb[FALLBACK_TRIES]; int nc = 0, nfb = 0, n_anchor = 0;
    float kept[MAXK][4]; int nk = 0, n = 0;
    if (max_k > MAXK) max_k = MAXK;
    if (max_k < 1 || src_w <= 0 || src_h <= 0) return 0;
    for (int l = 0; l < 3; l++) n_anchor += GH[l] * GW[l];
    for (int idx = 0; idx < n_anchor; idx++) {
        float s = score_of(outs, idx);
        cand_t x = {s, idx};
        if (fallback) {                            /* best FALLBACK_TRIES anchors at any score */
            int pos = nfb;
            while (pos > 0 && better(&x, &fb[pos - 1])) pos--;
            if (pos < FALLBACK_TRIES) {
                if (nfb < FALLBACK_TRIES) nfb++;
                for (int k = nfb - 1; k > pos; k--) fb[k] = fb[k - 1];
                fb[pos] = x;
            }
        }
        if (s < thr) continue;
        int pos = nc;
        while (pos > 0 && better(&x, &c[pos - 1])) pos--;
        if (pos >= PRE_NMS) continue;
        if (nc < PRE_NMS) nc++;
        for (int k = nc - 1; k > pos; k--) c[k] = c[k - 1];
        c[pos] = x;
    }
    {
        double sc = fmin((double)IN_W / src_w, (double)IN_H / src_h);
        int nw = (int)(src_w * sc), nh = (int)(src_h * sc);
        if (nw < 1) nw = 1;
        if (nh < 1) nh = 1;
        int px = (IN_W - nw) / 2, py = (IN_H - nh) / 2;
        float sx = (float)nw / src_w, sy = (float)nh / src_h;
        for (int k = 0; k < nc && nk < max_k; k++) {
            float b[4]; box_of(outs, c[k].idx, b);
            int ok = 1;
            for (int m = 0; m < nk && ok; m++) {
                float ix = fmaxf(0.f, fminf(b[2], kept[m][2]) - fmaxf(b[0], kept[m][0]));
                float iy = fmaxf(0.f, fminf(b[3], kept[m][3]) - fmaxf(b[1], kept[m][1]));
                float inter = ix * iy;
                float u = (b[2] - b[0]) * (b[3] - b[1]) + (kept[m][2] - kept[m][0]) * (kept[m][3] - kept[m][1]) - inter;
                if (u > 0.f && inter / u > nms_iou) ok = 0;
            }
            if (!ok) continue;
            for (int q = 0; q < 4; q++) kept[nk][q] = b[q];
            nk++;
            bear_box_t o;
            o.x1 = clampf((b[0] - px) / sx, 0.f, (float)src_w); o.y1 = clampf((b[1] - py) / sy, 0.f, (float)src_h);
            o.x2 = clampf((b[2] - px) / sx, 0.f, (float)src_w); o.y2 = clampf((b[3] - py) / sy, 0.f, (float)src_h);
            o.score = c[k].s;
            if (o.x2 - o.x1 >= 1.f && o.y2 - o.y1 >= 1.f) out[n++] = o;
        }
        for (int k = 0; n == 0 && fallback && k < nfb; k++) {   /* V19 said bear: one valid box */
            float b[4]; box_of(outs, fb[k].idx, b);
            bear_box_t o;
            o.x1 = clampf((b[0] - px) / sx, 0.f, (float)src_w); o.y1 = clampf((b[1] - py) / sy, 0.f, (float)src_h);
            o.x2 = clampf((b[2] - px) / sx, 0.f, (float)src_w); o.y2 = clampf((b[3] - py) / sy, 0.f, (float)src_h);
            o.score = fb[k].s;
            if (o.x2 - o.x1 >= 1.f && o.y2 - o.y1 >= 1.f) out[n++] = o;
        }
    }
    return n;
}
