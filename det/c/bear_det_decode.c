/* bear_det_decode.c -- mirrors det/decode.py exactly. */
#include <math.h>
#include "bear_det_decode.h"

#define MAXC 64
#define FALLBACK_TRIES 64   /* strongest peaks tried when nothing valid survives (decode.py) */
#define IN_W 640
#define IN_H 360

typedef struct { float v; int i, j; } cand_t;

static int better(const cand_t *a, const cand_t *b) {          /* score desc, then i, j asc */
    if (a->v != b->v) return a->v > b->v;
    if (a->i != b->i) return a->i < b->i;
    return a->j < b->j;
}

static float iou(const bear_box_t *a, const bear_box_t *b) {
    float ix = fmaxf(0.f, fminf(a->x2, b->x2) - fmaxf(a->x1, b->x1));
    float iy = fmaxf(0.f, fminf(a->y2, b->y2) - fmaxf(a->y1, b->y1));
    float inter = ix * iy;
    float u = (a->x2 - a->x1) * (a->y2 - a->y1) + (b->x2 - b->x1) * (b->y2 - b->y1) - inter;
    return u > 0.f ? inter / u : 0.f;
}

static float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

int bear_det_decode(const float *logits2, const float *heat, const float *box,
                    int gh, int gw, int stride, int src_w, int src_h,
                    float thr, int max_k, float nms_iou, int gate, bear_box_t *out) {
    cand_t top[MAXC], fb[FALLBACK_TRIES];
    int ntop = 0, nfb = 0, n = 0, hw = gh * gw;
    float zthr = logf(thr / (1.f - thr));
    double s;
    int nw, nh, px, py;
    float sx, sy;

    if (max_k > MAXC) max_k = MAXC;
    if (max_k < 1 || src_w <= 0 || src_h <= 0 || gh <= 0 || gw <= 0 || !(thr > 0.f && thr < 1.f)) return 0;
    if (gate && !(logits2[1] > logits2[0])) return 0;

    for (int i = 0; i < gh; i++) for (int j = 0; j < gw; j++) {
        float v = heat[i * gw + j];
        int peak = 1;
        for (int di = -1; di <= 1 && peak; di++) for (int dj = -1; dj <= 1; dj++) {
            int y = i + di, x = j + dj;
            if (y < 0 || y >= gh || x < 0 || x >= gw) continue;
            if (heat[y * gw + x] > v) { peak = 0; break; }
        }
        if (!peak) continue;
        cand_t c = {v, i, j};
        {                                         /* all peaks, best FALLBACK_TRIES */
            int pos = nfb;
            while (pos > 0 && better(&c, &fb[pos - 1])) pos--;
            if (pos < FALLBACK_TRIES) {
                if (nfb < FALLBACK_TRIES) nfb++;
                for (int k = nfb - 1; k > pos; k--) fb[k] = fb[k - 1];
                fb[pos] = c;
            }
        }
        if (v < zthr) continue;
        int pos = ntop;                           /* insert into the score-sorted list */
        while (pos > 0 && better(&c, &top[pos - 1])) pos--;
        if (pos >= max_k) continue;
        if (ntop < max_k) ntop++;
        for (int k = ntop - 1; k > pos; k--) top[k] = top[k - 1];
        top[pos] = c;
    }

    s = fmin((double)IN_W / src_w, (double)IN_H / src_h);
    nw = (int)(src_w * s); if (nw < 1) nw = 1;
    nh = (int)(src_h * s); if (nh < 1) nh = 1;
    px = (IN_W - nw) / 2; py = (IN_H - nh) / 2;
    sx = (float)nw / src_w; sy = (float)nh / src_h;

    for (int pass = 0; pass < 2 && n == 0; pass++) {
        /* pass 0: candidates >= thr with NMS; pass 1 (only if nothing valid): the strongest
           valid peak at any score -- the classifier said bear, so one box is always returned */
        const cand_t *list = pass == 0 ? top : fb;
        int nl = pass == 0 ? ntop : nfb;
        for (int k = 0; k < nl; k++) {
            int i = list[k].i, j = list[k].j;
            float cx = (j + 0.5f) * stride, cy = (i + 0.5f) * stride;
            float l = expf(box[0 * hw + i * gw + j]) * stride, t = expf(box[1 * hw + i * gw + j]) * stride;
            float r = expf(box[2 * hw + i * gw + j]) * stride, b = expf(box[3 * hw + i * gw + j]) * stride;
            bear_box_t bb;
            bb.x1 = clampf((cx - l - px) / sx, 0.f, (float)src_w);
            bb.y1 = clampf((cy - t - py) / sy, 0.f, (float)src_h);
            bb.x2 = clampf((cx + r - px) / sx, 0.f, (float)src_w);
            bb.y2 = clampf((cy + b - py) / sy, 0.f, (float)src_h);
            bb.score = 1.f / (1.f + expf(-list[k].v));
            if (bb.x2 - bb.x1 < 1.f || bb.y2 - bb.y1 < 1.f) continue;
            if (pass == 1) { out[n++] = bb; break; }
            int keep = 1;
            for (int m = 0; m < n; m++) if (iou(&bb, &out[m]) > nms_iou) { keep = 0; break; }
            if (keep) out[n++] = bb;
        }
    }
    return n;
}
