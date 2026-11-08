/* bear_det_decode.h -- decode the V19 detection head on the camera CPU (C99, no malloc). */
#ifndef BEAR_DET_DECODE_H
#define BEAR_DET_DECODE_H

typedef struct { float x1, y1, x2, y2, score; } bear_box_t;

/* logits2: classifier output [2] (index 1 = bear).
   heat:    grid_h*grid_w centre logits.
   box:     4*grid_h*grid_w log-distances (l,t,r,b) in stride units, NCHW.
   src_w/src_h: original frame size in pixels (the network input is that frame
   letterboxed to 640x360).
   thr: probability threshold (0.3); max_k: at most this many boxes (<= 64);
   nms_iou: 0.5; gate: 1 = return 0 boxes unless logits2[1] > logits2[0].
   When the classifier says bear but no valid box survives, the strongest peak that gives a valid (clipped) box is
   still returned (up to 64 peaks tried). Boxes are in original pixels, clipped, score-descending.
   Returns the number of boxes written to out (capacity >= max_k). */
int bear_det_decode(const float *logits2, const float *heat, const float *box,
                    int grid_h, int grid_w, int stride, int src_w, int src_h,
                    float thr, int max_k, float nms_iou, int gate, bear_box_t *out);

#endif
