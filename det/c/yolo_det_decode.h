/* yolo_det_decode.h -- decode the raw YOLOv8n head (candidate b') on the camera CPU (C99).
   Inputs: the 6 network outputs, NCHW float, one image:
     box_p3 [64][48][80], cls_p3 [1][48][80], box_p4 [64][24][40], cls_p4 [1][24][40],
     box_p5 [64][12][20], cls_p5 [1][12][20]   (network input 640x384, letterboxed).
   box = 4 sides (l,t,r,b) x 16 DFL bins; cls = bear logit.
   Output: boxes in ORIGINAL frame pixels (src_w x src_h), score-descending.
   fallback=1: when nothing reaches thr, return the best anchor anyway (V19 said bear).
   Returns the number of boxes written to out (capacity >= max_k, max_k <= 64). */
#ifndef YOLO_DET_DECODE_H
#define YOLO_DET_DECODE_H
#include "bear_det_decode.h"   /* bear_box_t */
int yolo_det_decode(const float *const outs[6], int src_w, int src_h,
                    float thr, int max_k, float nms_iou, int fallback, bear_box_t *out);
#endif
