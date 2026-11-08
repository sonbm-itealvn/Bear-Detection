/* yolo_test_main.c -- read one YOLO test vector, decode, print boxes.
   Layout: int32 src_w, src_h; float32 box_p3, cls_p3, box_p4, cls_p4, box_p5, cls_p5 (NCHW). */
#include <stdio.h>
#include <stdlib.h>
#include "yolo_det_decode.h"

int main(int argc, char **argv) {
    static const int N[6] = {64 * 48 * 80, 48 * 80, 64 * 24 * 40, 24 * 40, 64 * 12 * 20, 12 * 20};
    int hdr[2]; float *o[6]; bear_box_t out[20]; FILE *f;
    if (argc != 2 || !(f = fopen(argv[1], "rb"))) { fprintf(stderr, "usage: %s vector.bin\n", argv[0]); return 2; }
    if (fread(hdr, sizeof(int), 2, f) != 2) { fclose(f); return 3; }
    for (int k = 0; k < 6; k++) {
        o[k] = malloc(N[k] * sizeof(float));
        if (!o[k] || fread(o[k], sizeof(float), N[k], f) != (size_t)N[k]) { fclose(f); return 3; }
    }
    fclose(f);
    int n = yolo_det_decode((const float *const *)o, hdr[0], hdr[1], 0.3f, 20, 0.5f, 1, out);
    for (int k = 0; k < n; k++)
        printf("%.3f %.3f %.3f %.3f %.6f\n", out[k].x1, out[k].y1, out[k].x2, out[k].y2, out[k].score);
    for (int k = 0; k < 6; k++) free(o[k]);
    return 0;
}
