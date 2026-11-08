/* test_main.c -- read one test vector, decode, print boxes.
   File layout (little endian): int32 gh, gw, stride, src_w, src_h;
   float32 logits[2]; float32 heat[gh*gw]; float32 box[4*gh*gw]. */
#include <stdio.h>
#include <stdlib.h>
#include "bear_det_decode.h"

int main(int argc, char **argv) {
    int hdr[5];
    float logits[2];
    bear_box_t out[20];
    FILE *f;
    if (argc != 2 || !(f = fopen(argv[1], "rb"))) { fprintf(stderr, "usage: %s vector.bin\n", argv[0]); return 2; }
    if (fread(hdr, sizeof(int), 5, f) != 5 || fread(logits, sizeof(float), 2, f) != 2) { fclose(f); return 3; }
    size_t hw = (size_t)hdr[0] * hdr[1];
    float *heat = malloc(hw * sizeof(float)), *box = malloc(4 * hw * sizeof(float));
    if (!heat || !box || fread(heat, sizeof(float), hw, f) != hw || fread(box, sizeof(float), 4 * hw, f) != 4 * hw) {
        fclose(f); free(heat); free(box); return 3;
    }
    fclose(f);
    int n = bear_det_decode(logits, heat, box, hdr[0], hdr[1], hdr[2], hdr[3], hdr[4],
                            0.3f, 20, 0.5f, 1, out);
    for (int k = 0; k < n; k++)
        printf("%.3f %.3f %.3f %.3f %.5f\n", out[k].x1, out[k].y1, out[k].x2, out[k].y2, out[k].score);
    free(heat); free(box);
    return 0;
}
