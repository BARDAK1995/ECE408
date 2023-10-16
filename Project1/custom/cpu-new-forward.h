#ifndef SRC_LAYER_CPU_NEW_FORWARD_H
#define SRC_LAYER_CPU_NEW_FORWARD_H
#define out_4d(i3, i2, i1, i0) output[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define in_4d(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define mask_4d(i3, i2, i1, i0) mask[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
void conv_forward_cpu(float *output, const float *input, const float *mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S = 1);

#endif // SRC_LAYER_CPU_NEW_FORWARD_H
