#include <torch/extension.h>
#include "common.h"

at::Tensor compute_cost_volume_forward(
    at::Tensor feat1,
    at::Tensor feat2,
    int max_offset_h,
    int max_offset_w) {

  int B = feat1.size(0);
  int C = feat1.size(1);
  int H = feat1.size(2);
  int W = feat1.size(3);
  int L = (2*max_offset_h+1) * (2*max_offset_w+1);

  at::Tensor cost = torch::zeros({B,L,H,W}, feat1.type());

  AT_DISPATCH_FLOATING_TYPES(cost.type(), "comp_cost_volume_forward", ([&] {

    auto feat1_data = feat1.data<scalar_t>();
    auto feat2_data = feat2.data<scalar_t>();
    auto cost_data = cost.data<scalar_t>();

    for (int i = 0; i < B; ++i) {
      auto feat1_i = feat1_data + i*C*H*W;
      auto feat2_i = feat2_data + i*C*H*W;
      auto cost_i = cost_data + i*L*H*W;

      for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
          for (int dh = -max_offset_h; dh <= max_offset_h; ++dh) {
            for (int dw = -max_offset_w; dw <= max_offset_w; ++dw) {
              for (int c = 0; c < C; ++c) {
                int l = (dh+max_offset_h) * (2*max_offset_w+1) + (dw+max_offset_w);
                int h2 = h + dh;
                int w2 = w + dw;
                if (h2 >= 0 && h2 < H && w2 >= 0 && w2 < W) {
                  cost_i[l*H*W + h*W + w] += abs(feat1_i[c*H*W + h*W + w] - feat2_i[c*H*W + h2*W + w2]);                 
                } else {
                  cost_i[l*H*W + h*W + w] += abs(feat1_i[c*H*W + h*W + w]);
                }
              }
            }
          }
        }
      }
    }

  }));

  return cost;
}


vector<at::Tensor> compute_cost_volume_backward(
    at::Tensor grad_cost,
    at::Tensor feat1,
    at::Tensor feat2,
    int max_offset_h,
    int max_offset_w) {

  int B = feat1.size(0);
  int C = feat1.size(1);
  int H = feat1.size(2);
  int W = feat1.size(3);
  int L = (2*max_offset_h+1) * (2*max_offset_w+1);

  at::Tensor grad_feat1 = torch::zeros_like(feat1);
  at::Tensor grad_feat2 = torch::zeros_like(feat2);

  AT_DISPATCH_FLOATING_TYPES(grad_cost.type(), "comp_cost_volume_backward", ([&] {

    auto feat1_data = feat1.data<scalar_t>();
    auto feat2_data = feat2.data<scalar_t>();
    auto grad_feat1_data = grad_feat1.data<scalar_t>();
    auto grad_feat2_data = grad_feat2.data<scalar_t>();
    auto grad_output_data = grad_cost.data<scalar_t>();

    for (int i = 0; i < B; ++i) {
      auto feat1_i = feat1_data + i*C*H*W;
      auto feat2_i = feat2_data + i*C*H*W;
      auto grad_feat1_i = grad_feat1_data + i*C*H*W;
      auto grad_feat2_i = grad_feat2_data + i*C*H*W;      
      auto grad_cost_i = grad_output_data + i*L*H*W;

      for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
          for (int dh = -max_offset_h; dh <= max_offset_h; ++dh) {
            for (int dw = -max_offset_w; dw <= max_offset_w; ++dw) {
              for (int c = 0; c < C; ++c) {
                int l = (dh+max_offset_h) * (2*max_offset_w+1) + (dw+max_offset_w);
                int h2 = h + dh;
                int w2 = w + dw;
                if (h2 >= 0 && h2 < H && w2 >= 0 && w2 < W) {        
//                  printf("cpu: %d %d %d %d %lf\n", h, w, h2, w2, grad_feat1_i[c*H*W + h*W + w]);
                  scalar_t diff = sign(feat1_i[c*H*W + h*W + w] - feat2_i[c*H*W + h2*W + w2]);          
                  grad_feat1_i[c*H*W + h*W + w] += grad_cost_i[l*H*W + h*W + w] * diff;
                  grad_feat2_i[c*H*W + h2*W + w2] -= grad_cost_i[l*H*W + h*W + w] * diff;
//                  printf("cpu: %d %d %d %d\n", h, w, h2, w2);
//                  printf("cpu: %d %d %d %d %lf\n", h, w, h2, w2, grad_feat1_i[c*H*W + h*W + w]);
                } else {
                  grad_feat1_i[c*H*W + h*W + w] += grad_cost_i[l*H*W + h*W + w] * sign(feat1_i[c*H*W + h*W + w]);
//                  grad_feat1_i[c*H*W + h*W + w] += 0;
//          printf("cpu: %d %d %d %d %lf\n", h, w, h2, w2, grad_cost_i[l*H*W + h*W + w]*sign(feat1_i[c*H*W + h*W + w]));
                }
              }
            }
          }
        }
      }
    }
    
  }));

  return {grad_feat1, grad_feat2};
}
