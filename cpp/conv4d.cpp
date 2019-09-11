#include <limits>
#include <torch/extension.h>
#include <torch/csrc/autograd/variable.h>
#include "common.h"
#include "cost_volume.h"


at::Tensor conv4d_forward(
    at::Tensor inputs,
    at::Tensor weight,
    int channels_in,
    int channels_out,
    int ksize,
    int stride,
    int padding,
    int dilation) {

  int B_in = inputs.size(0);
  int C_in = inputs.size(1);
  int U_in = inputs.size(2);
  int V_in = inputs.size(3);
  int H_in = inputs.size(4);
  int W_in = inputs.size(5);

  int B_out = B_in;
  int C_out = channels_out;
  int U_out = (U_in + 2*padding - dilation*(ksize - 1) - 1) / stride + 1;
  int V_out = (V_in + 2*padding - dilation*(ksize - 1) - 1) / stride + 1;
  int H_out = (H_in + 2*padding - dilation*(ksize - 1) - 1) / stride + 1;
  int W_out = (W_in + 2*padding - dilation*(ksize - 1) - 1) / stride + 1;

  at::Tensor outputs = torch::zeros({B_out, C_out, U_out, V_out, H_out, W_out}, inputs.type());

  AT_DISPATCH_FLOATING_TYPES(inputs.type(), "forward", ([&] {

    auto inputs_data = inputs.data<scalar_t>();
    auto outputs_data = outputs.data<scalar_t>();
    auto weight_data = weight.data<scalar_t>();

    for (int i = 0; i < B_out; ++i) {

      auto input = inputs_data + i * C_in*U_in*V_in*H_in*W_in;
      auto output = outputs_data + i * C_out*U_out*V_out*H_out*W_out;

      for (int u = 0; u < U_out; ++u) {
        for (int v = 0; v < V_out; ++v) {
          for (int h = 0; h < H_out; ++h) {
            for (int w = 0; w < W_out; ++w) {

              for (int du = 0; du < ksize; ++du) {
                for (int dv = 0; dv < ksize; ++dv) {
                  for (int dh = 0; dh < ksize; ++dh) {
                    for (int dw = 0; dw < ksize; ++dw) {

                      int u1 = u*stride + du*dilation - padding;
                      int v1 = v*stride + dv*dilation - padding;
                      int h1 = h*stride + dh*dilation - padding;
                      int w1 = w*stride + dw*dilation - padding;

                      if (u1 < 0 || u1 >= U_in) {
                        continue;
                      }
                      if (v1 < 0 || v1 >= V_in) {
                        continue;
                      }
                      if (h1 < 0 || h1 >= H_in) {
                        continue;
                      }
                      if (w1 < 0 || w1 >= W_in) {
                        continue;
                      }

                      for (int c_out = 0; c_out < C_out; c_out++) {
                        for (int c_in = 0; c_in < C_in; c_in++) {

                          int out_idx = c_out*U_out*V_out*H_out*W_out
                                      + u*V_out*H_out*W_out
                                      + v*H_out*W_out
                                      + h*W_out + w;

                          int in_idx = c_in*U_in*V_in*H_in*W_in
                                     + u1*V_in*H_in*W_in
                                     + v1*H_in*W_in
                                     + h1*W_in + w1;
 
                          int weight_idx = c_out*C_in*ksize*ksize*ksize*ksize
                                         + c_in*ksize*ksize*ksize*ksize
                                         + du*ksize*ksize*ksize
                                         + dv*ksize*ksize
                                         + dh*ksize
                                         + dw;

                          output[out_idx] += input[in_idx] * weight_data[weight_idx];
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    
  }));

  return outputs;
}


vector<at::Tensor> conv4d_backward(
    at::Tensor grad_outputs,
    at::Tensor inputs,
    at::Tensor weight,
    int channels_in,
    int channels_out,
    int ksize,
    int stride,
    int padding,
    int dilation) {

  int B_out = grad_outputs.size(0);
  int C_out = grad_outputs.size(1);
  int U_out = grad_outputs.size(2);
  int V_out = grad_outputs.size(3);
  int H_out = grad_outputs.size(4);
  int W_out = grad_outputs.size(5);

  int B_in = inputs.size(0);
  int C_in = inputs.size(1);
  int U_in = inputs.size(2);
  int V_in = inputs.size(3);
  int H_in = inputs.size(4);
  int W_in = inputs.size(5);

  at::Tensor grad_inputs = torch::zeros({B_in, C_in, U_in, V_in, H_in, W_in}, grad_outputs.type());
  at::Tensor grad_weight = torch::zeros_like(weight);

  AT_DISPATCH_FLOATING_TYPES(grad_outputs.type(), "backward", ([&] {

    auto grad_inputs_data = grad_inputs.data<scalar_t>();
    auto grad_outputs_data = grad_outputs.data<scalar_t>();
    auto grad_weight_data = grad_weight.data<scalar_t>();
    auto inputs_data = inputs.data<scalar_t>();
    auto weight_data = weight.data<scalar_t>();

    for (int i = 0; i < B_out; ++i) {

      auto grad_input = grad_inputs_data + i * C_in*U_in*V_in*H_in*W_in;
      auto grad_output = grad_outputs_data + i * C_out*U_out*V_out*H_out*W_out;
      auto input = inputs_data + i * C_in*U_in*V_in*H_in*W_in;

      for (int u = 0; u < U_out; ++u) {
        for (int v = 0; v < V_out; ++v) {
          for (int h = 0; h < H_out; ++h) {
            for (int w = 0; w < W_out; ++w) {

              for (int du = 0; du < ksize; ++du) {
                for (int dv = 0; dv < ksize; ++dv) {
                  for (int dh = 0; dh < ksize; ++dh) {
                    for (int dw = 0; dw < ksize; ++dw) {

                      int u1 = u*stride + du*dilation - padding;
                      int v1 = v*stride + dv*dilation - padding;
                      int h1 = h*stride + dh*dilation - padding;
                      int w1 = w*stride + dw*dilation - padding;

                      if (u1 < 0 || u1 >= U_in) {
                        continue;
                      }
                      if (v1 < 0 || v1 >= V_in) {
                        continue;
                      }
                      if (h1 < 0 || h1 >= H_in) {
                        continue;
                      }
                      if (w1 < 0 || w1 >= W_in) {
                        continue;
                      }

                      for (int c_out = 0; c_out < C_out; ++c_out) {
                        for (int c_in = 0; c_in < C_in; ++c_in) {

                          int out_idx = c_out*U_out*V_out*H_out*W_out
                                      + u*V_out*H_out*W_out
                                      + v*H_out*W_out
                                      + h*W_out + w;

                          int in_idx = c_in*U_in*V_in*H_in*W_in
                                     + u1*V_in*H_in*W_in
                                     + v1*H_in*W_in
                                     + h1*W_in + w1;
 
                          int weight_idx = c_out*C_in*ksize*ksize*ksize*ksize
                                         + c_in*ksize*ksize*ksize*ksize
                                         + du*ksize*ksize*ksize
                                         + dv*ksize*ksize
                                         + dh*ksize
                                         + dw;

                          grad_input[in_idx] += grad_output[out_idx] * weight_data[weight_idx];
                          grad_weight_data[weight_idx] += input[in_idx] * grad_output[out_idx];
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    
  }));


  return {grad_inputs, grad_weight};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("compute_cost_volume_forward", &compute_cost_volume_forward, "match forward");
  m.def("compute_cost_volume_backward", &compute_cost_volume_backward, "match backward");
  m.def("conv4d_forward", &conv4d_forward, "conv4d forward");
  m.def("conv4d_backward", &conv4d_backward, "conv4d backward");
}

