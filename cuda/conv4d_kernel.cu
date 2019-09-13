#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <torch/csrc/autograd/variable.h>

using namespace std;

const int CUDA_NUM_THREADS = 1024;
const int BSIZE_H = 1;
const int BSIZE_W = 32;

#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

inline int GET_BLOCKS(const int N) {
    return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

__device__ __constant__ float const_weight[3*3*3*3];

template <typename T>
__global__ void conv4d_forward_kernel(T *output,
                                      T *input,
                                      T *weight,
                                      int channels_in,
                                      int channels_out,
                                      int ksize,
                                      int stride,
                                      int padding,
                                      int dilation,
                                      int C_in,
                                      int U_in,
                                      int V_in,
                                      int H_in,
                                      int W_in,
                                      int C_out,
                                      int U_out,
                                      int V_out,
                                      int H_out,
                                      int W_out) {

  int u = blockIdx.z / V_out;
  int v = blockIdx.z % V_out;
  int h = blockIdx.y * blockDim.y + threadIdx.y;
  int w = blockIdx.x * blockDim.x + threadIdx.x;

  int tid = threadIdx.y * BSIZE_W + threadIdx.x;

  const int n = 3*3*3*3;

  __shared__ T smem_input[n*BSIZE_W];

  for (int c_out = 0; c_out < C_out; ++c_out) {

    T output_local = 0.0;

    for (int c_in = 0; c_in < C_in; ++c_in) {

      /*
      for (int i = 0; i < n; i++) {
        smem_input[threadIdx.x + i*BSIZE_W] = input[threadIdx.x];
      }
      */

      for (int i = tid; i < n*BSIZE_W; i += BSIZE_W) {

        int du = i / (3*3*3*BSIZE_W);
        int dv = i % (3*3*3*BSIZE_W) / (3*3*BSIZE_W);
        int dh = i % (3*3*BSIZE_W) / (3*BSIZE_W);
        int dw = i % (3*BSIZE_W) / (BSIZE_W);
        int w1 = i % BSIZE_W;

        int u2 = u + du - 1;
        int v2 = v + dv - 1;
        int h2 = h + dh - 1;
        int w2 = blockIdx.x * blockDim.x + w1 + dw - 1;

        int idx_in = c_in*U_in*V_in*H_in*W_in
                   + u2*V_in*H_in*W_in
                   + v2*H_in*W_in
                   + h2*W_in + w2;

        if (u2 < 0 || u2 >= U_out || 
            v2 < 0 || v2 >= V_out || 
            h2 < 0 || h2 >= H_out || 
            w2 < 0 || w2 >= W_out) {
          smem_input[i] = 0;
        } else {
          smem_input[i] = input[idx_in];
        }
      }

      __syncthreads();

      for (int i = 0; i < n; ++i) {
        output_local += smem_input[i*BSIZE_W + threadIdx.x] * const_weight[i];
      }
    }

    if (u >= U_out || v >= V_out || w >= W_out || h >= H_out) {
      continue;
    }

    int idx_out = c_out*U_out*V_out*H_out*W_out
                + u*V_out*H_out*W_out
                + v*H_out*W_out
                + h*W_out + w;

    output[idx_out] = output_local;
  }


  /*

  for (int du = -1; du <= 1; ++du) {
    for (int dv = -1; dv <= 1; ++dv) {
      for (int dh = -1; dh <= 1; ++dh) {

        int u1 = u + du;
        int v1 = v + dv;
        int h1 = h + dh;
        int w1 = w;

        if (u1 < 0 || u1 >= U_out || 
            v1 < 0 || v1 >= V_out || 
            h1 < 0 || h1 >= H_out || 
            w1 < 0 || w1 >= W_out) {
          continue;
        }

        int idx_sm = (du+1) * 3 * 3 * (BSIZE_W + 2)
                   + (dv+1) * 3 * (BSIZE_W + 2)
                   + (dh+1) * (BSIZE_W + 2)
                   + tid + 1;

        int idx_in = u1 * 3 * 3 * BSIZE_W
                   + v1 * 3 * BSIZE_W
                   + h1 * BSIZE_W
                   + w1;

        smem_input[idx_sm] = input[idx_in];
      }
    }
  }

  for (int du = -1; du <= 1; ++du) {
    for (int dv = -1; dv <= 1; ++dv) {
      for (int dh = -1; dh <= 1; ++dh) {

        if (tid == 0) {

          int u1 = u + du;
          int v1 = v + dv;
          int h1 = h + dh;
          int w1 = w - 1;
  
          if (u1 < 0 || u1 >= U_out ||
              v1 < 0 || v1 >= V_out ||
              h1 < 0 || h1 >= H_out ||
              w1 < 0 || w1 >= W_out) {
            smem_input[tid] = 0;
          } else {

            int idx_in = u1 * 3 * 3 * BSIZE_W
                       + v1 * 3 * BSIZE_W
                       + h1 * BSIZE_W
                       + w1;

            smem_input[tid] = input[idx_in];
          }
        }

        if (tid == BSIZE_W) {

          int u1 = u + du;
          int v1 = v + dv;
          int h1 = h + dh;
          int w1 = w + 1;

          if (u1 < 0 || u1 >= U_out ||
              v1 < 0 || v1 >= V_out ||
              h1 < 0 || h1 >= H_out ||
              w1 < 0 || w1 >= W_out) {
            smem_input[tid] = 0;
          } else {

            int idx_in = u1 * 3 * 3 * BSIZE_W
                       + v1 * 3 * BSIZE_W
                       + h1 * BSIZE_W
                       + w1;

            smem_input[threadIdx.x] = input[idx_in];
          }
        }
      }
    }
  }


  for (int i = tid; i < 3*3*3*3; i += BSIZE_H*BSIZE_W) {
    smem_weight[i] = weight[i];
  }

  __syncthreads();


  for (int c_out = 0; c_out < C_out; ++c_out) {

    T output_local = 0.0;

    for (int c_in = 0; c_in < C_in; ++c_in) {

      for (int i = 0; i < ksize*ksize*ksize*ksize; ++i) {

        int du = i / (ksize*ksize*ksize);
        int dv = i % (ksize*ksize*ksize) / (ksize*ksize);
        int dh = i % (ksize*ksize) / ksize;
        int dw = i % ksize;

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

        int idx_in = c_in*U_in*V_in*H_in*W_in
                   + u1*V_in*H_in*W_in
                   + v1*H_in*W_in
                   + h1*W_in + w1;

        int idx_weight = c_out*C_in*ksize*ksize*ksize*ksize
                       + c_in*ksize*ksize*ksize*ksize
                       + du*ksize*ksize*ksize
                       + dv*ksize*ksize
                       + dh*ksize
                       + dw;

        int idx_sm = (du) * 3 * 3 * (BSIZE_W + 2)
                   + (dv) * 3 * (BSIZE_W + 2)
                   + (dh) * (BSIZE_W + 2)
                   + threadIdx.x;

//        output_local += input[idx_in] * smem_weight[i];
//        output_local += input[idx_in] * weight[idx_weight];
//        output_local += smem_input[tid] * weight[i];
        output_local += smem_input[threadIdx.x + i] * smem_weight[i];
      }
    }

    int idx_out = c_out*U_out*V_out*H_out*W_out
                + u*V_out*H_out*W_out
                + v*H_out*W_out
                + h*W_out + w;

    if (u >= U_out || v >= V_out || w >= W_out || h >= H_out) {
      return;
    }

    output[idx_out] = output_local;
  }


  */



/*  
  int n = U_out*V_out*H_out*W_out;

  CUDA_KERNEL_LOOP(index, n) {

    int u = index / (V_out*H_out*W_out);
    int v = index % (V_out*H_out*W_out) / (H_out*W_out);
    int h = index % (H_out*W_out) / W_out;
    int w = index % W_out;

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

                int idx_out = c_out*U_out*V_out*H_out*W_out
                            + u*V_out*H_out*W_out
                            + v*H_out*W_out
                            + h*W_out + w;

                int idx_in = c_in*U_in*V_in*H_in*W_in
                           + u1*V_in*H_in*W_in
                           + v1*H_in*W_in
                           + h1*W_in + w1;

                int idx_weight = c_out*C_in*ksize*ksize*ksize*ksize
                               + c_in*ksize*ksize*ksize*ksize
                               + du*ksize*ksize*ksize
                               + dv*ksize*ksize
                               + dh*ksize
                               + dw;

                output[idx_out] += input[idx_in] * weight[idx_weight];
              }
            }
          }
        }
      }
    }
  }
  */
}


at::Tensor conv4d_forward_cuda(
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

  at::Tensor outputs = at::zeros({B_out, C_out, U_out, V_out, H_out, W_out}, inputs.type());

  AT_DISPATCH_FLOATING_TYPES(inputs.type(), "forward", ([&] {

    auto inputs_data = inputs.data<scalar_t>();
    auto outputs_data = outputs.data<scalar_t>();
    auto weight_data = weight.data<scalar_t>();

    cudaMemcpyToSymbol(const_weight, weight_data, channels_in*3*3*3*3*sizeof(float), 0, cudaMemcpyDeviceToDevice);

    for (int i = 0; i < B_out; ++i) {

      auto input = inputs_data + i * C_in*U_in*V_in*H_in*W_in;
      auto output = outputs_data + i * C_out*U_out*V_out*H_out*W_out;

      dim3 gdim((W_out+BSIZE_W-1)/BSIZE_W, (H_out+BSIZE_H-1)/BSIZE_H, U_out*V_out);
      dim3 bdim(BSIZE_W, BSIZE_H, 1);

      conv4d_forward_kernel<<<gdim, bdim>>> (
        output, input, weight_data, channels_in, channels_out,
        ksize, stride, padding, dilation, 
        C_in, U_in, V_in, H_in, W_in, 
        C_out, U_out, V_out, H_out, W_out);

//      conv4d_forward_kernel<<<GET_BLOCKS(U_out*V_out*H_out*W_out), CUDA_NUM_THREADS>>> (
//        output, input, weight_data, channels_in, channels_out,
//        ksize, stride, padding, dilation, 
//        C_in, U_in, V_in, H_in, W_in, 
//        C_out, U_out, V_out, H_out, W_out);

    }
    
  }));

  return outputs;
}


template <typename T>
__global__ void conv4d_backward_kernel(T *grad_input,
                                       T *grad_weight,
                                       T *grad_output,
                                       T *input, 
                                       T *weight, 
                                       int channels_in,
                                       int channels_out,
                                       int ksize,
                                       int stride,
                                       int padding,
                                       int dilation,
                                       int C_in,
                                       int U_in,
                                       int V_in,
                                       int H_in,
                                       int W_in,
                                       int C_out,
                                       int U_out,
                                       int V_out,
                                       int H_out,
                                       int W_out) {
  int n = U_out*V_out*H_out*W_out;

  CUDA_KERNEL_LOOP(index, n) {

    int u = index / (V_out*H_out*W_out);
    int v = index % (V_out*H_out*W_out) / (H_out*W_out);
    int h = index % (H_out*W_out) / W_out;
    int w = index % W_out;

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

                int idx_out = c_out*U_out*V_out*H_out*W_out
                            + u*V_out*H_out*W_out
                            + v*H_out*W_out
                            + h*W_out + w;

                int idx_in = c_in*U_in*V_in*H_in*W_in
                           + u1*V_in*H_in*W_in
                           + v1*H_in*W_in
                           + h1*W_in + w1;

                int idx_weight = c_out*C_in*ksize*ksize*ksize*ksize
                               + c_in*ksize*ksize*ksize*ksize
                               + du*ksize*ksize*ksize
                               + dv*ksize*ksize
                               + dh*ksize
                               + dw;

                atomicAdd(grad_input + idx_in, grad_output[idx_out] * weight[idx_weight]);
                atomicAdd(grad_weight + idx_weight, input[idx_in] * grad_output[idx_out]);
              }
            }
          }
        }
      }
    }
  }
}


vector<at::Tensor> conv4d_backward_cuda(
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

  at::Tensor grad_inputs = at::zeros({B_in, C_in, U_in, V_in, H_in, W_in}, grad_outputs.type());
  at::Tensor grad_weight = at::zeros_like(weight);

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

      conv4d_backward_kernel<<<GET_BLOCKS(U_out*V_out*H_out*W_out), CUDA_NUM_THREADS>>> (
        grad_input, grad_weight_data, grad_output, input, weight_data,
        channels_in, channels_out,
        ksize, stride, padding, dilation,
        C_in, U_in, V_in, H_in, W_in,
        C_out, U_out, V_out, H_out, W_out);
    }

  }));

  return {grad_inputs, grad_weight};
}
