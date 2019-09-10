#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <torch/csrc/autograd/variable.h>

using namespace std;

constexpr int BLOCK_SIZE = 32;
constexpr int FEAT_DIM = 32;

template <typename scalar_t> 
__device__ inline scalar_t sign(scalar_t val) {
  return (static_cast<scalar_t>(0) < val) - (val < static_cast<scalar_t>(0));
}

template <typename scalar_t>
__global__ void compute_cost_volume_forward_kernel(scalar_t *cost,
                                                   const scalar_t *feat0,
                                                   const scalar_t *feat1,
                                                   const int C, 
                                                   const int L, 
                                                   const int H, 
                                                   const int W,
                                                   const int max_dh,
                                                   const int max_dw) {

  const int h = blockIdx.x*blockDim.x + threadIdx.x;
  const int w = blockIdx.y*blockDim.y + threadIdx.y;
  const int tid = threadIdx.y;

  __shared__ scalar_t smem_feat0[FEAT_DIM*BLOCK_SIZE];
  __shared__ scalar_t smem_feat1[2*FEAT_DIM*BLOCK_SIZE];
  __shared__ scalar_t smem_cost[BLOCK_SIZE];

  for (int c = 0; c < C; ++c) {
    if (w < W) {
      smem_feat0[c*BLOCK_SIZE + tid] = feat0[c*H*W + h*W + w];
    } else {
      smem_feat0[c*BLOCK_SIZE + tid] = static_cast<scalar_t>(0);
    }
  }

  for (int dh = -max_dh; dh <= max_dh; ++dh) {
    for (int dw = -max_dw; dw <= max_dw; dw += BLOCK_SIZE) {

      for (int i = tid; i < 2*C*BLOCK_SIZE; i += BLOCK_SIZE) {
        smem_feat1[i] = static_cast<scalar_t>(0);
      }

      __syncthreads();

      int h2 = h + dh;
      int w2 = w + dw;

      if (h2 >= 0 && h2 < H) {
        for (int c = 0; c < C; ++c) {

          if (w2 >= 0 && w2 < W) {
            smem_feat1[c*2*BLOCK_SIZE + tid] = feat1[c*H*W + h2*W + w2];
          }

          if (w2 + BLOCK_SIZE >= 0 && w2 + BLOCK_SIZE < W) {
            smem_feat1[c*2*BLOCK_SIZE + tid + BLOCK_SIZE] = feat1[c*H*W + h2*W + w2 + BLOCK_SIZE];
          }
        }
      }

      __syncthreads();

      for (int i = 0; i < BLOCK_SIZE; ++i) {

        int l = (dh+max_dh)*(2*max_dw+1) + (dw+max_dw+i);

        if (l >= L || w >= W) {
          break;
        }

        smem_cost[tid] = static_cast<scalar_t>(0);
        for (int c = 0; c < C; ++c) {
          smem_cost[tid] += abs(smem_feat0[c*BLOCK_SIZE + tid] - smem_feat1[c*2*BLOCK_SIZE + tid + i]);
        }

        cost[l*H*W + h*W + w] = smem_cost[tid];
      }
    }
  }
}


at::Tensor compute_cost_volume_forward_cuda(
  at::Tensor feat0,
  at::Tensor feat1,
  int max_dh,
  int max_dw) {

  int B = feat0.size(0);
  int C = feat0.size(1);
  int H = feat0.size(2);
  int W = feat0.size(3);
  int L = (2*max_dh+1) * (2*max_dw+1);

  at::Tensor cost = at::zeros({B,L,H,W}, feat0.type());

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(cost.type(), "compute_cost_volume_forward_cuda", ([&] {

    auto feat0_data = feat0.data<scalar_t>();
    auto feat1_data = feat1.data<scalar_t>();
    auto cost_data = cost.data<scalar_t>();

    for (int i = 0; i < B; ++i) {
      auto feat0_i = feat0_data + i*C*H*W;
      auto feat1_i = feat1_data + i*C*H*W;
      auto cost_i = cost_data + i*L*H*W;

      dim3 gdim(H, (W+BLOCK_SIZE-1)/BLOCK_SIZE);
      dim3 bdim(1, BLOCK_SIZE);

      compute_cost_volume_forward_kernel<scalar_t><<<gdim, bdim>>>
        (cost_i, feat0_i, feat1_i, C, L, H, W, max_dh, max_dw);        
    }

  }));

  return cost;
}


template <typename scalar_t>
__global__ void compute_cost_volume_backward_kernel_feat0(scalar_t *grad_feat0,
                                                          const scalar_t *grad_output,
                                                          const scalar_t *feat0,
                                                          const scalar_t *feat1,
                                                          const int C,
                                                          const int H,
                                                          const int W,
                                                          const int max_dh,
                                                          const int max_dw) {

  const int h = blockIdx.x*blockDim.x + threadIdx.x;
  const int w = blockIdx.y*blockDim.y + threadIdx.y;
  const int tid = threadIdx.y;

  __shared__ scalar_t smem_feat0[FEAT_DIM*BLOCK_SIZE];
  __shared__ scalar_t smem_feat1[2*FEAT_DIM*BLOCK_SIZE];
  __shared__ scalar_t smem_grad_feat0[FEAT_DIM*BLOCK_SIZE];

  for (int c = 0; c < C; ++c) {
    if (w < W) {
      smem_feat0[c*BLOCK_SIZE + tid] = feat0[c*H*W + h*W + w];
    } else {
      smem_feat0[c*BLOCK_SIZE + tid] = static_cast<scalar_t>(0);
    }
  }

  for (int i = tid; i < FEAT_DIM*BLOCK_SIZE; i += BLOCK_SIZE) {
    smem_grad_feat0[i] = static_cast<scalar_t>(0);
  }

  for (int dh = -max_dh; dh <= max_dh; ++dh) {
    for (int dw = -max_dw; dw <= max_dw; dw += BLOCK_SIZE) {

      int h2 = h + dh;
      int w2 = w + dw;

      for (int i = tid; i < 2*C*BLOCK_SIZE; i += BLOCK_SIZE) {
        smem_feat1[i] = static_cast<scalar_t>(0);
      }

      if (h2 >= 0 && h2 < H) {
        for (int c = 0; c < C; ++c) {

          if (w2 >= 0 && w2 < W) {
            smem_feat1[c*2*BLOCK_SIZE + tid] = feat1[c*H*W + h2*W + w2];
          }

          if (w2 + BLOCK_SIZE >= 0 && w2 + BLOCK_SIZE < W) {
            smem_feat1[c*2*BLOCK_SIZE + tid + BLOCK_SIZE] = feat1[c*H*W + h2*W + w2 + BLOCK_SIZE];
          }
        }
      }

      __syncthreads();

      for (int i = 0; i < BLOCK_SIZE; ++i) {

        if (i + dw > max_dw) {
          break;
        }

        int l = (dh+max_dh)*(2*max_dw+1) + (dw+max_dw+i);

        scalar_t grad_o = grad_output[l*H*W + h*W + w];

        for (int c = 0; c < C; c++) {
          scalar_t delta = sign(smem_feat0[c*BLOCK_SIZE + tid] - smem_feat1[c*2*BLOCK_SIZE + tid + i]);
          smem_grad_feat0[c*BLOCK_SIZE + tid] += grad_o * delta;
        }
      }
    }
  }

  if (w < W) {
    for (int c = 0; c < C; c++) {
      grad_feat0[c*H*W + h*W + w] += smem_grad_feat0[c*BLOCK_SIZE + tid];
    }
  }
}


template <typename scalar_t>
__global__ void compute_cost_volume_backward_kernel_feat1(scalar_t *grad_feat1,
                                                          const scalar_t *grad_output,
                                                          const scalar_t *feat0,
                                                          const scalar_t *feat1,
                                                          const int C,
                                                          const int H,
                                                          const int W,
                                                          const int max_dh,
                                                          const int max_dw) {

  const int h = blockIdx.x*blockDim.x + threadIdx.x;
  const int w = blockIdx.y*blockDim.y + threadIdx.y;
  const int tid = threadIdx.y;

  __shared__ scalar_t smem_feat0[2*FEAT_DIM*BLOCK_SIZE];
  __shared__ scalar_t smem_feat1[FEAT_DIM*BLOCK_SIZE];
  __shared__ scalar_t smem_grad_feat1[FEAT_DIM*BLOCK_SIZE];

  for (int c = 0; c < C; ++c) {
    if (w < W) {
      smem_feat1[c*BLOCK_SIZE + tid] = feat1[c*H*W + h*W + w];
    } else {
      smem_feat1[c*BLOCK_SIZE + tid] = static_cast<scalar_t>(0);
    }
  }

  for (int i = tid; i < FEAT_DIM*BLOCK_SIZE; i += BLOCK_SIZE) {
    smem_grad_feat1[i] = static_cast<scalar_t>(0);
  }

  for (int dh = -max_dh; dh <= max_dh; ++dh) {
    for (int dw = -max_dw; dw <= max_dw; dw += BLOCK_SIZE) {

      int h2 = h + dh;
      int w2 = w + dw;

      for (int i = tid; i < 2*C*BLOCK_SIZE; i += BLOCK_SIZE) {
        smem_feat0[i] = static_cast<scalar_t>(0);
      }

      if (h2 >= 0 && h2 < H) {
        for (int c = 0; c < C; ++c) {

          if (w2 >= 0 && w2 < W) {
            smem_feat0[c*2*BLOCK_SIZE + tid] = feat0[c*H*W + h2*W + w2];
          }

          if (w2 + BLOCK_SIZE >= 0 && w2 + BLOCK_SIZE < W) {
            smem_feat0[c*2*BLOCK_SIZE + tid + BLOCK_SIZE] = feat0[c*H*W + h2*W + w2 + BLOCK_SIZE];
          }
        }
      }

      __syncthreads();

      for (int i = 0; i < BLOCK_SIZE; ++i) {

        if (i + dw > max_dw) {
          break;
        }

        if (h2 < 0 || h2 >= H || w2 + i < 0 || w2 + i >= W) {
          continue;
        }

        int l = (-dh+max_dh)*(2*max_dw+1) + (-dw-i+max_dw);

        scalar_t grad_o = grad_output[l*H*W + h2*W + w2 + i];

        for (int c = 0; c < C; c++) {
          scalar_t delta = sign(smem_feat1[c*BLOCK_SIZE + tid] - smem_feat0[c*2*BLOCK_SIZE + tid + i]);
          smem_grad_feat1[c*BLOCK_SIZE + tid] += grad_o * delta;
        }
      }
    }
  }

  if (w < W) {
    for (int c = 0; c < C; c++) {
      grad_feat1[c*H*W + h*W + w] += smem_grad_feat1[c*BLOCK_SIZE + tid];
    }
  }
}


vector<at::Tensor> compute_cost_volume_backward_cuda(
  at::Tensor grad_output,
  at::Tensor feat0,
  at::Tensor feat1,
  int max_dh,
  int max_dw) {

  int B = feat0.size(0);
  int C = feat0.size(1);
  int H = feat0.size(2);
  int W = feat0.size(3);
  int L = (2*max_dh+1) * (2*max_dw+1);

  at::Tensor grad_feat0 = at::zeros_like(feat0);
  at::Tensor grad_feat1 = at::zeros_like(feat1);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_output.type(), "compute_cost_volume_backward_cuda", ([&] {

    auto grad_feat0_data = grad_feat0.data<scalar_t>();
    auto grad_feat1_data = grad_feat1.data<scalar_t>();
    auto grad_output_data = grad_output.data<scalar_t>();
    auto feat0_data = feat0.data<scalar_t>();
    auto feat1_data = feat1.data<scalar_t>();

    for (int i = 0; i < B; ++i) {
      auto grad_feat0_i = grad_feat0_data + i*C*H*W;
      auto grad_feat1_i = grad_feat1_data + i*C*H*W;
      auto grad_output_i = grad_output_data + i*L*H*W;
      auto feat0_i = feat0_data + i*C*H*W;
      auto feat1_i = feat1_data + i*C*H*W;

      dim3 gdim(H, (W+BLOCK_SIZE-1)/BLOCK_SIZE);
      dim3 bdim(1, BLOCK_SIZE);

      compute_cost_volume_backward_kernel_feat0<<<gdim, bdim>>>
        (grad_feat0_i, grad_output_i, feat0_i, feat1_i, C, H, W, max_dh, max_dw);

      compute_cost_volume_backward_kernel_feat1<<<gdim, bdim>>>
        (grad_feat1_i, grad_output_i, feat0_i, feat1_i, C, H, W, max_dh, max_dw);
    }
    
  }));
  
  return {grad_feat0, grad_feat1};
}
