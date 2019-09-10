#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <torch/csrc/autograd/variable.h>
#include "common.h"

using namespace std;

const int BLOCK_SIZE = 256;

template <typename T> __device__ void inline swap_cuda(T& a, T& b) {
  T c(a); 
  a = b; 
  b = c;
}


template <typename T>
__global__ void BLHW_to_BHWL_kernel(T *dst,
                                    T *src,
                                    int B,
                                    int L,
                                    int H,
                                    int W) {
  CUDA_KERNEL_LOOP(i, B*L*H*W) {

    int b = i / (L*H*W);
    int l = (i % (L*H*W)) / (H*W);
    int h = (i % (L*H*W)) % (H*W) / W;
    int w = i % W;

    int j = b*(H*W*L) + h*W*L + w*L + l;

    dst[j] = src[i];
  }
}


template <typename T>
__global__ void BHWL_to_BLHW_kernel(T *dst,
                                    T *src,
                                    int B,
                                    int L,
                                    int H,
                                    int W) {
  CUDA_KERNEL_LOOP(i, B*L*H*W) {

    int b = i / (L*H*W);
    int l = (i % (L*H*W)) / (H*W);
    int h = (i % (L*H*W)) % (H*W) / W;
    int w = i % W;

    int j = b*(H*W*L) + h*W*L + w*L + l;

    dst[i] = src[j];
  }
}


at::Tensor BLHW_to_BHWL(at::Tensor &src) {
  int B = src.size(0);
  int L = src.size(1);
  int H = src.size(2);
  int W = src.size(3);

  at::Tensor dst = at::zeros_like(src).view({B,H,W,L});

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(src.type(), "BLHW_to_BHWL", ([&] {

    scalar_t *src_data = src.data<scalar_t>();
    scalar_t *dst_data = dst.data<scalar_t>();

    BLHW_to_BHWL_kernel<<<GET_BLOCKS(B*H*W*L), CUDA_NUM_THREADS>>>(dst_data, src_data, B, L, H, W);

  }));

  return dst;
}

at::Tensor BHWL_to_BLHW(at::Tensor &src) {
  int B = src.size(0);
  int H = src.size(1);
  int W = src.size(2);
  int L = src.size(3);

  at::Tensor dst = at::zeros_like(src).view({B,L,H,W});

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(src.type(), "BLHW_to_BHWL", ([&] {

    scalar_t *src_data = src.data<scalar_t>();
    scalar_t *dst_data = dst.data<scalar_t>();

    BHWL_to_BLHW_kernel<<<GET_BLOCKS(B*H*W*L), CUDA_NUM_THREADS>>>(dst_data, src_data, B, L, H, W);

  }));

  return dst;
}




template <typename scalar_t>
__global__ void prop_hori_pos_forward_kernel(int16_t *indx,
                                             scalar_t *aggr,
                                             scalar_t *buff,
                                             scalar_t *cost,
                                             scalar_t *edge,
                                             int C, int L, int H, int W,
                                             int max_dh, int max_dw) {

  int h = blockIdx.x;
  int t = threadIdx.x;

  for (int l = t; l < L; l += blockDim.x) {
    scalar_t cost_l = cost[(h*W + 0)*L + l];
    buff[h*L + l] = cost_l;
    aggr[(h*W + 0)*L + l] += cost_l;
  }

  __syncthreads();

  for (int w = 1; w < W; ++w) {

    scalar_t weight = 0;
    for (int c = 0; c < C; ++c) {
      weight += abs(edge[c*H*W + h*W + w] - edge[c*H*W + h*W + w-1]); 
    }
    weight = exp(-weight);

    for (int l = t; l < L; l += blockDim.x) {
      indx[(h*W + w)*L + l] = l;
    }

    __syncthreads();


    for (int dw = -max_dw + t; dw <= max_dw; dw += blockDim.x) { 
      for (int dh = -max_dh+1; dh <= max_dh; ++dh) {
        int l1 = hw2label(dh, dw, max_dh, max_dw);
        int l2 = hw2label(dh-1, dw, max_dh, max_dw);                        
        scalar_t v1 = buff[h*L + l1];
        scalar_t v2 = buff[h*L + l2] + weight;
        if (v1 > v2) {
          buff[h*L + l1] = v2;
          indx[(h*W + w)*L + l1] = indx[(h*W + w)*L + l2]; 
        }
      }
    }

    __syncthreads();
      
    for (int dw = -max_dw + t; dw <= max_dw; dw += blockDim.x) { 
      for (int dh = max_dh-1; dh >= -max_dh; --dh) {
        int l1 = hw2label(dh, dw, max_dh, max_dw);
        int l2 = hw2label(dh+1, dw, max_dh, max_dw);                    
        scalar_t v1 = buff[h*L + l1];
        scalar_t v2 = buff[h*L + l2] + weight;
        if (v1 > v2) {
          buff[h*L + l1] = v2;
          indx[(h*W + w)*L + l1] = indx[(h*W + w)*L + l2]; 
        }
      }
    }

    __syncthreads();

    for (int dh = -max_dh + t; dh <= max_dh; dh += blockDim.x) {
      for (int dw = -max_dw+1; dw <= max_dw; ++dw) {     
        int l1 = hw2label(dh, dw, max_dh, max_dw);
        int l2 = hw2label(dh, dw-1, max_dh, max_dw);
        scalar_t v1 = buff[h*L + l1];
        scalar_t v2 = buff[h*L + l2] + weight;
        if (v1 > v2) {
          buff[h*L + l1] = v2;
          indx[(h*W + w)*L + l1] = indx[(h*W + w)*L + l2];
        }
      }
    }

    __syncthreads();

    for (int dh = -max_dh + t; dh <= max_dh; dh += blockDim.x) {
      for (int dw = max_dw-1; dw >= -max_dw; --dw) {
        int l1 = hw2label(dh, dw, max_dh, max_dw);
        int l2 = hw2label(dh, dw+1, max_dh, max_dw);
        scalar_t v1 = buff[h*L + l1];
        scalar_t v2 = buff[h*L + l2] + weight;
        if (v1 > v2) {
          buff[h*L + l1] = v2;
          indx[(h*W + w)*L + l1] = indx[(h*W + w)*L + l2];
        }
      }
    }

    __syncthreads();

    for (int l = t; l < L; l += blockDim.x) {
      buff[h*L + l] += cost[(h*W + w)*L + l];
      aggr[(h*W + w)*L + l] += buff[h*L + l];
    }
  }
}


template <typename scalar_t> 
__global__ void prop_hori_neg_forward_kernel(int16_t *indx,
                                             scalar_t *aggr, 
                                             scalar_t *buff,
                                             scalar_t *cost, 
                                             scalar_t *edge, 
                                             int C, int L, int H, int W, 
                                             int max_dh, int max_dw) {

  int h = blockIdx.x;
  int t = threadIdx.x;

  for (int l = t; l < L; l += blockDim.x) {
    scalar_t cost_l = cost[(h*W + W-1)*L + l];
    buff[h*L + l] = cost_l;
    aggr[(h*W + W-1)*L + l] += cost_l;
  }

  __syncthreads();

  for (int w = W-2; w >= 0; --w) {

    scalar_t weight = 0;
    for (int c = 0; c < C; ++c) {
      weight += abs(edge[c*H*W + h*W + w] - edge[c*H*W + h*W + w+1]); 
    }
    weight = exp(-weight);

    for (int l = t; l < L; l += blockDim.x) {
      indx[(h*W + w)*L + l] = l;
    }

    __syncthreads();

    for (int dh = -max_dh+1; dh <= max_dh; ++dh) {
      for (int dw = -max_dw + t; dw <= max_dw; dw += blockDim.x) { 
        int l1 = hw2label(dh, dw, max_dh, max_dw);
        int l2 = hw2label(dh-1, dw, max_dh, max_dw);                        
        if (buff[h*L + l1] > buff[h*L + l2] + weight) {
          buff[h*L + l1] = buff[h*L + l2] + weight;                
          indx[(h*W + w)*L + l1] = indx[(h*W + w)*L + l2]; 
        }
      }
    }

    __syncthreads();
      
    for (int dh = max_dh-1; dh >= -max_dh; --dh) {
      for (int dw = -max_dw + t; dw <= max_dw; dw += blockDim.x) { 
        int l1 = hw2label(dh, dw, max_dh, max_dw);
        int l2 = hw2label(dh+1, dw, max_dh, max_dw);                    
        if (buff[h*L + l1] > buff[h*L + l2] + weight) {
          buff[h*L + l1] = buff[h*L + l2] + weight;                
          indx[(h*W + w)*L + l1] = indx[(h*W + w)*L + l2]; 
        }
      }
    }

    __syncthreads();

    for (int dh = -max_dh + t; dh <= max_dh; dh += blockDim.x) {
      for (int dw = -max_dw+1; dw <= max_dw; ++dw) {     
        int l1 = hw2label(dh, dw, max_dh, max_dw);
        int l2 = hw2label(dh, dw-1, max_dh, max_dw);  
        if (buff[h*L + l1] > buff[h*L + l2] + weight) {
          buff[h*L + l1] = buff[h*L + l2] + weight;
          indx[(h*W + w)*L + l1] = indx[(h*W + w)*L + l2];
        }
      }
    }

    __syncthreads();

    for (int dh = -max_dh + t; dh <= max_dh; dh += blockDim.x) {
      for (int dw = max_dw-1; dw >= -max_dw; --dw) {     
        int l1 = hw2label(dh, dw, max_dh, max_dw);
        int l2 = hw2label(dh, dw+1, max_dh, max_dw);
        if (buff[h*L + l1] > buff[h*L + l2] + weight) {
          buff[h*L + l1] = buff[h*L + l2] + weight;
          indx[(h*W + w)*L + l1] = indx[(h*W + w)*L + l2];
        }
      }
    }

    __syncthreads();

    for (int l = t; l < L; l += blockDim.x) {
      buff[h*L + l] += cost[(h*W + w)*L + l];
      aggr[(h*W + w)*L + l] += buff[h*L + l];
    }
  }                
}


template <typename scalar_t> 
__global__ void prop_vert_pos_forward_kernel(int16_t *indx,
                                             scalar_t *aggr, 
                                             scalar_t *buff,
                                             scalar_t *cost, 
                                             scalar_t *edge, 
                                             int C, int L, int H, int W, 
                                             int max_dh, int max_dw) {

  int w = blockIdx.x;
  int t = threadIdx.x;

  for (int l = t; l < L; l += blockDim.x) {
    scalar_t cost_l = cost[(0*W + w)*L + l];
    buff[w*L + l] = cost_l;
    aggr[(0*W + w)*L + l] += cost_l;
  }

  __syncthreads();

  for (int h = 1; h < H; ++h) {

    scalar_t weight = 0;
    for (int c = 0; c < C; ++c) {
      weight += abs(edge[c*H*W + h*W + w] - edge[c*H*W + (h-1)*W + w]);
    }
    weight = exp(-weight);

    for (int l = t; l < L; l += blockDim.x) {
      indx[(h*W + w)*L + l] = l;
    }

    __syncthreads();

    for (int dh = -max_dh+1; dh <= max_dh; ++dh) {
      for (int dw = -max_dw + t; dw <= max_dw; dw += blockDim.x) { 
        int l1 = hw2label(dh, dw, max_dh, max_dw);
        int l2 = hw2label(dh-1, dw, max_dh, max_dw);                        
        if (buff[w*L + l1] > buff[w*L + l2] + weight) {
          buff[w*L + l1] = buff[w*L + l2] + weight;                
          indx[(h*W + w)*L + l1] = indx[(h*W + w)*L + l2]; 
        }
      }
    }

    __syncthreads();
      
    for (int dh = max_dh-1; dh >= -max_dh; --dh) {
      for (int dw = -max_dw + t; dw <= max_dw; dw += blockDim.x) { 
        int l1 = hw2label(dh, dw, max_dh, max_dw);
        int l2 = hw2label(dh+1, dw, max_dh, max_dw);                    
        if (buff[w*L + l1] > buff[w*L + l2] + weight) {
          buff[w*L + l1] = buff[w*L + l2] + weight;                
          indx[(h*W + w)*L + l1] = indx[(h*W + w)*L + l2]; 
        }
      }
    }

    __syncthreads();

    for (int dh = -max_dh + t; dh <= max_dh; dh += blockDim.x) {
      for (int dw = -max_dw+1; dw <= max_dw; ++dw) {     
        int l1 = hw2label(dh, dw, max_dh, max_dw);
        int l2 = hw2label(dh, dw-1, max_dh, max_dw);  
        if (buff[w*L + l1] > buff[w*L + l2] + weight) {
          buff[w*L + l1] = buff[w*L + l2] + weight;
          indx[(h*W + w)*L + l1] = indx[(h*W + w)*L + l2];
        }
      }
    }

    __syncthreads();

    for (int dh = -max_dh + t; dh <= max_dh; dh += blockDim.x) {
      for (int dw = max_dw-1; dw >= -max_dw; --dw) {     
        int l1 = hw2label(dh, dw, max_dh, max_dw);
        int l2 = hw2label(dh, dw+1, max_dh, max_dw);
        if (buff[w*L + l1] > buff[w*L + l2] + weight) {
          buff[w*L + l1] = buff[w*L + l2] + weight;
          indx[(h*W + w)*L + l1] = indx[(h*W + w)*L + l2];
        }
      }
    }

    __syncthreads();

    for (int l = t; l < L; l += blockDim.x) {
      buff[w*L + l] += cost[(h*W + w)*L + l];
      aggr[(h*W + w)*L + l] += buff[w*L + l];
    }
  }
}


template <typename scalar_t> 
__global__ void prop_vert_neg_forward_kernel(int16_t *indx,
                                             scalar_t *aggr, 
                                             scalar_t *buff,
                                             scalar_t *cost, 
                                             scalar_t *edge, 
                                             int C, int L, int H, int W, 
                                             int max_dh, int max_dw) {
  int w = blockIdx.x;
  int t = threadIdx.x;

  for (int l = t; l < L; l += blockDim.x) {
    scalar_t cost_l = cost[((H-1)*W + w)*L + l];
    buff[w*L + l] = cost_l;
    aggr[((H-1)*W + w)*L + l] += cost_l;
  }

  __syncthreads();

  for (int h = H-2; h >= 0; --h) {

    scalar_t weight = 0;
    for (int c = 0; c < C; ++c) {
      weight += abs(edge[c*H*W + h*W + w] - edge[c*H*W + (h+1)*W + w]); 
    }
    weight = exp(-weight);

    for (int l = t; l < L; l += blockDim.x) {
      indx[(h*W + w)*L + l] = l;
    }

    __syncthreads();

    for (int dh = -max_dh+1; dh <= max_dh; ++dh) {
      for (int dw = -max_dw + t; dw <= max_dw; dw += blockDim.x) { 
        int l1 = hw2label(dh, dw, max_dh, max_dw);
        int l2 = hw2label(dh-1, dw, max_dh, max_dw);                        
        if (buff[w*L + l1] > buff[w*L + l2] + weight) {
          buff[w*L + l1] = buff[w*L + l2] + weight;                
          indx[(h*W + w)*L + l1] = indx[(h*W + w)*L + l2]; 
        }
      }
    }

    __syncthreads();
      
    for (int dh = max_dh-1; dh >= -max_dh; --dh) {
      for (int dw = -max_dw + t; dw <= max_dw; dw += blockDim.x) { 
        int l1 = hw2label(dh, dw, max_dh, max_dw);
        int l2 = hw2label(dh+1, dw, max_dh, max_dw);                    
        if (buff[w*L + l1] > buff[w*L + l2] + weight) {
          buff[w*L + l1] = buff[w*L + l2] + weight;                
          indx[(h*W + w)*L + l1] = indx[(h*W + w)*L + l2]; 
        }
      }
    }

    __syncthreads();

    for (int dh = -max_dh + t; dh <= max_dh; dh += blockDim.x) {
      for (int dw = -max_dw+1; dw <= max_dw; ++dw) {     
        int l1 = hw2label(dh, dw, max_dh, max_dw);
        int l2 = hw2label(dh, dw-1, max_dh, max_dw);  
        if (buff[w*L + l1] > buff[w*L + l2] + weight) {
          buff[w*L + l1] = buff[w*L + l2] + weight;
          indx[(h*W + w)*L + l1] = indx[(h*W + w)*L + l2];
        }
      }
    }

    __syncthreads();

    for (int dh = -max_dh + t; dh <= max_dh; dh += blockDim.x) {
      for (int dw = max_dw-1; dw >= -max_dw; --dw) {     
        int l1 = hw2label(dh, dw, max_dh, max_dw);
        int l2 = hw2label(dh, dw+1, max_dh, max_dw);
        if (buff[w*L + l1] > buff[w*L + l2] + weight) {
          buff[w*L + l1] = buff[w*L + l2] + weight;
          indx[(h*W + w)*L + l1] = indx[(h*W + w)*L + l2];
        }
      }
    }

    __syncthreads();

    for (int l = t; l < L; l += blockDim.x) {
      buff[w*L + l] += cost[(h*W + w)*L + l];
      aggr[(h*W + w)*L + l] += buff[w*L + l];
    }
  }
}


void prop_hori_pos_forward_cuda(
  at::Tensor indx,
  at::Tensor aggr,
  at::Tensor cost,
  at::Tensor edge, 
  int max_dh,
  int max_dw) {

  int B = cost.size(0);
  int H = cost.size(1);
  int W = cost.size(2);
  int L = cost.size(3);
  int C = edge.size(1);

  AT_DISPATCH_FLOATING_TYPES(cost.type(), "prop_hori_pos_forward", ([&] {

    auto indx_data = indx.data<int16_t>();
    auto aggr_data = aggr.data<scalar_t>();
    auto cost_data = cost.data<scalar_t>();
    auto edge_data = edge.data<scalar_t>();

    scalar_t *buffer;
    cudaMalloc(&buffer, H*L*sizeof(scalar_t));

    for (int i = 0; i < B; ++i) {
      auto indx_i = indx_data + i*L*H*W;      
      auto aggr_i = aggr_data + i*L*H*W;      
      auto cost_i = cost_data + i*L*H*W;
      auto edge_i = edge_data + i*C*H*W;

      prop_hori_pos_forward_kernel<scalar_t><<<H, BLOCK_SIZE>>> 
        (indx_i, aggr_i, buffer, cost_i, edge_i, C, L, H, W, max_dh, max_dw);
    }  

    cudaFree(buffer);
  
  }));
}


void prop_hori_neg_forward_cuda(
  at::Tensor indx,
  at::Tensor aggr,
  at::Tensor cost,
  at::Tensor edge, 
  int max_dh,
  int max_dw) {

  int B = cost.size(0);
  int H = cost.size(1);
  int W = cost.size(2);
  int L = cost.size(3);
  int C = edge.size(1);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(cost.type(), "prop_hori_neg_forward", ([&] {

    auto indx_data = indx.data<int16_t>();
    auto aggr_data = aggr.data<scalar_t>();
    auto cost_data = cost.data<scalar_t>();
    auto edge_data = edge.data<scalar_t>();    

    scalar_t *buffer;
    cudaMalloc(&buffer, H*L*sizeof(scalar_t));

    for (int i = 0; i < B; ++i) {
      auto indx_i = indx_data + i*L*H*W;      
      auto aggr_i = aggr_data + i*L*H*W;      
      auto cost_i = cost_data + i*L*H*W;
      auto edge_i = edge_data + i*C*H*W;

      prop_hori_neg_forward_kernel<scalar_t><<<H, BLOCK_SIZE>>> 
        (indx_i, aggr_i, buffer, cost_i, edge_i, C, L, H, W, max_dh, max_dw);
    } 
  
    cudaFree(buffer);

  }));
}


void prop_vert_pos_forward_cuda(
  at::Tensor indx,
  at::Tensor aggr,
  at::Tensor cost,
  at::Tensor edge, 
  int max_dh,
  int max_dw) {

  int B = cost.size(0);
  int H = cost.size(1);
  int W = cost.size(2);
  int L = cost.size(3);
  int C = edge.size(1);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(cost.type(), "prop_vert_pos_forward", ([&] {

    auto indx_data = indx.data<int16_t>();
    auto aggr_data = aggr.data<scalar_t>();
    auto cost_data = cost.data<scalar_t>();
    auto edge_data = edge.data<scalar_t>();    

    scalar_t *buffer;
    cudaMalloc(&buffer, W*L*sizeof(scalar_t));

    for (int i = 0; i < B; ++i) {
      auto indx_i = indx_data + i*L*H*W;      
      auto aggr_i = aggr_data + i*L*H*W;
      auto cost_i = cost_data + i*L*H*W;
      auto edge_i = edge_data + i*C*H*W;

      prop_vert_pos_forward_kernel<scalar_t><<<W, BLOCK_SIZE>>> 
        (indx_i, aggr_i, buffer, cost_i, edge_i, C, L, H, W, max_dh, max_dw);
    }  
  
    cudaFree(buffer);

  }));
}


void prop_vert_neg_forward_cuda(
  at::Tensor indx,
  at::Tensor aggr,
  at::Tensor cost,
  at::Tensor edge, 
  int max_dh,
  int max_dw) {

  int B = cost.size(0);
  int H = cost.size(1);
  int W = cost.size(2);
  int L = cost.size(3);
  int C = edge.size(1);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(cost.type(), "prop_vert_neg_forward", ([&] {

    auto indx_data = indx.data<int16_t>();
    auto aggr_data = aggr.data<scalar_t>();
    auto cost_data = cost.data<scalar_t>();
    auto edge_data = edge.data<scalar_t>();    

    scalar_t *buffer;
    cudaMalloc(&buffer, W*L*sizeof(scalar_t));

    for (int i = 0; i < B; ++i) {
      auto indx_i = indx_data + i*L*H*W;      
      auto aggr_i = aggr_data + i*L*H*W;
      auto cost_i = cost_data + i*L*H*W;
      auto edge_i = edge_data + i*C*H*W;

      prop_vert_neg_forward_kernel<scalar_t><<<W, BLOCK_SIZE>>> 
        (indx_i, aggr_i, buffer, cost_i, edge_i, C, L, H, W, max_dh, max_dw);
    }  
  
    cudaFree(buffer);

  }));
}


template <typename scalar_t> 
__global__ void prop_hori_pos_backward_kernel(scalar_t *grad_cost, 
                                              scalar_t *grad_edge, 
                                              scalar_t *grad_aggr,                                               
                                              scalar_t *edge,
                                              int16_t  *indx,
                                              scalar_t *buffer_prev,
                                              scalar_t *buffer_curr,
                                              int C, int L, int H, int W, 
                                              int max_dh, int max_dw) {

  int h = blockIdx.x;
  int t = threadIdx.x;

  for (int l = t; l < L; l += blockDim.x) {
    grad_cost[(h*W + W-1)*L + l] += grad_aggr[(h*W + W-1)*L + l];
    buffer_prev[h*L + l] = grad_aggr[(h*W + W-1)*L + l];
  }

  for (int w = W-1; w > 0; --w) {

    __shared__ scalar_t weight;
    __shared__ scalar_t acc_grad;

    if (t == 0) {
      acc_grad = 0;
      weight = 0;
    }

    __syncthreads();

    for (int c = t; c < C; c += blockDim.x) {
      atomicAdd(&weight, abs(edge[c*H*W + h*W + w] - edge[c*H*W + h*W + w-1]));
    }

    __syncthreads();

    if (t == 0) {
      weight = exp(-weight);
    }

    __syncthreads();

    for (int l = t; l < L; l += blockDim.x) {
      buffer_curr[h*L + l] = grad_aggr[(h*W + w-1)*L + l];
    }

    __syncthreads();

    for (int l = t; l < L; l += blockDim.x) {

      int min_indx = indx[(h*W + w)*L + l];

      scalar_t grad_cost_prev = buffer_prev[h*L + l];
      atomicAdd(buffer_curr + h*L + min_indx, grad_cost_prev);

      int dh1 = l / (2*max_dw+1) - max_dh;
      int dw1 = l % (2*max_dw+1) - max_dw;
      int dh2 = min_indx / (2*max_dw+1) - max_dh;
      int dw2 = min_indx % (2*max_dw+1) - max_dw;
      scalar_t dist = (scalar_t) abs(dh1 - dh2) + abs(dw1 - dw2);
      atomicAdd(&acc_grad, dist * grad_cost_prev);
    } 

    __syncthreads();

    if (t == 0) {
      acc_grad *= weight;
    }

    __syncthreads();

    for (int l = t; l < L; l += blockDim.x) {
      grad_cost[(h*W + w-1)*L + l] += buffer_curr[h*L + l];
    }

    swap_cuda(buffer_curr, buffer_prev);

    __syncthreads();

    for (int c = t; c < C; c += blockDim.x) {
      scalar_t diff = acc_grad * sign(edge[c*H*W + h*W + w] - edge[c*H*W + h*W + w-1]);
      grad_edge[c*H*W + h*W + w] -= diff;
      grad_edge[c*H*W + h*W + w-1] += diff;
    }
  }
}


template <typename scalar_t> 
__global__ void prop_hori_neg_backward_kernel(scalar_t *grad_cost,
                                              scalar_t *grad_edge,
                                              scalar_t *grad_aggr,
                                              scalar_t *edge, 
                                              int16_t  *indx,
                                              scalar_t *buffer_prev,
                                              scalar_t *buffer_curr,
                                              int C, int L, int H, int W, 
                                              int max_dh, int max_dw) {
  int h = blockIdx.x;
  int t = threadIdx.x;

  for (int l = t; l < L; l += blockDim.x) {
    grad_cost[(h*W + 0)*L + l] += grad_aggr[(h*W + 0)*L + l];
    buffer_prev[h*L + l] = grad_aggr[(h*W + 0)*L + l];
  }

  for (int w = 0; w < W-1; ++w) {

    __shared__ scalar_t weight;
    __shared__ scalar_t acc_grad;

    if (t == 0) {
      acc_grad = 0;
      weight = 0;
    }

    __syncthreads();

    for (int c = t; c < C; c += blockDim.x) {
      atomicAdd(&weight, abs(edge[c*H*W + h*W + w] - edge[c*H*W + h*W + w+1])); 
    }

    __syncthreads();

    if (t == 0) {
      weight = exp(-weight);
    }

    __syncthreads();

    for (int l = t; l < L; l += blockDim.x) {
      buffer_curr[h*L + l] = grad_aggr[(h*W + w+1)*L + l];
    }

    __syncthreads();

    for (int l = t; l < L; l += blockDim.x) {

      int min_indx = indx[(h*W + w)*L + l];

      scalar_t grad_cost_prev = buffer_prev[h*L + l];
      atomicAdd(buffer_curr + h*L + min_indx, grad_cost_prev);

      int dh1 = l / (2*max_dw+1) - max_dh;
      int dw1 = l % (2*max_dw+1) - max_dw;
      int dh2 = min_indx / (2*max_dw+1) - max_dh;
      int dw2 = min_indx % (2*max_dw+1) - max_dw;
      scalar_t dist = (scalar_t) abs(dh1 - dh2) + abs(dw1 - dw2);
      atomicAdd(&acc_grad, dist * grad_cost_prev);
    } 

    __syncthreads();

    if (t == 0) {
      acc_grad *= weight;
    }

    __syncthreads();

    for (int l = t; l < L; l += blockDim.x) {
      grad_cost[(h*W + w+1)*L + l] += buffer_curr[h*L + l];
    }

    swap_cuda(buffer_curr, buffer_prev);

    __syncthreads();

    for (int c = t; c < C; c += blockDim.x) {
      scalar_t diff = acc_grad * sign(edge[c*H*W + h*W + w] - edge[c*H*W + h*W + w+1]);
      grad_edge[c*H*W + h*W + w] -= diff;
      grad_edge[c*H*W + h*W + w+1] += diff;
    }
  }
}


template <typename scalar_t> 
__global__ void prop_vert_pos_backward_kernel(scalar_t *grad_cost,
                                              scalar_t *grad_edge,
                                              scalar_t *grad_aggr,
                                              scalar_t *edge, 
                                              int16_t  *indx,
                                              scalar_t *buffer_prev,
                                              scalar_t *buffer_curr,
                                              int C, int L, int H, int W, 
                                              int max_dh, int max_dw) {

  int w = blockIdx.x;
  int t = threadIdx.x;

  for (int l = t; l < L; l += blockDim.x) {
    grad_cost[((H-1)*W + w)*L + l] += grad_aggr[((H-1)*W + w)*L + l];
    buffer_prev[w*L + l] = grad_aggr[((H-1)*W + w)*L + l];
  }

  for (int h = H-1; h > 0; --h) {

    __shared__ scalar_t weight;
    __shared__ scalar_t acc_grad;

    if (t == 0) {
      acc_grad = 0;
      weight = 0;
    }

    __syncthreads();

    for (int c = t; c < C; c += blockDim.x) {
      atomicAdd(&weight, abs(edge[c*H*W + h*W + w] - edge[c*H*W + (h-1)*W + w])); 
    }

    __syncthreads();

    if (t == 0) {
      weight = exp(-weight);
    }

    __syncthreads();

    for (int l = t; l < L; l += blockDim.x) {
      buffer_curr[w*L + l] = grad_aggr[((h-1)*W + w)*L + l];
    }

    __syncthreads();

    for (int l = t; l < L; l += blockDim.x) {

      int min_indx = indx[(h*W + w)*L + l];

      scalar_t grad_cost_prev = buffer_prev[w*L + l];
      atomicAdd(buffer_curr + w*L + min_indx, grad_cost_prev);

      int dh1 = l / (2*max_dw+1) - max_dh;
      int dw1 = l % (2*max_dw+1) - max_dw;
      int dh2 = min_indx / (2*max_dw+1) - max_dh;
      int dw2 = min_indx % (2*max_dw+1) - max_dw;
      scalar_t dist = (scalar_t) abs(dh1 - dh2) + abs(dw1 - dw2);
      atomicAdd(&acc_grad, dist * grad_cost_prev);
    } 

    __syncthreads();

    if (t == 0) {
      acc_grad *= weight;
    }

    __syncthreads();

    for (int l = t; l < L; l += blockDim.x) {
      grad_cost[((h-1)*W + w)*L + l] += buffer_curr[w*L + l];
    }

    swap_cuda(buffer_curr, buffer_prev);

    __syncthreads();

    for (int c = t; c < C; c += blockDim.x) {
      scalar_t diff = acc_grad * sign(edge[c*H*W + h*W + w] - edge[c*H*W + (h-1)*W + w]);
      grad_edge[c*H*W + h*W + w] -= diff;
      grad_edge[c*H*W + (h-1)*W + w] += diff;
    }
  }
}

template <typename scalar_t> 
__global__ void prop_vert_neg_backward_kernel(scalar_t *grad_cost,
                                              scalar_t *grad_edge,
                                              scalar_t *grad_aggr,
                                              scalar_t *edge, 
                                              int16_t  *indx,
                                              scalar_t *buffer_prev,
                                              scalar_t *buffer_curr,
                                              int C, int L, int H, int W, 
                                              int max_dh, int max_dw) {
  int w = blockIdx.x;
  int t = threadIdx.x;

  for (int l = t; l < L; l += blockDim.x) {
    grad_cost[(0*W + w)*L + l] += grad_aggr[(0*W + w)*L + l];
    buffer_prev[w*L + l] = grad_aggr[(0*W + w)*L + l];
  }

  for (int h = 0; h < H-1; ++h) {

    __shared__ scalar_t weight;
    __shared__ scalar_t acc_grad;

    if (t == 0) {
      acc_grad = 0;
      weight = 0;
    }

    __syncthreads();

    for (int c = t; c < C; c += blockDim.x) {
      atomicAdd(&weight, abs(edge[c*H*W + h*W + w] - edge[c*H*W + (h+1)*W + w])); 
    }

    __syncthreads();

    if (t == 0) {
      weight = exp(-weight);
    }

    __syncthreads();

    for (int l = t; l < L; l += blockDim.x) {
      buffer_curr[w*L + l] = grad_aggr[((h+1)*W + w)*L + l];
    }

    __syncthreads();

    for (int l = t; l < L; l += blockDim.x) {

      int min_indx = indx[(h*W + w)*L + l];

      scalar_t grad_cost_prev = buffer_prev[w*L + l];
      atomicAdd(buffer_curr + w*L + min_indx, grad_cost_prev);

      int dh1 = l / (2*max_dw+1) - max_dh;
      int dw1 = l % (2*max_dw+1) - max_dw;
      int dh2 = min_indx / (2*max_dw+1) - max_dh;
      int dw2 = min_indx % (2*max_dw+1) - max_dw;
      scalar_t dist = (scalar_t) abs(dh1 - dh2) + abs(dw1 - dw2);
      atomicAdd(&acc_grad, dist * grad_cost_prev);
    } 

    __syncthreads();

    if (t == 0) {
      acc_grad *= weight;
    }

    __syncthreads();

    for (int l = t; l < L; l += blockDim.x) {
      grad_cost[((h+1)*W + w)*L + l] += buffer_curr[w*L + l];
    }

    swap_cuda(buffer_curr, buffer_prev);

    __syncthreads();

    for (int c = t; c < C; c += blockDim.x) {
      scalar_t diff = acc_grad * sign(edge[c*H*W + h*W + w] - edge[c*H*W + (h+1)*W + w]);
      grad_edge[c*H*W + h*W + w] -= diff;
      grad_edge[c*H*W + (h+1)*W + w] += diff;
    }
  }
}
                                            

void prop_hori_pos_backward_cuda(
  at::Tensor grad_cost,
  at::Tensor grad_edge,
  at::Tensor grad_aggr,
  at::Tensor edge,     
  at::Tensor indx,
  int max_dh,
  int max_dw) {

  int B = grad_aggr.size(0);
  int H = grad_aggr.size(1);
  int W = grad_aggr.size(2);
  int L = grad_aggr.size(3);
  int C = edge.size(1);      

  AT_DISPATCH_FLOATING_TYPES(grad_aggr.type(), "prop_hori_pos_backward", ([&] {
    
    auto grad_cost_data = grad_cost.data<scalar_t>();
    auto grad_edge_data = grad_edge.data<scalar_t>();        
    auto grad_aggr_data = grad_aggr.data<scalar_t>();        
    auto edge_data = edge.data<scalar_t>();      
    auto indx_data = indx.data<int16_t>();      
    
    scalar_t *buffer_prev, *buffer_curr;
    cudaMalloc(&buffer_prev, H*L*sizeof(scalar_t));
    cudaMalloc(&buffer_curr, H*L*sizeof(scalar_t));

    for (int i = 0; i < B; ++i) {      
      auto grad_cost_i = grad_cost_data + i*L*H*W;
      auto grad_edge_i = grad_edge_data + i*C*H*W;
      auto grad_aggr_i = grad_aggr_data + i*L*H*W;      
      auto edge_i = edge_data + i*C*H*W;
      auto indx_i = indx_data + i*L*H*W;
      
      prop_hori_pos_backward_kernel<<<H, BLOCK_SIZE>>>
        (grad_cost_i, grad_edge_i, grad_aggr_i,
         edge_i, indx_i, buffer_prev, buffer_curr,
         C, L, H, W, max_dh, max_dw);        
        
    }

    cudaFree(buffer_prev);
    cudaFree(buffer_curr);
    
  }));
}


void prop_hori_neg_backward_cuda(
  at::Tensor grad_cost,
  at::Tensor grad_edge,
  at::Tensor grad_aggr,
  at::Tensor edge,     
  at::Tensor indx,
  int max_dh,
  int max_dw) {

  int B = grad_aggr.size(0);
  int H = grad_aggr.size(1);
  int W = grad_aggr.size(2);
  int L = grad_aggr.size(3);
  int C = edge.size(1);      

  AT_DISPATCH_FLOATING_TYPES(grad_aggr.type(), "prop_hori_neg_backward", ([&] {
    
    auto grad_cost_data = grad_cost.data<scalar_t>();
    auto grad_edge_data = grad_edge.data<scalar_t>();    
    auto grad_aggr_data = grad_aggr.data<scalar_t>();          
    auto edge_data = edge.data<scalar_t>();     
    auto indx_data = indx.data<int16_t>();      
    
    scalar_t *buffer_prev, *buffer_curr;
    cudaMalloc(&buffer_prev, H*L*sizeof(scalar_t));
    cudaMalloc(&buffer_curr, H*L*sizeof(scalar_t));
    
    for (int i = 0; i < B; ++i) {      
      auto grad_cost_i = grad_cost_data + i*L*H*W;
      auto grad_edge_i = grad_edge_data + i*C*H*W;
      auto grad_aggr_i = grad_aggr_data + i*L*H*W;      
      auto edge_i = edge_data + i*C*H*W;      
      auto indx_i = indx_data + i*L*H*W;

      prop_hori_neg_backward_kernel<<<H, BLOCK_SIZE>>>
        (grad_cost_i, grad_edge_i, grad_aggr_i,
         edge_i, indx_i, buffer_prev, buffer_curr,
         C, L, H, W, max_dh, max_dw);        
        
    }

    cudaFree(buffer_prev);
    cudaFree(buffer_curr);
    
  }));
}


void prop_vert_pos_backward_cuda(
  at::Tensor grad_cost,
  at::Tensor grad_edge,
  at::Tensor grad_aggr,
  at::Tensor edge,     
  at::Tensor indx,
  int max_dh,
  int max_dw) {

  int B = grad_aggr.size(0);
  int H = grad_aggr.size(1);
  int W = grad_aggr.size(2);
  int L = grad_aggr.size(3);
  int C = edge.size(1);      

  AT_DISPATCH_FLOATING_TYPES(grad_aggr.type(), "prop_vert_pos_backward", ([&] {
    
    auto grad_cost_data = grad_cost.data<scalar_t>();
    auto grad_edge_data = grad_edge.data<scalar_t>();    
    auto grad_aggr_data = grad_aggr.data<scalar_t>();          
    auto edge_data = edge.data<scalar_t>();     
    auto indx_data = indx.data<int16_t>();      
    
    scalar_t *buffer_prev, *buffer_curr;
    cudaMalloc(&buffer_prev, W*L*sizeof(scalar_t));
    cudaMalloc(&buffer_curr, W*L*sizeof(scalar_t));

    for (int i = 0; i < B; ++i) {      
      auto grad_cost_i = grad_cost_data + i*L*H*W;
      auto grad_edge_i = grad_edge_data + i*C*H*W;
      auto grad_aggr_i = grad_aggr_data + i*L*H*W;      
      auto edge_i = edge_data + i*C*H*W;      
      auto indx_i = indx_data + i*L*H*W;

      prop_vert_pos_backward_kernel<<<W, BLOCK_SIZE>>>
        (grad_cost_i, grad_edge_i, grad_aggr_i,
         edge_i, indx_i, buffer_prev, buffer_curr,
         C, L, H, W, max_dh, max_dw);        
           
    }

    cudaFree(buffer_prev);
    cudaFree(buffer_curr);
    
  }));
}


void prop_vert_neg_backward_cuda(
  at::Tensor grad_cost,
  at::Tensor grad_edge,
  at::Tensor grad_aggr,
  at::Tensor edge,     
  at::Tensor indx,
  int max_dh,
  int max_dw) {

  int B = grad_aggr.size(0);
  int H = grad_aggr.size(1);
  int W = grad_aggr.size(2);
  int L = grad_aggr.size(3);
  int C = edge.size(1);      

  AT_DISPATCH_FLOATING_TYPES(grad_aggr.type(), "prop_vert_neg_backward", ([&] {

    auto grad_cost_data = grad_cost.data<scalar_t>();
    auto grad_edge_data = grad_edge.data<scalar_t>();    
    auto grad_aggr_data = grad_aggr.data<scalar_t>();          
    auto edge_data = edge.data<scalar_t>();     
    auto indx_data = indx.data<int16_t>();      
    
    scalar_t *buffer_prev, *buffer_curr;
    cudaMalloc(&buffer_prev, W*L*sizeof(scalar_t));
    cudaMalloc(&buffer_curr, W*L*sizeof(scalar_t));

    for (int i = 0; i < B; ++i) {      

      auto grad_cost_i = grad_cost_data + i*L*H*W;
      auto grad_edge_i = grad_edge_data + i*C*H*W;
      auto grad_aggr_i = grad_aggr_data + i*L*H*W;      
      auto edge_i = edge_data + i*C*H*W;      
      auto indx_i = indx_data + i*L*H*W;

      prop_vert_neg_backward_kernel<<<W, BLOCK_SIZE>>>
        (grad_cost_i, grad_edge_i, grad_aggr_i,
         edge_i, indx_i, buffer_prev, buffer_curr,
         C, L, H, W, max_dh, max_dw);        
    }

    cudaFree(buffer_prev);
    cudaFree(buffer_curr);

  }));
}
