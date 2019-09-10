#ifndef COMMON_H
#define COMMON_H

const int CUDA_NUM_THREADS = 1024;

#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

inline int GET_BLOCKS(const int N) {
    return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

template <typename T> 
inline __device__ T sign(T val) {
      return (T(0) < val) - (val < T(0));
}

inline __device__ int hw2label(int h, int w, int max_dh, int max_dw) {
    int l = (h+max_dh)*(2*max_dw+1) + w+max_dw;
      return l;
}

__device__ at::Half min(at::Half a, at::Half b) {
    return a < b ? a : b;
}

#endif
