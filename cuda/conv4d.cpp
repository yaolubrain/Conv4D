#include <torch/torch.h>
#include <vector>

using namespace std;
 
at::Tensor compute_cost_volume_forward_cuda(
    at::Tensor feat1,
    at::Tensor feat2,
    int max_offset_h,
    int max_offset_w);

vector<at::Tensor> compute_cost_volume_backward_cuda(
    at::Tensor grad_output,
    at::Tensor feat1,
    at::Tensor feat2,
    int max_offset_h,
    int max_offset_w);

at::Tensor conv4d_forward_cuda(
    at::Tensor inputs,
    at::Tensor weight,
    int channels_in,
    int channels_out,
    int ksize,
    int stride,
    int padding,
    int dilation);

vector<at::Tensor> conv4d_backward_cuda(
    at::Tensor grad_outputs,
    at::Tensor inputs,
    at::Tensor weight,
    int channels_in,
    int channels_out,
    int ksize,
    int stride,
    int padding,
    int dilation);


// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor compute_cost_volume_forward(
    at::Tensor feat1,
    at::Tensor feat2,
    int max_offset_h,
    int max_offset_w) {
  
  return compute_cost_volume_forward_cuda(feat1, feat2, max_offset_h, max_offset_w);  
}

vector<at::Tensor> compute_cost_volume_backward(
    at::Tensor grad_output,
    at::Tensor feat1,
    at::Tensor feat2,
    int max_offset_h,
    int max_offset_w) {

  return compute_cost_volume_backward_cuda(grad_output, feat1, feat2, max_offset_h, max_offset_w);    
}  

at::Tensor conv4d_forward(
    at::Tensor inputs,
    at::Tensor weight,
    int channels_in,
    int channels_out,
    int ksize,
    int stride,
    int padding,
    int dilation) {

  return conv4d_forward_cuda(inputs, weight, channels_in, channels_out, ksize, stride, padding, dilation);
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

  return conv4d_backward_cuda(grad_outputs, inputs, weight, 
                              channels_in, channels_out, 
                              ksize, stride, padding, dilation);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("compute_cost_volume_forward", &compute_cost_volume_forward, "cost volume forward (CUDA)");  
  m.def("compute_cost_volume_backward", &compute_cost_volume_backward, "cost volume backward (CUDA)");  
  m.def("conv4d_forward", &conv4d_forward, "conv4d forward");
  m.def("conv4d_backward", &conv4d_backward, "conv4d backward");
}
