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

void prop_hori_pos_forward_cuda(
    at::Tensor indx,
    at::Tensor aggr,
    at::Tensor cost,
    at::Tensor edge, 
    int max_offset_h,
    int max_offset_w);

void prop_hori_neg_forward_cuda(
    at::Tensor indx,
    at::Tensor aggr,
    at::Tensor cost,
    at::Tensor edge, 
    int max_offset_h,
    int max_offset_w);    

void prop_vert_pos_forward_cuda(
    at::Tensor indx,
    at::Tensor aggr,
    at::Tensor cost,
    at::Tensor edge, 
    int max_offset_h,
    int max_offset_w);

void prop_vert_neg_forward_cuda(
    at::Tensor indx,
    at::Tensor aggr,
    at::Tensor cost,
    at::Tensor edge, 
    int max_offset_h,
    int max_offset_w);       

void prop_hori_pos_backward_cuda(
    at::Tensor grad_cost,
    at::Tensor grad_edge,
    at::Tensor grad_aggr,
    at::Tensor edge,     
    at::Tensor indx,
    int max_offset_h,
    int max_offset_w);    

void prop_hori_neg_backward_cuda(
    at::Tensor grad_cost,
    at::Tensor grad_edge,
    at::Tensor grad_aggr,
    at::Tensor edge,     
    at::Tensor indx,
    int max_offset_h,
    int max_offset_w);    

void prop_vert_pos_backward_cuda(
    at::Tensor grad_cost,
    at::Tensor grad_edge,
    at::Tensor grad_aggr,
    at::Tensor edge,     
    at::Tensor indx,
    int max_offset_h,
    int max_offset_w);    

void prop_vert_neg_backward_cuda(
    at::Tensor grad_cost,
    at::Tensor grad_edge,
    at::Tensor grad_aggr,
    at::Tensor edge,     
    at::Tensor indx,
    int max_offset_h,
    int max_offset_w);                

//void rearrange_BLHW_to_BHWL(at::Tensor &src);
//void rearrange_BHWL_to_BLHW(at::Tensor &src);
at::Tensor BLHW_to_BHWL(at::Tensor &src);
at::Tensor BHWL_to_BLHW(at::Tensor &src);


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


vector<at::Tensor> prop_forward(
    at::Tensor cost, 
    at::Tensor edge, 
    int max_offset_h,
    int max_offset_w) {

  int B = cost.size(0);
  int L = cost.size(1);
  int H = cost.size(2);
  int W = cost.size(3);
  int C = edge.size(1);

  cost = BLHW_to_BHWL(cost);
  cost = cost.view({B,H,W,L});

  //at::Tensor aggr = at::zeros_like(cost);
  at::Tensor aggr = -3 * cost.clone();

  at::Tensor hori_pos_idx = at::zeros({B,H,W,L}, torch::dtype(torch::kInt16).device(torch::kCUDA));
  at::Tensor hori_neg_idx = at::zeros({B,H,W,L}, torch::dtype(torch::kInt16).device(torch::kCUDA));
  at::Tensor vert_pos_idx = at::zeros({B,H,W,L}, torch::dtype(torch::kInt16).device(torch::kCUDA));
  at::Tensor vert_neg_idx = at::zeros({B,H,W,L}, torch::dtype(torch::kInt16).device(torch::kCUDA));

  prop_hori_pos_forward_cuda(hori_pos_idx, aggr, cost, edge, max_offset_h, max_offset_w);
  prop_hori_neg_forward_cuda(hori_neg_idx, aggr, cost, edge, max_offset_h, max_offset_w);
  prop_vert_pos_forward_cuda(vert_pos_idx, aggr, cost, edge, max_offset_h, max_offset_w);
  prop_vert_neg_forward_cuda(vert_neg_idx, aggr, cost, edge, max_offset_h, max_offset_w);

  hori_pos_idx = torch::autograd::make_variable(hori_pos_idx);
  hori_neg_idx = torch::autograd::make_variable(hori_neg_idx);
  vert_pos_idx = torch::autograd::make_variable(vert_pos_idx);
  vert_neg_idx = torch::autograd::make_variable(vert_neg_idx);

  aggr = BHWL_to_BLHW(aggr);
  aggr = aggr.view({B,L,H,W});

  return {aggr, hori_pos_idx, hori_neg_idx, vert_pos_idx, vert_neg_idx};
}


vector<at::Tensor> prop_backward(
    at::Tensor grad_aggr,
    at::Tensor edge, 
    at::Tensor hori_pos_indx, 
    at::Tensor hori_neg_indx, 
    at::Tensor vert_pos_indx, 
    at::Tensor vert_neg_indx,
    int max_offset_h,
    int max_offset_w) {

  int B = grad_aggr.size(0);
  int L = grad_aggr.size(1);
  int H = grad_aggr.size(2);
  int W = grad_aggr.size(3);
  int C = edge.size(1);

  grad_aggr = BLHW_to_BHWL(grad_aggr);

  //at::Tensor grad_cost = torch::zeros_like(grad_aggr);
  at::Tensor grad_cost = -3 * grad_aggr.clone();
  at::Tensor grad_edge = torch::zeros_like(edge);

  prop_hori_pos_backward_cuda(grad_cost, grad_edge, grad_aggr, edge, hori_pos_indx, max_offset_h, max_offset_w);
  prop_hori_neg_backward_cuda(grad_cost, grad_edge, grad_aggr, edge, hori_neg_indx, max_offset_h, max_offset_w);
  prop_vert_pos_backward_cuda(grad_cost, grad_edge, grad_aggr, edge, vert_pos_indx, max_offset_h, max_offset_w);
  prop_vert_neg_backward_cuda(grad_cost, grad_edge, grad_aggr, edge, vert_neg_indx, max_offset_h, max_offset_w);

  grad_cost = BHWL_to_BLHW(grad_cost);

  return {grad_cost, grad_edge};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("compute_cost_volume_forward", &compute_cost_volume_forward, "blsolver hori forward (CUDA)");  
  m.def("compute_cost_volume_backward", &compute_cost_volume_backward, "blsolver hori backward (CUDA)");  
  m.def("prop_forward", &prop_forward, "SGMFlow prop forward");    
  m.def("prop_backward", &prop_backward, "SGMFlow prop backward");
}
