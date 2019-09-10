#ifndef COST_VOLUME_H
#define COST_VOLUME_H

#include <torch/extension.h>
#include "common.h"

at::Tensor compute_cost_volume_forward(
    at::Tensor feat1,
    at::Tensor feat2,
    int max_offset_h,
    int max_offset_w);

vector<at::Tensor> compute_cost_volume_backward(
    at::Tensor grad_cost,
    at::Tensor feat1,
    at::Tensor feat2,
    int max_offset_h,
    int max_offset_w);

#endif