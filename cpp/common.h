#ifndef COMMON_H
#define COMMON_H

using namespace std;

template <typename scalar_t>
scalar_t sigmoid(scalar_t x) {
  return 0.5 * x / (1 + abs(x)) + 0.5;
}

template <typename scalar_t>
scalar_t softplus(scalar_t x) {
  if (x > 0) {
    scalar_t exp_minus_x = exp(-x);
    return -log( exp_minus_x / (1 + exp_minus_x) );
  } else {
    return log( 1 + exp(x) );
  }
}

template <typename scalar_t> 
scalar_t sign(scalar_t val) {
    return (scalar_t(0) < val) - (val < scalar_t(0));
}

inline vector<int> label2hw(int l, int max_offset_h, int max_offset_w) {
  vector<int> hw(2);
  hw[0] = l / (2*max_offset_w+1) - max_offset_h;
  hw[1] = l % (2*max_offset_w+1) - max_offset_w;
  return hw;
}

inline int hw2label(int h, int w, int max_offset_h, int max_offset_w) {
  int l = (h+max_offset_h)*(2*max_offset_w+1) + w+max_offset_w;
  return l;
}



#endif