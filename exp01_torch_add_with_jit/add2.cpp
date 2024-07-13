#include <torch/extension.h>
#include "add2.h"

void torch_launch_add2(torch::Tensor &d,
                       const torch::Tensor &a,
                       const torch::Tensor &b,
                       const torch::Tensor &c,
                       int64_t x,
                       int64_t y) {
    launch_add2((float *)d.data_ptr(),
                (const float *)a.data_ptr(),
                (const float *)b.data_ptr(),
                (const float *)c.data_ptr(),
                x,y);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("torch_launch_add2",
          &torch_launch_add2,
          "add2 kernel warpper");
}

TORCH_LIBRARY(add2, m) {
    m.def("torch_launch_add2", torch_launch_add2);
}