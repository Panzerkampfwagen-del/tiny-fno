// PyTorch C++ extension binding for the spectral_mm CUDA kernels.
// The implementations live in spectral_mm.cu; here we only expose them.

#include <torch/extension.h>
#include <vector>

torch::Tensor spectral_mm_forward(torch::Tensor u, torch::Tensor w);
std::vector<torch::Tensor> spectral_mm_backward(torch::Tensor gv,
                                                torch::Tensor u,
                                                torch::Tensor w);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &spectral_mm_forward, "spectral_mm forward (CUDA)");
  m.def("backward", &spectral_mm_backward, "spectral_mm backward (CUDA)");
}
