// Custom CUDA kernels for the FNO spectral channel-mixing multiply.
//
// Forward:  v[b,o,k] = sum_i w[i,o,k] * u[b,i,k]                 (complex)
// Backward: grad_u[b,i,k] = sum_o conj(w[i,o,k]) * gv[b,o,k]
//           grad_w[i,o,k] = sum_b conj(u[b,i,k]) * gv[b,o,k]
//
// Tensors are torch complex viewed as real: interleaved float pairs [...,2]
// with real/imag adjacent, the contiguous mode axis k last. We assign one
// thread per output element with k as the fastest thread index, and each thread
// serially reduces over the contraction dimension. Because k is contiguous in
// memory, the 32 threads of a warp read 32 consecutive complex values per step,
// so the loads coalesce. This beats the textbook one-warp-per-(b,k) warp-shuffle
// reduction here: the contraction dims (C_in, C_out, B) are only 16-64 wide, so
// a warp reduction would leave most lanes idle and, worse, stride the loads.
// Templated on scalar_t so the same code serves float32 and float64 (gradcheck).

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

constexpr int BLOCK = 256;

static inline int grid_for(long n) { return (int)((n + BLOCK - 1) / BLOCK); }

// v[b,o,k] = sum_i w[i,o,k] * u[b,i,k]
template <typename scalar_t>
__global__ void forward_kernel(const scalar_t* __restrict__ u,
                               const scalar_t* __restrict__ w,
                               scalar_t* __restrict__ v,
                               int B, int Ci, int Co, int K) {
  long tid = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long total = (long)B * Co * K;
  if (tid >= total) return;
  int k = tid % K;
  int o = (tid / K) % Co;
  int b = tid / ((long)K * Co);

  scalar_t accr = 0, acci = 0;
  long ubase = (long)b * Ci * K + k;          // u[b,0,k] element index
  long wbase = (long)o * K + k;               // w[0,o,k]
  for (int i = 0; i < Ci; ++i) {
    long ui = (ubase + (long)i * K) * 2;
    long wi = (wbase + (long)i * Co * K) * 2;
    scalar_t ur = u[ui], uii = u[ui + 1];
    scalar_t wr = w[wi], wii = w[wi + 1];
    accr += wr * ur - wii * uii;
    acci += wr * uii + wii * ur;
  }
  long vi = (((long)b * Co + o) * K + k) * 2;
  v[vi] = accr;
  v[vi + 1] = acci;
}

// grad_u[b,i,k] = sum_o conj(w[i,o,k]) * gv[b,o,k]
template <typename scalar_t>
__global__ void grad_u_kernel(const scalar_t* __restrict__ gv,
                              const scalar_t* __restrict__ w,
                              scalar_t* __restrict__ gu,
                              int B, int Ci, int Co, int K) {
  long tid = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long total = (long)B * Ci * K;
  if (tid >= total) return;
  int k = tid % K;
  int i = (tid / K) % Ci;
  int b = tid / ((long)K * Ci);

  scalar_t accr = 0, acci = 0;
  long gbase = (long)b * Co * K + k;          // gv[b,0,k]
  long wbase = (long)i * Co * K + k;          // w[i,0,k]
  for (int o = 0; o < Co; ++o) {
    long gi = (gbase + (long)o * K) * 2;
    long wi = (wbase + (long)o * K) * 2;
    scalar_t gr = gv[gi], gii = gv[gi + 1];
    scalar_t wr = w[wi], wii = w[wi + 1];     // conj(w) = (wr, -wii)
    accr += wr * gr + wii * gii;
    acci += wr * gii - wii * gr;
  }
  long gui = (((long)b * Ci + i) * K + k) * 2;
  gu[gui] = accr;
  gu[gui + 1] = acci;
}

// grad_w[i,o,k] = sum_b conj(u[b,i,k]) * gv[b,o,k]
template <typename scalar_t>
__global__ void grad_w_kernel(const scalar_t* __restrict__ gv,
                              const scalar_t* __restrict__ u,
                              scalar_t* __restrict__ gw,
                              int B, int Ci, int Co, int K) {
  long tid = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long total = (long)Ci * Co * K;
  if (tid >= total) return;
  int k = tid % K;
  int o = (tid / K) % Co;
  int i = tid / ((long)K * Co);

  scalar_t accr = 0, acci = 0;
  long gbase = (long)o * K + k;               // gv[0,o,k]
  long ubase = (long)i * K + k;               // u[0,i,k]
  for (int b = 0; b < B; ++b) {
    long gi = (gbase + (long)b * Co * K) * 2;
    long ui = (ubase + (long)b * Ci * K) * 2;
    scalar_t gr = gv[gi], gii = gv[gi + 1];
    scalar_t ur = u[ui], uii = u[ui + 1];     // conj(u) = (ur, -uii)
    accr += ur * gr + uii * gii;
    acci += ur * gii - uii * gr;
  }
  long gwi = (((long)i * Co + o) * K + k) * 2;
  gw[gwi] = accr;
  gw[gwi + 1] = acci;
}

// u: [B,Ci,K] complex, w: [Ci,Co,K] complex -> v: [B,Co,K] complex.
torch::Tensor spectral_mm_forward(torch::Tensor u, torch::Tensor w) {
  TORCH_CHECK(u.is_cuda() && w.is_cuda(), "inputs must be CUDA tensors");
  TORCH_CHECK(u.is_complex() && w.is_complex(), "inputs must be complex");
  u = u.contiguous();
  w = w.contiguous();
  int B = u.size(0), Ci = u.size(1), K = u.size(2), Co = w.size(1);

  auto v = torch::empty({B, Co, K}, u.options());
  auto ur = torch::view_as_real(u), wr = torch::view_as_real(w);
  auto vr = torch::view_as_real(v);

  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(ur.scalar_type(), "spectral_mm_forward", [&] {
    forward_kernel<scalar_t><<<grid_for((long)B * Co * K), BLOCK, 0, stream>>>(
        ur.data_ptr<scalar_t>(), wr.data_ptr<scalar_t>(),
        vr.data_ptr<scalar_t>(), B, Ci, Co, K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
  return v;
}

std::vector<torch::Tensor> spectral_mm_backward(torch::Tensor gv,
                                                torch::Tensor u,
                                                torch::Tensor w) {
  gv = gv.contiguous();
  u = u.contiguous();
  w = w.contiguous();
  int B = u.size(0), Ci = u.size(1), K = u.size(2), Co = w.size(1);

  auto gu = torch::empty_like(u);
  auto gw = torch::empty_like(w);
  auto gvr = torch::view_as_real(gv);
  auto ur = torch::view_as_real(u), wr = torch::view_as_real(w);
  auto gur = torch::view_as_real(gu), gwr = torch::view_as_real(gw);

  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(gvr.scalar_type(), "spectral_mm_backward", [&] {
    grad_u_kernel<scalar_t><<<grid_for((long)B * Ci * K), BLOCK, 0, stream>>>(
        gvr.data_ptr<scalar_t>(), wr.data_ptr<scalar_t>(),
        gur.data_ptr<scalar_t>(), B, Ci, Co, K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    grad_w_kernel<scalar_t><<<grid_for((long)Ci * Co * K), BLOCK, 0, stream>>>(
        gvr.data_ptr<scalar_t>(), ur.data_ptr<scalar_t>(),
        gwr.data_ptr<scalar_t>(), B, Ci, Co, K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
  return {gu, gw};
}
