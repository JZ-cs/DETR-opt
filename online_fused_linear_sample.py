import os
import subprocess
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load, load_inline

cpp_src = """
#include <torch/extension.h>
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor set_increase_cuda(
    at::Tensor input
);

at::Tensor set_increase_forward(
    at::Tensor input
){
    CHECK_INPUT(input);
    return set_increase_cuda(input);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//   m.def("forward", &fused_linear_sample_forward, "fused linear sample forward function");
  m.def("set_increase", &set_increase_forward, "test func set increase");
//   m.def("sample_forward_2", &only_sample_forward_2, "ONLY sample forward function 2 func");
}

"""

cuda_src = """
template <typename scalar_t> 
__global__ void set_increase_cuda_kernel(
    scalar_t*  input,    
    scalar_t*  output
){
    const int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    output[idx] = idx + 1;
}

at::Tensor set_increase_cuda(
    at::Tensor input
){
    const int batch_size = input.size(0);
    const int query_size = input.size(1);
    const int dim_size = input.size(2);
    const torch::OptionalDeviceGuard device_guard(input.device());
    auto output = torch::empty({batch_size, query_size, dim_size}, input.options());

    const int threads = dim_size;
    const dim3 blocks(query_size, batch_size);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "set increase cuda", ([&] {
        set_increase_cuda_kernel<<<blocks, threads>>>(
            input.data<scalar_t>(),
            output.data<scalar_t>()
        );
    }));
    return output;
}
"""

fused_linear_sample_cuda = load_inline(name='fused_linear_sample_cuda', cpp_sources=[cpp_src],
                   cuda_sources=[cuda_src])
B = 64
H = 16
W = 16
s_len = H*W
nheads = 16
d_head = 64
npoints = 8
d = nheads * d_head
len_q = 256


if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    fake_x = torch.Tensor([x+1 for x in range(B*s_len*d)])\
                .to(torch.device('cuda'))\
                .to(dtype=torch.float16)\
                .view((B, s_len, d))
    fake_x = torch.ones((B, s_len, d), dtype=torch.float16, device = torch.device('cuda'))
    fake_projw = nn.Parameter(torch.ones((d,d), dtype=torch.float16, device=torch.device('cuda')))

    # x = torch.randn((B, s_len, d), dtype=torch.float16, device = torch.device('cuda'))
    # projw = nn.Parameter(torch.randn((d,d), dtype=torch.float16, device=torch.device('cuda')))
    x = fake_x
    projw = fake_projw

    samp_pts = torch.ones((B, len_q, nheads, npoints, 2), dtype=torch.float16, device=torch.device('cuda'))
    sp_offs = torch.Tensor([x+1 for x in range(B*len_q*nheads*npoints)])\
                .to(torch.device('cuda'))\
                .view((B, len_q, nheads, npoints, 1))
    attn_weights = torch.ones((B, len_q, nheads, npoints), dtype=torch.float16, device=torch.device('cuda'))
    # samp_pts = torch.ones((B, nheads, len_q, npoints, 2), dtype=torch.float16, device=torch.device('cuda'))
    # attn_weights = torch.ones((B, nheads, len_q, npoints), dtype=torch.float16, device=torch.device('cuda'))
    # _t_cuda_os, res_cuda_os = test_cuda(only_sample, x, projw, samp_pts, attn_weights)
    # _t_cuda, res_cuda = test_cuda(fused_sample, x, projw, samp_pts, attn_weights)
    # print(f'os-time= {_t_cuda_os}ms  res.shape:{res_cuda_os.shape} res.sum:{torch.sum(res_cuda_os)}')
    # print(f'fs-time= {_t_cuda}ms  res.shape:{res_cuda.shape} res.sum:{torch.sum(res_cuda)}')

    # _t_cuda, res_cuda = test_cuda(fused_sample, x, projw, samp_pts, attn_weights)
    # print(f'fs-time= {_t_cuda}ms  res.shape:{res_cuda.shape} res.sum:{torch.sum(res_cuda[0,0,:])}')
    # print(res_cuda[0,0,0])

    print('-' * 100)
    inp = torch.zeros((8, 16, 128), dtype=torch.float16, device=torch.device('cuda'))
    output = fused_linear_sample_cuda.set_increase(inp)
    print(output.shape, torch.sum(output), output[0,0,9])

# cpp_src = """
# void add_cu(torch::Tensor a, torch::Tensor b, torch::Tensor c);

# #define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
# #define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
# #define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

# void add(torch::Tensor a, torch::Tensor b, torch::Tensor c){
#     CHECK_INPUT(a);
#     CHECK_INPUT(b);
#     CHECK_INPUT(c);
#     add_cu(a, b, c);
# }

# PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#   m.def("add", &add, "add(CUDA)");
# }
# """

# cuda_src = """
# #include <torch/extension.h>
# #include <cuda.h>
# #include <cuda_runtime.h>




# template <typename scalar_t>
# __global__ void add_kernel(
#     scalar_t* __restrict__ a, 
#     scalar_t* __restrict__ b, 
#     scalar_t* __restrict__ c, 
#     size_t size
# ){
#     const int index = blockIdx.x * blockDim.x + threadIdx.x;
#     const int stride = blockDim.x * gridDim.x;
#     for (int i = index; i < size; i += stride){
#         a[i] = b[i] + c[i];
#     }
# }


# void add_cu(torch::Tensor a, torch::Tensor b, torch::Tensor c){
#     const auto size = a.size(0);

#     const int threads = 8;
#     const dim3 blocks((size + threads - 1) / threads);

#     AT_DISPATCH_FLOATING_TYPES_AND_HALF(a.type(), "add cuda", ([&] {
#         add_kernel<scalar_t><<<blocks, threads>>>(
#             a.data<scalar_t>(), 
#             b.data<scalar_t>(), 
#             c.data<scalar_t>(), 
#             size
#         );
#     }));
# }
# """

# add = load_inline(name='add', cpp_sources=[cpp_src],
#                    cuda_sources=[cuda_src])

# a = torch.zeros((100, ),dtype=torch.float16,device=torch.device('cuda'))
# b = torch.ones((100, ),dtype=torch.float16,device=torch.device('cuda')) * 10
# c = torch.ones((100, ),dtype=torch.float16,device=torch.device('cuda'))
# # a = a.cuda(0)
# # b = b.cuda(0)
# # c = c.cuda(0)
# add.add(a, b, c)
# print(a)