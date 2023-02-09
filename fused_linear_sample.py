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
from ms_utils import tracktime
import triton
import triton.language as tl
# ------------------------------------------------------------------------------------
#                                   jit load 
# ------------------------------------------------------------------------------------
# os.system('rm -rf /mnt/cache/jiangzhen/.cache/torch_extensions/py38_cu116/fused_linear_sample/lock')
generator_flag = []
torch_dir = torch.__path__[0]
if os.path.exists(os.path.join(torch_dir, "include", "ATen", "CUDAGeneratorImpl.h")):
    generator_flag = ["-DOLD_GENERATOR_PATH"]

cc_flag = []

cc_flag.append("-gencode")
cc_flag.append("arch=compute_75,code=sm_75")

def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    release = output[release_idx].split(".")
    bare_metal_major = release[0]
    bare_metal_minor = release[1][0]

    return raw_output, bare_metal_major, bare_metal_minor

def append_nvcc_threads(nvcc_extra_args):
    _, bare_metal_major, bare_metal_minor = get_cuda_bare_metal_version(CUDA_HOME)
    if int(bare_metal_major) >= 11 and int(bare_metal_minor) >= 2:
        return nvcc_extra_args + ["--threads", "4"]
    return nvcc_extra_args

this_dir = '/home/jz/DETR-opt'
fused_linear_sample_cuda = load(
    name="fused_linear_sample", 
    sources=[str(this_dir) + "/fused_linear_sample.cpp", 
        str(this_dir)+'/fused_linear_sample_kernel.cu',
        str(this_dir)+'/fmhead_sample_kernel.cu',
        str(this_dir)+'/fused_sample_kernel_opt.cu'], 
    extra_include_paths=['/home/jz/demo/cutlass/include/'],
    extra_cflags = ["-O3", "-std=c++17"],# + generator_flag,
    extra_cuda_cflags = append_nvcc_threads(
                [
                    "-O3",
                    "-std=c++17",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "--use_fast_math",
                    "--ptxas-options=-v",
                    "-lineinfo",
                    "-DCUTLASS_NVCC_ARCHS=80"
                ]
                + generator_flag
                + cc_flag
            ),
    with_cuda=True)
# # ------------------------------------------------------------------------------------
# #                                   build first 
# # ------------------------------------------------------------------------------------
# import fused_linear_sample_cuda

B = 512
H = 16
W = 16
s_len = H*W
nheads = 16
d_head = 64
npoints = 9
d = nheads * d_head
len_q = 256

@torch.no_grad()
def sampling_torch(x: torch.Tensor, projw, samp_pts: torch.Tensor, attn_weights: torch.Tensor):
    """
    x.shape: B, H*W, d
    samp_pts.shape: B, len_q, nheads, npoints, 2
    attn_weights.shape: B, len_q, nheads, npoints
    """
    #(B, H*W, d)->(B, H, W, nheads, d_head)->(B, nheads, d_head, H, W)->(B*nheads, d_head, H, W)
    x_ = x.reshape(B,H,W,nheads,d_head).permute(0,3,4,1,2).reshape(B*nheads, d_head, H, W)

    #(B,len_q,nheads,npoints,2)->(B,nheads,len_q,n,points,2)->(B*nheads,len_q,npoints,2)
    samp_pts_ = samp_pts.permute(0,2,1,3,4).reshape(B*nheads,len_q,npoints,2)
    samp_grids = 2 * samp_pts_ - 1
    # (B*nheads, d_head, H, W) and (B*nheads,len_q,npoints,2) -> (B*nheads, d_head, len_q, npoints) 
    res = F.grid_sample(x_, samp_grids, 
                        mode='bilinear', padding_mode='zeros', align_corners=True)
    #(B*nheads, d_head, len_q, npoints)->(B, nheads, d_head, len_q, npoints)->(B, len_q, nheads, d_head, npoints)
    res = res.view(B,nheads,d_head, len_q, npoints).permute(0, 3, 1, 2, 4)

    attn_weights_ = attn_weights.view(B, len_q, nheads, 1, npoints)
    #(B, len_q, nheads, d_head or 1, npoints)->(B, len_q, nheads, d_head)
    output = (res * attn_weights_).sum(4)#.view(B, len_q, d)
    return output

def only_sample(x: torch.Tensor, projw: torch.Tensor, samp_pts: torch.Tensor, attn_weights: torch.Tensor):
    output = fused_linear_sample_cuda.only_sample(x, projw, samp_pts, attn_weights, H, W)
    return output

def only_sample_opt(x: torch.Tensor, projw: torch.Tensor, samp_pts: torch.Tensor, attn_weights: torch.Tensor):
    output = fused_linear_sample_cuda.only_sample_opt(x, projw, samp_pts, attn_weights, H, W)
    return output

def test_cuda(func, x, projw, samp_pts, attn_weights, warmup=50, iters=50, ):
    for _ in range(warmup):
        _ = func(x, projw, samp_pts, attn_weights)

    tracktime.cuda_record_start('cuda')
    for _ in range(iters):
        output_cuda = func(x, projw, samp_pts, attn_weights)
    _t = tracktime.cuda_record_end('cuda')
    return _t/iters, output_cuda

def test_torch(func, x, projw, samp_pts, attn_weights, warmup=50, iters=50, ):
    for _ in range(warmup):
        _ = func(x, projw, samp_pts, attn_weights)

    tracktime.cuda_record_start('torch')
    for _ in range(iters):
        output_torch = func(x, projw, samp_pts, attn_weights)
    _t = tracktime.cuda_record_end('torch')
    return _t/iters, output_torch

if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    fake_projw = nn.Parameter(torch.ones((d,d), dtype=torch.float16, device=torch.device('cuda:0')))
    projw = fake_projw

    x = torch.ones((B, s_len, d), dtype=torch.float16, device = torch.device('cuda:0'))

    x_offs = torch.Tensor([i+1 for i in range(s_len)])\
        .to(dtype=torch.float16, device=torch.device('cuda:0'))\
        .view(1, s_len, 1)
    x = x * x_offs

    attn_weights = torch.ones((B, len_q, nheads, npoints), dtype=torch.float16, device=torch.device('cuda:0'))

    attn_offs = torch.Tensor([i+1 for i in range(npoints)])\
        .to(dtype=torch.float16, device=torch.device('cuda:0'))\
        .view(1,1,1,-1)
    attn_weights = attn_weights * attn_offs

    samp_pts = torch.ones((B, len_q, nheads, npoints, 2), dtype=torch.float16, device=torch.device('cuda:0'))

    samp_offs = torch.Tensor([i+1 for i in range(nheads*2)])\
        .to(dtype=torch.float16, device=torch.device('cuda:0'))\
        .view(1, 1, nheads, 1, -1)
    samp_pts = samp_pts/samp_offs
    
    # _t_cuda, res_cuda = test_cuda(only_sample, x, projw, samp_pts, attn_weights)
    # print(f'sample-time= {_t_cuda}ms  res: shape:{res_cuda.shape} sum:{torch.sum(res_cuda)}\
    #      max:{torch.max(res_cuda)} min:{torch.min(res_cuda)}')
    # print(res_cuda[0,0,0,0], res_cuda[B-1, len_q-1, nheads-1, 0])

    _t_cuda, res_cuda = test_cuda(only_sample_opt, x, projw, samp_pts, attn_weights)
    print(f'sample-opt-time= {_t_cuda}ms  res: shape:{res_cuda.shape} sum:{torch.sum(res_cuda)}\
         max:{torch.max(res_cuda)} min:{torch.min(res_cuda)}')
    print(res_cuda[0,0,0,0], res_cuda[B-1, len_q-1, nheads-1, 0])

    # _t_torch, res_torch = test_torch(sampling_torch, x, projw, samp_pts, attn_weights)
    # print(f'torch-sample-time= {_t_torch}ms  res: shape:{res_torch.shape} sum:{torch.sum(res_torch)}\
    #      max:{torch.max(res_torch)} min:{torch.min(res_torch)}')
    # print(res_torch[0,0,0,0], res_torch[B-1, len_q-1, nheads-1, 0])