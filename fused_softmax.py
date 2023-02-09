import random 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# @triton.autotune(
#     configs=[
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
#         triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
#         triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
#         triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
#     ],
#     key=['M', 'N', 'K'],
# )
@triton.jit
def fused_softmax_kernel(
    # Pointers to matrices
    q_ptr, pw_ptr, out_ptr,
    # Matrix dimensions
    M, N, K,
    # The stride variables
    stride_qm, stride_qk,
    stride_pwk, stride_pwn,
    stride_om, stride_on,
    N_HEADS: tl.constexpr, N_POINTS: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    offs_qm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_pwn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    q_ptrs = q_ptr + (offs_qm[:, None] * stride_qm + offs_k[None, :] * stride_qk)
    pw_ptrs = pw_ptr + (offs_k[:, None] * stride_pwk + offs_pwn[None, :] * stride_pwn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        q = tl.load(q_ptrs)
        pw = tl.load(pw_ptrs)

        accumulator += tl.dot(q, pw)

        q_ptrs += BLOCK_SIZE_K * stride_qk
        pw_ptrs += BLOCK_SIZE_K * stride_pwk
    accumulator = tl.reshape(accumulator, (BLOCK_SIZE_M, N_HEADS, N_POINTS))
    # accumulator = accumulator.to(tl.float16)
    # softmax over points
    # res = accumulator
    vals_minusmax = accumulator - tl.broadcast_to(tl.max(accumulator, axis=2), (BLOCK_SIZE_M, N_HEADS, 1))
    exp_vals = tl.exp(vals_minusmax)
    sum_vals = tl.sum(exp_vals, 2)
    res = exp_vals# / tl.reshape(sum_vals, accumulator.shape)
    res = res.to(tl.float16)

    res = tl.reshape(res, (BLOCK_SIZE_M,BLOCK_SIZE_N))

    offs_om = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_on = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    out_ptrs = out_ptr + stride_om * offs_om[:, None] + stride_on * offs_on[None, :]
    # out_mask = (offs_om[:, None] < M) & (offs_on[None, :] < N)
    tl.store(out_ptrs, res)
     


B = 512
H = 16
W = 16
len_s = H*W
nheads = 16
d_head = 64
d = 1024
npoints = 16
len_q = 256

def tri_fused_softmax(x: torch.Tensor, projw: torch.Tensor):
    """
    q.shape: B, len_s, dmodel
    proj weights: dmodel, nheads * npoints
    """
    Bx, lx, C = x.shape
    outputs = torch.empty((Bx, lx, nheads*npoints), 
                            device=x.device, dtype=x.dtype)
    M = Bx * lx
    N = nheads * npoints
    K = C
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    grid = lambda META: (
        triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),
    )
    fused_softmax_kernel[grid](
        x, projw , outputs,
        M, N, K,
        x.stride(1), x.stride(2),
        projw.stride(0), projw.stride(1),
        outputs.stride(1), outputs.stride(2),
        2, npoints,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
        GROUP_SIZE_M
    )
    return outputs.view(Bx, lx, nheads, npoints)

def fused_softmax_torch(q:torch.Tensor, projw:torch.Tensor):
    """
    q.shape: B, len_q, dmodel
    proj-weights.shape: dmodel, nheads * npoints
    """
    Bq, lq, dm = q.shape
    attn_weights = torch.matmul(q, projw) # B, len_q, nheads, npoints
    attn_weights = attn_weights.reshape(Bq, lq, nheads, npoints)
    attn_weights = F.softmax(attn_weights, dim=3)
    return attn_weights


from opt_utils import tracktime
def test_torch(func, q, projw, warmup=50, iters=50, ):
    for _ in range(warmup):
        outputs_torch = func(q, projw)

    tracktime.cuda_record_start('torch')
    for _ in range(iters):
        outputs_torch = func(q, projw)
    _t = tracktime.cuda_record_end('torch')
    return _t/iters, outputs_torch

def test_triton(func, q, projw, warmup=50, iters=50, ):
    for _ in range(warmup):
        outputs_tri = func(q, projw)

    tracktime.cuda_record_start('tri')
    for _ in range(iters):
        outputs_tri = func(q, projw)
    _t = tracktime.cuda_record_end('tri')
    return _t/iters, outputs_tri


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['lq'],  # argument names to use as an x-axis for the plot
        x_vals=[
            32*i for i in range(1,9)
        ],  # different possible values for `x_name`
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        # possible values for `line_arg``
        line_vals=['PyTorch', 'Triton'],
        # label name for the lines
        line_names=['PyTorch', 'Triton'],
        # line styles
        styles=[('green', '-'), ('blue', '-')],
        ylabel="TFLOPS",  # label name for the y-axis
        plot_name="fused-softmax-performance",  # name for the plot. Used also as a file name for saving the plot.
        args={},
    )
)
def benchmark(bsz, lq, nh, provider):
    # print(bsz, lq, nh)
    q = torch.randn((B, lq, d), dtype=torch.float16, device=torch.device('cuda'))
    projw = torch.randn((d, nheads*npoints), dtype=torch.float16, device=torch.device('cuda'))
    if provider == 'PyTorch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: fused_softmax_torch(q, projw))
    if provider == 'Triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: tri_fused_softmax(q, projw))
    perf = lambda ms: 2 * bsz * lq * d * nh * 16 * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)

def test_multiple(batch_list, len_q_list, d_list):
    for _bsz in batch_list:
        for _lq in len_q_list:
            for _dm in d_list:
                random.seed(0)
                np.random.seed(0)
                torch.manual_seed(0)
                torch.cuda.manual_seed(0)
                
                q = torch.randn((_bsz, _lq, _dm), dtype=torch.float16, device = torch.device('cuda'))
                # proj = nn.Linear(d, nheads * npoints, bias=False).to(torch.device('cuda'))
                projw = torch.randn((_dm, nheads * npoints), dtype=torch.float16, device = torch.device('cuda'))
                fake_v1 = torch.FloatTensor([i for i in range(_bsz*_lq*_dm)]).to(torch.device('cuda'))\
                            .view((_bsz, _lq, _dm))
                fake_v2 = torch.FloatTensor([i for i in range(_dm*nheads*npoints)]).to(torch.device('cuda'))\
                            .view((_dm, nheads * npoints))
                q = fake_v1
                projw = fake_v2
                warmup_iters = 20
                test_iters = 20
                _t_tri, res_tri = test_triton(tri_fused_softmax, q, projw, 
                                            warmup=warmup_iters, iters=test_iters)
                _t_torch, res_torch = test_torch(fused_softmax_torch, q, projw, 
                                            warmup=warmup_iters, iters=test_iters)

                # print(torch.allclose(res_tri, res_torch))
                # if triton.testing.allclose(res_tri, res_torch):
                #     print("✅ Triton and Torch match")
                # else:
                #     print("❌ Triton and Torch differ")
                # print(f'diff sum= {torch.sum(torch.abs(res_torch-res_tri))}')
                # print(f'diff max= {torch.max(torch.abs(res_torch-res_tri))}')
                print(f'{_bsz}, {_lq}, {_dm} PyTorch:{_t_torch:.6f} vs Triton:{_t_tri:.6f}')

if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    dev = torch.device('cuda:0')
    
    x = torch.randn((B, len_s, d), dtype=torch.float16, device=dev)

    # x_offs = torch.Tensor([i+1 for i in range(len_s)])\
    #     .to(dtype=torch.float16, device=dev)\
    #     .view(1, len_s, 1)
    # x = x*x_offs

    projw = torch.ones((d, nheads * npoints), dtype=torch.float16, device=dev)
    # projw_offs = torch.Tensor([i+1 for i in range(nheads*npoints)])\
    #     .to(dtype=torch.float16, device=dev)\
    #     .view(1, nheads*npoints)
    # projw = projw / projw_offs

    _t_tri, res_tri = test_triton(tri_fused_softmax, x, projw)
    _t_torch, res_torch = test_torch(fused_softmax_torch, x, projw)

    # print(torch.allclose(res_tri, res_torch))
    if triton.testing.allclose(res_tri, res_torch):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")
    # print(f'diff sum= {torch.sum(torch.abs(res_torch-res_tri))}')
    # print(f'diff max= {torch.max(torch.abs(res_torch-res_tri))}')
    print(f"Triton res shape:{res_tri.shape} max:{torch.max(res_tri)} min:{torch.min(res_tri)}")
    print(f"PyTorch res shape:{res_torch.shape} max:{torch.max(res_torch)} min:{torch.min(res_torch)}")
    print(f'Time- PyTorch:{_t_torch:.6f} Triton:{_t_tri:.6f}')
    # test_multiple([512], len_q_list=[256], d_list=[1024])
    # benchmark.run(show_plots=False, print_data=True, save_path='./')



