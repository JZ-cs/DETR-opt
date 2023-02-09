#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <fstream>
#include <string>

#include "cutlass/aligned_buffer.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/layout/vector.h"
#include "cutlass/numeric_types.h"
#include "cutlass/cutlass.h"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/wmma.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/transform/pitch_linear_thread_map.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator_2dthreadtile.h"
#include "cutlass/gemm/threadblock/default_mma_core_wmma.h"
// #include "cutlass/gemm/threadblock/default_mma_core_sm70.h"
// #include "cutlass/gemm/warp/mma_tensor_op_policy.h"
// #include "cutlass/gemm/warp/mma_tensor_op_wmma.h"
#include "cutlass/gemm/threadblock/mma_pipelined.h"
#include "cutlass/platform/platform.h"

#define MEM_ALIGN 256
#define TH_DIM_READ_STRIDE 8
#define WARP_READ_ELEMENTS (TH_DIM_READ_STRIDE * 32) 
#define TH_DIM_CS_STRIDE 8
#define BUFF_SIZE TH_DIM_CS_STRIDE
#define SHMEM_PAD 0
#define BLOCK_COVER_HEADS 8

template <typename scalar_t> 
__global__ void only_sample_forward_mhead_kernel(    
    const scalar_t*  __restrict__ input,    
    const scalar_t*  __restrict__ proj_weights, 
    const scalar_t* __restrict__ sample_locations,   
    const scalar_t* __restrict__ attn_weights,    
    scalar_t*  output,    
    const int std_sp_b, const int std_sp_lq, const int std_sp_h,    
    const int std_aw_b, const int std_aw_lq, const int std_aw_h,    
    const int std_o_b, const int std_o_lq, const int std_o_h,
    const int H, const int W,
    const int batch_size, const int len_q, const int d_model, const int d_head, const int npoints
){      
    // extern __shared__ float4 dynamic_shmem1[];
    // at::Half* fmap_shmem = (at::Half*)dynamic_shmem1;
    // // __shared__ at::Half fmap_shmem[256 * 64];
    // const int len_s = H*W;
    // int n_threads = blockDim.x * blockDim.y;
    // int tb_thread_id = threadIdx.y * blockDim.x + threadIdx.x;   
    // int bid = blockIdx.y;
    // int hid = blockIdx.x * BLOCK_COVER_HEADS;

    // // prologue->read data from global to shared memory
    // // read fmap into shared memory
    // int elements_per_loop = n_threads * TH_DIM_READ_STRIDE;
    // int block_cover_dims = d_head;
    // int shmem_stride = block_cover_dims + SHMEM_PAD; 
    // int pix_per_loop = elements_per_loop / block_cover_dims;
    // int total_elements = len_s * block_cover_dims;
    // int rk_loops = (total_elements + elements_per_loop - 1) / elements_per_loop;

    // int thread_read_start_idx = tb_thread_id * TH_DIM_READ_STRIDE;
    // int hdim_start_idx = thread_read_start_idx % block_cover_dims;
    // int pix_id = thread_read_start_idx / block_cover_dims;

    // int fmap_shmem_index = pix_id * shmem_stride + hdim_start_idx;
    // // int fmap_shmem_index_adding = shmem_stride * pix_per_loop;
    // int input_read_index = bid*len_s*d_model + hid*d_head + pix_id*d_model + hdim_start_idx;
    // // int input_read_index_adding = d_model * pix_per_loop;
    
    // for(int i = 0; i < rk_loops; i++){
    //     // int pix_id = thread_read_start_idx / block_cover_dims;
    //     // fmap_shmem[pix_id * shmem_stride + hdim_start_idx] = input[read_base + pix_id*d_model + hdim_start_idx];
    //     *((float4*)(fmap_shmem + fmap_shmem_index)) = *((float4*)(input + input_read_index));


    //     fmap_shmem_index += shmem_stride * pix_per_loop;
    //     input_read_index += d_model * pix_per_loop;
    //     // pix_id += pix_per_loop;
    //     // thread_read_start_idx += elements_per_loop;
    // }

    // //read attn weights and locations into shared memory
    // int read_id = tb_thread_id;
    // #pragma unroll
    // for(int j = 0; j < r_ca_loops; j++){
    //     if(read_id < 32 * 9){
    //         int coord_offset = r_stage * 32 * 9 * 2;
    //         int attnw_offset = r_stage * 32 * 9;
    //         int rq_id = read_id / 9;
    //         int rpt_id = read_id % 9;
    //         int attnw_read_idx = bid * std_aw_b + rq_id * std_aw_lq + hid * std_aw_h + rpt_id;
    //         int h_coord_read_idx = bid * std_sp_b + rq_id * std_sp_lq + hid * std_sp_h + rpt_id * 2;
    //         coord_shmem[coord_offset + read_id * 2] = sample_locations[h_coord_read_idx];
    //         coord_shmem[coord_offset + read_id * 2 + 1] = sample_locations[h_coord_read_idx + 1];
    //         attnw_shmem[attnw_offset + read_id] = attn_weights[attnw_read_idx];
    //         read_id += n_threads;
    //     }
    // }
    // __syncthreads();

    // // process multiple heads
    // #pragma unroll
    // for(int i = 0; i < BLOCK_COVER_HEADS; i++){
    //     // read into shared memory

    //     //increase hid
    //     hid++;
    // }
    // __shared__ at::Half coord_shmem[2 * 32 * 9 * 2];
    // __shared__ at::Half attnw_shmem[2 * 32 * 9];
    

    // //compute and store
    // // float4 res_reg[2 * BUFF_SIZE];
    // at::Half res_reg[2 *  BUFF_SIZE];
    // at::Half samp_res[TH_DIM_CS_STRIDE];
    // elements_per_loop = n_threads * TH_DIM_CS_STRIDE;
    // int q_per_loop = elements_per_loop / block_cover_dims;
    // int k_loops = (len_q * block_cover_dims + elements_per_loop - 1) / elements_per_loop;
    // int thread_start_idx = tb_thread_id * TH_DIM_CS_STRIDE;

    // // int aw_batch_offset = bid * std_aw_b; int aw_head_offset = hid * std_aw_h;
    // // int sp_batch_offset = bid * std_sp_b; int sp_head_offset = hid * std_sp_h;
    // int qid = thread_start_idx / block_cover_dims;
    // // int aw_idx = bid * std_aw_b + qid * std_aw_lq + hid * std_aw_h;
    // // int sp_idx = bid * std_sp_b + qid * std_sp_lq + hid * std_sp_h;
    // hdim_start_idx = thread_start_idx % block_cover_dims;
    
    // int out_write_idx = bid*std_o_b + qid*std_o_lq + hid*std_o_h + hdim_start_idx;
    // int write_idx_adding = q_per_loop * std_o_lq;
    // int stage = 0;

    // int r_stage = 0;
    // int th_qid = qid;
    // int r_ca_loops = (32 * 9 + n_threads - 1) / (n_threads);
    // __syncthreads();

    // //main loop starts
    // for(int i = 0; i < k_loops; i++){
    //     // int qid = thread_start_idx / block_cover_dims;
    //     if(i > 0){//store
    //         //output[bid*std_o_b + qid*std_o_lq + hid*std_o_h + hdim_start_idx] = res_reg[(stage^1)*BUFF_SIZE];
    //         *((int4*)(output + out_write_idx)) = *((int4*)(res_reg + (stage^1)*8));
    //         // #pragma unroll
    //         // for(int ti = 0; ti < TH_DIM_CS_STRIDE; ti++){
    //         //     // *(output + out_write_idx + ti) = ((at::Half*)res_reg + 8*(stage^1))[ti];
    //         //     *(output + out_write_idx + ti) = res_reg[8*(stage^1)+ti];
    //         // }
    //         out_write_idx += write_idx_adding;
    //     }



    //     // initialize res register of each loop;
    //     int res_reg_offset = 8*stage;
    //     #pragma unroll
    //     for(int ti = 0; ti < TH_DIM_CS_STRIDE; ti++){
    //         res_reg[res_reg_offset + ti] = 0;
    //     }
    //     // int attn_v_idx = bid * std_aw_b + qid * std_aw_lq + hid * std_aw_h;
    //     // int h_coord_idx = bid * std_sp_b + qid * std_sp_lq + hid * std_sp_h;
    //     int out_idx = BUFF_SIZE * stage;

    //     int coord_offset = (r_stage * 32 + th_qid)* 9 * 2;
    //     int attnw_offset = (r_stage * 32 + th_qid )* 9;

    //     #pragma unroll
    //     for(int pt_id = 0; pt_id < npoints; pt_id++){
    //         if(pt_id == npoints-1){
    //             //read coord and attn weigths into shared memory for next loop
    //             r_stage ^= 1;
    //             int read_id = tb_thread_id;
    //             #pragma unroll
    //             for(int j = 0; j < r_ca_loops; j++){
    //                 if(read_id < 32 * 9){
    //                     int coord_offset = r_stage * 32 * 9 * 2;
    //                     int attnw_offset = r_stage * 32 * 9;
    //                     int rq_id = read_id / 9;
    //                     int rpt_id = read_id % 9;
    //                     int attnw_read_idx = bid * std_aw_b + rq_id * std_aw_lq + hid * std_aw_h + rpt_id;
    //                     int h_coord_read_idx = bid * std_sp_b + rq_id * std_sp_lq + hid * std_sp_h + rpt_id * 2;
    //                     coord_shmem[coord_offset + read_id * 2] = sample_locations[h_coord_read_idx];
    //                     coord_shmem[coord_offset + read_id * 2 + 1] = sample_locations[h_coord_read_idx + 1];
    //                     attnw_shmem[attnw_offset + read_id] = attn_weights[attnw_read_idx];
    //                     read_id += n_threads;
    //                 }
    //             }
    //             __syncthreads();
    //         }

    //         #pragma unroll
    //         for(int ti = 0; ti < TH_DIM_CS_STRIDE; ti++){
    //             samp_res[ti] = 0;
    //         }
            
    //         // const scalar_t h_coord = sample_locations[sp_idx] * (H-1);                
    //         // const scalar_t w_coord = sample_locations[sp_idx+1] * (W-1);
    //         // const scalar_t attn_v = attn_weights[aw_idx];

    //         // const scalar_t attn_v = attn_weights[attn_v_idx];
    //         // const scalar_t h_coord = sample_locations[h_coord_idx] * (H-1);                
    //         // const scalar_t w_coord = sample_locations[h_coord_idx+1] * (W-1); 
    //         //shared memory
    //         const scalar_t attn_v = attnw_shmem[attnw_offset + pt_id];
    //         const scalar_t h_coord = coord_shmem[coord_offset + pt_id * 2] * (H-1);                
    //         const scalar_t w_coord = coord_shmem[coord_offset + pt_id * 2 + 1] * (W-1); 
    //         if(h_coord > -1 && w_coord > -1 && h_coord < H && w_coord < W){
    //             const int h_low = floor(h_coord);                
    //             const float r_h_low = h_coord - h_low;                
    //             const int w_low = floor(w_coord);                
    //             const float r_w_low = w_coord - w_low;                
    //             const int h_high = h_low + 1;                
    //             const float r_h_high = 1.0 - r_h_low;                
    //             const int w_high = w_low + 1;                
    //             const float r_w_high = 1.0 - r_w_low;
    //             int idx_h_low_w_ = (h_high*W + w_low) * shmem_stride + hdim_start_idx;
    //             if(h_low > -1 && w_low > -1){
    //                 // (h_low, w_low)
    //                 const float r_hh_wh = r_h_high * r_w_high;
    //                 #pragma unroll
    //                 for(int ti = 0; ti < TH_DIM_CS_STRIDE; ti++){
    //                     samp_res[ti] += r_hh_wh * fmap_shmem[idx_h_low_w_ + ti]; 
    //                 }
    //             }
    //             idx_h_low_w_ += shmem_stride;
    //             if(h_low > -1 && w_high < W){
    //                 // (h_low, w_high)
    //                 const float r_hh_wl = r_h_high * r_w_low;
    //                 #pragma unroll
    //                 for(int ti = 0; ti < TH_DIM_CS_STRIDE; ti++){
    //                     samp_res[ti] += r_h_high * r_w_low * fmap_shmem[idx_h_low_w_ + ti]; 
    //                 }
    //             }

    //             int idx_h_high_w_ = idx_h_low_w_ + W * shmem_stride;
    //             if(h_high < H && w_high < W){
    //                 // (h_high, w_high)
    //                 const float r_hl_wl = r_h_low * r_w_low;
    //                 #pragma unroll
    //                 for(int ti = 0; ti < TH_DIM_CS_STRIDE; ti++){
    //                     samp_res[ti] += r_hl_wl * fmap_shmem[idx_h_high_w_ + ti];
    //                 }
    //             } 

    //             idx_h_high_w_ -= shmem_stride;
    //             if(h_high < H && w_low > -1){
    //                 // (h_high, w_low)
    //                 const float r_hl_wh = r_h_low * r_w_high;
    //                 #pragma unroll
    //                 for(int ti = 0; ti < TH_DIM_CS_STRIDE; ti++){
    //                     samp_res[ti] += r_hl_wh * fmap_shmem[idx_h_high_w_ + ti];
    //                 }
    //             } 
    //         }//end if(h_coord > -1 && w_coord > -1 && h_coord < H && w_coord < W)
    //         #pragma unroll
    //         for(int ti = 0; ti < TH_DIM_CS_STRIDE; ti++){
    //             // ((at::Half*)res_reg + 8*out_idx)[ti] += samp_res[ti] * attn_v;
    //             res_reg[res_reg_offset+ti] += samp_res[ti] * attn_v;
    //         }
    //         // attn_v_idx += 1;
    //         // h_coord_idx += 2;
    //         // aw_idx += 1;
    //         // sp_idx += 2;
    //     }//end for(int pt_id = 0; )
    //     // aw_idx += q_per_loop * std_aw_lq - (npoints);
    //     // sp_idx += q_per_loop * std_sp_lq - (npoints * 2);
    //     // // thread_start_idx += elements_per_loop;
    //     stage ^= 1;
    //     qid += q_per_loop;

    // }
    // // store left
    // // *((float4*)(output + out_write_idx)) = *((float4*)(res_reg + (stage^1)*BUFF_SIZE));
    // #pragma unroll
    // for(int ti = 0; ti < TH_DIM_CS_STRIDE; ti++){
    //     // *(output + out_write_idx + ti) = ((at::Half*)res_reg + 8*(stage^1))[ti];
    //     *(output + out_write_idx + ti) = res_reg[8*(stage^1)+ti];
    // }
}



at::Tensor only_sample_forward_mhead_cuda(
    at::Tensor input,
    at::Tensor proj_weights,
    at::Tensor sample_locations,
    at::Tensor attn_weights,
    const int H, const int W
){
    const int batch_size = sample_locations.size(0);
    const int len_q = sample_locations.size(1);
    const int nheads = sample_locations.size(2);
    const int npoints = sample_locations.size(3);
    const int d_model = input.size(2);
    const int d_head = d_model / nheads;
    const int len_s = H * W;

    auto output = torch::empty({batch_size, len_q, nheads, d_head}, input.options());


    const dim3 threads(32, 8);
    const dim3 blocks(nheads/BLOCK_COVER_HEADS, batch_size);

    int shared_maxbytes = 64 * 1024; // 100 KB
    cudaFuncSetAttribute(only_sample_forward_mhead_kernel<at::Half>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_maxbytes);
    cudaFuncSetAttribute(only_sample_forward_mhead_kernel<float>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_maxbytes);
    // printf("<<<blocks, threads, shared_maxbytes=%d>>>\n", shared_maxbytes);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "sample multi-heads", ([&] {
        only_sample_forward_mhead_kernel<<<blocks, threads, shared_maxbytes>>>(
            input.data_ptr<at::Half>(),
            proj_weights.data_ptr<at::Half>(),
            sample_locations.data_ptr<at::Half>(),
            attn_weights.data_ptr<at::Half>(),
            output.data_ptr<at::Half>(),
            sample_locations.stride(0), sample_locations.stride(1), sample_locations.stride(2),
            attn_weights.stride(0),  attn_weights.stride(1), attn_weights.stride(2),
            output.stride(0), output.stride(1), output.stride(2),
            H, W,
            batch_size, len_q, d_model, d_head, npoints
        );
    }));
    return output;
}

