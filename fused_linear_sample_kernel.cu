#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/pipeline>
#include <fstream>
#include <string>

#define BLOCK_COVER_HEADS 1
#define TH_DIM_READ_LA_STRIDE 8
#define MEM_ALIGN 256
#define TH_DIM_READ_STRIDE 16
#define TH_DIM_CS_STRIDE 16
#define SHMEM_PAD 0
#define N_POINTS 8
#define WAPR_SIZE 32
#define WARPS 16
#define N_THREADS (WAPR_SIZE * WARPS)
#define MASK 0xffffffff

bool printed_1 = false;

template <typename scalar_t> 
__global__ void only_sample_forward_kernel(    
    const scalar_t*  __restrict__ input,    
    const scalar_t*  __restrict__ proj_weights, 
    const scalar_t* __restrict__ sample_locations,   
    const scalar_t* __restrict__ attn_weights,    
    scalar_t*  output,    
    const int std_sp_b, const int std_sp_lq,  const int std_sp_h,  
    const int std_aw_b, const int std_aw_lq, const int std_aw_h,   
    const int std_o_b, const int std_o_lq, const int std_o_h,
    const int H, const int W,
    const int batch_size, const int len_q, const int d_model, const int d_head, const int npoints
){      
    extern __shared__ float4 dynamic_shmem1[];
    at::Half* fmap_shmem = (at::Half*)dynamic_shmem1;
    at::Half* la_shmem = &((at::Half*)dynamic_shmem1)[256*(64+SHMEM_PAD)];

    // auto block = cooperative_groups::this_thread_block();
    // auto thread = cooperative_groups::this_thread();
    // cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();

    const int len_s = H*W;
    int n_threads = blockDim.x * blockDim.y;
    int tb_thread_id = threadIdx.y * blockDim.x + threadIdx.x;  
    int bid = blockIdx.y;
    int hid_base = blockIdx.x * BLOCK_COVER_HEADS;
    int block_cover_dims = d_head;
    
    int elements_per_loop = n_threads * TH_DIM_CS_STRIDE;
    int q_per_loop = elements_per_loop / block_cover_dims;
    int k_loops = (len_q * block_cover_dims + elements_per_loop - 1) / elements_per_loop;
    int thread_start_idx = tb_thread_id * TH_DIM_CS_STRIDE;
    int hdim_start_idx = thread_start_idx % block_cover_dims;

    int r_elements_per_loop = n_threads * TH_DIM_READ_STRIDE;
    int shmem_stride = block_cover_dims + SHMEM_PAD; 
    int pix_per_loop = r_elements_per_loop / block_cover_dims;
    int total_elements = len_s * block_cover_dims;
    int rk_loops = (total_elements + r_elements_per_loop - 1) / r_elements_per_loop;
    int thread_read_start_idx = tb_thread_id * TH_DIM_READ_STRIDE;
    int hdim_read_start_idx = thread_read_start_idx % block_cover_dims;
    int pix_id = thread_read_start_idx / block_cover_dims;

    // //used for location reads and attn reads
    // int _attns = (len_q * npoints);
    // int _locations = _attns << 1;
    // int _rla_elements_per_loop = n_threads * TH_DIM_READ_LA_STRIDE;
    // int read_loc_loops = (_locations + _rla_elements_per_loop - 1) / _rla_elements_per_loop;
    // int read_attn_loops = (_attns + _rla_elements_per_loop - 1) / _rla_elements_per_loop;
    // int dim_read_loc_start = (tb_thread_id * TH_DIM_READ_LA_STRIDE) % (npoints << 1);
    // int qid_loc_start = (tb_thread_id * TH_DIM_READ_LA_STRIDE) / (npoints << 1);
    // int q_loc_per_loop = _rla_elements_per_loop / (npoints << 1);
    // int dim_read_attn_start = (tb_thread_id * TH_DIM_READ_LA_STRIDE) % npoints;
    // int qid_attn_start = (tb_thread_id * TH_DIM_READ_LA_STRIDE) / npoints;
    // int q_attn_per_loop = _rla_elements_per_loop / npoints;

    int sf_loops = len_q * npoints / N_THREADS;
    int div_offs = -(tb_thread_id % npoints);
    
    int stage_size = 0;
    for(int hid = hid_base; hid < hid_base + BLOCK_COVER_HEADS; hid++){
        //compute and store
        float res_reg[TH_DIM_CS_STRIDE];
        // float samp_res[TH_DIM_CS_STRIDE];
        at::Half fmap_reg_hl_wl[TH_DIM_CS_STRIDE];
        at::Half loc_buf[N_POINTS << 2];
        at::Half at_buf[N_POINTS << 1];
        // at::Half fmap_reg_hl_wh[TH_DIM_CS_STRIDE];
        // at::Half fmap_reg_hh_wl[TH_DIM_CS_STRIDE];
        // at::Half fmap_reg_hh_wh[TH_DIM_CS_STRIDE];

        int qid = thread_start_idx / block_cover_dims;
        int out_write_idx = bid*std_o_b + qid*std_o_lq + hid*std_o_h + hdim_start_idx;
        int write_idx_adding = q_per_loop * std_o_lq;

        int _attn_v_idx = bid * std_aw_b + qid * std_aw_lq + hid * std_aw_h;
        int _h_coord_idx = bid * std_sp_b + qid * std_sp_lq + hid * std_sp_h;
        for(int i = 0; i < k_loops; i++){
            if(i == 0){
                // read fmap into shared memory
                int read_base = bid*len_s*d_model + hid*d_head;
                int fmap_shmem_index = pix_id * shmem_stride + hdim_read_start_idx;
                int input_read_index = read_base + pix_id*d_model + hdim_read_start_idx;
                // read into shared memory
                for(int i = 0; i < rk_loops; i++){
                    #pragma unroll
                    for(int j = 0; j < TH_DIM_READ_STRIDE>>3; j++){
                        *((float4*)(fmap_shmem + fmap_shmem_index + (j<<3))) = *((float4*)(input + input_read_index + (j<<3)));
                    }

                    fmap_shmem_index += shmem_stride * pix_per_loop;
                    input_read_index += d_model * pix_per_loop;
                }

                __syncthreads();
            }
            #pragma unroll
            for(int ti = 0; ti < TH_DIM_CS_STRIDE; ti++){
                res_reg[ti] = 0;
            }
            int attn_v_idx = bid * std_aw_b + qid * std_aw_lq + hid * std_aw_h;
            int h_coord_idx = bid * std_sp_b + qid * std_sp_lq + hid * std_sp_h;
            
            #pragma unroll
            for(int pt_id = 0; pt_id < npoints; pt_id++){
                // if(pt_id == 0){
                //     #pragma unroll
                //     for(int j = 0; j < (N_POINTS>>2); j++){
                //         *((float4*)(loc_buf + (j<<3))) = *((float4*)(sample_locations + _h_coord_idx + (j<<3)));
                //     }
                //     #pragma unroll
                //     for(int j = 0; j < (N_POINTS>>1); j++){
                //         *((float4*)(at_buf + (j<<3))) = *((float4*)(attn_weights + _attn_v_idx + (j<<3)));
                //     }
                //     __syncthreads();
                //     _h_coord_idx += std_sp_lq * q_per_loop;
                //     _attn_v_idx += std_aw_lq * q_per_loop;
                // }
                const float attn_v = attn_weights[attn_v_idx];
                const float h_coord = sample_locations[h_coord_idx+1] * (H-1);              
                const float w_coord = sample_locations[h_coord_idx] * (W-1);
                // const float attn_v = at_buf[pt_id];
                // const float h_coord = loc_buf[(pt_id<<1) + 1] * (H-1);             
                // const float w_coord = loc_buf[(pt_id<<1)] * (W-1);
                if(h_coord > -1 && w_coord > -1 && h_coord < H && w_coord < W){
                    int h_low = floor(h_coord);                
                    float r_h_low = h_coord - h_low;                
                    int w_low = floor(w_coord);                
                    float r_w_low = w_coord - w_low;                
                    int h_high = h_low + 1;                
                    float r_h_high = 1.0 - r_h_low;                
                    int w_high = w_low + 1;                
                    float r_w_high = 1.0 - r_w_low;
                    int idx_ = (h_low*W + w_low) * shmem_stride + hdim_start_idx;
                    float r_hh_wh = r_h_high * r_w_high*attn_v;
                    float r_hh_wl = r_h_high * r_w_low*attn_v;
                    float r_hl_wl = r_h_low * r_w_low*attn_v;
                    float r_hl_wh = r_h_low * r_w_high*attn_v;
                    // *((float4*)fmap_reg_hl_wl) = make_float4(0,0,0,0);
                    // *((float4*)fmap_reg_hh_wl) = make_float4(0,0,0,0);
                    // *((float4*)fmap_reg_hl_wh) = make_float4(0,0,0,0);
                    // *((float4*)fmap_reg_hh_wh) = make_float4(0,0,0,0);

                    #pragma unroll
                    for(int ti = 0; ti < TH_DIM_CS_STRIDE; ti++){
                        fmap_reg_hl_wl[ti] = 0;//fmap_reg_hh_wl[ti] = fmap_reg_hl_wh[ti] = fmap_reg_hh_wh[ti] = 0;
                    }
                    if(h_low > -1 && w_low > -1){
                        // (h_low, w_low)
                        #pragma unroll
                        for(int j = 0; j < TH_DIM_CS_STRIDE>>3; j++){
                            *((float4*)(fmap_reg_hl_wl+(j<<3))) = *((float4*)(fmap_shmem + idx_+(j<<3)));
                        }
                        #pragma unroll
                        for(int ti = 0; ti < TH_DIM_CS_STRIDE; ti++){
                            res_reg[ti] += r_hh_wh*fmap_reg_hl_wl[ti];
                        }
                        // *((float4*)fmap_reg_hl_wl) = *((float4*)(fmap_shmem + idx_));
                    }
                    
                    idx_ += shmem_stride;
                    if(h_low > -1 && w_high < W){
                        // (h_low, w_high)
                        #pragma unroll
                        for(int j = 0; j < TH_DIM_CS_STRIDE>>3; j++){
                            *((float4*)(fmap_reg_hl_wl+(j<<3))) = *((float4*)(fmap_shmem + idx_+(j<<3)));
                        }
                        #pragma unroll
                        for(int ti = 0; ti < TH_DIM_CS_STRIDE; ti++){
                            res_reg[ti] += r_hh_wl*fmap_reg_hl_wl[ti];
                        }
                    }
          

                    idx_ += W * shmem_stride;
                    if(h_high < H && w_high < W){
                        // (h_high, w_high)
                        // *((float4*)fmap_reg_hh_wh) = *((float4*)(fmap_shmem + idx_));
                        #pragma unroll
                        for(int j = 0; j < TH_DIM_CS_STRIDE>>3; j++){
                            *((float4*)(fmap_reg_hl_wl+(j<<3))) = *((float4*)(fmap_shmem + idx_+(j<<3)));
                        }
                        #pragma unroll
                        for(int ti = 0; ti < TH_DIM_CS_STRIDE; ti++){
                            res_reg[ti] += r_hl_wl*fmap_reg_hl_wl[ti];
                        }
                    } 
                    

                    idx_ -= shmem_stride;
                    if(h_high < H && w_low > -1){
                        // (h_high, w_low)
                        // *((float4*)fmap_reg_hh_wl) = *((float4*)(fmap_shmem + idx_));
                        #pragma unroll
                        for(int j = 0; j < TH_DIM_CS_STRIDE>>3; j++){
                            *((float4*)(fmap_reg_hl_wl+(j<<3))) = *((float4*)(fmap_shmem + idx_+(j<<3)));
                        }
                        #pragma unroll
                        for(int ti = 0; ti < TH_DIM_CS_STRIDE; ti++){
                            res_reg[ti] += r_hl_wh*fmap_reg_hl_wl[ti];
                        }
                    } 
                    
                }//end if(h_coord > -1 && w_coord > -1 && h_coord < H && w_coord < W)
                // #pragma unroll
                // for(int ti = 0; ti < TH_DIM_CS_STRIDE; ti++){
                //     res_reg[ti] += samp_res[ti] * attn_v;
                // }
                attn_v_idx += 1;
                h_coord_idx += 2;

            }//end for(int pt_id = 0; )

            //store results
            #pragma unroll
            for(int ti = 0; ti < TH_DIM_CS_STRIDE; ti++){
                fmap_reg_hl_wl[ti] = (at::Half)res_reg[ti];
            }
            #pragma unroll
            for(int j = 0; j < TH_DIM_CS_STRIDE>>3; j++){
                *((float4*)(output + out_write_idx+(j<<3))) = *((float4*)(fmap_reg_hl_wl+(j<<3)));
            }

            out_write_idx += write_idx_adding;
            qid += q_per_loop;

        }
        // stage ^= 1;
    }//end for(hid = )
}



at::Tensor only_sample_forward_cuda(
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


    const dim3 threads(32, 16);
    const dim3 blocks(nheads/BLOCK_COVER_HEADS, batch_size);

    int shared_maxbytes = 64 * 1024; // 100 KB
    cudaFuncSetAttribute(only_sample_forward_kernel<at::Half>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_maxbytes);
    cudaFuncSetAttribute(only_sample_forward_kernel<float>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_maxbytes);
    if(printed_1 == false){
        printf("--------------------  sample configs  -------------------------\n");
        printf("<<<blocks, threads, shared_maxbytes=%d>>>\n", shared_maxbytes);
        printf("threads:(%d, %d, %d)\n", threads.x, threads.y, threads.z);
        printf("BLOCK_COVER_HEADS=%d, TH_DIM_CS_STRIDE=%d TH_DIM_READ_STRIDE=%d\n", 
                BLOCK_COVER_HEADS, TH_DIM_CS_STRIDE, TH_DIM_READ_STRIDE);
        printf("-------------------------------------------------------\n");
        printed_1 = true;
    }
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "fused linear sample cuda", ([&] {
        only_sample_forward_kernel<<<blocks, threads, shared_maxbytes>>>(
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
// #include <torch/extension.h>
// #include <ATen/ATen.h>
// #include <ATen/cuda/CUDAContext.h>
// #include <stdio.h>
// #include <cuda.h>
// #include <cuda_runtime.h>
// #include <fstream>
// #include <string>


// #define BLOCK_COVER_HEADS 4
// #define THRD_DIM_STRIDE 8
// #define WARP_ELEMENTS (THRD_DIM_STRIDE * 32) 
// #define MEM_ALIGN 256
// #define TH_DIM_READ_STRIDE 16
// #define TH_DIM_CS_STRIDE 16
// #define BUFF_SIZE TH_DIM_CS_STRIDE
// #define SHMEM_PAD 0
// #define AT_SIZE 16
// #define LOC_SIZE 24
// bool printed_ = false;

// template <typename scalar_t> 
// __global__ void only_sample_forward_tensorcore_kernel(    
//     const scalar_t*  __restrict__ input,    
//     const scalar_t*  __restrict__ proj_weights, 
//     const scalar_t* __restrict__ sample_locations,   
//     const scalar_t* __restrict__ attn_weights,    
//     scalar_t*  output,    
//     const int std_sp_b, const int std_sp_lq, const int std_sp_h,    
//     const int std_aw_b, const int std_aw_lq, const int std_aw_h,    
//     const int std_o_b, const int std_o_lq, const int std_o_h,
//     const int H, const int W,
//     const int batch_size, const int len_q, const int d_model, const int d_head, const int npoints
// ){      
//     extern __shared__ float4 dynamic_shmem1[];
//     at::Half* fmap_shmem = (at::Half*)dynamic_shmem1;
//     at::Half* la_shmem = &((at::Half*)dynamic_shmem1)[256*64];
//     const int len_s = H*W;
//     int n_threads = blockDim.x * blockDim.y;
//     int tb_thread_id = threadIdx.y * blockDim.x + threadIdx.x;  
//     int bid = blockIdx.y;
//     int hid_base = blockIdx.x * BLOCK_COVER_HEADS;

//     int block_cover_dims = d_head;
//     int shmem_stride = block_cover_dims + SHMEM_PAD; 

//     int r_elements_per_loop = blockDim.x * blockDim.y * TH_DIM_READ_STRIDE;
//     int pix_per_loop = r_elements_per_loop / block_cover_dims;
//     int total_elements = len_s * block_cover_dims;
//     int rk_loops = (total_elements + r_elements_per_loop - 1) / r_elements_per_loop;
//     int thread_read_start_idx = tb_thread_id * TH_DIM_READ_STRIDE;
//     int hdim_read_start_idx = thread_read_start_idx % block_cover_dims;
//     int pix_id = thread_read_start_idx / block_cover_dims;

//     int fmap_size = len_s * shmem_stride;
//     int la_size = len_q * (LOC_SIZE + AT_SIZE);
//     int attn_shmem_offset = len_q * LOC_SIZE;

//     int stage = 0;

//     for(int hid = hid_base; hid < hid_base + BLOCK_COVER_HEADS; hid++){
//         //compute and store
//         // float4 res_reg[2 * BUFF_SIZE];
//         float res_reg[TH_DIM_CS_STRIDE];
//         // float samp_res[TH_DIM_CS_STRIDE];
//         at::Half fmap_reg_hl_wl[TH_DIM_CS_STRIDE];
//         // at::Half fmap_reg_hl_wh[TH_DIM_CS_STRIDE];
//         // at::Half fmap_reg_hh_wl[TH_DIM_CS_STRIDE];
//         // at::Half fmap_reg_hh_wh[TH_DIM_CS_STRIDE];
//         // at::Half fmap_reg[TH_DIM_CS_STRIDE];
//         int elements_per_loop = blockDim.x * blockDim.y * TH_DIM_CS_STRIDE;
//         int q_per_loop = elements_per_loop / block_cover_dims;
//         int k_loops = (len_q * block_cover_dims + elements_per_loop - 1) / elements_per_loop;
//         int thread_start_idx = tb_thread_id * TH_DIM_CS_STRIDE;
//         int qid = thread_start_idx / block_cover_dims;
//         int hdim_start_idx = thread_start_idx % block_cover_dims;
        
//         int out_write_idx = bid*std_o_b + qid*std_o_lq + hid*std_o_h + hdim_start_idx;
//         int write_idx_adding = q_per_loop * std_o_lq;
//         for(int i = 0; i < k_loops; i++){
//             if(i == 0){
//                 // read fmap into shared memory
//                 int read_base = bid*len_s*d_model + hid*d_head;

//                 int fmap_shmem_index = pix_id * shmem_stride + hdim_read_start_idx;
//                 int input_read_index = read_base + pix_id*d_model + hdim_read_start_idx;
//                 #pragma unroll
//                 for(int i = 0; i < rk_loops; i++){
//                     // int pix_id = thread_read_start_idx / block_cover_dims;
//                     // fmap_shmem[pix_id * shmem_stride + hdim_start_idx] = input[read_base + pix_id*d_model + hdim_start_idx];
//                     #pragma unroll
//                     for(int j = 0; j < TH_DIM_READ_STRIDE/8; j++){
//                         *((float4*)(fmap_shmem + fmap_shmem_index + (j<<3))) = *((float4*)(input + input_read_index + (j<<3)));
//                     }


//                     fmap_shmem_index += shmem_stride * pix_per_loop;
//                     input_read_index += d_model * pix_per_loop;
//                 }

//                 // //read locations and attn weigths into shared memory
//                 // // 1. locations
//                 // int rrk_loops = (len_q + n_threads - 1) / n_threads;
//                 // int _shmem_index = thread_read_start_idx;
//                 // int _read_index = bid * std_sp_b + hid * std_sp_h + thread_read_start_idx;
//                 // #pragma unroll
//                 // for(int ri = 0; ri < rrk_loops; ri++){
//                 //     if(_shmem_index < _total_elements){
//                 //         #pragma unroll
//                 //         for(int j = 0; j < TH_DIM_READ_STRIDE/8; j++){
//                 //             *((float4*)(la_shmem + _shmem_index +(j<<3) )) = *((float4*)(sample_locations + _read_index +(j<<3) ));
//                 //         }
//                 //         _shmem_index += r_elements_per_loop;
//                 //         _read_index += r_elements_per_loop;
                        
//                 //     }
//                 // }
//                 // // 2. attn weights
//                 // _total_elements = len_q*npoints;
//                 // rrk_loops = (_total_elements + r_elements_per_loop - 1) / r_elements_per_loop;
//                 // _shmem_index = attn_shmem_offset + thread_read_start_idx;
//                 // _read_index = bid * std_aw_b + hid * std_aw_h + thread_read_start_idx;
//                 // #pragma unroll
//                 // for(int ri = 0; ri < rrk_loops; ri++){
//                 //     if(_shmem_index - attn_shmem_offset < _total_elements){
//                 //         #pragma unroll
//                 //         for(int j = 0; j < TH_DIM_READ_STRIDE/8; j++){
//                 //             *((float4*)(la_shmem + _shmem_index+(j<<3) )) = *((float4*)(attn_weights + _read_index +(j<<3) ));
//                 //         }
//                 //         _shmem_index += r_elements_per_loop;
//                 //         _read_index += r_elements_per_loop;
//                 //     }
//                 // }
                
//                 __syncthreads();
//             }
//             #pragma unroll
//             for(int ti = 0; ti < TH_DIM_CS_STRIDE; ti++){
//                 res_reg[ti] = 0;
//             }
//             int attn_v_idx = bid * std_aw_b + qid * std_aw_lq + hid * std_aw_h;
//             int h_coord_idx = bid * std_sp_b + qid * std_sp_lq + hid * std_sp_h;
//             #pragma unroll
//             for(int pt_id = 0; pt_id < npoints; pt_id++){
                
//                 const scalar_t attn_v = attn_weights[attn_v_idx];
//                 const scalar_t h_coord = sample_locations[h_coord_idx+1] * (H-1);                
//                 const scalar_t w_coord = sample_locations[h_coord_idx] * (W-1); 
//                 if(h_coord > -1 && w_coord > -1 && h_coord < H && w_coord < W){
//                     int h_low = floor(h_coord);                
//                     float r_h_low = h_coord - h_low;                
//                     int w_low = floor(w_coord);                
//                     float r_w_low = w_coord - w_low;                
//                     int h_high = h_low + 1;                
//                     float r_h_high = 1.0 - r_h_low;                
//                     int w_high = w_low + 1;                
//                     float r_w_high = 1.0 - r_w_low;
//                     int idx_ = (h_low*W + w_low) * shmem_stride + hdim_start_idx;
//                     float r_hh_wh = r_h_high * r_w_high*attn_v;
//                     float r_hh_wl = r_h_high * r_w_low*attn_v;
//                     float r_hl_wl = r_h_low * r_w_low*attn_v;
//                     float r_hl_wh = r_h_low * r_w_high*attn_v;
//                     // *((float4*)fmap_reg_hl_wl) = make_float4(0,0,0,0);
//                     // *((float4*)fmap_reg_hh_wl) = make_float4(0,0,0,0);
//                     // *((float4*)fmap_reg_hl_wh) = make_float4(0,0,0,0);
//                     // *((float4*)fmap_reg_hh_wh) = make_float4(0,0,0,0);

//                     #pragma unroll
//                     for(int ti = 0; ti < TH_DIM_CS_STRIDE; ti++){
//                         fmap_reg_hl_wl[ti] = 0;//fmap_reg_hh_wl[ti] = fmap_reg_hl_wh[ti] = fmap_reg_hh_wh[ti] = 0;
//                     }
//                     if(h_low > -1 && w_low > -1){
//                         // (h_low, w_low)
//                         #pragma unroll
//                         for(int j = 0; j < TH_DIM_CS_STRIDE/8; j++){
//                             *((float4*)(fmap_reg_hl_wl+(j<<3))) = *((float4*)(fmap_shmem + idx_+(j<<3)));
//                         }
//                         #pragma unroll
//                         for(int ti = 0; ti < TH_DIM_CS_STRIDE; ti++){
//                             res_reg[ti] += r_hh_wh*fmap_reg_hl_wl[ti];
//                         }
//                         // *((float4*)fmap_reg_hl_wl) = *((float4*)(fmap_shmem + idx_));
//                     }
                    
//                     idx_ += shmem_stride;
//                     if(h_low > -1 && w_high < W){
//                         // (h_low, w_high)
//                         #pragma unroll
//                         for(int j = 0; j < TH_DIM_CS_STRIDE/8; j++){
//                             *((float4*)(fmap_reg_hl_wl+(j<<3))) = *((float4*)(fmap_shmem + idx_+(j<<3)));
//                         }
//                         #pragma unroll
//                         for(int ti = 0; ti < TH_DIM_CS_STRIDE; ti++){
//                             res_reg[ti] += r_hh_wl*fmap_reg_hl_wl[ti];
//                         }
//                     }
          

//                     idx_ += W * shmem_stride;
//                     if(h_high < H && w_high < W){
//                         // (h_high, w_high)
//                         // *((float4*)fmap_reg_hh_wh) = *((float4*)(fmap_shmem + idx_));
//                         #pragma unroll
//                         for(int j = 0; j < TH_DIM_CS_STRIDE/8; j++){
//                             *((float4*)(fmap_reg_hl_wl+(j<<3))) = *((float4*)(fmap_shmem + idx_+(j<<3)));
//                         }
//                         #pragma unroll
//                         for(int ti = 0; ti < TH_DIM_CS_STRIDE; ti++){
//                             res_reg[ti] += r_hl_wl*fmap_reg_hl_wl[ti];
//                         }
//                     } 
                    

//                     idx_ -= shmem_stride;
//                     if(h_high < H && w_low > -1){
//                         // (h_high, w_low)
//                         // *((float4*)fmap_reg_hh_wl) = *((float4*)(fmap_shmem + idx_));
//                         #pragma unroll
//                         for(int j = 0; j < TH_DIM_CS_STRIDE/8; j++){
//                             *((float4*)(fmap_reg_hl_wl+(j<<3))) = *((float4*)(fmap_shmem + idx_+(j<<3)));
//                         }
//                         #pragma unroll
//                         for(int ti = 0; ti < TH_DIM_CS_STRIDE; ti++){
//                             res_reg[ti] += r_hl_wh*fmap_reg_hl_wl[ti];
//                         }
//                     } 
                    
//                 }//end if(h_coord > -1 && w_coord > -1 && h_coord < H && w_coord < W)
//                 // #pragma unroll
//                 // for(int ti = 0; ti < TH_DIM_CS_STRIDE; ti++){
//                 //     res_reg[ti] += samp_res[ti] * attn_v;
//                 // }
//                 attn_v_idx += 1;
//                 h_coord_idx += 2;
//                 // aw_idx += 1;
//                 // sp_idx += 2;
//             }//end for(int pt_id = 0; )


//             //store results
//             #pragma unroll
//             for(int ti = 0; ti < TH_DIM_CS_STRIDE; ti++){
//                 fmap_reg_hl_wl[ti] = (at::Half)res_reg[ti];
//             }
//             #pragma unroll
//             for(int j = 0; j < TH_DIM_CS_STRIDE/8; j++){
//                 *((float4*)(output + out_write_idx+(j<<3))) = *((float4*)(fmap_reg_hl_wl+(j<<3)));
//             }
//             out_write_idx += write_idx_adding;
//             qid += q_per_loop;

//         }
//         stage ^= 1;
//     }//end for(hid = )
// }



// at::Tensor only_sample_forward_cuda(
//     at::Tensor input,
//     at::Tensor proj_weights,
//     at::Tensor sample_locations,
//     at::Tensor attn_weights,
//     const int H, const int W
// ){
//     const int batch_size = sample_locations.size(0);
//     const int len_q = sample_locations.size(1);
//     const int nheads = sample_locations.size(2);
//     const int npoints = sample_locations.size(3);
//     const int d_model = input.size(2);
//     const int d_head = d_model / nheads;
//     const int len_s = H * W;

//     auto output = torch::empty({batch_size, len_q, nheads, d_head}, input.options());


//     const dim3 threads(32, 32);
//     const dim3 blocks(nheads/BLOCK_COVER_HEADS, batch_size);

//     int shared_maxbytes = 64 * 1024; // 100 KB
//     cudaFuncSetAttribute(only_sample_forward_tensorcore_kernel<at::Half>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_maxbytes);
//     cudaFuncSetAttribute(only_sample_forward_tensorcore_kernel<float>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_maxbytes);
//     if(printed_ == false){
//         printf("--------------------  sample configs  -------------------------\n");
//         printf("<<<blocks, threads, shared_maxbytes=%d>>>\n", shared_maxbytes);
//         printf("threads:(%d, %d, %d)\n", threads.x, threads.y, threads.z);
//         printf("BLOCK_COVER_HEADS=%d, TH_DIM_CS_STRIDE=%d TH_DIM_READ_STRIDE=%d\n", 
//                 BLOCK_COVER_HEADS, TH_DIM_CS_STRIDE, TH_DIM_READ_STRIDE);
//         printf("-------------------------------------------------------\n");
//         printed_ = true;
//     }
//     AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "fused linear sample cuda", ([&] {
//         only_sample_forward_tensorcore_kernel<<<blocks, threads, shared_maxbytes>>>(
//             input.data_ptr<at::Half>(),
//             proj_weights.data_ptr<at::Half>(),
//             sample_locations.data_ptr<at::Half>(),
//             attn_weights.data_ptr<at::Half>(),
//             output.data_ptr<at::Half>(),
//             sample_locations.stride(0), sample_locations.stride(1), sample_locations.stride(2),
//             attn_weights.stride(0),  attn_weights.stride(1), attn_weights.stride(2),
//             output.stride(0), output.stride(1), output.stride(2),
//             H, W,
//             batch_size, len_q, d_model, d_head, npoints
//         );
//     }));
//     return output;
// }

