#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <fstream>
#include <string>


#define BLOCK_COVER_HEADS 4
#define THRD_DIM_STRIDE 8
#define WARP_ELEMENTS (THRD_DIM_STRIDE * 32) 
#define MEM_ALIGN 256
#define TH_DIM_READ_STRIDE 8
#define TH_DIM_CS_STRIDE 16
#define BUFF_SIZE TH_DIM_CS_STRIDE
#define SHMEM_PAD 0

bool printed = false;

template <typename scalar_t> 
__global__ void only_sample_forward_tensorcore_opt_kernel(    
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
    extern __shared__ float4 dynamic_shmem1[];
    at::Half* fmap_shmem = (at::Half*)dynamic_shmem1;
    const int len_s = H*W;
    int tb_thread_id = threadIdx.y * blockDim.x + threadIdx.x;  
    int bid = blockIdx.y;
    int hid_base = blockIdx.x * BLOCK_COVER_HEADS;
    int elements_per_loop = blockDim.x * blockDim.y * TH_DIM_READ_STRIDE;
    int block_cover_dims = d_head;
    int shmem_stride = block_cover_dims + SHMEM_PAD; 
    int pix_per_loop = elements_per_loop / block_cover_dims;
    int total_elements = len_s * block_cover_dims;
    int rk_loops = (total_elements + elements_per_loop - 1) / elements_per_loop;
    int fmap_size = len_s * shmem_stride;

    int thread_read_start_idx = tb_thread_id * TH_DIM_READ_STRIDE;
    int read_base = bid*len_s*d_model;
    int hdim_start_idx = thread_read_start_idx % block_cover_dims;
    int pix_id = thread_read_start_idx / block_cover_dims;

    int fmap_shmem_index = pix_id * shmem_stride + hdim_start_idx;
    // int fmap_shmem_index_adding = shmem_stride * pix_per_loop;
    int input_read_index = read_base + pix_id*d_model + hdim_start_idx;
    // int input_read_index_adding = d_model * pix_per_loop;
    //read into shared memory
    for(int i = 0; i < rk_loops; i++){
        // int pix_id = thread_read_start_idx / block_cover_dims;
        // fmap_shmem[pix_id * shmem_stride + hdim_start_idx] = input[read_base + pix_id*d_model + hdim_start_idx];
        *((float4*)(fmap_shmem + fmap_shmem_index)) = *((float4*)(input + input_read_index));


        fmap_shmem_index += shmem_stride * pix_per_loop;
        input_read_index += d_model * pix_per_loop;
        // pix_id += pix_per_loop;
        // thread_read_start_idx += elements_per_loop;
    }
    __syncthreads();
    int stage = 0;

    for(int hid = hid_base; hid < hid_base + BLOCK_COVER_HEADS; hid++){
        //compute and store
        // float4 res_reg[2 * BUFF_SIZE];
        float res_reg[BUFF_SIZE];
        float samp_res[TH_DIM_CS_STRIDE];
        at::Half fmap_reg_hl_wl[TH_DIM_CS_STRIDE];
        at::Half fmap_reg_hl_wh[TH_DIM_CS_STRIDE];
        at::Half fmap_reg_hh_wl[TH_DIM_CS_STRIDE];
        at::Half fmap_reg_hh_wh[TH_DIM_CS_STRIDE];
        // at::Half fmap_reg[TH_DIM_CS_STRIDE];
        elements_per_loop = blockDim.x * blockDim.y * TH_DIM_CS_STRIDE;
        int q_per_loop = elements_per_loop / block_cover_dims;
        int k_loops = (len_q * block_cover_dims + elements_per_loop - 1) / elements_per_loop;
        int thread_start_idx = tb_thread_id * TH_DIM_CS_STRIDE;
        // int thread_access_shift = threadIdx.x / (block_cover_dims / TH_DIM_CS_STRIDE) * 2;
        // int aw_batch_offset = bid * std_aw_b; int aw_head_offset = hid * std_aw_h;
        // int sp_batch_offset = bid * std_sp_b; int sp_head_offset = hid * std_sp_h;
        int qid = thread_start_idx / block_cover_dims;
        // int aw_idx = bid * std_aw_b + qid * std_aw_lq + hid * std_aw_h;
        // int sp_idx = bid * std_sp_b + qid * std_sp_lq + hid * std_sp_h;
        hdim_start_idx = thread_start_idx % block_cover_dims;
        
        int out_write_idx = bid*std_o_b + qid*std_o_lq + hid*std_o_h + hdim_start_idx;
        int write_idx_adding = q_per_loop * std_o_lq;
        for(int i = 0; i < k_loops; i++){
            if(i == 0 && hid < hid_base + BLOCK_COVER_HEADS - 1){
                int thread_read_start_idx = tb_thread_id * TH_DIM_READ_STRIDE;
                int read_base = bid*len_s*d_model + (hid+1)*d_head;
                int hdim_start_idx = thread_read_start_idx % block_cover_dims;
                int pix_id = thread_read_start_idx / block_cover_dims;

                int fmap_shmem_index = (stage^1)*fmap_size + pix_id * shmem_stride + hdim_start_idx;
                // int fmap_shmem_index_adding = shmem_stride * pix_per_loop;
                int input_read_index = read_base + pix_id*d_model + hdim_start_idx;
                // int input_read_index_adding = d_model * pix_per_loop;
                // read into shared memory
                for(int i = 0; i < rk_loops; i++){
                    // int pix_id = thread_read_start_idx / block_cover_dims;
                    // fmap_shmem[pix_id * shmem_stride + hdim_start_idx] = input[read_base + pix_id*d_model + hdim_start_idx];
                    *((float4*)(fmap_shmem + fmap_shmem_index)) = *((float4*)(input + input_read_index));


                    fmap_shmem_index += shmem_stride * pix_per_loop;
                    input_read_index += d_model * pix_per_loop;
                    // pix_id += pix_per_loop;
                    // thread_read_start_idx += elements_per_loop;
                }
            }
            if(i == k_loops - 1){
                __syncthreads();
            }
            // if(i > 0){//store
            //     //output[bid*std_o_b + qid*std_o_lq + hid*std_o_h + hdim_start_idx] = res_reg[(stage^1)*BUFF_SIZE];
            //     // *((int4*)(output + out_write_idx)) = *((int4*)(res_reg + (stage^1)*TH_DIM_CS_STRIDE));
            //     // out_write_idx += write_idx_adding;
            //     #pragma unroll
            //     for(int ti = 0; ti < TH_DIM_CS_STRIDE; ti++){
            //         // *(output + out_write_idx + ti) = ((at::Half*)res_reg + 8*(stage^1))[ti];
            //         *(output + out_write_idx + ti) = res_reg[ti];
            //     }
            // }
            // initialize res register of each loop;
            // int res_reg_offset = TH_DIM_CS_STRIDE*stage;
            #pragma unroll
            for(int ti = 0; ti < TH_DIM_CS_STRIDE; ti++){
                res_reg[ti] = 0;
            }
            int attn_v_idx = bid * std_aw_b + qid * std_aw_lq + hid * std_aw_h;
            int h_coord_idx = bid * std_sp_b + qid * std_sp_lq + hid * std_sp_h;
            #pragma unroll
            for(int pt_id = 0; pt_id < npoints; pt_id++){
                // //initialize samp_res to 0
                // #pragma unroll
                // for(int ti = 0; ti < TH_DIM_CS_STRIDE; ti++){
                //     samp_res[ti] = 0;
                // }
                const scalar_t attn_v = attn_weights[attn_v_idx];
                const scalar_t h_coord = sample_locations[h_coord_idx+1] * (H-1);                
                const scalar_t w_coord = sample_locations[h_coord_idx] * (W-1); 
                if(h_coord > -1 && w_coord > -1 && h_coord < H && w_coord < W){
                    int h_low = floor(h_coord);                
                    float r_h_low = h_coord - h_low;                
                    int w_low = floor(w_coord);                
                    float r_w_low = w_coord - w_low;                
                    int h_high = h_low + 1;                
                    float r_h_high = 1.0 - r_h_low;                
                    int w_high = w_low + 1;                
                    float r_w_high = 1.0 - r_w_low;
                    int idx_ = stage*fmap_size + (h_low*W + w_low) * shmem_stride + hdim_start_idx;
                    float r_hh_wh = r_h_high * r_w_high;
                    float r_hh_wl = r_h_high * r_w_low;;
                    float r_hl_wl = r_h_low * r_w_low;
                    float r_hl_wh = r_h_low * r_w_high;
                    // *((float4*)fmap_reg_hl_wl) = make_float4(0,0,0,0);
                    // *((float4*)fmap_reg_hh_wl) = make_float4(0,0,0,0);
                    // *((float4*)fmap_reg_hl_wh) = make_float4(0,0,0,0);
                    // *((float4*)fmap_reg_hh_wh) = make_float4(0,0,0,0);
                    #pragma unroll
                    for(int ti = 0; ti < TH_DIM_CS_STRIDE; ti++){
                        fmap_reg_hl_wl[ti] = fmap_reg_hh_wl[ti] = fmap_reg_hl_wh[ti] = fmap_reg_hh_wh[ti] = 0;
                    }
                    if(h_low > -1 && w_low > -1){
                        // (h_low, w_low)
                        #pragma unroll
                        for(int j = 0; j < TH_DIM_CS_STRIDE/8; j++){
                            *((float4*)(fmap_reg_hl_wl+j*8)) = *((float4*)(fmap_shmem + idx_+j*8));
                        }
                        // *((float4*)fmap_reg_hl_wl) = *((float4*)(fmap_shmem + idx_));
                    }
                    
                    idx_ += shmem_stride;
                    if(h_low > -1 && w_high < W){
                        // (h_low, w_high)
                        #pragma unroll
                        for(int j = 0; j < TH_DIM_CS_STRIDE/8; j++){
                            *((float4*)(fmap_reg_hl_wh+j*8)) = *((float4*)(fmap_shmem + idx_+j*8));
                        }
                        // *((float4*)fmap_reg_hl_wh) = *((float4*)(fmap_shmem + idx_));
                        
                    }
          

                    idx_ += W * shmem_stride;
                    if(h_high < H && w_high < W){
                        // (h_high, w_high)
                        // *((float4*)fmap_reg_hh_wh) = *((float4*)(fmap_shmem + idx_));
                        #pragma unroll
                        for(int j = 0; j < TH_DIM_CS_STRIDE/8; j++){
                            *((float4*)(fmap_reg_hh_wh+j*8)) = *((float4*)(fmap_shmem + idx_+j*8));
                        }
                    } 
                    

                    idx_ -= shmem_stride;
                    if(h_high < H && w_low > -1){
                        // (h_high, w_low)
                        // *((float4*)fmap_reg_hh_wl) = *((float4*)(fmap_shmem + idx_));
                        #pragma unroll
                        for(int j = 0; j < TH_DIM_CS_STRIDE/8; j++){
                            *((float4*)(fmap_reg_hh_wl+j*8)) = *((float4*)(fmap_shmem + idx_+j*8));
                        }
                    } 
                    #pragma unroll
                    for(int ti = 0; ti < TH_DIM_CS_STRIDE; ti++){
                        samp_res[ti] = r_hh_wh*(fmap_reg_hl_wl[ti]-fmap_reg_hl_wh[ti]) + r_h_high * fmap_reg_hl_wh[ti] + 
                                        r_hl_wh*(fmap_reg_hh_wl[ti]-fmap_reg_hh_wh[ti]) + r_h_low * fmap_reg_hh_wh[ti];
                    }
                }//end if(h_coord > -1 && w_coord > -1 && h_coord < H && w_coord < W)
                #pragma unroll
                for(int ti = 0; ti < TH_DIM_CS_STRIDE; ti++){
                    res_reg[ti] += samp_res[ti] * attn_v;
                }
                attn_v_idx += 1;
                h_coord_idx += 2;
                // aw_idx += 1;
                // sp_idx += 2;
            }//end for(int pt_id = 0; )
            // aw_idx += q_per_loop * std_aw_lq - (npoints);
            // sp_idx += q_per_loop * std_sp_lq - (npoints * 2);
            // // thread_start_idx += elements_per_loop;
            //store results
            #pragma unroll
            for(int ti = 0; ti < TH_DIM_CS_STRIDE; ti++){
                fmap_reg_hl_wl[ti] = (at::Half)res_reg[ti];
            }
            #pragma unroll
            for(int j = 0; j < TH_DIM_CS_STRIDE/8; j++){
                *((float4*)(output + out_write_idx+j*8)) = *((float4*)(fmap_reg_hl_wl+j*8));
            }
            out_write_idx += write_idx_adding;
            qid += q_per_loop;

        }
        // store left
        // *((float4*)(output + out_write_idx)) = *((float4*)(res_reg + (stage^1)*BUFF_SIZE));
        // #pragma unroll
        // for(int ti = 0; ti < TH_DIM_CS_STRIDE; ti++){
        //     // *(output + out_write_idx + ti) = ((at::Half*)res_reg + 8*(stage^1))[ti];
        //     *(output + out_write_idx + ti) = res_reg[TH_DIM_CS_STRIDE+ti];
        // }
        stage ^= 1;
    }//end for(hid = )
}



at::Tensor only_sample_forward_cuda_opt(
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
    cudaFuncSetAttribute(only_sample_forward_tensorcore_opt_kernel<at::Half>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_maxbytes);
    cudaFuncSetAttribute(only_sample_forward_tensorcore_opt_kernel<float>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_maxbytes);
    if(printed == false){
        printf("--------------------  configs  -------------------------\n");
        printf("<<<blocks, threads, shared_maxbytes=%d>>>\n", shared_maxbytes);
        printf("threads:(%d, %d, %d)\n", threads.x, threads.y, threads.z);
        printf("BLOCK_COVER_HEADS:%d, TH_DIM_CS_STRIDE:%d\n", BLOCK_COVER_HEADS, TH_DIM_CS_STRIDE);
        printf("-------------------------------------------------------\n");
        printed = true;
    }
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "fused linear sample cuda", ([&] {
        only_sample_forward_tensorcore_opt_kernel<<<blocks, threads, shared_maxbytes>>>(
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

