#include <torch/extension.h>

at::Tensor only_sample_forward_cuda(
    at::Tensor input,
    at::Tensor proj_weights,
    at::Tensor sample_locations,
    at::Tensor attn_weights,
    const int H, const int W
);

at::Tensor only_sample_forward_cuda_opt(
    at::Tensor input,
    at::Tensor proj_weights,
    at::Tensor sample_locations,
    at::Tensor attn_weights,
    const int H, const int W
);

at::Tensor only_sample_forward_mhead_cuda(
    at::Tensor input,
    at::Tensor proj_weights,
    at::Tensor sample_locations,
    at::Tensor attn_weights,
    const int H, const int W
);


#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


at::Tensor only_sample_forward(
    at::Tensor input,
    at::Tensor proj_weights,
    at::Tensor sample_locations,
    at::Tensor attn_weights,
    const int H, const int W
){
    CHECK_INPUT(input);
    CHECK_INPUT(proj_weights);
    CHECK_INPUT(sample_locations);
    CHECK_INPUT(attn_weights);
    return only_sample_forward_cuda(input, proj_weights, sample_locations, attn_weights, H, W);
}

at::Tensor only_sample_forward_opt(
    at::Tensor input,
    at::Tensor proj_weights,
    at::Tensor sample_locations,
    at::Tensor attn_weights,
    const int H, const int W
){
    CHECK_INPUT(input);
    CHECK_INPUT(proj_weights);
    CHECK_INPUT(sample_locations);
    CHECK_INPUT(attn_weights);
    return only_sample_forward_cuda_opt(input, proj_weights, sample_locations, attn_weights, H, W);
}

at::Tensor only_sample_forward_mhead(
    at::Tensor input,
    at::Tensor proj_weights,
    at::Tensor sample_locations,
    at::Tensor attn_weights,
    const int H, const int W
){
    CHECK_INPUT(input);
    CHECK_INPUT(proj_weights);
    CHECK_INPUT(sample_locations);
    CHECK_INPUT(attn_weights);
    return only_sample_forward_mhead_cuda(input, proj_weights, sample_locations, attn_weights, H, W);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("only_sample", &only_sample_forward, "only sample forward function");
  m.def("only_sample_opt", &only_sample_forward_opt, "only sample opt forward function");
  m.def("only_sample_mhead", &only_sample_forward_mhead, "only sample mhead forward function");
}

