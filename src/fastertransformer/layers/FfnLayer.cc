/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "src/fastertransformer/layers/FfnLayer.h"
#include "src/fastertransformer/kernels/transpose_int8_kernels.h"
#include "src/fastertransformer/utils/nvtx_utils.h"

namespace fastertransformer {


template<typename T>
void FfnLayer<T>::reset_fetcher() {
}

template<typename T>
void FfnLayer<T>::forward(std::vector<fastertransformer::Tensor>*       output_tensors,
                          const std::vector<fastertransformer::Tensor>* input_tensors,
                          const FfnWeight<T>*                           ffn_weights)
{
    TensorMap input_tensor({{"ffn_input", input_tensors->at(0)}});
    TensorMap output_tensor({{"ffn_output", output_tensors->at(0)}});
    int activated_expert_num = 0;
    int iter_num = 0;
    int cache_hit_num = 0;
    forward(&output_tensor, &input_tensor, ffn_weights, -1, nullptr, nullptr, 0, 0, 0, 0, 0, 0, false, activated_expert_num, iter_num, cache_hit_num);
}

template<typename T>
void FfnLayer<T>::forward(std::vector<fastertransformer::Tensor>*       output_tensors,
                          const std::vector<fastertransformer::Tensor>* input_tensors,
                          const FfnWeight<T>*                           ffn_weights,
                          int layer_num, float* expert_priority, int* expert_in_cache, int cache_size, int fix_cache_size, float max_val, float threshold, float dec_in_cache, float dec_out_cache, bool use_cache, int& activated_expert_num, int& iter_num, int& cache_hit_num)
{
    TensorMap input_tensor({{"ffn_input", input_tensors->at(0)}});
    TensorMap output_tensor({{"ffn_output", output_tensors->at(0)}});
    forward(&output_tensor, &input_tensor, ffn_weights, layer_num, expert_priority, expert_in_cache, cache_size, fix_cache_size, max_val, threshold, dec_in_cache, dec_out_cache, use_cache, activated_expert_num, iter_num, cache_hit_num);
}

template<typename T>
const char* get_weight_src(const FfnWeight<T>* ffn_weights)
{
    if (ffn_weights == nullptr) return nullptr;
    return reinterpret_cast<const char*>(ffn_weights->all_weight);
}

template<typename T>
void FfnLayer<T>::forward(TensorMap* output_tensors, TensorMap* input_tensors, const FfnWeight<T>* ffn_weights){
    int activated_expert_num = 0;
    int iter_num = 0;
    int cache_hit_num = 0;
    forward(output_tensors, input_tensors, ffn_weights, -1, nullptr, nullptr, 0, 0, 0, 0, 0, 0, false, activated_expert_num, iter_num, cache_hit_num);
}

template<typename T>
void FfnLayer<T>::forward(TensorMap* output_tensors, TensorMap* input_tensors, const FfnWeight<T>* ffn_weights, int layer_num, float* expert_priority, int* expert_in_cache, int cache_size, int fix_cache_size, float max_val, float threshold, float dec_in_cache, float dec_out_cache, bool use_cache, int& activated_expert_num, int& iter_num, int& cache_hit_num)
{
    // input tensors:
    //      ffn_input [token_num, hidden_dimension],
    //      ia3_tasks [batch_size] (optional)
    //      moe_k     [1], uint64 (optional)
    //      padding_offset [token_num] (optional)
    //      seq_len [1], int32, (optional), only used for ia3

    // output tensors:
    //      ffn_output [token_num, hidden_dimension] or [moe_k * token_num, hidden_dimension] if use_moe
    //      expert_scales [token_num, moe_k] (optional)
    //      expanded_source_row_to_expanded_dest_row [token_num, moe_k] (optional)
    //      expert_for_source_row [token_num, moe_k] (optional)

    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    FT_CHECK(input_tensors->size() >= 1 && input_tensors->size() <= 5);
    FT_CHECK(output_tensors->size() >= 1 || output_tensors->size() <= 4);
    bool   use_moe = false;
    size_t moe_k   = 0;
    if (input_tensors->isExist("moe_k")) {
        use_moe = true;
        moe_k   = input_tensors->at("moe_k").getVal<size_t>();
    }
    allocateBuffer(input_tensors->at("ffn_input").shape[0], moe_k, use_moe);

    const int m             = input_tensors->at("ffn_input").shape[0];
    T*        output_tensor = output_tensors->at("ffn_output").getPtr<T>(); // moe_k_ * h_token_num, d_model_
    const T*  input_tensor  = input_tensors->at("ffn_input").getPtr<const T>();

    // for moe output
    T*   expert_scales    = nullptr;
    int* permuted_rows    = nullptr;
    int* permuted_experts = nullptr;

    // moe outputs should exist or not together
    FT_CHECK((use_moe && output_tensors->isExist("expert_scales")
              && output_tensors->isExist("expanded_source_row_to_expanded_dest_row")
              && output_tensors->isExist("expert_for_source_row"))
             || (!use_moe && !output_tensors->isExist("expert_scales")
                 && !output_tensors->isExist("expanded_source_row_to_expanded_dest_row")
                 && !output_tensors->isExist("expert_for_source_row")));

    if (use_moe) {
        expert_scales    = output_tensors->at("expert_scales").getPtr<T>();
        permuted_rows    = output_tensors->at("expanded_source_row_to_expanded_dest_row").getPtr<int>();
        permuted_experts = output_tensors->at("expert_for_source_row").getPtr<int>();
    }

    // TODO: INT8 and Sparsity are currently not implemented (geglu or reglu)
    const bool use_gated_activation = use_gated_activation_ && ffn_weights->intermediate_weight2.kernel != nullptr;

    // moe can't be used with use_gated_activation currently
    FT_CHECK(!(use_gated_activation && use_moe));
    auto activation_type = getActivationType();

    const int* ia3_tasks = input_tensors->getPtr<const int>("ia3_tasks", nullptr);

    sync_check_cuda_error();

    if (use_moe) {
        PUSH_RANGE("FFN moe");
        FT_CHECK(ia3_tasks == nullptr);
        FT_LOG_TRACE("=== milestones 100");

        cublas_wrapper_->Gemm(CUBLAS_OP_N,
                              CUBLAS_OP_N,
                              expert_num_,      // m
                              m,                // n
                              hidden_units_,    // k
                              ffn_weights->gating_weight.kernel,    // A[hidden_units，expert_num]
                              expert_num_,                          // lda
                              input_tensor,                         // B[m, hidden_units]
                              hidden_units_,                        // ldb
                              moe_gates_buf_,                       // C[m, expert_num]
                              expert_num_);                         // ldc

        sync_check_cuda_error();

        FT_LOG_TRACE("=== milestones 101 === %d", int8_mode_);
        FT_LOG_TRACE("=== milestones 101.5");

        if (int8_mode_ == 0) { //select this branch
            if (fetcher_context_.get() != nullptr) {
                fetcher_context_->allocateBuffer(allocator_, m);
                fetcher_context_->set_source(
                    get_weight_src(ffn_weights_of_the_next_moe_layer_),
                    get_weight_src(ffn_weights));
                FT_LOG_TRACE("=== milestones 101.6");
            }
            moe_fc_runner_->setFetcherContext(fetcher_context_.get());

            FT_LOG_TRACE("=== milestones 101.7");
            moe_fc_runner_->run_moe_fc(input_tensor,
                                       moe_gates_buf_,
                                       ffn_weights->intermediate_weight.kernel,
                                       ffn_weights->intermediate_weight.weight_only_quant_scale,
                                       ffn_weights->intermediate_weight.bias ? ffn_weights->intermediate_weight.bias : bias_buf_,
                                       activation_type,
                                       ffn_weights->output_weight.kernel,
                                       ffn_weights->output_weight.weight_only_quant_scale,
                                       ffn_weights->intermediate_weight_stay_on_GPU.kernel,
                                       ffn_weights->output_weight_stay_on_GPU.kernel,
                                       expert_priority,
                                       expert_in_cache,
                                       cache_size,
                                       fix_cache_size,
                                       max_val,
                                       threshold,
                                       dec_in_cache,
                                       dec_out_cache,
                                       m,
                                       hidden_units_,
                                       inter_size_,
                                       expert_num_,
                                       moe_k,
                                       moe_fc_workspace_,
                                       output_tensor, // moe_k_ * h_token_num, d_model_
                                       expert_scales,
                                       permuted_rows, // h_token_num, moe_k_
                                       permuted_experts, // h_token_num, moe_k_
                                       layer_num,   // current layer number
                                       use_cache,
                                       activated_expert_num,
                                       iter_num,
                                       cache_hit_num,
                                       stream_);
            FT_LOG_TRACE("=== milestones 102");
        }
        else if (int8_mode_ == 1) {
            if (int8_fetcher_context_.get() != nullptr) {
                int8_fetcher_context_->allocateBuffer(allocator_, m);
                int8_fetcher_context_->set_source(
                    get_weight_src(ffn_weights_of_the_next_moe_layer_),
                    get_weight_src(ffn_weights));
                FT_LOG_TRACE("=== milestones 101.6");
                moe_int8_weight_only_fc_runner_->setFetcherContext(int8_fetcher_context_.get());
            }

            FT_LOG_TRACE("=== milestones 101.8");
            FT_CHECK_WITH_INFO(moe_int8_weight_only_fc_runner_.get() != NULL,
                               "weight only runner was not initialized.");

            moe_int8_weight_only_fc_runner_->run_moe_fc(
                input_tensor,
                moe_gates_buf_,
                reinterpret_cast<const quant_t*>(ffn_weights->intermediate_weight.int8_kernel),
                ffn_weights->intermediate_weight.weight_only_quant_scale,
                ffn_weights->intermediate_weight.bias ? ffn_weights->intermediate_weight.bias : bias_buf_,
                activation_type,
                reinterpret_cast<const quant_t*>(ffn_weights->output_weight.int8_kernel),
                ffn_weights->output_weight.weight_only_quant_scale,
                reinterpret_cast<const quant_t*>(ffn_weights->intermediate_weight_stay_on_GPU.kernel),
                reinterpret_cast<const quant_t*>(ffn_weights->output_weight_stay_on_GPU.kernel),
                expert_priority,
                expert_in_cache,
                cache_size,
                fix_cache_size,
                max_val,
                threshold,
                dec_in_cache,
                dec_out_cache,
                m,
                hidden_units_,
                inter_size_,
                expert_num_,
                moe_k,
                moe_fc_workspace_,
                output_tensor,
                expert_scales,
                permuted_rows,
                permuted_experts,
                layer_num,   // current layer number
                use_cache,
                activated_expert_num,
                iter_num,
                cache_hit_num,
                stream_);
        }
        else {
            FT_CHECK_WITH_INFO(false, "Invalid int8 mode for MoE");
        }
        FT_LOG_TRACE("=== milestones 104");

        sync_check_cuda_error();
        if (is_free_buffer_after_forward_ == true) {
            freeBuffer();
        }
        sync_check_cuda_error();
        POP_RANGE;
        return;
    }

    PUSH_RANGE("FFN gemm 1");
    int m_tmp = input_tensors->at("ffn_input").shape[0];
    if (m_tmp % 8 != 0) {
        m_tmp = (m_tmp / 8 + 1) * 8;
    }
    const int m_padded = m_tmp;
#ifdef SPARSITY_ENABLED
    bool use_sparse_gemm = sparse_ && cublas_wrapper_->isUseSparse(1, inter_size_, m, hidden_units_);
#else
    constexpr bool use_sparse_gemm = false;
#endif

    if (use_sparse_gemm) {
        FT_CHECK(!use_gated_activation);
#ifdef SPARSITY_ENABLED
        cublas_wrapper_->SpGemm(CUBLAS_OP_N,
                                CUBLAS_OP_N,
                                inter_size_,
                                m_padded,
                                hidden_units_,
                                ffn_weights->intermediate_weight.sp_kernel,
                                input_tensor,
                                inter_buf_);
#endif
    }
    else {
        if (int8_mode_ == 1) {
            FT_LOG_TRACE("=== milestones 105");
            FT_CHECK_WITH_INFO(weight_only_int8_fc_runner_.get() != NULL, "weight only runner was not initialized.");
            FT_CHECK(ffn_weights->intermediate_weight.int8_kernel != NULL
                     && ffn_weights->intermediate_weight.weight_only_quant_scale != NULL);

            if (ia3_tasks == nullptr && !use_gated_activation) {
                // launch fused GEMM + activation
                weight_only_int8_fc_runner_->gemm_bias_act(
                    input_tensor,
                    reinterpret_cast<const quant_t*>(ffn_weights->intermediate_weight.int8_kernel),
                    ffn_weights->intermediate_weight.weight_only_quant_scale,
                    ffn_weights->intermediate_weight.bias,
                    inter_buf_,
                    m,
                    inter_size_,
                    hidden_units_,
                    activation_type,
                    mixed_gemm_workspace_,
                    mixed_gemm_ws_bytes_,
                    stream_);
            }
            else {
                FT_LOG_TRACE("=== milestones 106");
                // Otherwise, let FT handle activation
                weight_only_int8_fc_runner_->gemm(
                    input_tensor,
                    reinterpret_cast<const quant_t*>(ffn_weights->intermediate_weight.int8_kernel),
                    ffn_weights->intermediate_weight.weight_only_quant_scale,
                    inter_buf_,
                    m,
                    inter_size_,
                    hidden_units_,
                    mixed_gemm_workspace_,
                    mixed_gemm_ws_bytes_,
                    stream_);

                if (use_gated_activation) {
                    FT_CHECK(ffn_weights->intermediate_weight2.int8_kernel != NULL
                             && ffn_weights->intermediate_weight2.weight_only_quant_scale != NULL);

                    weight_only_int8_fc_runner_->gemm(
                        input_tensor,
                        reinterpret_cast<const quant_t*>(ffn_weights->intermediate_weight2.int8_kernel),
                        ffn_weights->intermediate_weight2.weight_only_quant_scale,
                        inter_buf_2_,
                        m,
                        inter_size_,
                        hidden_units_,
                        mixed_gemm_workspace_,
                        mixed_gemm_ws_bytes_,
                        stream_);
                }
            }
        }
        else if (int8_mode_ == 2) {
            FT_CHECK(!use_gated_activation);
            cublas_wrapper_->Int8Gemm(inter_size_,
                                      m,
                                      hidden_units_,
                                      ffn_weights->intermediate_weight.int8_kernel,
                                      hidden_units_,
                                      input_tensors->getPtr<int8_t>("ffn_input"),
                                      hidden_units_,
                                      reinterpret_cast<int8_t*>(inter_buf_),
                                      inter_size_,
                                      ffn_weights->intermediate_weight.scale_inter);
        }
        else {
            FT_LOG_TRACE("=== milestones 107");
            cublas_wrapper_->Gemm(CUBLAS_OP_N,
                                  CUBLAS_OP_N,
                                  inter_size_,
                                  m,
                                  hidden_units_,
                                  ffn_weights->intermediate_weight.kernel,
                                  inter_size_,
                                  input_tensor,
                                  hidden_units_,
                                  inter_buf_,
                                  inter_size_);
            if (use_gated_activation) {
                cublas_wrapper_->Gemm(CUBLAS_OP_N,
                                      CUBLAS_OP_N,
                                      inter_size_,
                                      m,
                                      hidden_units_,
                                      ffn_weights->intermediate_weight2.kernel,
                                      inter_size_,
                                      input_tensor,
                                      hidden_units_,
                                      inter_buf_2_,
                                      inter_size_);
            }
        }
    }

    POP_RANGE;

    if (int8_mode_ != 1 || int8_mode_ != 3 || ia3_tasks != nullptr || use_gated_activation) {
        // if int8_mode == 1 && ia3_tasks == nullptr && we don't use gated activations, we use cutlass
        // to fuse GEMM + bias + activation, so we skip the activation function here. In all
        // other cases, we must apply the activation function separately.
        PUSH_RANGE("add bias act");
        genericActivation(m,
                          ffn_weights->intermediate_weight.bias,
                          use_gated_activation ? ffn_weights->intermediate_weight2.bias : nullptr,
                          input_tensors->at("ia3_tasks", {MEMORY_GPU, TYPE_INT32, {}, nullptr}).getPtr<const int>(),
                          ffn_weights->ia3_weight.kernel,
                          int8_mode_ == 2 ? ffn_weights->intermediate_weight.scale_out : (float*)nullptr,
                          int8_mode_ == 2 ? ffn_weights->output_weight.scale : (float*)nullptr,
                          input_tensors->getPtr<int>("padding_offset", nullptr),
                          input_tensors->getVal<int>("seq_len", 1));
        POP_RANGE;
    }

    sync_check_cuda_error();
    FT_LOG_TRACE("=== milestones 108");

    PUSH_RANGE("FFN gemm 2");
#ifdef SPARSITY_ENABLED
    use_sparse_gemm = sparse_ && cublas_wrapper_->isUseSparse(1, hidden_units_, m, inter_size_);
#endif
    if (use_sparse_gemm) {
#ifdef SPARSITY_ENABLED
        cublas_wrapper_->SpGemm(CUBLAS_OP_N,
                                CUBLAS_OP_N,
                                hidden_units_,
                                m_padded,
                                inter_size_,
                                ffn_weights->output_weight.sp_kernel,
                                inter_buf_,
                                output_tensor);
#endif
    }
    else {
        if (int8_mode_ == 1) {
            FT_LOG_TRACE("=== milestones 109");
            FT_CHECK_WITH_INFO(weight_only_int8_fc_runner_.get() != NULL, "weight only runner was not initialized.");
            FT_CHECK(ffn_weights->output_weight.int8_kernel != NULL
                     && ffn_weights->output_weight.weight_only_quant_scale != NULL);
            weight_only_int8_fc_runner_->gemm(inter_buf_,
                                              reinterpret_cast<const quant_t*>(ffn_weights->output_weight.int8_kernel),
                                              ffn_weights->output_weight.weight_only_quant_scale,
                                              output_tensor,
                                              m,
                                              hidden_units_,
                                              inter_size_,
                                              mixed_gemm_workspace_,
                                              mixed_gemm_ws_bytes_,
                                              stream_);
        }
        else if (int8_mode_ == 2) {
            FT_LOG_TRACE("=== milestones 110");
            int8_fc_runner_->gemm(reinterpret_cast<int8_t*>(inter_buf_),
                                  ffn_weights->output_weight.int8_kernel,
                                  QuantMode::PerTensorQuant,
                                  ffn_weights->output_weight.scale_inter,
                                  ffn_weights->output_weight.scale_out,
                                  output_tensors->getPtr<T>("ffn_output"),
                                  m,
                                  hidden_units_,
                                  inter_size_,
                                  nullptr,
                                  0,
                                  stream_);
        }
        else {
            FT_LOG_TRACE("=== milestones 111");
            cublas_wrapper_->Gemm(CUBLAS_OP_N,
                                  CUBLAS_OP_N,
                                  hidden_units_,
                                  m,
                                  inter_size_,
                                  ffn_weights->output_weight.kernel,
                                  hidden_units_,
                                  inter_buf_,
                                  inter_size_,
                                  output_tensor,
                                  hidden_units_);
        }
    }
    sync_check_cuda_error();
    POP_RANGE;

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
    sync_check_cuda_error();
}

template<typename T>
FfnLayer<T>::FfnLayer(size_t           max_batch_size,
                      size_t           max_seq_len,
                      size_t           head_num,
                      size_t           size_per_head,
                      size_t           expert_num,
                      size_t           inter_size,
                      cudaStream_t     stream,
                      cublasMMWrapper* cublas_wrapper,
                      IAllocator*      allocator,
                      bool             is_free_buffer_after_forward,
                      bool             sparse,
                      int              int8_mode,
                      bool             use_gated_activation):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, nullptr, sparse),
    max_token_num_(max_batch_size * max_seq_len),
    head_num_(head_num),
    size_per_head_(size_per_head),
    expert_num_(expert_num),
    hidden_units_(head_num * size_per_head),
    max_inter_size_(inter_size),
    inter_size_(inter_size),
    int8_mode_(int8_mode),
    use_gated_activation_(use_gated_activation),
    int8_fc_runner_(int8_mode == 2 ? std::make_shared<CutlassInt8GemmRunner<T>>() : nullptr)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (int8_mode_ == 0) {
        moe_fc_runner_ = std::make_shared<CutlassMoeFCRunner<T, T>>();
    }
    else if (int8_mode_ == 1) {
        FT_CHECK_WITH_INFO(!(std::is_same<T, float>::value), "Weight only quant not supported for fp32.");
        moe_int8_weight_only_fc_runner_ = std::make_shared<CutlassMoeFCRunner<T, quant_t>>();
        weight_only_int8_fc_runner_     = std::make_shared<CutlassFpAIntBGemmRunner<T, quant_t>>();
    }
}

template<typename T>
FfnLayer<T>::FfnLayer(FfnLayer<T> const& ffn_layer):
    BaseLayer(ffn_layer.stream_,
              ffn_layer.cublas_wrapper_,
              ffn_layer.allocator_,
              ffn_layer.is_free_buffer_after_forward_,
              ffn_layer.cuda_device_prop_,
              ffn_layer.sparse_),
    max_token_num_(ffn_layer.max_token_num_),
    head_num_(ffn_layer.head_num_),
    size_per_head_(ffn_layer.size_per_head_),
    expert_num_(ffn_layer.expert_num_),
    hidden_units_(ffn_layer.hidden_units_),
    max_inter_size_(ffn_layer.max_inter_size_),
    inter_size_(ffn_layer.inter_size_),
    int8_mode_(ffn_layer.int8_mode_),
    use_gated_activation_(ffn_layer.use_gated_activation_),
    moe_fc_runner_(ffn_layer.moe_fc_runner_),
    moe_int8_weight_only_fc_runner_(ffn_layer.moe_int8_weight_only_fc_runner_),
    weight_only_int8_fc_runner_(ffn_layer.weight_only_int8_fc_runner_),
    int8_fc_runner_(ffn_layer.int8_fc_runner_)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
}

template<typename T>
FfnLayer<T>::~FfnLayer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    cublas_wrapper_ = nullptr;
    freeBuffer();
}

template<typename T>
void FfnLayer<T>::allocateBuffer()
{
    FT_CHECK_WITH_INFO(false,
                       "FfnLayer::allocateBuffer() is deprecated. Use `allocateBuffer(size_t token_num, ...)` instead");
}

template<typename T>
void FfnLayer<T>::initFetcherContext(FetchType mode, int moe_k, size_t arena_size) {
    if (mode != FetchType::GPU_ONLY) {
        FT_LOG_DEBUG(" === create fetcher context for moe MODE %d", mode);
        if (int8_mode_ == 0) { // select this branch
            fetcher_context_ = std::make_shared<FetcherContext<T>>(
                mode, 
                expert_num_,
                hidden_units_ * inter_size_,
                inter_size_ * hidden_units_,
                inter_size_,
                0,
                0,
                arena_size);
        } else {
            int8_fetcher_context_ = std::make_shared<FetcherContext<T, quant_t>>(
                mode, 
                expert_num_,
                hidden_units_ * inter_size_,
                inter_size_ * hidden_units_,
                inter_size_,
                inter_size_,
                hidden_units_,
                arena_size);
        }
    } else {
        fetcher_context_.reset();
        int8_fetcher_context_.reset();
    }
}

template<typename T>
void FfnLayer<T>::set_layer(const std::string &layer_name, int layer_index, const std::vector<int64_t>& moe_layer_index) {
    bool is_first_moe = (layer_index == moe_layer_index[0]);
    bool is_last_moe = (layer_index == moe_layer_index[moe_layer_index.size() - 1]);
    auto next_layer_it = std::upper_bound(moe_layer_index.begin(), moe_layer_index.end(), layer_index);
    int next_layer_index;
    if (next_layer_it != moe_layer_index.end()) {
        next_layer_index = *next_layer_it;
    } else {
        next_layer_index = layer_index;
    }

    if (int8_mode_ == 0) {
        if (fetcher_context_) {
            fetcher_context_->set_layer(
                layer_name + std::to_string(next_layer_index),
                layer_name + std::to_string(layer_index),
                is_first_moe,
                is_last_moe);
        }
    } else {
        if (int8_fetcher_context_) {
            int8_fetcher_context_->set_layer(
                layer_name + std::to_string(next_layer_index),
                layer_name + std::to_string(layer_index),
                is_first_moe,
                is_last_moe);
        }
    }
}

template<typename T>
void FfnLayer<T>::allocateBuffer(size_t token_num, int moe_k, bool use_moe)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (use_moe) {
        moe_gates_buf_ =
            (T*)allocator_->reMalloc(moe_gates_buf_, sizeof(T) * pad_to_multiple_of_16(token_num * expert_num_), false);
        size_t ws_size_moe = 0;
        if (int8_mode_ == 0) {
            FT_CHECK_WITH_INFO(moe_fc_runner_.get() != NULL, "moe runner was not initialized.");
            ws_size_moe = moe_fc_runner_->getWorkspaceSize(token_num, hidden_units_, inter_size_, expert_num_, moe_k);
        }
        else if (int8_mode_ == 1) {
            FT_CHECK_WITH_INFO(moe_int8_weight_only_fc_runner_.get() != NULL,
                               "weight only moe runner was not initialized.");
            ws_size_moe = moe_int8_weight_only_fc_runner_->getWorkspaceSize(
                token_num, hidden_units_, inter_size_, expert_num_, moe_k);
        }

        moe_fc_workspace_ = (char*)allocator_->reMalloc(moe_fc_workspace_, sizeof(char) * ws_size_moe, false);
        bias_buf_ = (T*)allocator_->reMalloc(bias_buf_, sizeof(T) * pad_to_multiple_of_16(expert_num_ * inter_size_), true);
    }
    else {
        const auto type_size = int8_mode_ == 2 ? sizeof(int8_t) : sizeof(T);
        inter_buf_           = (T*)allocator_->reMalloc(inter_buf_, type_size * token_num * max_inter_size_, false);
        if (use_gated_activation_) {
            inter_buf_2_ = (T*)allocator_->reMalloc(inter_buf_2_, sizeof(T) * token_num * max_inter_size_, false);
        }

        if (int8_mode_ == 1) {
            FT_CHECK_WITH_INFO(weight_only_int8_fc_runner_.get() != NULL, "weight only runner was not initialized.");
            // We use max_size for n and k since we reuse buffers for both FCs and want to allocate the max
            // possible memory that would be required by any of the individual gemms.
            const int max_size    = std::max(hidden_units_, inter_size_);
            mixed_gemm_ws_bytes_  = weight_only_int8_fc_runner_->getWorkspaceSize(token_num, max_size, max_size);
            mixed_gemm_workspace_ = (char*)allocator_->reMalloc(mixed_gemm_workspace_, mixed_gemm_ws_bytes_, false);
        }
        else if (int8_mode_ == 2) {
            const int max_size   = std::max(hidden_units_, inter_size_);
            int8_gemm_ws_bytes_  = int8_fc_runner_->getWorkspaceSize(token_num, max_size, max_size);
            int8_gemm_workspace_ = (char*)allocator_->reMalloc(int8_gemm_workspace_, int8_gemm_ws_bytes_, false);
        }
    }

    is_allocate_buffer_ = true;
}

template<typename T>
void FfnLayer<T>::freeBuffer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_allocate_buffer_) {
        allocator_->free((void**)(&inter_buf_));
        if (use_gated_activation_) {
            allocator_->free((void**)(&inter_buf_2_));
        }
        if (expert_num_ != 0) {
            allocator_->free((void**)(&moe_gates_buf_));
            allocator_->free((void**)(&moe_fc_workspace_));
        }

        if (mixed_gemm_workspace_) {
            allocator_->free((void**)(&mixed_gemm_workspace_));
            mixed_gemm_ws_bytes_ = 0;
        }

        is_allocate_buffer_ = false;
    }
}

#define INVOKE_GENERIC_ACT(ACT)                                                                                        \
    invokeGenericActivation<ACT>(inter_buf_,                                                                           \
                                 bias1,                                                                                \
                                 inter_buf_2_,                                                                         \
                                 bias2,                                                                                \
                                 ia3_tasks,                                                                            \
                                 ia3_weights,                                                                          \
                                 m,                                                                                    \
                                 inter_size_,                                                                          \
                                 int8_mode_,                                                                           \
                                 activation_in,                                                                        \
                                 activation_out,                                                                       \
                                 padding_offset,                                                                       \
                                 seq_len,                                                                              \
                                 stream_);

template<typename T>
void FfnLayer<T>::genericActivation(int          m,
                                    const T*     bias1,
                                    const T*     bias2,
                                    const int*   ia3_tasks,
                                    const T*     ia3_weights,
                                    const float* activation_in,
                                    const float* activation_out,
                                    const int*   padding_offset,
                                    const int    seq_len)
{
    if (ia3_tasks != nullptr) {
        FT_CHECK(seq_len > 0);
    }

    // dispatch according to actual activation
    switch (getActivationType()) {
        case ActivationType::Gelu:
        case ActivationType::GeGLU:
            if (inter_buf_2_ == nullptr && int8_mode_ <= 1) {
                invokeAddBiasGeluV2(
                    inter_buf_, bias1, ia3_tasks, ia3_weights, padding_offset, seq_len, m, inter_size_, stream_);
            }
            else {
                INVOKE_GENERIC_ACT(GeluActivation);
            }
            break;
        case ActivationType::Relu:
        case ActivationType::ReGLU:
            INVOKE_GENERIC_ACT(ReluActivation);
            break;
        case ActivationType::Silu:
        case ActivationType::SiGLU:
            INVOKE_GENERIC_ACT(SiluActivation);
            break;
        case ActivationType::Identity:
            INVOKE_GENERIC_ACT(IdentityActivation);
            break;
    }
}

#undef INVOKE_GENERIC_ACT

template class FfnLayer<float>;
template class FfnLayer<half>;
#ifdef ENABLE_BF16
template class FfnLayer<__nv_bfloat16>;
#endif

template<typename T>
GeluFfnLayer<T>::GeluFfnLayer(size_t           max_batch_size,
                              size_t           max_seq_len,
                              size_t           head_num,
                              size_t           size_per_head,
                              size_t           expert_num,
                              size_t           inter_size,
                              cudaStream_t     stream,
                              cublasMMWrapper* cublas_wrapper,
                              IAllocator*      allocator,
                              bool             is_free_buffer_after_forward,
                              bool             sparse,
                              int              int8_mode,
                              bool             use_gated_activation):
    FfnLayer<T>(max_batch_size,
                max_seq_len,
                head_num,
                size_per_head,
                expert_num,
                inter_size,
                stream,
                cublas_wrapper,
                allocator,
                is_free_buffer_after_forward,
                sparse,
                int8_mode,
                use_gated_activation)
{
}

template<typename T>
GeluFfnLayer<T>::GeluFfnLayer(GeluFfnLayer<T> const& gelu_ffn_layer): FfnLayer<T>(gelu_ffn_layer)
{
}

template class GeluFfnLayer<float>;
template class GeluFfnLayer<half>;
#ifdef ENABLE_BF16
template class GeluFfnLayer<__nv_bfloat16>;
#endif

template<typename T>
ReluFfnLayer<T>::ReluFfnLayer(size_t           max_batch_size,
                              size_t           max_seq_len,
                              size_t           head_num,
                              size_t           size_per_head,
                              size_t           expert_num,
                              size_t           inter_size,
                              cudaStream_t     stream,
                              cublasMMWrapper* cublas_wrapper,
                              IAllocator*      allocator,
                              bool             is_free_buffer_after_forward,
                              bool             sparse,
                              int              int8_mode,
                              bool             use_gated_activation):
    FfnLayer<T>(max_batch_size,
                max_seq_len,
                head_num,
                size_per_head,
                expert_num,
                inter_size,
                stream,
                cublas_wrapper,
                allocator,
                is_free_buffer_after_forward,
                sparse,
                int8_mode,
                use_gated_activation)
{
}

template<typename T>
ReluFfnLayer<T>::ReluFfnLayer(ReluFfnLayer<T> const& relu_ffn_layer): FfnLayer<T>(relu_ffn_layer)
{
}

template class ReluFfnLayer<float>;
template class ReluFfnLayer<half>;
#ifdef ENABLE_BF16
template class ReluFfnLayer<__nv_bfloat16>;
#endif

template<typename T>
SiluFfnLayer<T>::SiluFfnLayer(size_t           max_batch_size,
                              size_t           max_seq_len,
                              size_t           head_num,
                              size_t           size_per_head,
                              size_t           expert_num,
                              size_t           inter_size,
                              cudaStream_t     stream,
                              cublasMMWrapper* cublas_wrapper,
                              IAllocator*      allocator,
                              bool             is_free_buffer_after_forward,
                              bool             sparse,
                              bool             use_gated_activation):
    FfnLayer<T>(max_batch_size,
                max_seq_len,
                head_num,
                size_per_head,
                expert_num,
                inter_size,
                stream,
                cublas_wrapper,
                allocator,
                is_free_buffer_after_forward,
                sparse,
                0,
                use_gated_activation)
{
}

template<typename T>
SiluFfnLayer<T>::SiluFfnLayer(SiluFfnLayer<T> const& gelu_ffn_layer): FfnLayer<T>(gelu_ffn_layer)
{
}

template class SiluFfnLayer<float>;
template class SiluFfnLayer<half>;
#ifdef ENABLE_BF16
template class SiluFfnLayer<__nv_bfloat16>;
#endif

}  // namespace fastertransformer
