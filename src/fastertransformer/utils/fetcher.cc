#include "fetcher.h"

#include "cutlass/array.h"
#include "cutlass/numeric_types.h"
#include "src/fastertransformer/utils/cuda_utils.h"
#include "src/fastertransformer/utils/profiling.h"
#include "src/fastertransformer/utils/random.h"
#include <chrono>
#include <cuda.h>
#include <cuda_fp16.h>
#include <math.h>
#include <sstream>
#include <algorithm>  // for std::shuffle
#include <random>     // for std::random_device, std::mt19937

namespace fastertransformer {

// the linker asks me to do so

template class FetcherContext<float>;
template class FetcherContext<half>;

template class FetcherContext<float, cutlass::fp4_t>;
template class FetcherContext<float, cutlass::nf4_t>;
template class FetcherContext<float, cutlass::uint4b_t>;
template class FetcherContext<float, cutlass::int4b_t>;

template class FetcherContext<half, cutlass::fp4_t>;
template class FetcherContext<half, cutlass::nf4_t>;
template class FetcherContext<half, cutlass::uint4b_t>;
template class FetcherContext<half, cutlass::int4b_t>;

template class FetcherContext<float, uint8_t>;
template class FetcherContext<half, uint8_t>;

#ifdef ENABLE_BF16
template class FetcherContext<__nv_bfloat16>;

template class FetcherContext<float, __nv_bfloat16>;
template class FetcherContext<half, __nv_bfloat16>;

template class FetcherContext<__nv_bfloat16, float>;
template class FetcherContext<__nv_bfloat16, half>;

template class FetcherContext<__nv_bfloat16, cutlass::fp4_t>;
template class FetcherContext<__nv_bfloat16, cutlass::nf4_t>;
template class FetcherContext<__nv_bfloat16, cutlass::uint4b_t>;
template class FetcherContext<__nv_bfloat16, cutlass::int4b_t>;

template class FetcherContext<__nv_bfloat16, uint8_t>;
#endif

int64_t calc_sparse_time             = 0;  // microseconds
int64_t cpy_expert_array_to_cpu_time = 0;
int64_t total_row_cpy                = 0;
int64_t layer_1_fetch_time           = 0;

// 1. copy to expert_for_source_row_fetching
// 2. calc expert_sparse_idx_working
// 3. launch fetch on the stream, from source to working

template<class ActT, class WeightT, class BiasT>
/**
 * Fetches the permuted experts and allocates memory for intermediate and output working buffers.
 *
 * @param permuted_experts The array of permuted experts.
 * @param prefetch A boolean indicating whether prefetching is enabled.
 *
 * @return void
 */
void FetcherContext<ActT, WeightT, BiasT>::fetch(const int*      permuted_experts,
                                                 bool            prefetch,
                                                 const WeightT*& fc1_expert_weights_stay_on_GPU,
                                                 const WeightT*& fc2_expert_weights_stay_on_GPU,
                                                 float*          expert_priority,
                                                 int*            expert_in_cache,
                                                 int             cache_size,
                                                 int             fix_cache_size,
                                                 float           max_val,
                                                 float           threshold,
                                                 float           dec_in_cache,
                                                 float           dec_out_cache,
                                                 int             layer_num,
                                                 bool            use_cache,
                                                 int&            activated_expert_num,
                                                 int&            iter_num,
                                                 int&            cache_hit_num)
{
    // prtint export_priority address
    // if (layer_num != -1)
        // std::cout << "========================================expert_priority[0]: " << expert_priority[0] <<
        // std::endl;

        FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (last_time && prefetch) {
        FT_LOG_TRACE("Abandon prefetching at final layer");
        return;
    }

    check_cuda_error(cudaMemcpy(permuted_experts_, permuted_experts, sizeof(int) * num_rows_, cudaMemcpyDeviceToHost));

    auto new_end        = std::unique(permuted_experts_, permuted_experts_ + num_rows_);
    num_active_experts_ = new_end - permuted_experts_;
    // std::// std::cout << "num_active_experts_=" << num_active_experts_ << std::std::endl;
    //
    if (GlobalConfig::instance().profiling) {
        Profiling::instance().activeExperts(num_active_experts_);
    }

    if (GlobalConfig::instance().profiling) {
        Profiling::instance().insert(stream, EventType::MEM_START);
    }

    bool fetch_all            = GlobalConfig::instance().fetch_all;
    int  forced_num_experts   = GlobalConfig::instance().forced_num_experts;
    num_active_experts_       = forced_num_experts ? forced_num_experts : num_active_experts_;
    int _active_experts_count = fetch_all ? num_experts_ : num_active_experts_;

    static constexpr bool scales_required =
        std::is_same<WeightT, uint8_t>::value || std::is_same<WeightT, cutlass::uint4b_t>::value
        || std::is_same<WeightT, cutlass::fp4_t>::value || std::is_same<WeightT, cutlass::nf4_t>::value;

    if (layer_num != -1) {
        activated_expert_num += _active_experts_count;
        iter_num++;
    }
    int expert_using[_active_experts_count];
    for (int i = 0; i < _active_experts_count; i++) {
        expert_using[i] = -1;
    }
    for (int i = 0; i < _active_experts_count; i++) {
        int expert                   = (forced_num_experts || fetch_all) ? i : permuted_experts_[i];
        expert_using[i]              = expert;
        const char* fetch_weight_src = prefetch ? next_weight_src_ : current_weight_src_;
        std::string layer_name       = prefetch ? next_layer_name_ : current_layer_name_;
        int         layer            = prefetch ? layer_num + 2 : layer_num;
        // printf("layer: %d\n", layer);
        // TODO: ffn_layer_->set_layer("decoder::layer", l, moe_layer_index_);

        bool found      = false;
        int  expert_idx = -1;
        if (layer_num != -1) {  // decoder
            for (int j = 0; j < cache_size; j++) {
                if (expert == expert_in_cache[(layer / 2) * cache_size + j]) {
                    found      = true;
                    expert_idx = j;  // the cache space for each layer is separate
                    break;
                }
            }
            // std::// std::cout << "layer_num " << layer_name << " layer " << layer << "expert id " << expert << "
            // found value " << found <<std::std::endl;
        }
        if (scales_required) {
            futures_.push_back(GroupedMemoryArena::instance().allocate(
                layer_name + "expert" + std::to_string(expert),
                {reinterpret_cast<char*>(intermediate_working_) + i * intermediate_w_size_per_expert_,
                 reinterpret_cast<char*>(output_working_) + i * output_w_size_per_expert_,
                 reinterpret_cast<char*>(intermediate_scale_working_) + i * intermediate_scale_size_per_expert_,
                 reinterpret_cast<char*>(output_scale_working_) + i * output_scale_size_per_expert_},
                fetch_weight_src + expert * weight_size_per_expert_));
        }
        else {
            // GroupedMemoryArena::instance() is a singleton object that manages memory allocation.
            // allocate() allocates memory for the intermediate and output working buffers.
            // The memory is allocated from the arena managed by GroupedMemoryArena::instance().
            // The name of the memory allocation is "layer_name + "expert" + std::to_string(expert)".
            // "reinterpret_cast<char*>(intermediate_working_) + i * intermediate_w_size_per_expert_" is the pointer to
            // the intermediate working buffer. "reinterpret_cast<char*>(output_working_) + i *
            // output_w_size_per_expert_" is the pointer to the output working buffer. "fetch_weight_src + expert *
            // weight_size_per_expert_" is the pointer to the weight source.
            if (use_cache) {
                // std::cout << "decoder expert load" << expert <<  std::endl;
                if (!found) {
                    futures_.push_back(GroupedMemoryArena::instance().allocate(
                        layer_name + "expert" + std::to_string(expert),
                        {reinterpret_cast<char*>(intermediate_working_) + i * intermediate_w_size_per_expert_,
                         reinterpret_cast<char*>(output_working_) + i * output_w_size_per_expert_},
                        fetch_weight_src + expert * weight_size_per_expert_));
                }
                // printf("fc1_src: %p\n", fc1_expert_weights_stay_on_GPU);
                else {
                    // For weights on GPU, perform GPU memory copy
                    void* intermediate_dest =
                        reinterpret_cast<char*>(intermediate_working_) + i * intermediate_w_size_per_expert_;
                    void* output_dest = reinterpret_cast<char*>(output_working_) + i * output_w_size_per_expert_;
                    // printf("intermediate_dest: %p\n", intermediate_dest);
                    // printf("fc1_src: %p\n", fc1_expert_weights_stay_on_GPU);
                    // printf("expert_idx: %d, size: %d\n", expert_idx, intermediate_w_size_per_expert_);
                    check_cuda_error(cudaMemcpyAsync(intermediate_dest,
                                                     reinterpret_cast<const char*>(fc1_expert_weights_stay_on_GPU)
                                                         + expert_idx * intermediate_w_size_per_expert_,
                                                     intermediate_w_size_per_expert_,
                                                     cudaMemcpyDeviceToDevice,
                                                     stream));
                    check_cuda_error(cudaMemcpyAsync(output_dest,
                                                     reinterpret_cast<const char*>(fc2_expert_weights_stay_on_GPU)
                                                         + expert_idx * output_w_size_per_expert_,
                                                     output_w_size_per_expert_,
                                                     cudaMemcpyDeviceToDevice,
                                                     stream));
                    cache_hit_num++;
                    // std::cout << "copy expert from GPU" << expert << " to working" << " expert_idx: " << expert_idx
                    // << std::endl;
                }
            }
            else {
                futures_.push_back(GroupedMemoryArena::instance().allocate(
                    layer_name + "expert" + std::to_string(expert),
                    {reinterpret_cast<char*>(intermediate_working_) + i * intermediate_w_size_per_expert_,
                     reinterpret_cast<char*>(output_working_) + i * output_w_size_per_expert_},
                    fetch_weight_src + expert * weight_size_per_expert_));
            }
        }
    }
    // std::cout << "expert_using: ";
    for (int i = 0; i < _active_experts_count; i++) {
        // std::cout << expert_using[i];
    }
    // std::cout << std::endl;
    int cache_policy = 0;
    if (layer_num != -1 && use_cache) {  // deal decoder cache
        if (cache_policy == 0) { // Priority-driven cache policy
            int layer = prefetch ? (layer_num + 2) / 2 : layer_num / 2;
            for (int i = 0; i < _active_experts_count; i++) {  // deal cache
                int expert_index = layer * 128 + expert_using[i];
                expert_priority[expert_index] += 1.0;
                if (expert_priority[expert_index] > max_val) {
                    expert_priority[expert_index] = max_val;
                }
            }
            for (int i = 0; i < 128; i++) {
                int  expert_index = layer * 128 + i;
                bool is_in_using  = false;
                // bool is_in_cache = false;

                // Check if it's in expert_using
                
                is_in_using = std::find(expert_using, expert_using + _active_experts_count, i) != expert_using + _active_experts_count;
                int* cache_begin = expert_in_cache + layer * cache_size;
                int* cache_end = cache_begin + cache_size;
                bool in_cache_flag = std::find(cache_begin, cache_end, expert_index) != cache_end;
                if (!is_in_using && in_cache_flag) {
                    expert_priority[expert_index] -= dec_in_cache;
                }
                else if (!is_in_using && !in_cache_flag) {
                    expert_priority[expert_index] -= dec_out_cache;
                }
            }
            // Sort expert_using in descending order

            // Use pair to sort expert ID and its index simultaneously
            std::vector<int> original_indices_expert_using(_active_experts_count);
            std::vector<std::pair<int, int>> expert_pairs_using(_active_experts_count);
            for (int i = 0; i < _active_experts_count; i++) {
                expert_pairs_using[i] = {expert_using[i], i};
            }

            // Sort by expert ID priority
            std::sort(expert_pairs_using.begin(), expert_pairs_using.end(), 
                [&](const std::pair<int, int>& a, const std::pair<int, int>& b) {
                    float priority_a = expert_priority[layer * 128 + a.first];
                    float priority_b = expert_priority[layer * 128 + b.first];
                    return priority_a > priority_b;  // descending order
                });

            // Update expert_using and original_indices
            for (int i = 0; i < _active_experts_count; i++) {
                expert_using[i] = expert_pairs_using[i].first;
                original_indices_expert_using[i] = expert_pairs_using[i].second;
            }

            // Sort cache by priority in descending order
            std::sort(expert_in_cache + layer * cache_size + fix_cache_size,
                expert_in_cache + (layer + 1) * cache_size,
                [&](const int& a, const int& b) {
                    float priority_a = expert_priority[layer * 128 + a];
                    float priority_b = expert_priority[layer * 128 + b];
                    return priority_a > priority_b; 
                });
            for (int i = 0, j = 0; (i < cache_size - fix_cache_size) && (j < _active_experts_count); i++, j++) {
                if (expert_priority[layer * 128 + expert_in_cache[(layer + 1) * cache_size - i - 1]] >= threshold) {
                // if (expert_priority[layer * 128 + expert_in_cache[(layer + 1) * cache_size - i - 1]] >= expert_priority[layer * 128 + expert_using[j]]) {
                    break;
                }
                else {
                    if (expert_priority[layer * 128 + expert_using[j]] < threshold) {
                        break;
                    }
                    else {                    
                        int* cache_begin = expert_in_cache + layer * cache_size;
                        int* cache_end = cache_begin + cache_size;
                        bool already_in_cache = std::find(cache_begin, cache_end, expert_using[j]) != cache_end;
                        if (already_in_cache) {
                            i--;
                            continue;
                        }
                        expert_in_cache[(layer + 1) * cache_size - i - 1] = expert_using[j];
                        int original_indices_expert_using_temp = original_indices_expert_using[j];
                        int expert_idx_in_this_layer = cache_size - i - 1;
                        void* intermediate_dest =
                            reinterpret_cast<char*>(intermediate_working_) + original_indices_expert_using_temp * intermediate_w_size_per_expert_;
                        void* output_dest = reinterpret_cast<char*>(output_working_) + original_indices_expert_using_temp * output_w_size_per_expert_;
                        futures_.push_back(GroupedMemoryArena::instance().cache_move(
                            reinterpret_cast<char*>(intermediate_dest),
                            reinterpret_cast<const char*>(fc1_expert_weights_stay_on_GPU)
                                + expert_idx_in_this_layer * intermediate_w_size_per_expert_,
                            intermediate_w_size_per_expert_));
                        futures_.push_back(GroupedMemoryArena::instance().cache_move(
                            reinterpret_cast<char*>(output_dest),
                            reinterpret_cast<const char*>(fc2_expert_weights_stay_on_GPU)
                                + expert_idx_in_this_layer * output_w_size_per_expert_,
                            output_w_size_per_expert_));
                    }
                }
            }
        }
        else if (cache_policy == 1) {  // LRU (Least Recently Used) cache policy
            int layer = prefetch ? (layer_num + 2) / 2 : layer_num / 2;

            // Iterate through currently used experts
            // int cache_replace = 0;
            int num_to_pick = std::min(cache_size, _active_experts_count);

            // Create an index array 0, 1, ..., _active_experts_count - 1
            std::vector<int> indices(_active_experts_count);
            for (int i = 0; i < _active_experts_count; i++) {
                indices[i] = i;
            }
            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(indices.begin(), indices.end(), g);
            // for (int i = 0; i < _active_experts_count; i++) {
            //     int expert_to_add = expert_using[i];
            for (int i = 0; i < num_to_pick; i++) {
                // int expert_to_add = expert_using[indices[i]];
                int expert_to_add = expert_using[i];
                int expert_index = layer * 128 + expert_to_add;

                // Update usage timestamp (usage count)
                expert_priority[expert_index] = iter_num;  // Use current iteration number as timestamp

                // Check if already in cache
                int* cache_begin = expert_in_cache + layer * cache_size;
                int* cache_end = cache_begin + cache_size;
                auto it = std::find(cache_begin, cache_end, expert_to_add);
                    
                if (it == cache_end) {  // if not in cache
                // cache_replace ++;
                    // Find LRU (least recently used) cache item with minimum timestamp
                    int lru_index = -1;
                    float min_priority = std::numeric_limits<float>::max();
                    for (int j = 0; j < cache_size; j++) {
                        int cached_expert = expert_in_cache[layer * cache_size + j];
                        int cached_index = layer * 128 + cached_expert;
                        if (expert_priority[cached_index] < min_priority) {
                            min_priority = expert_priority[cached_index];
                            lru_index = j;
                        }
                    }
                    
                    // Replace LRU cache item
                    if (lru_index != -1) {
                        expert_in_cache[layer * cache_size + lru_index] = expert_to_add;

                        // Update weights and working areas
                        void* intermediate_dest =
                            reinterpret_cast<char*>(intermediate_working_) + i * intermediate_w_size_per_expert_;
                        void* output_dest = reinterpret_cast<char*>(output_working_) + i * output_w_size_per_expert_;
                        futures_.push_back(GroupedMemoryArena::instance().cache_move(
                            reinterpret_cast<char*>(intermediate_dest),
                            reinterpret_cast<const char*>(fc1_expert_weights_stay_on_GPU)
                                + lru_index * intermediate_w_size_per_expert_,
                            intermediate_w_size_per_expert_));
                        futures_.push_back(GroupedMemoryArena::instance().cache_move(
                            reinterpret_cast<char*>(output_dest),
                            reinterpret_cast<const char*>(fc2_expert_weights_stay_on_GPU)
                                + lru_index * output_w_size_per_expert_,
                            output_w_size_per_expert_));
                    }
                }
            }
        }
    }
#ifndef NDEBUG
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError());
#endif
}

int64_t fetcher_sync_wait_time = 0;  // microseconds

template<class ActT, class WeightT, class BiasT>
void FetcherContext<ActT, WeightT, BiasT>::sync()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    for (auto& future : futures_) {
        future.wait();
    }
    if (GlobalConfig::instance().profiling) {
        Profiling::instance().insert(stream, EventType::MEM_END);
    }
    futures_.clear();
    check_cuda_error(cudaStreamSynchronize(stream));

    // update dst from working (swap them)
    std::swap(intermediate_dst_, intermediate_working_);
    std::swap(output_dst_, output_working_);
    std::swap(intermediate_bias_dst_, intermediate_bias_working_);
    std::swap(intermediate_scale_dst_, intermediate_scale_working_);
    std::swap(output_scale_dst_, output_scale_working_);
}

// called in FfnLayer.cc
//
template<class ActT, class WeightT, class BiasT>
void FetcherContext<ActT, WeightT, BiasT>::set_source(const char* next_weight_src, const char* current_weight_src)
{
    next_weight_src_    = next_weight_src;
    current_weight_src_ = current_weight_src;
}

template<class ActT, class WeightT, class BiasT>
void FetcherContext<ActT, WeightT, BiasT>::set_layer(const std::string& next_layer_name,
                                                     const std::string& current_layer_name,
                                                     bool               is_first_moe,
                                                     bool               is_last_moe)
{
    next_layer_name_    = next_layer_name;
    current_layer_name_ = current_layer_name;
    first_time          = is_first_moe;
    last_time           = is_last_moe;
}

template<class ActT, class WeightT, class BiasT>
void FetcherContext<ActT, WeightT, BiasT>::get_weights(int&            num_active_experts,
                                                       const WeightT*& fc1_expert_weights,
                                                       const WeightT*& fc2_expert_weights,
                                                       const BiasT*&   fc1_expert_biases,
                                                       const ActT*&    fc1_scales,
                                                       const ActT*&    fc2_scales) const
{
    num_active_experts = num_active_experts_;
    fc1_expert_weights = intermediate_dst_;
    fc2_expert_weights = output_dst_;
    fc1_expert_biases  = intermediate_bias_dst_;
    if (scales_required) {
        fc1_scales = intermediate_scale_dst_;
        fc2_scales = output_scale_dst_;
    }
}

int64_t expert_for_row_backup_time = 0;  // microseconds

template<class ActT, class WeightT, class BiasT>
FetcherContext<ActT, WeightT, BiasT>::~FetcherContext()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    FT_LOG_TRACE("futures left: %d", futures_.size());
    freeBuffer();
    check_cuda_error(cudaStreamDestroy(stream));
}

template<class ActT, class WeightT, class BiasT>
FetcherContext<ActT, WeightT, BiasT>::FetcherContext(FetchType mode,
                                                     int       num_experts,
                                                     size_t    intermediate_w_size_per_expert,
                                                     size_t    output_w_size_per_expert,
                                                     size_t    intermediate_b_size_per_expert,
                                                     size_t    intermediate_scale_size_per_expert,
                                                     size_t    output_scale_size_per_expert,
                                                     size_t    arena_size):
    mode(mode),
    first_time(true),
    num_experts_(num_experts),
    intermediate_w_size_per_expert_(cutlass::get_real_size<WeightT>(intermediate_w_size_per_expert)),
    output_w_size_per_expert_(cutlass::get_real_size<WeightT>(output_w_size_per_expert)),
    intermediate_b_size_per_expert_(cutlass::get_real_size<BiasT>(intermediate_b_size_per_expert)),
    intermediate_scale_size_per_expert_(cutlass::get_real_size<ActT>(intermediate_scale_size_per_expert)),
    output_scale_size_per_expert_(cutlass::get_real_size<ActT>(output_scale_size_per_expert)),
    is_allocate_buffer_(false)
{
    // create cuda stream
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    check_cuda_error(cudaStreamCreate(&this->stream));
    weight_size_per_expert_ = intermediate_w_size_per_expert_ + output_w_size_per_expert_
                              + intermediate_scale_size_per_expert_ + output_scale_size_per_expert_;
    if (scales_required) {
        GroupedMemoryArena::instance().initIfUninit(arena_size,
                                                    {intermediate_w_size_per_expert_,
                                                     output_w_size_per_expert_,
                                                     intermediate_scale_size_per_expert_,
                                                     output_scale_size_per_expert_},
                                                    stream);
    }
    else {
        GroupedMemoryArena::instance().initIfUninit(
            arena_size, {intermediate_w_size_per_expert_, output_w_size_per_expert_}, stream);
    }
    Profiling::instance().reset();
}

template<class ActT, class WeightT, class BiasT>
void FetcherContext<ActT, WeightT, BiasT>::allocateBuffer(IAllocator* allocator, size_t num_rows)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_allocate_buffer_) {
        return;
    }

    allocator_ = allocator;
    num_rows_  = num_rows;

    // TODO: refactor with reMalloc
    intermediate_dst_ =
        (WeightT*)allocator_->reMalloc(intermediate_dst_, intermediate_w_size_per_expert_ * num_experts_);
    output_dst_ = (WeightT*)allocator_->reMalloc(output_dst_, output_w_size_per_expert_ * num_experts_);
    intermediate_bias_dst_ =
        (BiasT*)allocator_->reMalloc(intermediate_bias_dst_, intermediate_b_size_per_expert_ * num_experts_);
    intermediate_working_ =
        (WeightT*)allocator_->reMalloc(intermediate_working_, intermediate_w_size_per_expert_ * num_experts_);
    output_working_ = (WeightT*)allocator_->reMalloc(output_working_, output_w_size_per_expert_ * num_experts_);
    intermediate_bias_working_ =
        (BiasT*)allocator_->reMalloc(intermediate_bias_working_, intermediate_b_size_per_expert_ * num_experts_);
    if (scales_required) {
        intermediate_scale_dst_ =
            (ActT*)allocator_->reMalloc(intermediate_scale_dst_, intermediate_scale_size_per_expert_ * num_experts_);
        output_scale_dst_ =
            (ActT*)allocator_->reMalloc(output_scale_dst_, output_scale_size_per_expert_ * num_experts_);
        intermediate_scale_working_ = (ActT*)allocator_->reMalloc(intermediate_scale_working_,
                                                                  intermediate_scale_size_per_expert_ * num_experts_);
        output_scale_working_ =
            (ActT*)allocator_->reMalloc(output_scale_working_, output_scale_size_per_expert_ * num_experts_);
    }

    permuted_experts_ = (int*)allocator_->reMalloc(permuted_experts_, sizeof(int) * num_rows, false, true);

    is_allocate_buffer_ = true;

    if (GlobalConfig::instance().profiling) {
        Profiling::instance().recordMemoryUsage();
    }
}

template<class ActT, class WeightT, class BiasT>
void FetcherContext<ActT, WeightT, BiasT>::freeBuffer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (is_allocate_buffer_) {
        allocator_->free((void**)&intermediate_dst_);
        allocator_->free((void**)&output_dst_);
        allocator_->free((void**)&intermediate_bias_dst_);
        allocator_->free((void**)&intermediate_working_);
        allocator_->free((void**)&output_working_);
        allocator_->free((void**)&intermediate_bias_working_);
        if (scales_required) {
            allocator_->free((void**)&intermediate_scale_dst_);
            allocator_->free((void**)&output_scale_dst_);
            allocator_->free((void**)&intermediate_scale_working_);
            allocator_->free((void**)&output_scale_working_);
        }

        allocator_->free((void**)&permuted_experts_, true);

        is_allocate_buffer_ = false;
    }
}

}  // namespace fastertransformer