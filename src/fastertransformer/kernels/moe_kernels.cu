/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include <cuda.h>
#include <cuda_fp16.h>
#include <math.h>
#include <sstream>

// Ignore CUTLASS warnings about type punning
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"

#include "cutlass/array.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"

#pragma GCC diagnostic pop

#include "src/fastertransformer/kernels/moe_kernels.h"
#include "src/fastertransformer/utils/cuda_utils.h"
#include "src/fastertransformer/utils/fetcher.h"
#include "src/fastertransformer/utils/profiling.h"
#include "src/fastertransformer/utils/config.h"

#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#elif (CUDART_VERSION >= 11050)
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/util_type.cuh>
#include <thrust/unique.h>
#else
#include "3rdparty/cub/cub.cuh"
#include "3rdparty/cub/device/device_radix_sort.cuh"
#include "3rdparty/cub/util_type.cuh"
#endif

namespace fastertransformer {

static constexpr int WARP_SIZE = 32;

// ====================== Softmax things ===============================
// We have our own implementation of softmax here so we can support transposing the output
// in the softmax kernel when we extend this module to support expert-choice routing.
template<typename T, int TPB>
__launch_bounds__(TPB) __global__ void moe_softmax(const T* input, const bool* finished, T* output, const int num_cols)
{
    using BlockReduce = cub::BlockReduce<float, TPB>;
    __shared__ typename BlockReduce::TempStorage tmpStorage;

    __shared__ float normalizing_factor;
    __shared__ float float_max;

    const int thread_row_offset = blockIdx.x * num_cols;

    cub::Sum sum;
    float    threadData(-FLT_MAX);

    // Don't touch finished rows.
    if ((finished != nullptr) && finished[blockIdx.x]) {
        return;
    }

    for (int ii = threadIdx.x; ii < num_cols; ii += TPB) {
        const int idx = thread_row_offset + ii;
        threadData    = max(static_cast<float>(input[idx]), threadData);
    }

    const float maxElem = BlockReduce(tmpStorage).Reduce(threadData, cub::Max());
    if (threadIdx.x == 0) {
        float_max = maxElem;
    }
    __syncthreads();

    threadData = 0;

    for (int ii = threadIdx.x; ii < num_cols; ii += TPB) {
        const int idx = thread_row_offset + ii;
        threadData += exp((static_cast<float>(input[idx]) - float_max));
    }

    const auto Z = BlockReduce(tmpStorage).Reduce(threadData, sum);

    if (threadIdx.x == 0) {
        normalizing_factor = 1.f / Z;
    }
    __syncthreads();

    for (int ii = threadIdx.x; ii < num_cols; ii += TPB) {
        const int   idx = thread_row_offset + ii;
        const float val = exp((static_cast<float>(input[idx]) - float_max)) * normalizing_factor;
        output[idx]     = T(val);
    }
}

template<typename T, int TPB>
__launch_bounds__(TPB) __global__ void moe_top_k(const T*    inputs_after_softmax,
                                                 const bool* finished,
                                                 T*          output,
                                                 int*        indices,
                                                 int*        source_rows,
                                                 const int   num_experts,
                                                 const int   k)
{

    using cub_kvp     = cub::KeyValuePair<int, T>;
    using BlockReduce = cub::BlockReduce<cub_kvp, TPB>;
    __shared__ typename BlockReduce::TempStorage tmpStorage;

    cub_kvp     thread_kvp;
    cub::ArgMax arg_max;

    const int num_rows  = gridDim.x;
    const int block_row = blockIdx.x;

    const bool should_process_row = finished ? !finished[block_row] : true;
    const int  thread_read_offset = blockIdx.x * num_experts;
    for (int k_idx = 0; k_idx < k; ++k_idx) {
        thread_kvp.key   = 0;
        thread_kvp.value = T(-1.f);  // This is OK because inputs are probabilities

        cub_kvp inp_kvp;
        for (int expert = threadIdx.x; expert < num_experts; expert += TPB) {
            const int idx = thread_read_offset + expert;
            inp_kvp.key   = expert;
            inp_kvp.value = inputs_after_softmax[idx];

            for (int prior_k = 0; prior_k < k_idx; ++prior_k) {
                const int prior_winning_expert = indices[k * block_row + prior_k];

                if (prior_winning_expert == expert) {
                    inp_kvp = thread_kvp;
                }
            }

            thread_kvp = arg_max(inp_kvp, thread_kvp);
        }

        const cub_kvp result_kvp = BlockReduce(tmpStorage).Reduce(thread_kvp, arg_max);
        if (threadIdx.x == 0) {
            const int idx    = k * block_row + k_idx;
            output[idx]      = result_kvp.value;
            indices[idx]     = should_process_row ? result_kvp.key : num_experts;
            source_rows[idx] = k_idx * num_rows + block_row;
        }
        __syncthreads();
    }
}

// ====================== TopK softmax things ===============================

/*
  A Top-K gating softmax written to exploit when the number of experts in the MoE layers
  are a small power of 2. This allows us to cleanly share the rows among the threads in
  a single warp and eliminate communication between warps (so no need to use shared mem).

  It fuses the softmax, max and argmax into a single kernel.

  Limitations:
  1) This implementation is intended for when the number of experts is a small power of 2.
  2) This implementation assumes k is small, but will work for any k.
*/

/**
 * @brief Performs top-k gating softmax operation on the input tensor.
 *
 * This CUDA kernel computes the top-k gating softmax operation on the input tensor.
 * It takes the input tensor, the finished flag, and the number of rows as input.
 * The kernel computes the softmax of each row, finds the top-k elements in each row,
 * and writes the results to the output tensor along with the corresponding indices and source rows.
 *
 * @tparam T The data type of the input tensor.
 * @tparam VPT The number of values processed per thread.
 * @tparam NUM_EXPERTS The number of experts per row.
 * @tparam WARPS_PER_CTA The number of warps per CTA (Cooperative Thread Array).
 * @tparam BYTES_PER_LDG The number of bytes loaded per thread.
 * 
 * @param input The input tensor.
 * @param finished The finished flag indicating whether each row is finished.
 * @param output The output tensor.
 * @param num_rows The number of rows in the input tensor.
 * @param indices The tensor to store the indices of the top-k elements.
 * @param source_rows The tensor to store the source rows of the top-k elements.
 * @param k The number of top-k elements to find in each row.
 */
template<typename T, int VPT, int NUM_EXPERTS, int WARPS_PER_CTA, int BYTES_PER_LDG>
__launch_bounds__(WARPS_PER_CTA* WARP_SIZE) __global__ void topk_gating_softmax( // 线程块（CTA，Cooperative Thread Array）中的线程数目为 WARPS_PER_CTA * WARP_SIZE
    const T* input, const bool* finished, T* output, const int num_rows, int* indices, int* source_rows, const int k)
{
    // We begin by enforcing compile time assertions and setting up compile time constants.
    static_assert(VPT == (VPT & -VPT), "VPT must be power of 2");   // 每个线程处理的元素数量（Values Per Thread）
    static_assert(NUM_EXPERTS == (NUM_EXPERTS & -NUM_EXPERTS), "NUM_EXPERTS must be power of 2");
    static_assert(BYTES_PER_LDG == (BYTES_PER_LDG & -BYTES_PER_LDG), "BYTES_PER_LDG must be power of 2");
    static_assert(BYTES_PER_LDG <= 16, "BYTES_PER_LDG must be leq 16");

    // Number of bytes each thread pulls in per load
    static constexpr int ELTS_PER_LDG    = BYTES_PER_LDG / sizeof(T);
    static constexpr int ELTS_PER_ROW    = NUM_EXPERTS;
    static constexpr int THREADS_PER_ROW = ELTS_PER_ROW / VPT;
    static constexpr int LDG_PER_THREAD  = VPT / ELTS_PER_LDG;

    // Restrictions based on previous section.
    static_assert(VPT % ELTS_PER_LDG == 0, "The elements per thread must be a multiple of the elements per ldg");
    static_assert(WARP_SIZE % THREADS_PER_ROW == 0, "The threads per row must cleanly divide the threads per warp");
    static_assert(THREADS_PER_ROW == (THREADS_PER_ROW & -THREADS_PER_ROW), "THREADS_PER_ROW must be power of 2");
    static_assert(THREADS_PER_ROW <= WARP_SIZE, "THREADS_PER_ROW can be at most warp size");

    // We have NUM_EXPERTS elements per row. We specialize for small #experts
    static constexpr int ELTS_PER_WARP = WARP_SIZE * VPT;
    static constexpr int ROWS_PER_WARP = ELTS_PER_WARP / ELTS_PER_ROW;
    static constexpr int ROWS_PER_CTA  = WARPS_PER_CTA * ROWS_PER_WARP;

    // Restrictions for previous section.
    static_assert(ELTS_PER_WARP % ELTS_PER_ROW == 0, "The elts per row must cleanly divide the total elt per warp");

    // ===================== From this point, we finally start computing run-time variables. ========================

    // Compute CTA and warp rows. We pack multiple rows into a single warp, and a block contains WARPS_PER_CTA warps.
    // This, each block processes a chunk of rows. We start by computing the start row for each block.
    const int cta_base_row = blockIdx.x * ROWS_PER_CTA;

    // Now, using the base row per thread block, we compute the    base row per warp.
    const int warp_base_row = cta_base_row + threadIdx.y * ROWS_PER_WARP;

    // The threads in a warp are split into sub-groups that will work on a row.
    // We compute row offset for each thread sub-group
    const int thread_row_in_warp = threadIdx.x / THREADS_PER_ROW;
    const int thread_row         = warp_base_row + thread_row_in_warp;

    // Threads with indices out of bounds should early exit here.
    if (thread_row >= num_rows)
        return;
    const bool should_process_row = finished ? !finished[thread_row] : true;

    // We finally start setting up the read pointers for each thread. First, each thread jumps to the start of the
    // row it will read.
    const T* thread_row_ptr = input + thread_row * ELTS_PER_ROW;

    // Now, we compute the group each thread belong to in order to determine the first column to start loads.
    const int thread_group_idx         = threadIdx.x % THREADS_PER_ROW;
    const int first_elt_read_by_thread = thread_group_idx * ELTS_PER_LDG;
    const T*  thread_read_ptr          = thread_row_ptr + first_elt_read_by_thread;

    // Determine the pointer type to use to read in the data depending on the BYTES_PER_LDG template param. In theory,
    // this can support all powers of 2 up to 16.
    using AccessType = cutlass::AlignedArray<T, ELTS_PER_LDG>;

    // Finally, we pull in the data from global mem
    cutlass::Array<T, VPT> row_chunk_input;
    AccessType*            row_chunk_vec_ptr   = reinterpret_cast<AccessType*>(&row_chunk_input);
    const AccessType*      vec_thread_read_ptr = reinterpret_cast<const AccessType*>(thread_read_ptr);
#pragma unroll  // 从全局内存中读取数据
    for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
        row_chunk_vec_ptr[ii] = vec_thread_read_ptr[ii * THREADS_PER_ROW];
    }

    using ComputeType = float;
    using Converter   = cutlass::NumericArrayConverter<ComputeType, T, VPT>;
    Converter                        compute_type_converter;
    cutlass::Array<ComputeType, VPT> row_chunk = compute_type_converter(row_chunk_input);

    // First, we perform a max reduce within the thread. We can do the max in fp16 safely (I think) and just
    // convert to float afterwards for the exp + sum reduction.
    ComputeType thread_max = row_chunk[0];
#pragma unroll
    for (int ii = 1; ii < VPT; ++ii) {
        thread_max = max(thread_max, row_chunk[ii]);
    }

// Now, we find the max within the thread group and distribute among the threads. We use a butterfly reduce.
#pragma unroll
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
        thread_max = max(thread_max, __shfl_xor_sync(0xFFFFFFFF, thread_max, mask, THREADS_PER_ROW));
    }

    // From this point, thread max in all the threads have the max within the row.
    // Now, we subtract the max from each element in the thread and take the exp. We also compute the thread local sum.
    float row_sum = 0;
#pragma unroll
    for (int ii = 0; ii < VPT; ++ii) {
        row_chunk[ii] = expf(row_chunk[ii] - thread_max);
        row_sum += row_chunk[ii];
    }

// Now, we perform the sum reduce within each thread group. Similar to the max reduce, we use a bufferfly pattern.
#pragma unroll
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
        row_sum += __shfl_xor_sync(0xFFFFFFFF, row_sum, mask, THREADS_PER_ROW);
    }

    // From this point, all threads have the max and the sum for their rows in the thread_max and thread_sum variables
    // respectively. Finally, we can scale the rows for the softmax. Technically, for top-k gating we don't need to
    // compute the entire softmax row. We can likely look at the maxes and only compute for the top-k values in the row.
    // However, this kernel will likely not be a bottle neck and it seems better to closer match torch and find the
    // argmax after computing the softmax.
    const float reciprocal_row_sum = 1.f / row_sum;

#pragma unroll
    for (int ii = 0; ii < VPT; ++ii) {
        row_chunk[ii] = row_chunk[ii] * reciprocal_row_sum;
    }

    // Now, softmax_res contains the softmax of the row chunk. Now, I want to find the topk elements in each row, along
    // with the max index.​
    int                  start_col          = first_elt_read_by_thread;
    static constexpr int COLS_PER_GROUP_LDG = ELTS_PER_LDG * THREADS_PER_ROW;

    for (int k_idx = 0; k_idx < k; ++k_idx) {   //选出来k个专家
        // First, each thread does the local argmax
        float max_val = row_chunk[0];
        int   expert  = start_col;
#pragma unroll
        for (int ldg = 0, col = start_col; ldg < LDG_PER_THREAD; ++ldg, col += COLS_PER_GROUP_LDG) {
#pragma unroll
            for (int ii = 0; ii < ELTS_PER_LDG; ++ii) {
                float val = row_chunk[ldg * ELTS_PER_LDG + ii];

                // No check on the experts here since columns with the smallest index are processed first and only
                // updated if > (not >=)
                if (val > max_val) {
                    max_val = val;
                    expert  = col + ii;
                }
            }
        }

// Now, we perform the argmax reduce. We use the butterfly pattern so threads reach consensus about the max.
// This will be useful for K > 1 so that the threads can agree on "who" had the max value. That thread can
// then blank out their max with -inf and the warp can run more iterations...
#pragma unroll
        for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
            float other_max    = __shfl_xor_sync(0xFFFFFFFF, max_val, mask, THREADS_PER_ROW);
            int   other_expert = __shfl_xor_sync(0xFFFFFFFF, expert, mask, THREADS_PER_ROW);

            // We want lower indices to "win" in every thread so we break ties this way
            if (other_max > max_val || (other_max == max_val && other_expert < expert)) {
                max_val = other_max;
                expert  = other_expert;
            }
        }

        // Write the max for this k iteration to global memory.
        if (thread_group_idx == 0) {
            // The lead thread from each sub-group will write out the final results to global memory. (This will be a
            // single) thread per row of the input/output matrices.
            const int idx    = k * thread_row + k_idx;
            output[idx]      = T(max_val);
            indices[idx]     = should_process_row ? expert : NUM_EXPERTS;
            source_rows[idx] = k_idx * num_rows + thread_row;
        }

        // Finally, we clear the value in the thread with the current max if there is another iteration to run.
        if (k_idx + 1 < k) {
            const int ldg_group_for_expert     = expert / COLS_PER_GROUP_LDG;
            const int thread_to_clear_in_group = (expert / ELTS_PER_LDG) % THREADS_PER_ROW;

            // Only the thread in the group which produced the max will reset the "winning" value to -inf.
            if (thread_group_idx == thread_to_clear_in_group) {
                const int offset_for_expert = expert % ELTS_PER_LDG;
                // Safe to set to any negative value since row_chunk values must be between 0 and 1.
                row_chunk[ldg_group_for_expert * ELTS_PER_LDG + offset_for_expert] = ComputeType(-10000.f);
            }
        }
    }
}

namespace detail {
// Constructs some constants needed to partition the work across threads at compile time.
template<typename T, int EXPERTS, int BYTES_PER_LDG>
struct TopkConstants {
    static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(T);
    static_assert(EXPERTS / (ELTS_PER_LDG * WARP_SIZE) == 0 || EXPERTS % (ELTS_PER_LDG * WARP_SIZE) == 0, "");
    static constexpr int VECs_PER_THREAD = std::max(1, EXPERTS / (ELTS_PER_LDG * WARP_SIZE));
    static constexpr int VPT             = VECs_PER_THREAD * ELTS_PER_LDG;
    static constexpr int THREADS_PER_ROW = EXPERTS / VPT;
    static constexpr int ROWS_PER_WARP   = WARP_SIZE / THREADS_PER_ROW;
};
}  // namespace detail

template<typename T, int EXPERTS, int WARPS_PER_TB>
void topk_gating_softmax_launcher_helper(const T*     input,
                                         const bool*  finished,
                                         T*           output,
                                         int*         indices,
                                         int*         source_row,
                                         const int    num_rows,
                                         const int    num_experts,
                                         const int    k,
                                         cudaStream_t stream)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    static constexpr unsigned long MAX_BYTES_PER_LDG = 16;

    static constexpr int BYTES_PER_LDG = std::min(MAX_BYTES_PER_LDG, sizeof(T) * EXPERTS);
    using Constants                    = detail::TopkConstants<T, EXPERTS, BYTES_PER_LDG>;
    static constexpr int VPT           = Constants::VPT;
    static constexpr int ROWS_PER_WARP = Constants::ROWS_PER_WARP;
    const int            num_warps     = (num_rows + ROWS_PER_WARP - 1) / ROWS_PER_WARP;
    const int            num_blocks    = (num_warps + WARPS_PER_TB - 1) / WARPS_PER_TB;

    dim3 block_dim(WARP_SIZE, WARPS_PER_TB);
    // print the args
    FT_LOG_TRACE("topk_gating_softmax_launcher_helper kernel args: input: %x, \
         finished: %x, output: %x, num_rows: %d, indices: %x, source_row: %x, k: %d",
                 input,
                 finished,
                 output,
                 num_rows,
                 indices,
                 source_row,
                 k);
    topk_gating_softmax<T, VPT, EXPERTS, WARPS_PER_TB, BYTES_PER_LDG>
        <<<num_blocks, block_dim, 0, stream>>>(input, finished, output, num_rows, indices, source_row, k);
    FT_LOG_TRACE("kernel finished");
}

template<typename T>
void topk_gating_softmax_kernelLauncher(const T*     input,
                                        const bool*  finished,
                                        T*           output,
                                        T*           softmax_temp_output,
                                        int*         indices,
                                        int*         source_row,
                                        const int    num_rows,
                                        const int    num_experts,
                                        const int    k,
                                        cudaStream_t stream)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    static constexpr int WARPS_PER_TB = 4;

    switch (num_experts) {
        case 2: {
            topk_gating_softmax_launcher_helper<T, 2, WARPS_PER_TB>(
                input, finished, output, indices, source_row, num_rows, num_experts, k, stream);
            break;
        }
        case 4: {
            topk_gating_softmax_launcher_helper<T, 4, WARPS_PER_TB>(
                input, finished, output, indices, source_row, num_rows, num_experts, k, stream);
            break;
        }
        case 8: {
            topk_gating_softmax_launcher_helper<T, 8, WARPS_PER_TB>(
                input, finished, output, indices, source_row, num_rows, num_experts, k, stream);
            break;
        }
        case 16: {
            topk_gating_softmax_launcher_helper<T, 16, WARPS_PER_TB>(
                input, finished, output, indices, source_row, num_rows, num_experts, k, stream);
            break;
        }
        case 32: {
            topk_gating_softmax_launcher_helper<T, 32, WARPS_PER_TB>(
                input, finished, output, indices, source_row, num_rows, num_experts, k, stream);
            break;
        }
        case 64: {
            topk_gating_softmax_launcher_helper<T, 64, WARPS_PER_TB>(
                input, finished, output, indices, source_row, num_rows, num_experts, k, stream);
            break;
        }
        case 128: {
            topk_gating_softmax_launcher_helper<T, 128, WARPS_PER_TB>(
                input, finished, output, indices, source_row, num_rows, num_experts, k, stream);
            break;
        }
        case 256: {
            topk_gating_softmax_launcher_helper<T, 256, WARPS_PER_TB>(
                input, finished, output, indices, source_row, num_rows, num_experts, k, stream);
            break;
        }
        default: {
            static constexpr int TPB = 256;
            FT_CHECK(softmax_temp_output != nullptr);
            moe_softmax<T, TPB><<<num_rows, TPB, 0, stream>>>(input, finished, softmax_temp_output, num_experts);
            moe_top_k<T, TPB><<<num_rows, TPB, 0, stream>>>(
                softmax_temp_output, finished, output, indices, source_row, num_experts, k);
        }
    }
}

// ========================== CUB Sorting things ====================================
CubKeyValueSorter::CubKeyValueSorter(): num_experts_(0), num_bits_(sizeof(int) * 8)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
}

CubKeyValueSorter::CubKeyValueSorter(const int num_experts):
    num_experts_(num_experts), num_bits_((int)log2(num_experts) + 1)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
}

void CubKeyValueSorter::update_num_experts(const int num_experts)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    num_experts_ = num_experts;
    num_bits_    = (int)log2(num_experts) + 1;
}

size_t CubKeyValueSorter::getWorkspaceSize(const size_t num_key_value_pairs)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    num_key_value_pairs_    = num_key_value_pairs;
    size_t required_storage = 0;
    int*   null_int         = nullptr;
    cub::DeviceRadixSort::SortPairs(
        NULL, required_storage, null_int, null_int, null_int, null_int, num_key_value_pairs, 0, num_bits_);
    return required_storage;
}

void CubKeyValueSorter::run(void*        workspace,
                            const size_t workspace_size,
                            const int*   keys_in,
                            int*         keys_out,
                            const int*   values_in,
                            int*         values_out,
                            const size_t num_key_value_pairs,
                            cudaStream_t stream)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    size_t expected_ws_size = getWorkspaceSize(num_key_value_pairs);
    size_t actual_ws_size   = workspace_size;

    if (expected_ws_size > workspace_size) {
        std::stringstream err_ss;
        err_ss << "[FT Error][CubKeyValueSorter::run]\n";
        err_ss << "Error. The allocated workspace is too small to run this problem.\n";
        err_ss << "Expected workspace size of at least " << expected_ws_size << " but got problem size "
               << workspace_size << "\n";
        throw std::runtime_error(err_ss.str());
    }
    cub::DeviceRadixSort::SortPairs(
        workspace, actual_ws_size, keys_in, keys_out, values_in, values_out, num_key_value_pairs, 0, num_bits_, stream);
}

// ============================== Infer GEMM sizes =================================
__device__ inline int find_total_elts_leq_target(const int* sorted_indices, const int arr_length, const int target)
{
    int64_t low = 0, high = arr_length - 1, target_location = -1;
    while (low <= high) {
        int64_t mid = (low + high) / 2;

        if (sorted_indices[mid] > target) {
            high = mid - 1;
        }
        else {
            low             = mid + 1;
            target_location = mid;
        }
    }
    return target_location + 1;
}

// Sets up the gemm assuming the inputs, experts and outputs are stored in row major order.
// Assumes we want to perform output = matmul(inputs, experts) + bias
__global__ void compute_total_rows_before_expert_kernel(const int*    sorted_experts,
                                                        const int     sorted_experts_len,
                                                        const int64_t num_experts,
                                                        int64_t*      total_rows_before_expert)
{

    // First, compute the global tid. We only need 1 thread per expert.
    const int expert = blockIdx.x * blockDim.x + threadIdx.x;
    if (expert >= num_experts)
        return;

    // This should construct the last index where each expert occurs.
    total_rows_before_expert[expert] = find_total_elts_leq_target(sorted_experts, sorted_experts_len, expert);
}

template<typename T>
__global__ void unique_and_remove_zero_kernel(T* array, int arr_length)
{
    int start_idx;
    // First, find non-zero element
    for (start_idx = 0; start_idx < arr_length; ++start_idx) {
        if (array[start_idx] != 0) break;
    }

    // Then, unique
    T cur_val = 0;
    int cur_idx = 0;
    for (int i = start_idx; i < arr_length; ++i) {
        if (array[i] != cur_val) {
            cur_val = array[i];
            array[cur_idx++] = cur_val;
        }
    }
}

template<typename T>
void unique_and_remove_zero(T* array, int arr_length, cudaStream_t stream)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    unique_and_remove_zero_kernel<<<1, 1, 0, stream>>>(array, arr_length);
}

template<typename T, typename WeightType, typename Enable>
CutlassMoeFCRunner<T, WeightType, Enable>::CutlassMoeFCRunner()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
}

template<typename T, typename WeightType, typename Enable>
size_t CutlassMoeFCRunner<T, WeightType, Enable>::getWorkspaceSize(
    const int num_rows, const int hidden_size, const int inter_size, const int num_experts, const int k)
{

    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    const int buf_size         = pad_to_multiple_of_16(k * num_rows * hidden_size);
    const int interbuf_size    = pad_to_multiple_of_16(k * num_rows * inter_size);
    const int padded_experts   = pad_to_multiple_of_16(num_experts);
    const int num_moe_inputs   = pad_to_multiple_of_16(k * num_rows);
    int       num_softmax_outs = 0;

    const bool is_pow_2 = (num_experts != 0) && ((num_experts & (num_experts - 1)) == 0);
    if (!is_pow_2 || num_experts > 256) {
        num_softmax_outs = pad_to_multiple_of_16(num_rows * num_experts);
    }

    // softmax output, permuted_rows and permuted_experts have moved to outside of moe kernel, allocate them
    // in Encoder or Decoder before invoking FfnLayer forward.
    size_t total_ws_bytes = 3 * num_moe_inputs * sizeof(int);  // source_rows_, permuted_rows_, permuted_experts_,
    total_ws_bytes += num_moe_inputs * sizeof(T);              // next_expert_scales_
    total_ws_bytes += buf_size * sizeof(T);                    // permuted_data
    total_ws_bytes += padded_experts * sizeof(int64_t);        // Hold total_rows_before_expert_
    total_ws_bytes += num_softmax_outs * sizeof(T);
    const int bytes_for_fc1_result = interbuf_size * sizeof(T);
    const int sorter_ws_size_bytes = pad_to_multiple_of_16(sorter_.getWorkspaceSize(num_rows));
    sorter_.update_num_experts(num_experts);

    int bytes_for_intermediate_and_sorting = bytes_for_fc1_result;
    if (sorter_ws_size_bytes > bytes_for_fc1_result) {
        int remaining_bytes = pad_to_multiple_of_16(sorter_ws_size_bytes - bytes_for_fc1_result);
        bytes_for_intermediate_and_sorting += remaining_bytes;
    }

    total_ws_bytes += bytes_for_intermediate_and_sorting;  // intermediate (fc1) output + cub sorting workspace
    return total_ws_bytes;
}

template<typename T, typename WeightType, typename Enable>
void CutlassMoeFCRunner<T, WeightType, Enable>::configure_ws_ptrs(
    char* ws_ptr, const int num_rows, const int hidden_size, const int inter_size, const int num_experts, const int k)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    const int buf_size       = pad_to_multiple_of_16(k * num_rows * hidden_size);
    const int interbuf_size  = pad_to_multiple_of_16(k * num_rows * inter_size);
    const int padded_experts = pad_to_multiple_of_16(num_experts);
    const int num_moe_inputs = 128 + pad_to_multiple_of_16(k * num_rows);
    // const int num_softmax_outs = pad_to_multiple_of_16(num_rows * num_experts);

    source_rows_      = (int*)ws_ptr;
    permuted_rows_    = source_rows_ + num_moe_inputs;
    permuted_experts_ = permuted_rows_ + num_moe_inputs;
    next_expert_scales_ = (T*)(permuted_experts_ + num_moe_inputs);
    permuted_data_    = (T*)(next_expert_scales_ + num_moe_inputs);

    total_rows_before_expert_ = (int64_t*)(permuted_data_ + buf_size);
    fc1_result_ = (T*)(total_rows_before_expert_ + padded_experts);


    const bool is_pow_2 = (num_experts != 0) && ((num_experts & (num_experts - 1)) == 0);
    if (!is_pow_2 || num_experts > 256) {
        softmax_out_ = (T*)(fc1_result_ + interbuf_size);
    }
    else {
        softmax_out_ = nullptr;
    }
}

template<typename T, typename WeightType, typename Enable>
void CutlassMoeFCRunner<T, WeightType, Enable>::setFetcherContext(FetcherContext<T, WeightType> *fetcher_ctx)
{
    this->fetcher_context_ = fetcher_ctx;
}

template<typename T>
__global__ void
check_memory(const T* ptr)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ T s[64];
    s[0] = ptr[idx];
}

__global__ void
force_total_rows_before_expert_kernel(int64_t*  total_rows_before_expert, 
                                      const int num_experts,
                                      const int total_rows)
{

    // First, compute the global tid. We only need 1 thread per expert.
    const int expert = blockIdx.x * blockDim.x + threadIdx.x;
    if (expert >= num_experts)
        return;

    // This should construct the last index where each expert occurs.
    if (expert == num_experts - 1) {
        total_rows_before_expert[expert] = total_rows;
    } else {
        total_rows_before_expert[expert] = expert + 1;
    }
}

void force_total_rows_before_expert(int64_t*     total_rows_before_expert, 
                                    const int    num_experts,
                                    const int    total_rows,
                                    cudaStream_t stream)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    const int threads = std::min(1024, num_experts);
    const int blocks  = (num_experts + threads - 1) / threads;

    force_total_rows_before_expert_kernel<<<blocks, threads, 0, stream>>>(
        total_rows_before_expert, num_experts, total_rows);
}


template<typename T, typename WeightType, typename Enable>
void CutlassMoeFCRunner<T, WeightType, Enable>::run_moe_fc(const T*          input_activations,     // [num_rows, hidden_size]
                                                           const T*          gating_output,         // [num_rows, expert_nums]
                                                           const WeightType* fc1_expert_weights,    // [num_experts, hidden_size, inter_size]
                                                           const T*          fc1_scales,            
                                                           const T*          fc1_expert_biases,     // [num_experts, inter_size]
                                                           ActivationType    fc1_activation_type,   
                                                           const WeightType* fc2_expert_weights,    // [num_experts, inter_size, hidden_size]
                                                           const T*          fc2_scales,
                                                           const WeightType* fc1_expert_weights_stay_on_GPU,
                                                           const WeightType* fc2_expert_weights_stay_on_GPU,
                                                           float*              expert_priority,
                                                           int*              expert_in_cache,
                                                           int               cache_size,
                                                           int               fix_cache_size,
                                                           float             max_val,
                                                           float             threshold,
                                                           float             dec_in_cache,
                                                           float             dec_out_cache,
                                                           const int         num_rows,              // h_token_num
                                                           const int         hidden_size,
                                                           const int         inter_size,
                                                                 int         num_experts,
                                                           const int         k,
                                                           char*             workspace_ptr,
                                                           T*                fc2_result,            // [num_rows, hidden_size]
                                                           const bool*       finished,
                                                           const int         active_rows,           // num_rows
                                                           T*                expert_scales,
                                                           int*              expanded_source_row_to_expanded_dest_row, // h_token_num, moe_k_
                                                           int*              expert_for_source_row, // h_token_num, moe_k_
                                                           int               layer_num,
                                                           bool              use_cache,
                                                           int&               activated_expert_num,
                                                           int&               iter_num,
                                                           int&               cache_hit_num,
                                                           cudaStream_t      stream)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    static constexpr bool scales_required =
        std::is_same<WeightType, uint8_t>::value || std::is_same<WeightType, cutlass::uint4b_t>::value ||
        std::is_same<WeightType, cutlass::fp4_t>::value || std::is_same<WeightType, cutlass::nf4_t>::value;

    configure_ws_ptrs(workspace_ptr, num_rows, hidden_size, inter_size, num_experts, k);

    bool prefetch = false;
    if (fetcher_context_) {
        prefetch = (fetcher_context_->mode == FetchType::PREFETCH) && (!fetcher_context_->first_time);
    }
// 在预取模式下，代码会使用上一次路由的结果next_expert_scales_,并将其复制到当前的专家比例数组（expert_scales）,所以先做初始化规约
// encoder里是FETCH_ON_DEMAND模式，不进prefetch
    if (prefetch) { 
        // Use result of last routing
        cudaMemcpyAsync(expert_scales, next_expert_scales_, sizeof(T) * num_rows * k, cudaMemcpyDeviceToDevice, stream);
        // 举例：使用expert0的token编号是46 61 67 87 95 105 106 111...
        // permuted_data_[expanded_dest_row] = input_activations[source_row], 即输出的是token46的数据, token61的数据...
        initialize_moe_routing_kernelLauncher(input_activations,            // [num_rows, hidden_size] input
                                              permuted_data_,               // [k * num_rows, hidden_size] output
                                              permuted_rows_,               // [k * num_rows]  input expanded_dest_row_to_expanded_source_row
                                              expanded_source_row_to_expanded_dest_row, // [k * num_rows] output
                                              num_rows,
                                              active_rows,
                                              hidden_size,
                                              k,
                                              stream);

    }

    topk_gating_softmax_kernelLauncher<T>(gating_output,
                                          finished,
                                          prefetch ? next_expert_scales_ : expert_scales,
                                          softmax_out_,   // k * num_rows * inter_size OR NULL
                                          expert_for_source_row, // [num_rows * k]
                                          source_rows_, // [k * num_rows]
                                          num_rows,
                                          num_experts,
                                          k,
                                          stream);

#ifndef NDEBUG
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError());
#endif 

    const int sorter_ws_size_bytes = pad_to_multiple_of_16(sorter_.getWorkspaceSize(k * num_rows));
    sorter_.run((void*)fc1_result_,
                sorter_ws_size_bytes,
                expert_for_source_row,  // [num_rows, k]  input, key
                permuted_experts_,      // [num_rows * k] output
                source_rows_,           // [k * num_rows] input , value
                permuted_rows_,    // [k * num_rows] output
                k * num_rows,
                stream);
    //------------------print experts-------------------------------
    // cudaDeviceSynchronize();
    // std::cout <<"****expert_for_source_row ";
    // int*  h_data= new int[num_rows * k];
    // printf("k: %d", k);
    // cudaMemcpy(h_data, expert_for_source_row, num_rows * k * sizeof(int), cudaMemcpyDeviceToHost);
    // for(int i = 0; i < num_rows * k; i++) {

    //     std::cout << h_data[i] << " ";
    // }
    // std::cout << std::endl << "permuted_experts_ ";
    // cudaMemcpy(h_data, permuted_experts_, num_rows * k * sizeof(int), cudaMemcpyDeviceToHost);
    // for(int i = 0; i < num_rows * k; i++) {
    //     std::cout << h_data[i] << " ";
    // }
    // std::cout << std::endl << "source_rows_ ";
    // cudaMemcpy(h_data, source_rows_, num_rows * k * sizeof(int), cudaMemcpyDeviceToHost);
    // for(int i = 0; i < num_rows * k; i++) {
    //     std::cout << h_data[i] << " ";
    // }
    // std::cout << std::endl << "permuted_rows_ ";
    // cudaMemcpy(h_data, permuted_rows_, num_rows * k * sizeof(int), cudaMemcpyDeviceToHost);
    // for(int i = 0; i < num_rows * k; i++) {
    //     std::cout << h_data[i] << " ";
    // }
    
    // std::cout << std::endl ;
    // delete[] h_data;
    // printf("\n");
    // //----------------------------------------------------------------
    // for(int i = 0; i < num_rows * k; i++) {
    //     printf("[%d] = %f,", i, reinterpret_cast<float*>(permuted_experts_)[i]);
    // }
    // // printf("\nsource_rows_");
    // for(int i = 0; i < num_rows * k; i++) {
    //     printf("[%d] = %d,", i, reinterpret_cast<int>(source_rows_[i]));
    // }
    // printf("\npermuted_rows_");
    // for(int i = 0; i < num_rows * k; i++) {
    //     printf("[%d] = %d,", i, reinterpret_cast<int>(permuted_rows_[i]));
    // }

#ifndef NDEBUG
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError());
#endif 

    if (!prefetch) {
        initialize_moe_routing_kernelLauncher(input_activations,            // [num_rows, hidden_size] input
                                              permuted_data_,               // [k * num_rows, hidden_size] output
                                              permuted_rows_,               // [k * num_rows]  input expanded_dest_row_to_expanded_source_row
                                              expanded_source_row_to_expanded_dest_row, // [k * num_rows] output
                                              num_rows,
                                              active_rows,
                                              hidden_size,
                                              k,
                                              stream);
    }

#ifndef NDEBUG
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError());
#endif 

    const int expanded_active_expert_rows = k * active_rows;    

    // 用二分查找法，确定所有的专家在permuted_experts_里的最后一个索引所在的位置
    compute_total_rows_before_expert(
        permuted_experts_, expanded_active_expert_rows, num_experts, total_rows_before_expert_, stream);

    // std::cout <<"compute_total_rows_before_expert ";

    // int h_size = num_experts;
    // int* h_data_rows_before_expert= new int[h_size];
    // cudaMemcpy(h_data_rows_before_expert, total_rows_before_expert_, h_size * sizeof(int64_t), cudaMemcpyDeviceToHost);
    // for(int i = 0; i < h_size; i++) {
    //     std::cout << h_data_rows_before_expert[i] << " ";
    // }
    // delete[] h_data_rows_before_expert;

#ifndef NDEBUG
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError());
#endif 

    if (fetcher_context_) {
        // Remove redundant experts and leading zero
        unique_and_remove_zero(total_rows_before_expert_, num_experts, stream);

        if (fetcher_context_->mode == FetchType::PREFETCH) {
            if (fetcher_context_->first_time) {
                FT_LOG_DEBUG("Start fetch layer 1");
                fetcher_context_->fetch(permuted_experts_, false, fc1_expert_weights_stay_on_GPU, fc2_expert_weights_stay_on_GPU, expert_priority, expert_in_cache, cache_size, fix_cache_size, max_val, threshold, dec_in_cache, dec_out_cache, layer_num, use_cache, activated_expert_num, iter_num, cache_hit_num); //获取专家权重，将专家索引复制到CPU上，并且在GPU内分配内存
                fetcher_context_->sync();
                FT_LOG_DEBUG("Fetch layer 1 end");
            }
            else {
                fetcher_context_->sync();
            }
            // Get weights of last fetch
            fetcher_context_->get_weights(num_experts,
                                          fc1_expert_weights,
                                          fc2_expert_weights,
                                          fc1_expert_biases, 
                                          fc1_scales,
                                          fc2_scales);
            // Prefetch next layer
            fetcher_context_->fetch(permuted_experts_, true, fc1_expert_weights_stay_on_GPU, fc2_expert_weights_stay_on_GPU, expert_priority, expert_in_cache, cache_size, fix_cache_size, max_val, threshold, dec_in_cache, dec_out_cache, layer_num, use_cache, activated_expert_num, iter_num, cache_hit_num);
        }
        else if (fetcher_context_->mode == FetchType::FETCH_ON_DEMAND){
            fetcher_context_->fetch(permuted_experts_, false, fc1_expert_weights_stay_on_GPU, fc2_expert_weights_stay_on_GPU, expert_priority, expert_in_cache, cache_size, fix_cache_size, max_val, threshold, dec_in_cache, dec_out_cache, layer_num, use_cache, activated_expert_num, iter_num, cache_hit_num);
            fetcher_context_->sync();
            fetcher_context_->get_weights(num_experts,
                                          fc1_expert_weights,
                                          fc2_expert_weights,
                                          fc1_expert_biases, 
                                          fc1_scales,
                                          fc2_scales);
        }
    }

    if (scales_required) {
        if (fc1_scales == nullptr) {
            throw std::runtime_error(
                "[FT Error][Run MoE FC] Scales expected but scale for first matmul is a null pointer");
        }
        else if (fc2_scales == nullptr) {
            throw std::runtime_error(
                "[FT Error][Run MoE FC] Scales expected but scale for second matmul is a null pointer");
        }
    }
    else {
        if (fc1_scales != nullptr) {
            throw std::runtime_error(
                "[FT Error][Run MoE FC] Scales are ignored for fp32/fp16/bf16 but received scale for FC1");
        }
        else if (fc2_scales != nullptr) {
            throw std::runtime_error(
                "[FT Error][Run MoE FC] Scales are ignored for fp32/fp16/bf16 but received scale for FC2");
        }
    }

    if (GlobalConfig::instance().profiling) {
        Profiling::instance().insert(stream, EventType::COMP_START);
    }

#ifndef NDEBUG
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError());
#endif

    if (GlobalConfig::instance().forced_num_experts) {
        num_experts = GlobalConfig::instance().forced_num_experts;
        force_total_rows_before_expert(total_rows_before_expert_, num_experts, expanded_active_expert_rows, stream);
    }

#ifndef NDEBUG
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError());
#endif

    // std::cout << "T Type: " << typeid(T).name() << std::endl;
    // std::cout << "WeightType Type: " << typeid(WeightType).name() << std::endl;
    
    // half temp_half = 0.0;
    // std::cout << "temp_half Type: " << typeid(temp_half).name() << std::endl;
    // float temp_float= 0.0;
    // std::cout << "temp_float Type: " << typeid(temp_float).name() << std::endl;
    
    //  执行带有偏置和激活操作的矩阵乘法（GEMM），并根据不同的激活类型调用相应的操作
    //  执行带有偏置和激活操作的矩阵乘法（GEMM），并根据不同的激活类型调用相应的操作
    // /data/FasterTransformer/src/fastertransformer/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels_template.h
    // run the first matmul with bias and activation
    // permuted_data_是按照permuted_experts_的顺序排列的，即[使用了expert0的token0, 使用了expert1的token0, 使用了expert2的token0, ...]
    // total_rows_before_expert_是每个专家在permuted_experts_中的最后一个索引所在的位置
    moe_gemm_runner_.moe_gemm_bias_act(permuted_data_,              // [k * num_rows, hidden_size] input
                                       fc1_expert_weights,          // [num_experts, hidden_size, inter_size] input
                                       fc1_scales,                  // NULL
                                       fc1_expert_biases,           // [num_experts, inter_size] input
                                       fc1_result_,                 // [k * num_rows, inter_size] output
                                       total_rows_before_expert_,   // [num_experts] input
                                       expanded_active_expert_rows, // = k * num_rows
                                       inter_size,
                                       hidden_size,
                                       num_experts,
                                       fc1_activation_type,
                                       stream);

#ifndef NDEBUG
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError());
#endif
    // run the second matmul
    moe_gemm_runner_.moe_gemm(fc1_result_,                      // [k * num_rows, inter_size] input
                              fc2_expert_weights,               // [num_experts, inter_size, hidden_size] input
                              fc2_scales,                       // NULL
                              fc2_result,                       // [k * num_rows, hidden_size] output
                              total_rows_before_expert_,        // [num_experts] input
                              expanded_active_expert_rows,      // = k * num_rows
                              hidden_size,
                              inter_size,
                              num_experts,
                              stream);
    if (GlobalConfig::instance().profiling) {
        Profiling::instance().insert(stream, EventType::COMP_END);
    }

#ifndef NDEBUG
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError());
#endif
}

template<typename T, typename WeightType, typename Enable>
void CutlassMoeFCRunner<T, WeightType, Enable>::run_moe_fc(const T*          input_activations,     // 输入激活值的指针
                                                           const T*          gating_output,         // 门控输出的指针
                                                           const WeightType* fc1_expert_weights,    // 第一个全连接层专家权重的指针
                                                           const T*          fc1_scales,            // 第一个全连接层的缩放因子的指针
                                                           const T*          fc1_expert_biases,     // 第一个全连接层专家偏置的指针
                                                           ActivationType    fc1_activation_type,   // 第一个全连接层的激活类型
                                                           const WeightType* fc2_expert_weights,
                                                           const T*          fc2_scales,
                                                           const WeightType* fc1_expert_weights_stay_on_GPU,
                                                           const WeightType* fc2_expert_weights_stay_on_GPU,
                                                           float*              expert_priority,
                                                           int*              expert_in_cache,
                                                           int               cache_size,
                                                           int               fix_cache_size,
                                                           float             max_val,
                                                           float             threshold,
                                                           float             dec_in_cache,
                                                           float             dec_out_cache,
                                                           const int         num_rows,              // 输入数据的行数
                                                           const int         hidden_size,           // 隐藏层的大小 d_modle
                                                           const int         inter_size,            // 中间层的大小 d_ff
                                                                 int         num_experts,
                                                           const int         k,                     // 选择的专家数量
                                                           char*             workspace_ptr,
                                                           T*                fc2_result,
                                                           T*                expert_scales,         // 专家缩放因子的指针
                                                           int*              expanded_source_row_to_expanded_dest_row, // 扩展源行到扩展目标行的映射
                                                           int*              expert_for_source_row, // 源行对应的专家
                                                           int               layer_num,
                                                           bool              use_cache,
                                                           int&               activated_expert_num,
                                                           int&               iter_num,
                                                           int&               cache_hit_num,
                                                           cudaStream_t      stream)                // CUDA 流，用于异步执行 
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    run_moe_fc(input_activations,
               gating_output,
               fc1_expert_weights,
               fc1_scales,
               fc1_expert_biases,
               fc1_activation_type,
               fc2_expert_weights,
               fc2_scales,
               fc1_expert_weights_stay_on_GPU,
               fc2_expert_weights_stay_on_GPU,
               expert_priority,
               expert_in_cache,
               cache_size,
               fix_cache_size,
               max_val,
               threshold,
               dec_in_cache,
               dec_out_cache,
               num_rows,
               hidden_size,
               inter_size,
               num_experts,
               k,
               workspace_ptr,
               fc2_result,
               nullptr,
               num_rows,
               expert_scales,
               expanded_source_row_to_expanded_dest_row,
               expert_for_source_row,
               layer_num,
               use_cache,
               activated_expert_num,
                iter_num,
                cache_hit_num,
               stream);
}

template<typename T, typename WeightType, typename Enable>
void CutlassMoeFCRunner<T, WeightType, Enable>::compute_total_rows_before_expert(const int*   sorted_indices,
                                                                                 const int    total_indices,
                                                                                 const int    num_experts,
                                                                                 int64_t*     total_rows_before_expert,
                                                                                 cudaStream_t stream)
{

    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    const int threads = std::min(1024, num_experts);
    const int blocks  = (num_experts + threads - 1) / threads;

    compute_total_rows_before_expert_kernel<<<blocks, threads, 0, stream>>>(
        sorted_indices, total_indices, num_experts, total_rows_before_expert);
}

// ========================== Permutation things =======================================

// Duplicated and permutes rows for MoE. In addition, reverse the permutation map to help with finalizing routing.

// "expanded_x_row" simply means that the number of values is num_rows x k. It is "expanded" since we will have to
// duplicate some rows in the input matrix to match the dimensions. Duplicates will always get routed to separate
// experts in the end.

// Note that the expanded_dest_row_to_expanded_source_row map referred to here has indices in the range (0,
// k*rows_in_input - 1). However, it is set up so that index 0, rows_in_input, 2*rows_in_input ... (k-1)*rows_in_input
// all map to row 0 in the original matrix. Thus, to know where to read in the source matrix, we simply take the modulus
// of the expanded index.

template<typename T>
__global__ void initialize_moe_routing_kernel(const T*   unpermuted_input,
                                              T*         permuted_output,
                                              const int* expanded_dest_row_to_expanded_source_row,
                                              int*       expanded_source_row_to_expanded_dest_row,
                                              const int  num_rows,
                                              const int  active_rows,
                                              const int  cols) // hidden_size = d_model
{

    // Reverse permutation map.
    // I do this so that later, we can use the source -> dest map to do the k-way reduction and unpermuting. I need the
    // reverse map for that reduction to allow each threadblock to do 1 k-way reduce without atomics later in MoE. 1
    // thread block will be responsible for all k summations.

    // expanded_dest_row_to_expanded_source_row is the token number of the source row, (the tokens using expert0), (the tokens using expert1), (the tokens using expert2)...
    // expanded_source_row_to_expanded_dest_row is the offset of the tokens in the expanded_dest_row_to_expanded_source_row matrix
    // (the location of token0), (the location of token1), (the location of token2)...
    // dest_row: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    // source_row: token 46 61 67 87 95 105 106 111 119...
    const int expanded_dest_row   = blockIdx.x;
    const int expanded_source_row = expanded_dest_row_to_expanded_source_row[expanded_dest_row];
    
    if (threadIdx.x == 0) {
        expanded_source_row_to_expanded_dest_row[expanded_source_row] = expanded_dest_row;
        // printf("expanded_dest_row=%d, expanded_source_row = %d\n",expanded_dest_row, expanded_source_row);
        // printf("num_rows=%d, cols = %d\n",num_rows, cols);
    }

    if (blockIdx.x < active_rows) {
        // Duplicate and permute rows
        // when k = 1, source_row = expanded_source_row
        // 举例：使用expert0的token编号是46 61 67 87 95 105 106 111...
        // permuted_output[expanded_dest_row] = unpermuted_input[source_row], 即输出的是token46的数据, token61的数据...
        const int source_row = expanded_source_row % num_rows;

        const T* source_row_ptr = unpermuted_input + source_row * cols;
        T*       dest_row_ptr   = permuted_output + expanded_dest_row * cols;

        for (int tid = threadIdx.x; tid < cols; tid += blockDim.x) {
            dest_row_ptr[tid] = source_row_ptr[tid];
        }
    }

}

template<typename T>
void initialize_moe_routing_kernelLauncher(const T*     unpermuted_input,
                                           T*           permuted_output,
                                           const int*   expanded_dest_row_to_expanded_source_row,
                                           int*         expanded_source_row_to_expanded_dest_row,
                                           const int    num_rows,
                                           const int    active_rows,
                                           const int    cols,
                                           const int    k,
                                           cudaStream_t stream)
{
    const int blocks  = num_rows * k;
    const int threads = std::min(cols, 1024);
    // printf("num_rows=%d, active_rows=%d\n", num_rows, active_rows);
    // 打印规约前的专家编号
    // printf("Before reduction:\n");
    // for (int i = 0; i < active_rows; ++i) {
    //     printf("expanded_source_row: %d, expanded_dest_row: %d\n", expanded_dest_row_to_expanded_source_row[i], i);
    // }
    initialize_moe_routing_kernel<T><<<blocks, threads, 0, stream>>>(unpermuted_input,
                                                                     permuted_output,
                                                                     expanded_dest_row_to_expanded_source_row,
                                                                     expanded_source_row_to_expanded_dest_row,
                                                                     num_rows,
                                                                     k * active_rows,
                                                                     cols);
    //-------------------------print experts--------------------------------
    // int*  h_data_2 = new int[num_rows * k];
    // std::cout <<"****expanded_dest_row_to_expanded_source_row ";
    // cudaMemcpy(h_data_2, expanded_dest_row_to_expanded_source_row, num_rows * k * sizeof(int), cudaMemcpyDeviceToHost);
    // for(int i = 0; i < num_rows * k; i++) {
    //     std::cout << h_data_2[i] << " ";
    // }
    // std::cout <<std::endl<<"****expanded_source_row_to_expanded_dest_row ";
    // cudaMemcpy(h_data_2, expanded_source_row_to_expanded_dest_row, num_rows * k * sizeof(int), cudaMemcpyDeviceToHost);
    // for(int i = 0; i < num_rows * k; i++) {
    //     std::cout << h_data_2[i] << " ";
    // }
    // std::cout << std::endl;
    // delete[] h_data_2;
    //-----------------------------------------------------------------------
    // 打印规约后的专家编号
    // printf("After reduction:\n");
    // printf("expanded_source_row_to_expanded_dest_row=%p\n", expanded_source_row_to_expanded_dest_row);
    // printf("size=%lu", sizeof(expanded_source_row_to_expanded_dest_row)/sizeof(int));
    // for (int i = 0; i < num_rows; ++i) {
    //     printf("Expert %d ", expanded_source_row_to_expanded_dest_row[i]);
    // }
    // printf("\n");
}

// Final kernel to unpermute and scale
// This kernel unpermutes the original data, does the k-way reduction and performs the final skip connection.
template<typename T, int RESIDUAL_NUM>
__global__ void finalize_moe_routing_kernel(const T*   expanded_permuted_rows,
                                            T*         reduced_unpermuted_output,
                                            const T*   skip_1,
                                            const T*   skip_2,
                                            const T*   bias,
                                            const T*   scales,
                                            const int* expanded_source_row_to_expanded_dest_row,
                                            const int* expert_for_source_row,
                                            const int  cols,
                                            const int  k)
{

    const int original_row    = blockIdx.x;
    const int num_rows        = gridDim.x;
    T*        reduced_row_ptr = reduced_unpermuted_output + original_row * cols;
    const T*  skip_1_row_ptr  = skip_1 + original_row * cols;
    const T*  skip_2_row_ptr;
    if (RESIDUAL_NUM == 2) {
        skip_2_row_ptr = skip_2 + original_row * cols;
    }

    for (int tid = threadIdx.x; tid < cols; tid += blockDim.x) {
        T thread_output;
        if (RESIDUAL_NUM == 1) {
            thread_output = skip_1_row_ptr[tid];
        }
        else if (RESIDUAL_NUM == 2) {
            thread_output = skip_1_row_ptr[tid] + skip_2_row_ptr[tid];
        }
        for (int k_idx = 0; k_idx < k; ++k_idx) {
            const int expanded_original_row = original_row + k_idx * num_rows;
            const int expanded_permuted_row = expanded_source_row_to_expanded_dest_row[expanded_original_row];

            const int64_t k_offset                       = original_row * k + k_idx;
            const T       row_scale                      = scales[k_offset];
            const T*      expanded_permuted_rows_row_ptr = expanded_permuted_rows + expanded_permuted_row * cols;

            const int expert_idx = expert_for_source_row[k_offset];

            if (bias) {
                const T*  bias_ptr   = bias + expert_idx * cols;
                thread_output = thread_output + row_scale * (expanded_permuted_rows_row_ptr[tid] + bias_ptr[tid]);
            }
            else {
                thread_output = thread_output + row_scale * expanded_permuted_rows_row_ptr[tid];
            }
        }
        reduced_row_ptr[tid] = thread_output;
    }
}

template<typename T>
void finalize_moe_routing_kernelLauncher(const T*     expanded_permuted_rows,
                                         T*           reduced_unpermuted_output,
                                         const T*     skip,
                                         const T*     bias,
                                         const T*     scales,
                                         const int*   expanded_source_row_to_expanded_dest_row,
                                         const int*   expert_for_source_row,
                                         const int    num_rows,
                                         const int    cols,
                                         const int    k,
                                         cudaStream_t stream)
{
    const int blocks  = num_rows;
    const int threads = std::min(cols, 1024);
    finalize_moe_routing_kernel<T, 1><<<blocks, threads, 0, stream>>>(expanded_permuted_rows,
                                                                      reduced_unpermuted_output,
                                                                      skip,
                                                                      nullptr,
                                                                      bias,
                                                                      scales,
                                                                      expanded_source_row_to_expanded_dest_row,
                                                                      expert_for_source_row,
                                                                      cols,
                                                                      k);
}

template<typename T>
void finalize_moe_routing_kernelLauncher(const T*     expanded_permuted_rows,
                                         T*           reduced_unpermuted_output,
                                         const T*     skip_1,
                                         const T*     skip_2,
                                         const T*     bias,
                                         const T*     scales,
                                         const int*   expanded_source_row_to_expanded_dest_row,
                                         const int*   expert_for_source_row,
                                         const int    num_rows,
                                         const int    cols,
                                         const int    k,
                                         cudaStream_t stream)
{
    const int blocks  = num_rows;
    const int threads = std::min(cols, 1024);
    if (skip_2 == nullptr) {
        finalize_moe_routing_kernel<T, 1><<<blocks, threads, 0, stream>>>(expanded_permuted_rows,
                                                                          reduced_unpermuted_output,
                                                                          skip_1,
                                                                          skip_2,
                                                                          bias,
                                                                          scales,
                                                                          expanded_source_row_to_expanded_dest_row,
                                                                          expert_for_source_row,
                                                                          cols,
                                                                          k);
    }
    else {
        finalize_moe_routing_kernel<T, 2><<<blocks, threads, 0, stream>>>(expanded_permuted_rows,
                                                                          reduced_unpermuted_output,
                                                                          skip_1,
                                                                          skip_2,
                                                                          bias,
                                                                          scales,
                                                                          expanded_source_row_to_expanded_dest_row,
                                                                          expert_for_source_row,
                                                                          cols,
                                                                          k);
    }
}

// ========================= TopK Softmax specializations ===========================
template void topk_gating_softmax_kernelLauncher(
    const float*, const bool*, float*, float*, int*, int*, const int, const int, const int, cudaStream_t);
template void topk_gating_softmax_kernelLauncher(
    const half*, const bool*, half*, half*, int*, int*, const int, const int, const int, cudaStream_t);

#ifdef ENABLE_BF16
template void topk_gating_softmax_kernelLauncher(const __nv_bfloat16*,
                                                 const bool*,
                                                 __nv_bfloat16*,
                                                 __nv_bfloat16*,
                                                 int*,
                                                 int*,
                                                 const int,
                                                 const int,
                                                 const int,
                                                 cudaStream_t);
#endif

// ==================== Variable batched GEMM specializations ==================================
template class CutlassMoeFCRunner<float, float>;

#ifdef ENABLE_BF16
template class CutlassMoeFCRunner<__nv_bfloat16, __nv_bfloat16>;
template class CutlassMoeFCRunner<__nv_bfloat16, uint8_t>;
template class CutlassMoeFCRunner<__nv_bfloat16, cutlass::uint4b_t>;
template class CutlassMoeFCRunner<__nv_bfloat16, cutlass::fp4_t>;
template class CutlassMoeFCRunner<__nv_bfloat16, cutlass::nf4_t>;
#endif

template class CutlassMoeFCRunner<half, half>;
template class CutlassMoeFCRunner<half, uint8_t>;
template class CutlassMoeFCRunner<half, cutlass::uint4b_t>;
template class CutlassMoeFCRunner<half, cutlass::fp4_t>;
template class CutlassMoeFCRunner<half, cutlass::nf4_t>;

// ===================== Specializations for init routing =========================
template void initialize_moe_routing_kernelLauncher(
    const float*, float*, const int*, int*, const int, const int, const int, const int, cudaStream_t);
template void initialize_moe_routing_kernelLauncher(
    const half*, half*, const int*, int*, const int, const int, const int, const int, cudaStream_t);
#ifdef ENABLE_BF16
template void initialize_moe_routing_kernelLauncher(
    const __nv_bfloat16*, __nv_bfloat16*, const int*, int*, const int, const int, const int, const int, cudaStream_t);
#endif

// ==================== Specializations for final routing ===================================
template void finalize_moe_routing_kernelLauncher(const float*,
                                                  float*,
                                                  const float*,
                                                  const float*,
                                                  const float*,
                                                  const int*,
                                                  const int*,
                                                  const int,
                                                  const int,
                                                  const int,
                                                  cudaStream_t);
template void finalize_moe_routing_kernelLauncher(const half*,
                                                  half*,
                                                  const half*,
                                                  const half*,
                                                  const half*,
                                                  const int*,
                                                  const int*,
                                                  const int,
                                                  const int,
                                                  const int,
                                                  cudaStream_t);
template void finalize_moe_routing_kernelLauncher(const float*,
                                                  float*,
                                                  const float*,
                                                  const float*,
                                                  const float*,
                                                  const float*,
                                                  const int*,
                                                  const int*,
                                                  const int,
                                                  const int,
                                                  const int,
                                                  cudaStream_t);
template void finalize_moe_routing_kernelLauncher(const half*,
                                                  half*,
                                                  const half*,
                                                  const half*,
                                                  const half*,
                                                  const half*,
                                                  const int*,
                                                  const int*,
                                                  const int,
                                                  const int,
                                                  const int,
                                                  cudaStream_t);
#ifdef ENABLE_BF16
template void finalize_moe_routing_kernelLauncher(const __nv_bfloat16*,
                                                  __nv_bfloat16*,
                                                  const __nv_bfloat16*,
                                                  const __nv_bfloat16*,
                                                  const __nv_bfloat16*,
                                                  const int*,
                                                  const int*,
                                                  const int,
                                                  const int,
                                                  const int,
                                                  cudaStream_t);
template void finalize_moe_routing_kernelLauncher(const __nv_bfloat16*,
                                                  __nv_bfloat16*,
                                                  const __nv_bfloat16*,
                                                  const __nv_bfloat16*,
                                                  const __nv_bfloat16*,
                                                  const __nv_bfloat16*,
                                                  const int*,
                                                  const int*,
                                                  const int,
                                                  const int,
                                                  const int,
                                                  cudaStream_t);
#endif


// __global__ void tag_all_actiavted_experts(
//     int *expert_sparse_idx,
//     const int *expert_for_source_row,
//     const int num_rows) {

//     int row = blockIdx.x;
//     if (row < num_rows) {
//         expert_sparse_idx[expert_for_source_row[row]] = 1;
//     }
// }

// __global__ void prefix_sum_to_get_sparse_index(
//     int *expert_sparse_idx,
//     const int num_experts) {

//     int tid = threadIdx.x;
//     if (0 < tid && tid < num_experts) {
//         expert_sparse_idx[tid] += expert_sparse_idx[tid - 1];
//     }
// }

// void get_expert_sparse_idx_kernelLauncher(
//     int *expert_sparse_idx,
//     const int *expert_for_source_row,
//     const int num_rows,
//     const int num_experts,
//     int *active_expert_count // cpu
//     ) {
    
//     check_cuda_error(cudaMemset(expert_sparse_idx, 0, sizeof(int) * num_experts));
        
//     tag_all_actiavted_experts<<<1, num_rows>>>(
//         expert_sparse_idx, 
//         expert_for_source_row, 
//         num_rows);
//     prefix_sum_to_get_sparse_index<<<1, num_experts>>>
//         (expert_sparse_idx, num_experts);
// }


}  // namespace fastertransformer
