/*
 * Copyright 2024 Google LLC
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

// Metal runtime C API.
//
// This header is pure C++ with no Objective-C syntax so it can be included
// from any .cc or .cu translation unit.  All Objective-C / Metal API code
// lives exclusively in metal_runtime.mm.

#pragma once

#include <cstddef>
#include <cstdint>

// ---------------------------------------------------------------------------
// Buffer handle
//
// metal_buffer_alloc() creates an MTLBuffer with MTLResourceStorageModeShared,
// which places the buffer in the unified memory pool accessible by both the
// CPU and GPU without any copy.  The returned *out_ptr is the CPU-side
// address; the same physical memory is read/written by the GPU.
//
// The opaque handle is an ObjC object reference kept alive by a manual
// CFRetain.  Call metal_buffer_free() to release it.

extern "C" {

// Allocate a shared (unified-memory) Metal buffer of `bytes` bytes.
// Sets *out_ptr to the CPU-accessible start address.
// Returns an opaque handle for GPU dispatch.  Never returns NULL; aborts on
// allocation failure.
void* metal_buffer_alloc(size_t bytes, void** out_ptr);

// Release a buffer created by metal_buffer_alloc().
void metal_buffer_free(void* handle);

// ---------------------------------------------------------------------------
// Lifecycle

// Load the Metal shader library from `metallib_path`.  Must be called once
// before any dispatch function.  Returns 0 on success, non-zero on error.
int metal_init(const char* metallib_path);

// Locate cubff.metallib automatically (searches alongside the executable and
// the CUBFF_METALLIB environment variable) and call metal_init().  Aborts if
// the library cannot be found.  Safe to call multiple times; re-entrant calls
// after successful initialization are no-ops.
void metal_auto_init(void);

// ---------------------------------------------------------------------------
// Synchronization

// Wait for all GPU work submitted so far to complete and commit the current
// command buffer.  After this call, all shared buffers written by the GPU are
// visible to the CPU.
void metal_synchronize();

// ---------------------------------------------------------------------------
// Compute dispatches
//
// All dispatch functions encode work into the current command buffer without
// waiting.  Call metal_synchronize() to wait for completion.
//
// Buffer layout matches the kernel argument indices in the .metal source:
//
//   init_programs:
//     [0] programs   device uint8_t*
//     [1] params     constant (seed, num_programs, zero_init)
//
//   mutate_and_run_<lang>:
//     [0] programs         device uint8_t*
//     [1] shuf_idx         device const uint32_t*
//     [2] insn_per_thread  device uint64_t*
//     [3] params           constant (seed, num_programs, num_indices, mutation_prob)
//
//   check_selfrep_<lang>:
//     [0] programs  device uint8_t*
//     [1] result    device uint64_t*
//     [2] params    constant (seed, num_programs)

void metal_dispatch_init_programs(
    size_t   thread_count,
    void*    programs_handle,
    uint64_t seed,
    uint64_t num_programs,
    int      zero_init);

void metal_dispatch_mutate_and_run(
    const char* kernel_name,
    size_t      thread_count,
    void*       programs_handle,
    void*       shuf_idx_handle,
    void*       insn_per_thread_handle,
    uint64_t    seed,
    uint32_t    mutation_prob,
    uint64_t    num_programs,
    uint64_t    num_indices);

void metal_dispatch_check_selfrep(
    const char* kernel_name,
    size_t      thread_count,
    void*       programs_handle,
    void*       result_handle,
    uint64_t    seed,
    uint64_t    num_programs);

}  // extern "C"
