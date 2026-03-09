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

// Objective-C++ implementation of the Metal runtime C API.
//
// Only this file uses Metal/ObjC APIs.  All other C++ sources interact with
// Metal exclusively through the plain C interface declared in metal_runtime.h.

#import <CoreFoundation/CoreFoundation.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <string>
#include <mach-o/dyld.h>  // _NSGetExecutablePath

#include "metal_runtime.h"

// ---------------------------------------------------------------------------
// Internal state

static id<MTLDevice>       g_device        = nil;
static id<MTLCommandQueue> g_queue         = nil;
static id<MTLLibrary>      g_library       = nil;

// Current open command buffer + encoder (kept open across multiple dispatches;
// committed and waited on by metal_synchronize()).
static id<MTLCommandBuffer>         g_cmd_buf = nil;
static id<MTLComputeCommandEncoder> g_encoder = nil;

// Pipeline cache: kernel name → MTLComputePipelineState.
// The pipeline states are retained via __bridge_retained, released on exit.
static std::unordered_map<std::string, id<MTLComputePipelineState>>* g_pipelines;

// ---------------------------------------------------------------------------
// Kernel parameter structs (must match metal/sim_kernels.h exactly)

struct InitProgramsParams {
  uint64_t seed;
  uint64_t num_programs;
  int32_t  zero_init;
  int32_t  _pad;
};

struct MutateAndRunParams {
  uint64_t seed;
  uint64_t num_programs;
  uint64_t num_indices;
  uint32_t mutation_prob;
  uint32_t _pad;
};

struct CheckSelfRepParams {
  uint64_t seed;
  uint64_t num_programs;
};

// ---------------------------------------------------------------------------
// Helper: ensure the command buffer and encoder are open

static void ensure_encoder() {
  if (!g_encoder) {
    g_cmd_buf = [g_queue commandBuffer];
    g_encoder = [g_cmd_buf computeCommandEncoder];
  }
}

// ---------------------------------------------------------------------------
// Helper: look up (or create) a pipeline state for `kernel_name`

static id<MTLComputePipelineState> get_pipeline(const char* kernel_name) {
  auto it = g_pipelines->find(kernel_name);
  if (it != g_pipelines->end()) {
    return it->second;
  }

  NSString* name     = [NSString stringWithUTF8String:kernel_name];
  id<MTLFunction> fn = [g_library newFunctionWithName:name];
  if (!fn) {
    fprintf(stderr, "metal_runtime: kernel '%s' not found in metallib\n",
            kernel_name);
    abort();
  }

  NSError* error = nil;
  id<MTLComputePipelineState> pso =
      [g_device newComputePipelineStateWithFunction:fn error:&error];
  if (!pso) {
    fprintf(stderr, "metal_runtime: failed to create pipeline for '%s': %s\n",
            kernel_name,
            [[error localizedDescription] UTF8String]);
    abort();
  }

  (*g_pipelines)[kernel_name] = pso;
  return pso;
}

// ---------------------------------------------------------------------------
// Helper: dispatch with a given pipeline and thread count

static void dispatch(id<MTLComputePipelineState> pso, size_t thread_count) {
  NSUInteger tg_size = pso.maxTotalThreadsPerThreadgroup;
  if (tg_size > 128) tg_size = 128;
  if (tg_size > thread_count) tg_size = thread_count;

  [g_encoder setComputePipelineState:pso];
  // dispatchThreads:threadsPerThreadgroup: automatically handles non-uniform
  // grids (thread_count need not be a multiple of tg_size).
  [g_encoder dispatchThreads:MTLSizeMake(thread_count, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
}

// ---------------------------------------------------------------------------
// Public C API implementation

extern "C" {

void* metal_buffer_alloc(size_t bytes, void** out_ptr) {
  if (!g_device) {
    fprintf(stderr, "metal_runtime: metal_init() must be called first\n");
    abort();
  }
  // Shared mode: CPU and GPU access the same physical memory (unified memory
  // on Apple Silicon).  No explicit transfer is ever needed.
  id<MTLBuffer> buf =
      [g_device newBufferWithLength:(bytes ? bytes : 1)
                            options:MTLResourceStorageModeShared];
  if (!buf) {
    fprintf(stderr, "metal_runtime: failed to allocate %zu-byte buffer\n",
            bytes);
    abort();
  }
  *out_ptr = buf.contents;
  // Bridge to an unmanaged void* so the buffer survives outside of ObjC's ARC.
  return (__bridge_retained void*)buf;
}

void metal_buffer_free(void* handle) {
  if (handle) {
    // Transfer ownership back to ARC, which immediately releases.
    id<MTLBuffer> __unused released = (__bridge_transfer id<MTLBuffer>)handle;
  }
}

int metal_init(const char* metallib_path) {
  g_device = MTLCreateSystemDefaultDevice();
  if (!g_device) {
    fprintf(stderr, "metal_runtime: no Metal device found\n");
    return 1;
  }

  g_queue = [g_device newCommandQueue];
  if (!g_queue) {
    fprintf(stderr, "metal_runtime: could not create command queue\n");
    return 1;
  }

  NSURL*   url   = [NSURL fileURLWithPath:[NSString stringWithUTF8String:metallib_path]];
  NSError* error = nil;
  g_library = [g_device newLibraryWithURL:url error:&error];
  if (!g_library) {
    fprintf(stderr,
            "metal_runtime: could not load metallib '%s': %s\n",
            metallib_path,
            [[error localizedDescription] UTF8String]);
    return 1;
  }

  g_pipelines = new std::unordered_map<std::string, id<MTLComputePipelineState>>();
  return 0;
}

void metal_synchronize() {
  if (g_encoder) {
    [g_encoder endEncoding];
    g_encoder = nil;
    [g_cmd_buf commit];
    [g_cmd_buf waitUntilCompleted];
    g_cmd_buf = nil;
  }
}

void metal_dispatch_init_programs(
    size_t   thread_count,
    void*    programs_handle,
    uint64_t seed,
    uint64_t num_programs,
    int      zero_init)
{
  ensure_encoder();
  id<MTLComputePipelineState> pso = get_pipeline("init_programs");

  id<MTLBuffer> prog_buf = (__bridge id<MTLBuffer>)programs_handle;
  [g_encoder setBuffer:prog_buf offset:0 atIndex:0];

  InitProgramsParams params = {seed, num_programs, zero_init, 0};
  [g_encoder setBytes:&params length:sizeof(params) atIndex:1];

  dispatch(pso, thread_count);
}

void metal_dispatch_mutate_and_run(
    const char* kernel_name,
    size_t      thread_count,
    void*       programs_handle,
    void*       shuf_idx_handle,
    void*       insn_per_thread_handle,
    uint64_t    seed,
    uint32_t    mutation_prob,
    uint64_t    num_programs,
    uint64_t    num_indices)
{
  ensure_encoder();
  id<MTLComputePipelineState> pso = get_pipeline(kernel_name);

  [g_encoder setBuffer:(__bridge id<MTLBuffer>)programs_handle        offset:0 atIndex:0];
  [g_encoder setBuffer:(__bridge id<MTLBuffer>)shuf_idx_handle        offset:0 atIndex:1];
  [g_encoder setBuffer:(__bridge id<MTLBuffer>)insn_per_thread_handle offset:0 atIndex:2];

  MutateAndRunParams params = {seed, num_programs, num_indices, mutation_prob, 0};
  [g_encoder setBytes:&params length:sizeof(params) atIndex:3];

  dispatch(pso, thread_count);
}

void metal_dispatch_check_selfrep(
    const char* kernel_name,
    size_t      thread_count,
    void*       programs_handle,
    void*       result_handle,
    uint64_t    seed,
    uint64_t    num_programs)
{
  ensure_encoder();
  id<MTLComputePipelineState> pso = get_pipeline(kernel_name);

  [g_encoder setBuffer:(__bridge id<MTLBuffer>)programs_handle offset:0 atIndex:0];
  [g_encoder setBuffer:(__bridge id<MTLBuffer>)result_handle   offset:0 atIndex:1];

  CheckSelfRepParams params = {seed, num_programs};
  [g_encoder setBytes:&params length:sizeof(params) atIndex:2];

  dispatch(pso, thread_count);
}

void metal_auto_init() {
  static std::once_flag flag;
  std::call_once(flag, []() {
    if (g_device) return;  // already initialized

    auto try_init = [](const char* path) -> bool {
      return path && metal_init(path) == 0;
    };

    // 1. Try CUBFF_METALLIB environment variable.
    if (try_init(getenv("CUBFF_METALLIB"))) return;

    // 2. Try alongside the executable.
    char exec_buf[4096];
    uint32_t exec_size = sizeof(exec_buf);
    if (_NSGetExecutablePath(exec_buf, &exec_size) == 0) {
      NSString* exec_dir =
          [[NSString stringWithUTF8String:exec_buf]
              stringByDeletingLastPathComponent];
      NSString* lib_path =
          [exec_dir stringByAppendingPathComponent:@"cubff.metallib"];
      if (try_init([lib_path UTF8String])) return;

      // 3. Try bin/ relative to the parent of the executable's directory.
      NSString* parent_dir = [exec_dir stringByDeletingLastPathComponent];
      lib_path = [[parent_dir stringByAppendingPathComponent:@"bin"]
                      stringByAppendingPathComponent:@"cubff.metallib"];
      if (try_init([lib_path UTF8String])) return;
    }

    fprintf(stderr,
            "metal_runtime: could not find cubff.metallib\n"
            "  Set CUBFF_METALLIB=/path/to/cubff.metallib\n");
    abort();
  });
}

}  // extern "C"
