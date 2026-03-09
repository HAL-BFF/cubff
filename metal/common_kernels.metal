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

// Language-agnostic Metal compute kernels.
//
// This file is compiled into a separate .air and linked into cubff.metallib
// alongside the per-language kernel files.  Placing init_programs here (rather
// than in sim_kernels.h) prevents duplicate-symbol errors when all three
// per-language .metal files include the shared header.

#include <metal_stdlib>
using namespace metal;

#include "sim_kernels.h"

// ---------------------------------------------------------------------------
// init_programs — fills each program's tape with pseudo-random bytes or zeros.
// One GPU thread per program.

kernel void init_programs(
    device uint8_t*              programs  [[buffer(0)]],
    constant InitProgramsParams& params    [[buffer(1)]],
    uint                         index     [[thread_position_in_grid]])
{
  if (index >= (uint)params.num_programs) return;

  device uint8_t* prog = programs + index * kSingleTapeSize;

  if (params.zero_init) {
    for (int i = 0; i < kSingleTapeSize; i++) {
      prog[i] = 0;
    }
  } else {
    uint64_t base = (uint64_t)kSingleTapeSize * params.num_programs * params.seed
                    + (uint64_t)kSingleTapeSize * index;
    for (int i = 0; i < kSingleTapeSize; i++) {
      prog[i] = (uint8_t)(SplitMix64(base + i) % 256);
    }
  }
}
