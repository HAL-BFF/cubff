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

// Shared MSL header included by all cubff .metal shader files.
//
// Provides:
//  - C-compatible fixed-width integer typedefs
//  - Simulation constants
//  - SplitMix64 PRNG
//  - Kernel parameter structs (must match metal_runtime.mm exactly)
//  - The language-agnostic init_programs kernel
//
// This header is compiled as Metal Shading Language, not as C++.

#pragma once

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Simulation constants

constant int kSingleTapeSize = 64;

// ---------------------------------------------------------------------------
// PRNG (identical to the C++ version in common_language.h)

inline uint64_t SplitMix64(uint64_t seed) {
  uint64_t z = seed + 0x9e3779b97f4a7c15UL;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9UL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebUL;
  return z ^ (z >> 31);
}

// ---------------------------------------------------------------------------
// Kernel parameter structs (layout must match metal_runtime.mm)

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
// Shared mutate + evaluate logic called from per-language kernels.
//
// Each thread handles one pair of programs (indexed by shuf_idx[2*index] and
// shuf_idx[2*index+1]).  The language-specific Evaluate function is called
// on the combined tape.
//
// Template parameter LANG must provide:
//   static uint64_t Evaluate(thread uint8_t* tape, uint64_t stepcount);

template<typename LANG>
inline void mutate_and_run_impl(
    device uint8_t*               programs,
    device const uint32_t*        shuf_idx,
    device uint64_t*              insn_per_thread,
    constant MutateAndRunParams&  params,
    uint                          index)
{
  if (2 * (uint64_t)index >= params.num_programs) {
    insn_per_thread[index] = 0;
    return;
  }

  uint32_t p1 = shuf_idx[2 * index];
  uint32_t p2 = shuf_idx[2 * index + 1];

  // Load the two programs into a local tape.
  uint8_t tape[2 * 64];  // kSingleTapeSize = 64, compile-time constant
  for (int i = 0; i < kSingleTapeSize; i++) {
    tape[i]                 = programs[p1 * kSingleTapeSize + i];
    tape[i + kSingleTapeSize] = programs[p2 * kSingleTapeSize + i];
  }

  // Mutate both halves.
  uint64_t index_seed = params.num_programs * params.seed + index;
  for (int i = 0; i < 2 * kSingleTapeSize; i++) {
    uint64_t rng      = SplitMix64(index_seed * (uint64_t)(kSingleTapeSize * 2) + i);
    uint8_t  repl     = (uint8_t)(rng & 0xFF);
    uint64_t prob_rng = (rng >> 8) & ((1UL << 30) - 1);
    if (prob_rng < params.mutation_prob) {
      tape[i] = repl;
    }
  }

  // Evaluate (language-specific).
  uint64_t ops = 0;
  if ((uint64_t)index < params.num_indices) {
    ops = LANG::Evaluate(tape, 8 * 1024);
  }

  // Write programs back.
  for (int i = 0; i < kSingleTapeSize; i++) {
    programs[p1 * kSingleTapeSize + i]     = tape[i];
    programs[p2 * kSingleTapeSize + i] = tape[i + kSingleTapeSize];
  }

  // Accumulate into per-thread slot (no atomic needed: each thread owns its
  // slot exclusively across epochs).
  insn_per_thread[index] += ops;
}

// ---------------------------------------------------------------------------
// Shared check_selfrep logic
//
// Template parameter LANG must provide:
//   static uint64_t Evaluate(thread uint8_t* tape, uint64_t stepcount);

template<typename LANG>
inline void check_selfrep_impl(
    device const uint8_t*         programs,
    device uint64_t*              result,
    constant CheckSelfRepParams&  params,
    uint                          index)
{
  if ((uint64_t)index > params.num_programs) {
    return;
  }

  constexpr int kNumIters     = 13;
  constexpr int kNumExtraGens = 4;

  uint8_t tapes[kNumIters][2 * 64];

  uint64_t local_seed = SplitMix64(params.num_programs * params.seed + index);

  for (int i = 0; i < kNumIters; i++) {
    uint8_t noise[64];
    for (int j = 0; j < kSingleTapeSize; j++) {
      noise[j] = (uint8_t)(
          SplitMix64(local_seed ^ SplitMix64((uint64_t)(i + 1) * kSingleTapeSize + j))
          % 256);
    }

    thread uint8_t* tape = &tapes[i][0];
    for (int j = 0; j < kSingleTapeSize; j++) {
      tape[j]                 = programs[index * kSingleTapeSize + j];
      tape[j + kSingleTapeSize] = noise[j];
    }

    LANG::Evaluate(tape, 8 * 1024);

    for (int g = 0; g < kNumExtraGens; g++) {
      for (int j = 0; j < kSingleTapeSize; j++) {
        tape[j]                 = tape[j + kSingleTapeSize];
        tape[j + kSingleTapeSize] = noise[j];
      }
      LANG::Evaluate(tape, 8 * 1024);
    }
  }

  // Count how many tape positions are "stable" across iterations.
  uint64_t res[2] = {0, 0};
  for (int i = 0; i < 2 * kSingleTapeSize; i++) {
    for (int a = 0; a < kNumIters; a++) {
      uint64_t count = 1;
      if (i < kSingleTapeSize &&
          tapes[a][i] != programs[index * kSingleTapeSize + i]) {
        continue;
      }
      for (int b = a + 1; b < kNumIters; b++) {
        if (tapes[a][i] == tapes[b][i]) count++;
      }
      if (count > (uint64_t)kNumIters / 4) {
        res[i / kSingleTapeSize]++;
        break;
      }
    }
  }
  result[index] = res[0] < res[1] ? res[0] : res[1];
}
