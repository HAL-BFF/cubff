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

// Metal compute kernels for the SubLeq language family.
//
// Variants: subleq, rsubleq4

#include "sim_kernels.h"

// ---------------------------------------------------------------------------
// ============================================================
// Language: subleq
//   3-byte instruction: a, b, c
//   tape[a] -= tape[b]
//   if tape[a] <= 0: jump to tape[c]  (using tape[c] as destination)
//   else: pc += 3
// ============================================================

struct SubleqLang {
  static uint64_t Evaluate(thread uint8_t* tape, uint64_t stepcount) {
    int pos = 0;
    uint64_t i = 0;
    for (; i < stepcount; i++) {
      int sa = tape[pos]     % (2 * kSingleTapeSize);
      int sb = tape[pos + 1] % (2 * kSingleTapeSize);
      tape[sa] -= tape[sb];
      if ((tape[sa] & 0x80) || tape[sa] == 0) {
        pos = tape[pos + 2];
      } else {
        pos += 3;
      }
      if (pos + 3 > 2 * kSingleTapeSize || pos < 0) {
        i++;
        break;
      }
    }
    return i;
  }
};

// ---------------------------------------------------------------------------
// ============================================================
// Language: rsubleq4
//   4-byte instruction: da, db, dc, djump
//   Addresses are PC-relative.
//   sa = (pos + tape[pos])   % tape_size
//   sb = (pos + tape[pos+1]) % tape_size
//   sc = (pos + tape[pos+2]) % tape_size  (the result address)
//   tape[sa] = tape[sb] - tape[sc]
//   if tape[sa] <= 0: pos += (int8_t)tape[pos+3]
//   else: pos += 4
// ============================================================

struct RSubleq4Lang {
  static uint64_t Evaluate(thread uint8_t* tape, uint64_t stepcount) {
    int pos = 0;
    uint64_t i = 0;
    for (; i < stepcount; i++) {
      int sa = (pos + (int)tape[pos])     % (2 * kSingleTapeSize);
      int sb = (pos + (int)tape[pos + 1]) % (2 * kSingleTapeSize);
      int sc = (pos + (int)tape[pos + 2]) % (2 * kSingleTapeSize);
      tape[sa] = tape[sb] - tape[sc];
      if ((tape[sa] & 0x80) || tape[sa] == 0) {
        pos += (int)((int8_t)tape[pos + 3]);
      } else {
        pos += 4;
      }
      if (pos + 4 > 2 * kSingleTapeSize || pos < 0) {
        i++;
        break;
      }
    }
    return i;
  }
};

// ---------------------------------------------------------------------------
// Kernel entry points

#define DEFINE_SUBLEQ_KERNELS(SUFFIX, LANG)                                   \
kernel void mutate_and_run_##SUFFIX(                                          \
    device uint8_t*              programs         [[buffer(0)]],              \
    device const uint32_t*       shuf_idx         [[buffer(1)]],              \
    device uint64_t*             insn_per_thread  [[buffer(2)]],              \
    constant MutateAndRunParams& params           [[buffer(3)]],              \
    uint                         index            [[thread_position_in_grid]])\
{                                                                             \
  mutate_and_run_impl<LANG>(programs, shuf_idx, insn_per_thread, params,     \
                             index);                                          \
}                                                                             \
kernel void check_selfrep_##SUFFIX(                                           \
    device const uint8_t*         programs [[buffer(0)]],                     \
    device uint64_t*              result   [[buffer(1)]],                     \
    constant CheckSelfRepParams&  params   [[buffer(2)]],                     \
    uint                          index    [[thread_position_in_grid]])        \
{                                                                             \
  check_selfrep_impl<LANG>(programs, result, params, index);                 \
}

DEFINE_SUBLEQ_KERNELS(subleq,   SubleqLang)
DEFINE_SUBLEQ_KERNELS(rsubleq4, RSubleq4Lang)
