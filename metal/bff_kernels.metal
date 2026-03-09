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

// Metal compute kernels for the BFF language family.
//
// This file implements MSL equivalents of the CUDA kernels for:
//   bff, bff_noheads, bff_perm, bff_selfmove, bff_noheads_4bit, bff8, bff8_noheads
//
// Each language variant is a struct that provides:
//   - GetOpKind(uint8_t c) -> BffOp
//   - InitialState(tape, head0, head1, pc)
//   - EvaluateOne(tape, head0, head1, pc) -> bool (true if the op was a command)
//   - Evaluate(tape, stepcount) -> uint64_t (instruction count)

#include "sim_kernels.h"

// ---------------------------------------------------------------------------
// BFF opcode enum (shared across all BFF variants)

enum BffOp {
  kLoopStart,
  kLoopEnd,
  kPlus,
  kMinus,
  kCopy01,
  kCopy10,
  kDec0,
  kInc0,
  kDec1,
  kInc1,
  kNull,
  kNoop,
};

// ---------------------------------------------------------------------------
// Standard BFF EvaluateOne — used by most BFF variants.
//
// Returns true if the byte at `pc` was an active command (not a noop/comment).
// Modifies head0, head1, pc in-place.  Caller must wrap head positions with
// `& (2*kSingleTapeSize - 1)` before each step.

template<typename LANG>
inline bool bff_evaluate_one(thread uint8_t* tape, thread int& head0,
                              thread int& head1, thread int& pc) {
  uint8_t cmd = tape[pc];
  switch (LANG::GetOpKind(cmd)) {
    case kDec0:  head0--;                             break;
    case kInc0:  head0++;                             break;
    case kDec1:  head1--;                             break;
    case kInc1:  head1++;                             break;
    case kPlus:  tape[head0]++;                       break;
    case kMinus: tape[head0]--;                       break;
    case kCopy01: tape[head1] = tape[head0];          break;
    case kCopy10: tape[head0] = tape[head1];          break;
    case kLoopStart:
      if (LANG::GetOpKind(tape[head0]) == kNull) {
        int scanclosed = 1;
        pc++;
        for (; pc < 2 * kSingleTapeSize && scanclosed > 0; pc++) {
          if (LANG::GetOpKind(tape[pc]) == kLoopEnd)   scanclosed--;
          if (LANG::GetOpKind(tape[pc]) == kLoopStart) scanclosed++;
        }
        pc--;
        if (scanclosed != 0) pc = 2 * kSingleTapeSize;
      }
      break;
    case kLoopEnd:
      if (LANG::GetOpKind(tape[head0]) != kNull) {
        int scanopen = 1;
        pc--;
        for (; pc >= 0 && scanopen > 0; pc--) {
          if (LANG::GetOpKind(tape[pc]) == kLoopEnd)   scanopen++;
          if (LANG::GetOpKind(tape[pc]) == kLoopStart) scanopen--;
        }
        pc++;
        if (scanopen != 0) pc = -1;
      }
      break;
    default:
      return false;
  }
  return true;
}

// Generic BFF Evaluate loop — used by variants that use bff_evaluate_one.
template<typename LANG>
inline uint64_t bff_evaluate(thread uint8_t* tape, uint64_t stepcount) {
  int pos, head0, head1;
  LANG::InitialState(tape, head0, head1, pos);

  uint64_t i     = 0;
  uint64_t nskip = 0;
  for (; i < stepcount; i++) {
    head0 &= (2 * kSingleTapeSize - 1);
    head1 &= (2 * kSingleTapeSize - 1);
    if (!bff_evaluate_one<LANG>(tape, head0, head1, pos)) nskip++;
    if (pos < 0) { i++; break; }
    pos++;
    if (pos >= 2 * kSingleTapeSize) { i++; break; }
  }
  return i - nskip;
}

// ---------------------------------------------------------------------------
// ============================================================
// Language: bff
//   Standard BFF with head pointers (BFF_HEADS).
//   Opcode map: '[' ']' '+' '-' '.' ',' '<' '>' '{' '}'
// ============================================================

struct BffLang {
  static BffOp GetOpKind(uint8_t c) {
    switch (c) {
      case '[': return kLoopStart;
      case ']': return kLoopEnd;
      case '+': return kPlus;
      case '-': return kMinus;
      case '.': return kCopy01;
      case ',': return kCopy10;
      case '<': return kDec0;
      case '>': return kInc0;
      case '{': return kDec1;
      case '}': return kInc1;
      case 0:   return kNull;
      default:  return kNoop;
    }
  }
  static void InitialState(thread const uint8_t* tape,
                            thread int& head0, thread int& head1,
                            thread int& pc) {
    // BFF_HEADS: first two tape bytes encode initial head positions.
    head0 = tape[0] % (2 * kSingleTapeSize);
    head1 = tape[1] % (2 * kSingleTapeSize);
    pc    = 2;
  }
  static uint64_t Evaluate(thread uint8_t* tape, uint64_t stepcount) {
    return bff_evaluate<BffLang>(tape, stepcount);
  }
};

// ============================================================
// Language: bff_noheads
//   Standard BFF without head pointers (no BFF_HEADS).
//   Head positions start at 2*kSingleTapeSize (undefined/wrapping behavior
//   until explicitly moved).
// ============================================================

struct BffNoheadsLang {
  static BffOp GetOpKind(uint8_t c) { return BffLang::GetOpKind(c); }
  static void InitialState(thread const uint8_t*,
                            thread int& head0, thread int& head1,
                            thread int& pc) {
    head0 = 2 * kSingleTapeSize;
    head1 = 2 * kSingleTapeSize;
    pc    = 0;
  }
  static uint64_t Evaluate(thread uint8_t* tape, uint64_t stepcount) {
    return bff_evaluate<BffNoheadsLang>(tape, stepcount);
  }
};

// ============================================================
// Language: bff_perm
//   BFF_HEADS with a different opcode-to-byte mapping.
// ============================================================

struct BffPermLang {
  static BffOp GetOpKind(uint8_t c) {
    switch (c) {
      case 0:  return kNull;
      case 9:  return kLoopStart;
      case 10: return kLoopEnd;
      case 5:  return kPlus;
      case 6:  return kMinus;
      case 7:  return kCopy01;
      case 8:  return kCopy10;
      case 1:  return kDec0;
      case 2:  return kInc0;
      case 3:  return kDec1;
      case 4:  return kInc1;
      default: return kNoop;
    }
  }
  static void InitialState(thread const uint8_t* tape,
                            thread int& head0, thread int& head1,
                            thread int& pc) {
    head0 = tape[0] % (2 * kSingleTapeSize);
    head1 = tape[1] % (2 * kSingleTapeSize);
    pc    = 2;
  }
  static uint64_t Evaluate(thread uint8_t* tape, uint64_t stepcount) {
    return bff_evaluate<BffPermLang>(tape, stepcount);
  }
};

// ============================================================
// Language: bff_noheads_4bit
//   No head pointers.  Opcode is determined by (c % 16).
// ============================================================

struct BffNoheads4bitLang {
  static BffOp GetOpKind(uint8_t c) {
    switch (((int)c + 256) % 16) {
      case 6:  return kLoopStart;
      case 7:  return kLoopEnd;
      case 8:  return kPlus;
      case 9:  return kMinus;
      case 10: return kCopy01;
      case 11: return kCopy10;
      case 12: return kDec0;
      case 13: return kInc0;
      case 14: return kDec1;
      case 15: return kInc1;
      case 0:  return kNull;
      default: return kNoop;
    }
  }
  static void InitialState(thread const uint8_t*,
                            thread int& head0, thread int& head1,
                            thread int& pc) {
    head0 = 2 * kSingleTapeSize;
    head1 = 2 * kSingleTapeSize;
    pc    = 0;
  }
  static uint64_t Evaluate(thread uint8_t* tape, uint64_t stepcount) {
    return bff_evaluate<BffNoheads4bitLang>(tape, stepcount);
  }
};

// ============================================================
// Language: bff_selfmove
//   BFF_HEADS with a trimmed opcode set and a custom Copy10 that auto-
//   increments head1 after copying (hence "selfmove").
//   Loops check tape[head0] != 0 (truthiness) rather than kNull.
// ============================================================

struct BffSelfmoveLang {
  static BffOp GetOpKind(uint8_t c) {
    switch (c) {
      case 0: return kInc0;  // NOTE: byte 0 → Inc0, not Null
      case 1: return kDec0;
      case 2: return kPlus;
      case 3: return kMinus;
      case 4: return kCopy10;
      case 5: return kLoopStart;
      case 6: return kLoopEnd;
      default: return kNoop;
    }
  }
  static void InitialState(thread const uint8_t* tape,
                            thread int& head0, thread int& head1,
                            thread int& pc) {
    head0 = tape[0] % (2 * kSingleTapeSize);
    head1 = tape[1] % (2 * kSingleTapeSize);
    pc    = 2;
  }
  // Custom EvaluateOne: loops are truthy (not kNull-based), and Copy10
  // auto-increments head1.
  static bool EvaluateOne(thread uint8_t* tape, thread int& head0,
                           thread int& head1, thread int& pc) {
    uint8_t cmd = tape[pc];
    switch (GetOpKind(cmd)) {
      case kDec0:  head0--;                               break;
      case kInc0:  head0++;                               break;
      case kPlus:  tape[head0]++;                         break;
      case kMinus: tape[head0]--;                         break;
      case kCopy10:
        tape[head0] = tape[head1];
        head1++;
        break;
      case kLoopStart:
        if (!tape[head0]) {
          int scanclosed = 1;
          pc++;
          for (; pc < 2 * kSingleTapeSize && scanclosed > 0; pc++) {
            if (GetOpKind(tape[pc]) == kLoopEnd)   scanclosed--;
            if (GetOpKind(tape[pc]) == kLoopStart) scanclosed++;
          }
          pc--;
          if (scanclosed != 0) pc = 2 * kSingleTapeSize;
        }
        break;
      case kLoopEnd:
        if (tape[head0]) {
          int scanopen = 1;
          pc--;
          for (; pc >= 0 && scanopen > 0; pc--) {
            if (GetOpKind(tape[pc]) == kLoopEnd)   scanopen++;
            if (GetOpKind(tape[pc]) == kLoopStart) scanopen--;
          }
          pc++;
          if (scanopen != 0) pc = -1;
        }
        break;
      default:
        return false;
    }
    return true;
  }
  static uint64_t Evaluate(thread uint8_t* tape, uint64_t stepcount) {
    int pos, head0, head1;
    InitialState(tape, head0, head1, pos);
    uint64_t i = 0, nskip = 0;
    for (; i < stepcount; i++) {
      head0 &= (2 * kSingleTapeSize - 1);
      head1 &= (2 * kSingleTapeSize - 1);
      if (!EvaluateOne(tape, head0, head1, pos)) nskip++;
      if (pos < 0) { i++; break; }
      pos++;
      if (pos >= 2 * kSingleTapeSize) { i++; break; }
    }
    return i - nskip;
  }
};

// ============================================================
// Language: bff8
//   BFF_HEADS.  Loops jump by a fixed offset of 8 instead of
//   scanning for matching brackets.
// ============================================================

struct Bff8Lang {
  static BffOp GetOpKind(uint8_t c) { return BffLang::GetOpKind(c); }
  static void InitialState(thread const uint8_t* tape,
                            thread int& head0, thread int& head1,
                            thread int& pc) {
    head0 = tape[0] % (2 * kSingleTapeSize);
    head1 = tape[1] % (2 * kSingleTapeSize);
    pc    = 2;
  }
  static bool EvaluateOne(thread uint8_t* tape, thread int& head0,
                           thread int& head1, thread int& pc) {
    uint8_t cmd = tape[pc];
    switch (GetOpKind(cmd)) {
      case kDec0:   head0--;                      break;
      case kInc0:   head0++;                      break;
      case kDec1:   head1--;                      break;
      case kInc1:   head1++;                      break;
      case kPlus:   tape[head0]++;                break;
      case kMinus:  tape[head0]--;                break;
      case kCopy01: tape[head1] = tape[head0];    break;
      case kCopy10: tape[head0] = tape[head1];    break;
      case kLoopStart:
        if (!tape[head0]) pc += 8;
        break;
      case kLoopEnd:
        if (tape[head0]) pc -= 8;
        break;
      default:
        return false;
    }
    return true;
  }
  static uint64_t Evaluate(thread uint8_t* tape, uint64_t stepcount) {
    int pos, head0, head1;
    InitialState(tape, head0, head1, pos);
    uint64_t i = 0, nskip = 0;
    for (; i < stepcount; i++) {
      head0 &= (2 * kSingleTapeSize - 1);
      head1 &= (2 * kSingleTapeSize - 1);
      if (!EvaluateOne(tape, head0, head1, pos)) nskip++;
      if (pos < 0) { i++; break; }
      pos++;
      if (pos >= 2 * kSingleTapeSize) { i++; break; }
    }
    return i - nskip;
  }
};

// ============================================================
// Language: bff8_noheads
//   No head pointers.  Loops jump by fixed offset of 8.
// ============================================================

struct Bff8NoheadsLang {
  static BffOp GetOpKind(uint8_t c) { return BffLang::GetOpKind(c); }
  static void InitialState(thread const uint8_t*,
                            thread int& head0, thread int& head1,
                            thread int& pc) {
    head0 = 2 * kSingleTapeSize;
    head1 = 2 * kSingleTapeSize;
    pc    = 0;
  }
  static bool EvaluateOne(thread uint8_t* tape, thread int& head0,
                           thread int& head1, thread int& pc) {
    return Bff8Lang::EvaluateOne(tape, head0, head1, pc);
  }
  static uint64_t Evaluate(thread uint8_t* tape, uint64_t stepcount) {
    int pos, head0, head1;
    InitialState(tape, head0, head1, pos);
    uint64_t i = 0, nskip = 0;
    for (; i < stepcount; i++) {
      head0 &= (2 * kSingleTapeSize - 1);
      head1 &= (2 * kSingleTapeSize - 1);
      if (!EvaluateOne(tape, head0, head1, pos)) nskip++;
      if (pos < 0) { i++; break; }
      pos++;
      if (pos >= 2 * kSingleTapeSize) { i++; break; }
    }
    return i - nskip;
  }
};

// ---------------------------------------------------------------------------
// Kernel entry points — one pair per language variant

#define DEFINE_BFF_KERNELS(SUFFIX, LANG)                                      \
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

DEFINE_BFF_KERNELS(bff,              BffLang)
DEFINE_BFF_KERNELS(bff_noheads,      BffNoheadsLang)
DEFINE_BFF_KERNELS(bff_perm,         BffPermLang)
DEFINE_BFF_KERNELS(bff_selfmove,     BffSelfmoveLang)
DEFINE_BFF_KERNELS(bff_noheads_4bit, BffNoheads4bitLang)
DEFINE_BFF_KERNELS(bff8,             Bff8Lang)
DEFINE_BFF_KERNELS(bff8_noheads,     Bff8NoheadsLang)
