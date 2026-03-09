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

// Metal compute kernels for the Forth language family.
//
// Variants: forth, forthcopy, forthtrivial, forthtrivial_reset

#include "sim_kernels.h"

// ---------------------------------------------------------------------------
// Forth opcode enum

enum ForthOp {
  kWrite0,
  kWrite1,
  kWrite,
  kRead,
  kRead0,
  kRead1,
  kDup,
  kDrop,
  kSwap,
  kIf0,
  kInc,
  kDec,
  kAdd,
  kSub,
  kCopy,
  kCopy0,
  kCopy1,
  kXor,
  kConst,
  kJmp,
  kNoop,
};

// ---------------------------------------------------------------------------
// Stack (shared by all Forth variants)

struct ForthStack {
  static constant int kStackSize = 128;
  uint8_t data[128];
  int     stackpos;
  bool    overflow;

  ForthStack() : stackpos(0), overflow(false) {}

  void Push(uint8_t val) {
    if (stackpos != kStackSize) {
      data[stackpos++] = val;
    } else {
      overflow = true;
    }
  }

  uint8_t Pop() { return stackpos == 0 ? 0 : data[--stackpos]; }
};

// ---------------------------------------------------------------------------
// Generic Forth EvaluateOne — used by all variants that don't override it.
// Inlined so each variant's kernel specialization can compile efficiently.

template<typename LANG>
inline void forth_evaluate_one(thread uint8_t* tape, thread int& pos,
                               thread uint64_t& nops,
                               thread ForthStack& stack) {
  uint8_t command = tape[pos];
  switch (LANG::GetOpKind(command)) {
    case kRead0: {
      int addr = stack.Pop() % kSingleTapeSize;
      stack.Push(tape[0 + addr]);
      break;
    }
    case kRead1: {
      int addr = stack.Pop() % kSingleTapeSize;
      stack.Push(tape[kSingleTapeSize + addr]);
      break;
    }
    case kRead: {
      int addr = stack.Pop() % (2 * kSingleTapeSize);
      stack.Push(tape[addr]);
      break;
    }
    case kWrite: {
      int val  = stack.Pop();
      int addr = stack.Pop() % (2 * kSingleTapeSize);
      tape[addr] = val;
      break;
    }
    case kWrite0: {
      int val  = stack.Pop();
      int addr = stack.Pop() % kSingleTapeSize;
      tape[0 + addr] = val;
      break;
    }
    case kWrite1: {
      int val  = stack.Pop();
      int addr = stack.Pop() % kSingleTapeSize;
      tape[kSingleTapeSize + addr] = val;
      break;
    }
    case kDup: {
      int v = stack.Pop();
      stack.Push(v);
      stack.Push(v);
      break;
    }
    case kDrop:
      stack.Pop();
      break;
    case kSwap: {
      int a = stack.Pop();
      int b = stack.Pop();
      stack.Push(a);
      stack.Push(b);
      break;
    }
    case kIf0: {
      int v = stack.Pop();
      if (v) pos++;
      stack.Push(v);
      break;
    }
    case kInc:  stack.Push(stack.Pop() + 1); break;
    case kDec:  stack.Push(stack.Pop() - 1); break;
    case kAdd: {
      int a = stack.Pop();
      int b = stack.Pop();
      stack.Push(a + b);
      break;
    }
    case kSub: {
      int a = stack.Pop();
      int b = stack.Pop();
      stack.Push(a - b);
      break;
    }
    case kConst:
      stack.Push(command & 63);
      break;
    case kCopy0: {
      int addr = stack.Pop() % kSingleTapeSize;
      tape[kSingleTapeSize + addr] = tape[0 + addr];
      break;
    }
    case kCopy1: {
      int addr = stack.Pop() % kSingleTapeSize;
      tape[0 + addr] = tape[kSingleTapeSize + addr];
      break;
    }
    case kCopy: {
      int to   = stack.Pop() % (2 * kSingleTapeSize);
      int from = stack.Pop() % (2 * kSingleTapeSize);
      tape[to] = tape[from];
      break;
    }
    case kXor:
      stack.Push(stack.Pop() ^ 64);
      break;
    case kJmp: {
      int abs = (command & 63) + 1;
      int jmp = (command & 64) ? -abs : abs;
      pos += jmp;
      pos--;  // will be incremented by caller
      break;
    }
    default:
      nops++;
      break;
  }
  pos++;
}

template<typename LANG>
inline uint64_t forth_evaluate(thread uint8_t* tape, uint64_t stepcount) {
  ForthStack stack;
  int pos = 0;
  uint64_t i = 0, nops = 0;
  for (; i < stepcount; i++) {
    forth_evaluate_one<LANG>(tape, pos, nops, stack);
    if (pos >= 2 * kSingleTapeSize || pos < 0 || stack.overflow) {
      i++;
      break;
    }
  }
  return i - nops;
}

// ---------------------------------------------------------------------------
// ============================================================
// Language: forth
//   Standard opcodes from forth.inc.h (0x0–0xB + const/jmp).
// ============================================================

struct ForthLang {
  static ForthOp GetOpKind(uint8_t c) {
    switch (c) {
      case 0x0: return kRead0;
      case 0x1: return kRead1;
      case 0x2: return kWrite0;
      case 0x3: return kWrite1;
      case 0x4: return kDup;
      case 0x5: return kDrop;
      case 0x6: return kSwap;
      case 0x7: return kIf0;
      case 0x8: return kInc;
      case 0x9: return kDec;
      case 0xA: return kAdd;
      case 0xB: return kSub;
      default:
        return (c >= 128 ? kJmp : (c >= 64 ? kConst : kNoop));
    }
  }
  static uint64_t Evaluate(thread uint8_t* tape, uint64_t stepcount) {
    return forth_evaluate<ForthLang>(tape, stepcount);
  }
};

// ============================================================
// Language: forthcopy
//   Custom opcode map: read/write operate on the full tape;
//   adds copy, xor.
// ============================================================

struct ForthcopyLang {
  static ForthOp GetOpKind(uint8_t c) {
    switch (c) {
      case 0x0: return kRead;
      case 0x1: return kWrite;
      case 0x2: return kCopy;
      case 0x3: return kXor;
      case 0x4: return kDup;
      case 0x5: return kDrop;
      case 0x6: return kSwap;
      case 0x7: return kIf0;
      case 0x8: return kInc;
      case 0x9: return kDec;
      case 0xA: return kAdd;
      case 0xB: return kSub;
      default:
        return (c >= 128 ? kJmp : (c >= 64 ? kConst : kNoop));
    }
  }
  static uint64_t Evaluate(thread uint8_t* tape, uint64_t stepcount) {
    return forth_evaluate<ForthcopyLang>(tape, stepcount);
  }
};

// ============================================================
// Language: forthtrivial
//   Same opcode map as forth but adds 0xC → kCopy0, 0xD → kCopy1.
// ============================================================

struct ForthtrivialLang {
  static ForthOp GetOpKind(uint8_t c) {
    switch (c) {
      case 0x0: return kRead0;
      case 0x1: return kRead1;
      case 0x2: return kWrite0;
      case 0x3: return kWrite1;
      case 0x4: return kDup;
      case 0x5: return kDrop;
      case 0x6: return kSwap;
      case 0x7: return kIf0;
      case 0x8: return kInc;
      case 0x9: return kDec;
      case 0xA: return kAdd;
      case 0xB: return kSub;
      case 0xC: return kCopy0;
      case 0xD: return kCopy1;
      default:
        return (c >= 128 ? kJmp : (c >= 64 ? kConst : kNoop));
    }
  }
  static uint64_t Evaluate(thread uint8_t* tape, uint64_t stepcount) {
    return forth_evaluate<ForthtrivialLang>(tape, stepcount);
  }
};

// ============================================================
// Language: forthtrivial_reset
//   Same opcode map as forthtrivial, but writes to tape[1][addr] trigger a
//   hash-seeded reset of tape[0] if tape[1][addr] == 0x13.
// ============================================================

// Reset function: re-randomizes tape[0] using a hash of the full tape.
inline void forth_reset(thread uint8_t* tape, int pos) {
  uint64_t seed = 0;
  for (int i = 0; i < 2 * kSingleTapeSize; i++) {
    seed = SplitMix64(seed ^ tape[i]);
  }
  for (int i = 0; i < kSingleTapeSize; i++) {
    tape[i] = (uint8_t)(
        SplitMix64(seed * (uint64_t)kSingleTapeSize * kSingleTapeSize
                   + (uint64_t)pos * kSingleTapeSize + i)
        % 256);
  }
}

inline bool forth_reset_if(thread const uint8_t* tape, int addr) {
  return tape[kSingleTapeSize + addr] == 0x13;
}

struct ForthtrivialResetLang {
  static ForthOp GetOpKind(uint8_t c) {
    return ForthtrivialLang::GetOpKind(c);
  }

  // Custom EvaluateOne: kWrite1 and kCopy0 check for the reset trigger.
  static void EvaluateOne(thread uint8_t* tape, thread int& pos,
                           thread uint64_t& nops, thread ForthStack& stack) {
    uint8_t command = tape[pos];
    switch (GetOpKind(command)) {
      case kRead0: {
        int addr = stack.Pop() % kSingleTapeSize;
        stack.Push(tape[0 + addr]);
        break;
      }
      case kRead1: {
        int addr = stack.Pop() % kSingleTapeSize;
        stack.Push(tape[kSingleTapeSize + addr]);
        break;
      }
      case kRead: {
        int addr = stack.Pop() % (2 * kSingleTapeSize);
        stack.Push(tape[addr]);
        break;
      }
      case kWrite: {
        int val  = stack.Pop();
        int addr = stack.Pop() % (2 * kSingleTapeSize);
        tape[addr] = val;
        break;
      }
      case kWrite0: {
        int val  = stack.Pop();
        int addr = stack.Pop() % kSingleTapeSize;
        tape[0 + addr] = val;
        break;
      }
      case kWrite1: {
        int val  = stack.Pop();
        int addr = stack.Pop() % kSingleTapeSize;
        if (forth_reset_if(tape, addr)) {
          forth_reset(tape, pos);
        } else {
          tape[kSingleTapeSize + addr] = val;
        }
        break;
      }
      case kDup: {
        int v = stack.Pop();
        stack.Push(v);
        stack.Push(v);
        break;
      }
      case kDrop: stack.Pop(); break;
      case kSwap: {
        int a = stack.Pop();
        int b = stack.Pop();
        stack.Push(a);
        stack.Push(b);
        break;
      }
      case kIf0: {
        int v = stack.Pop();
        if (v) pos++;
        stack.Push(v);
        break;
      }
      case kInc:  stack.Push(stack.Pop() + 1); break;
      case kDec:  stack.Push(stack.Pop() - 1); break;
      case kAdd: {
        int a = stack.Pop();
        int b = stack.Pop();
        stack.Push(a + b);
        break;
      }
      case kSub: {
        int a = stack.Pop();
        int b = stack.Pop();
        stack.Push(a - b);
        break;
      }
      case kConst: stack.Push(command & 63); break;
      case kCopy0: {
        int addr = stack.Pop() % kSingleTapeSize;
        if (forth_reset_if(tape, addr)) {
          forth_reset(tape, pos);
        } else {
          tape[kSingleTapeSize + addr] = tape[0 + addr];
        }
        break;
      }
      case kCopy1: {
        int addr = stack.Pop() % kSingleTapeSize;
        tape[0 + addr] = tape[kSingleTapeSize + addr];
        break;
      }
      case kCopy: {
        int to   = stack.Pop() % (2 * kSingleTapeSize);
        int from = stack.Pop() % (2 * kSingleTapeSize);
        tape[to] = tape[from];
        break;
      }
      case kXor: stack.Push(stack.Pop() ^ 64); break;
      case kJmp: {
        int abs = (command & 63) + 1;
        int jmp = (command & 64) ? -abs : abs;
        pos += jmp;
        pos--;  // compensated by pos++ below
        break;
      }
      default:
        nops++;
        break;
    }
    pos++;
  }

  static uint64_t Evaluate(thread uint8_t* tape, uint64_t stepcount) {
    ForthStack stack;
    int pos = 0;
    uint64_t i = 0, nops = 0;
    for (; i < stepcount; i++) {
      EvaluateOne(tape, pos, nops, stack);
      if (pos >= 2 * kSingleTapeSize || pos < 0 || stack.overflow) {
        i++;
        break;
      }
    }
    return i - nops;
  }
};

// ---------------------------------------------------------------------------
// Kernel entry points

#define DEFINE_FORTH_KERNELS(SUFFIX, LANG)                                    \
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

DEFINE_FORTH_KERNELS(forth,              ForthLang)
DEFINE_FORTH_KERNELS(forthcopy,          ForthcopyLang)
DEFINE_FORTH_KERNELS(forthtrivial,       ForthtrivialLang)
DEFINE_FORTH_KERNELS(forthtrivial_reset, ForthtrivialResetLang)
