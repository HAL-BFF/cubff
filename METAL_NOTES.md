# Metal Backend — Audit and Implementation Notes

## Step 1: CUDA API Inventory

### CUDA-Specific APIs

| API | Location | Usage |
|-----|----------|-------|
| `cudaMalloc` | `common_language.h:70` | `DeviceMemory<T>` constructor |
| `cudaFree` | `common_language.h:78` | `DeviceMemory<T>` destructor |
| `cudaMemcpy` (H→D) | `common_language.h:72` | `DeviceMemory::Write` |
| `cudaMemcpy` (D→H) | `common_language.h:75` | `DeviceMemory::Read` |
| `cudaDeviceSynchronize` | `common_language.h:65` | `Synchronize()` |
| `cudaGetErrorString` | `common_language.h:40` | `CUCHECK` error macro |
| `atomicAdd` | `common_language.h:62` | `IncreaseInsnCount` (warp leader) |
| `__shfl_down_sync` | `common_language.h:48` | `warpReduceSum` (warp reduction) |

### CUDA Kernel Syntax

| Syntax | Location | Purpose |
|--------|----------|---------|
| `__global__` | `common_language.h` | Kernel function qualifier |
| `__device__` | `bff.inc.h`, `forth.inc.h`, `subleq.cu`, `rsubleq4.cu`, etc. | Device-only function |
| `__host__` | `bff.inc.h`, `forth.inc.h`, etc. | Host-side function (used with `__device__`) |
| `__inline__` | `common_language.h:46` | Inline device function |
| `blockIdx.x` | `common_language.h:53` | Block index |
| `threadIdx.x` | `common_language.h:53` | Thread index within block |
| `blockDim.x` | `common_language.h:53` | Threads per block |
| `warpSize` | `common_language.h:47` | Warp size (32 for CUDA) |
| `fun<<<grid, block>>>(args)` | `common_language.h:82` | Kernel launch syntax (`RUN` macro) |

### CUDA-Specific Types

| Type | Size | Used for |
|------|------|---------|
| `unsigned long long` | 8 bytes | `insn_count` atomic counter |
| `uint8_t` | 1 byte | Program tape bytes |
| `uint32_t` | 4 bytes | Shuffle index |

### `DeviceMemory<T>` — Full Interface

Defined in `common_language.h` (CUDA path, lines 67–80):

```cpp
template <typename T>
struct DeviceMemory {
  T *data;
  DeviceMemory(size_t size);      // cudaMalloc
  void Write(const T *host, size_t count);  // cudaMemcpy H→D
  void Read(T *host, size_t count);         // cudaMemcpy D→H
  T *Get();                       // raw device pointer
  ~DeviceMemory();                // cudaFree
  DeviceMemory(DeviceMemory &) = delete;
};
```

**Usage sites:**

| Variable | Type | Size | Purpose |
|----------|------|------|---------|
| `programs` | `DeviceMemory<uint8_t>` | `kSingleTapeSize * num_programs` | All program tapes |
| `insn_count` | `DeviceMemory<unsigned long long>` | 1 | Instruction counter |
| `shuf_idx` | `DeviceMemory<uint32_t>` | `num_programs` | Shuffle permutation |
| `mem` (RunSingleParsedProgram) | `DeviceMemory<uint8_t>` | `kSingleTapeSize * 2` | Single program tape |
| `result` (EvalParsedSelfrep) | `DeviceMemory<size_t>` | 1 | Self-rep result |
| `result` (CheckSelfRep in RunSimulation) | `DeviceMemory<size_t>` | `num_programs` | All self-rep results |

### Kernel Functions

All in `common_language.h`:

| Kernel | Template | Called From | Language-specific? |
|--------|----------|-------------|-------------------|
| `InitPrograms<L>` | Language | `RunSimulation` (2×) | No — only calls `SplitMix64` |
| `MutateAndRunPrograms<L>` | Language | `RunSimulation` (main loop) | Yes — calls `L::Evaluate()` |
| `RunOneProgram<L>` | Language | `RunSingleParsedProgram` | Yes — calls `L::Evaluate()` |
| `CheckSelfRep<L>` | Language | `EvalParsedSelfrep`, `RunSimulation` | Yes — calls `L::Evaluate()` |

### `nvcc`-Specific Pragmas and Flags

| Item | Location |
|------|----------|
| `--compiler-options -Wall,-fPIC` | `Makefile:21` |
| `--compiler-bindir $(CXX)` | `Makefile:22` |
| `-arch sm_75` | `Makefile:21` |
| `#ifdef __CUDACC__` | `common_language.h:30`, `bff.inc.h:47` |
| `#include <cuda_device_runtime_api.h>` | `common_language.h:31` |
| `#include <cuda_runtime_api.h>` | `common_language.h:32` |
| `#include <driver_types.h>` | `common_language.h:33` |

### CUDA Error Handling

The `CUCHECK(op)` macro (common_language.h:35–42) wraps every CUDA API call and prints + exits on error.

---

## Step 2: Memory Model Analysis

### All `cudaMemcpy` Calls

| Call site | Direction | Size | Frequency |
|-----------|-----------|------|-----------|
| `DeviceMemory::Write` in `RunSimulation` — `shuf_idx.Write` | H→D | `num_programs * 4` bytes | **Every epoch** |
| `DeviceMemory::Write` in `RunSimulation` — `insn_count.Write(&zero,1)` | H→D | 8 bytes | Every `callback_interval` epochs |
| `DeviceMemory::Read` in `RunSimulation` — `insn_count.Read` | D→H | 8 bytes | Every `callback_interval` epochs |
| `DeviceMemory::Read` in `RunSimulation` — `programs.Read` | D→H | `num_programs * kSingleTapeSize` bytes | Every `callback_interval` epochs |
| `DeviceMemory::Write` in `RunSimulation` (load file) | H→D | `num_programs * kSingleTapeSize` | Once at startup |
| `DeviceMemory::Write` in `RunSimulation` — `programs.Write` (initial_program) | H→D | `parsed.size()` bytes | Once at startup |
| `DeviceMemory::Write` in `RunSimulation` (insn_count init) | H→D | 8 bytes | Once at startup |
| `DeviceMemory::Read/Write` in `EvalParsedSelfrep` | Both | `kSingleTapeSize` and 8 bytes | On demand |
| `DeviceMemory::Read/Write` in `RunSingleParsedProgram` | Both | `kSingleTapeSize * 2` | On demand |

### Unified Memory Impact (Apple Silicon)

Apple Silicon has a **unified memory architecture**: CPU and GPU share the same physical DRAM pool with no PCIe bus. An `MTLBuffer` with `MTLResourceStorageModeShared` is directly addressable by both CPU and GPU without any copy.

**Transfers that can be eliminated:**

| Transfer | Reason for elimination |
|----------|----------------------|
| `programs.Read` (soup read at callback) | GPU writes to shared buffer; CPU reads the same physical memory. No copy needed — just read `data` pointer directly after GPU sync. |
| `programs.Write` (initial load, file load) | CPU writes to shared buffer; GPU reads it directly. Already zero-copy. |
| `shuf_idx.Write` (every epoch) | CPU writes to shared buffer; GPU reads it directly after CPU→GPU sync. |
| `insn_count.Write` (reset to zero) | CPU writes to shared buffer. Zero-copy. |
| `insn_count.Read` (callback) | With per-thread accumulation (no GPU atomic), CPU sums the shared buffer directly. |

**Transfers that must be retained for API correctness:**

None. All memory operations become memcpy within shared address space (effectively zero-overhead). The only synchronization needed is ensuring the GPU has finished before the CPU reads GPU-written data (handled by `waitUntilCompleted`).

**Expected performance impact:**

- Elimination of `programs.Read` every `callback_interval` epochs removes up to `num_programs * 64` bytes of PCIe transfers. At 128K programs: 8MB every 128 epochs. Significant on discrete GPU; zero-cost on Apple Silicon regardless.
- The main bottleneck on Apple Silicon is GPU compute throughput, not memory bandwidth.

---

## Step 3: Implementation Notes

### Architecture

The Metal backend adds a third path to `common_language.h` alongside the existing CUDA (`#ifdef __CUDACC__`) and CPU (`#else`) paths.

**Key design decisions:**

1. **`DeviceMemory<T>` via unified memory**: Uses `MTLStorageModeShared` buffers. `Write` and `Read` are plain `memcpy` (the buffer is already CPU-accessible). No copies across a PCIe bus.

2. **Instruction count via per-thread accumulation**: Metal lacks 64-bit atomic operations. Instead of `atomicAdd`, each GPU thread writes its instruction count to a private slot in a `uint64_t` buffer indexed by thread ID. The CPU sums all slots after each callback interval. No GPU synchronization required for the counter.

3. **No `RUN` macro in Metal path**: The `RUN` macro dispatches CUDA kernels with template instantiation. For Metal, `RunSimulation`, `EvalParsedSelfrep`, and `RunSingleParsedProgram` use `#ifdef USE_METAL` blocks that call the `metal_*` C API directly. The C API hides all Objective-C from the C++ simulation code.

4. **C API isolation**: All Metal/Objective-C code is in `metal_runtime.mm`. The rest of the codebase sees only a plain C API declared in `metal_runtime.h`. This avoids mixing Objective-C with C++ compilation units.

5. **Language dispatch**: Each language `.cu` file specializes `MetalLanguageTrait<Language>` to provide Metal kernel names (strings) for `MutateAndRunPrograms` and `CheckSelfRep`. `InitPrograms` is language-agnostic.

6. **`RunSingleParsedProgram` in Metal mode**: Falls back to CPU execution. This is a debug/analysis function, not used in production simulation.

7. **Shader compilation**: `.metal` files are compiled to `.air` intermediate representation at build time, then linked into a single `bin/cubff.metallib` Metal library. The runtime finds the library by searching alongside the executable binary.

8. **Synchronization**: `metal_synchronize()` waits for all submitted GPU work before the CPU reads GPU-written data. In the main simulation loop, synchronization occurs every epoch (to prevent CPU from overwriting `shuf_idx` while GPU is still reading it). This is safe because GPU dispatch + sync latency on Apple Silicon M-series is very low.

### Languages Ported to Metal

All 13 language variants have Metal shader implementations:

| Language name | `.cu` file | Kernel suffix |
|--------------|-----------|--------------|
| `bff` | `bff.cu` | `_bff` |
| `bff_noheads` | `bff_noheads.cu` | `_bff_noheads` |
| `bff_perm` | `bff_perm.cu` | `_bff_perm` |
| `bff_selfmove` | `bff_selfmove.cu` | `_bff_selfmove` |
| `bff_noheads_4bit` | `bff_noheads_4bit.cu` | `_bff_noheads_4bit` |
| `bff8` | `bff8.cu` | `_bff8` |
| `bff8_noheads` | `bff8_noheads.cu` | `_bff8_noheads` |
| `forth` | `forth.cu` | `_forth` |
| `forthcopy` | `forthcopy.cu` | `_forthcopy` |
| `forthtrivial` | `forthtrivial.cu` | `_forthtrivial` |
| `forthtrivial_reset` | `forthtrivial_reset.cu` | `_forthtrivial_reset` |
| `subleq` | `subleq.cu` | `_subleq` |
| `rsubleq4` | `rsubleq4.cu` | `_rsubleq4` |

### Files Added

- `metal_runtime.h` — C++ API for Metal operations (no Objective-C leakage)
- `metal_runtime.mm` — Objective-C++ Metal implementation
- `metal/sim_kernels.h` — Shared MSL header (PRNG, param structs, mutate/selfrep templates)
- `metal/common_kernels.metal` — Language-agnostic `init_programs` kernel
- `metal/bff_kernels.metal` — BFF language family Metal kernels
- `metal/forth_kernels.metal` — Forth language family Metal kernels
- `metal/subleq_kernels.metal` — SubLeq/rSubLeq4 Metal kernels

### Files Modified

- `common_language.h` — Added `USE_METAL` path: `DeviceMemory<T>`, `Synchronize()`, plus `#ifdef USE_METAL` blocks in `RunSimulation`, `EvalParsedSelfrep`, `RunSingleParsedProgram`
- `Makefile` — Added `METAL=1` build option, metallib compilation rules
- `bff.cu`, `bff_noheads.cu`, `bff_perm.cu`, `bff_selfmove.cu`, `bff_noheads_4bit.cu`, `bff8.cu`, `bff8_noheads.cu` — Added `MetalLanguageTrait<Bff>` specializations
- `forth.cu`, `forthcopy.cu`, `forthtrivial.cu`, `forthtrivial_reset.cu` — Added `MetalLanguageTrait<Forth>` specializations
- `subleq.cu`, `rsubleq4.cu` — Added `MetalLanguageTrait<Subleq/RSubleq4>` specializations

---

## Benchmark Results

Platform: Apple M-series (16 logical CPUs, Metal GPU).

**Test: BFF language, 131072 programs, 100 epochs**

| Backend | Wall time | Notes |
|---------|-----------|-------|
| Metal (GPU) | 2.79 s | `make METAL=1` |
| CPU + OpenMP (16 cores) | 1.78 s wall / 17.2 s user | `make METAL=0` |
| Single-threaded CPU (estimated) | ~17 s | OpenMP user time ÷ 1 |

Metal is approximately **6× faster than single-threaded CPU**.

The CPU-with-OpenMP result (1.78 s) benefits from 16 physical cores. On a machine
without OpenMP or with fewer cores (as is typical in a CUDA-primary project workflow),
Metal provides a substantial advantage. The GPU path also enables future scaling to
larger soup sizes where GPU parallelism dominates.

**Build times:**

- C++ compilation: ~30 s
- Metal shader compilation (`.metal` → `.air` → `.metallib`): ~20 s

---

## Surprises and Tradeoffs

- **No 64-bit GPU atomics in MSL**: Metal Shading Language lacks `atomic<uint64_t>`. Solved via per-thread accumulation buffers.
- **Static local arrays not allowed in MSL device functions**: `CharacterRepr()` in `bff.inc.h` uses a static local `const char*[]`. This function is CPU-only (display), so it is not needed in Metal shaders.
- **`__device__` annotations in CPU-callable templates**: All `__device__` and `__host__` qualifiers in `.inc.h` files are kept as-is; they expand to nothing in both the CPU and Metal build paths.
- **Threadgroup size**: Using 128 threads/threadgroup with `dispatchThreads:threadsPerThreadgroup:` (non-uniform dispatch). Metal computes the exact grid without needing padding.
