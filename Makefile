
MACOS := $(if $(filter Darwin,$(shell uname -s)),1)

# Default backend selection: on macOS use Metal; elsewhere use CUDA.
# Override with METAL=0 or CUDA=1 on the command line.
ifeq (${MACOS}, 1)
	# Explicit METAL=0 disables Metal even on macOS.
	ifeq (${METAL}, 0)
		CUDA ?= 0
	else
		METAL ?= 1
		CUDA   := 0
	endif
else
	METAL ?= 0
	CUDA  ?= 1
endif

PYTHON := 0

PYTHON_CONFIG ?= python3-config

# On macOS with Apple Silicon, native (arm64) Homebrew lives at /opt/homebrew.
# pkg-config may default to the Rosetta (x86_64) install at /usr/local, so we
# explicitly inject the arm64 search path before calling pkg-config.
ifeq (${MACOS}, 1)
PKGCFG := PKG_CONFIG_PATH=/opt/homebrew/lib/pkgconfig:/opt/homebrew/share/pkgconfig pkg-config
else
PKGCFG := pkg-config
endif

COMMON_FLAGS := -g -std=c++17 -O3 \
								$(shell ${PKGCFG} --cflags libbrotlienc libbrotlicommon) \
								${EXTRA_LDFLAGS}

LINK_FLAGS := $(shell ${PKGCFG} --libs libbrotlienc libbrotlicommon)


ifeq (${CUDA}, 1)
# -----------------------------------------------------------------------
# CUDA backend
# -----------------------------------------------------------------------
CUDA_PATH := $(shell which nvcc | xargs realpath | xargs dirname | xargs dirname)
FLAGS := ${COMMON_FLAGS} -arch sm_75 --compiler-options -Wall,-fPIC \
					--compiler-bindir $(shell which ${CXX}) \
					-I ${CUDA_PATH}/include -L ${CUDA_PATH}/lib
COMPILE_FLAGS := ${FLAGS}
COMPILER := nvcc

else ifeq (${METAL}, 1)
# -----------------------------------------------------------------------
# Metal backend
# -----------------------------------------------------------------------
FLAGS := ${COMMON_FLAGS} -Wall -fPIC -DUSE_METAL
# .mm files need the Metal and Foundation frameworks
METAL_LINK_FLAGS := -framework Metal -framework Foundation
LINK_FLAGS += ${METAL_LINK_FLAGS}

COMPILE_FLAGS     := ${FLAGS} -xc++
COMPILE_FLAGS_MM  := ${COMMON_FLAGS} -Wall -fPIC -DUSE_METAL
COMPILER := ${CXX}
# Allow Python extension modules to leave Python symbols unresolved at link
# time; the Python interpreter supplies them at dlopen() time.
LINK_FLAGS += -undefined dynamic_lookup

# Metal shader source files and their build products
METAL_SRCS := $(wildcard metal/*.metal)
METAL_AIRS  = $(patsubst metal/%.metal,build/%.air,$(METAL_SRCS))
METAL_LIB   = bin/cubff.metallib

else
# -----------------------------------------------------------------------
# CPU-only fallback backend
# -----------------------------------------------------------------------
FLAGS := ${COMMON_FLAGS} -Wall -fPIC

ifeq (${MACOS}, 1)
	FLAGS += -Xclang -fopenmp
	LINK_FLAGS += -lomp -undefined dynamic_lookup
else
	FLAGS += -fopenmp
endif

COMPILE_FLAGS := ${FLAGS} -xc++
COMPILER := ${CXX}
endif

# -----------------------------------------------------------------------
# Language object files (compiled from *.cu as plain C++)
# -----------------------------------------------------------------------
LANGS=$(patsubst %.cu,build/%.o,$(wildcard *.cu))

PYEXT=$(shell ${PYTHON_CONFIG} --extension-suffix)

# -----------------------------------------------------------------------
# Top-level targets
# -----------------------------------------------------------------------
ifeq (${PYTHON}, 0)
.PHONY:
ifeq (${METAL}, 1)
all: bin/main ${METAL_LIB}
else
all: bin/main
endif
else
ifeq (${METAL}, 1)
all: bin/main bin/cubff${PYEXT} ${METAL_LIB}
else
all: bin/main bin/cubff${PYEXT}
endif
endif

# -----------------------------------------------------------------------
# Binary link rules
# -----------------------------------------------------------------------
ifeq (${METAL}, 1)
bin/main: build/main.o build/common.o build/metal_runtime.o ${LANGS}
	${COMPILER} $^ ${FLAGS} ${LINK_FLAGS} -o $@
else
bin/main: build/main.o build/common.o ${LANGS}
	${COMPILER} $^ ${FLAGS} ${LINK_FLAGS} -o $@
endif

# -----------------------------------------------------------------------
# Compilation rules
# -----------------------------------------------------------------------
build/%.o: %.cc common.h common_language.h
	${COMPILER} -c ${COMPILE_FLAGS} $< -o $@

build/%.o: %.cu common.h common_language.h forth.inc.h bff.inc.h
	${COMPILER} -c ${COMPILE_FLAGS} $< -o $@

# metal_runtime.mm is compiled as Objective-C++ (clang, not nvcc).
build/metal_runtime.o: metal_runtime.mm metal_runtime.h
	${CXX} -c ${COMPILE_FLAGS_MM} -xobjective-c++ $< -o $@

ifeq (${PYTHON}, 1)
build/cubff_py.o: cubff_py.cc common.h
	${COMPILER} -c ${COMPILE_FLAGS} $< -o $@ $(shell ${PYTHON_CONFIG} --includes) $(shell python3 -m pybind11 --includes)

ifeq (${METAL}, 1)
bin/cubff${PYEXT}: build/cubff_py.o build/common.o build/metal_runtime.o ${LANGS}
	${COMPILER} -shared $^ ${FLAGS} ${LINK_FLAGS} -o $@
else
bin/cubff${PYEXT}: build/cubff_py.o build/common.o ${LANGS}
	${COMPILER} -shared $^ ${FLAGS} ${LINK_FLAGS} -o $@
endif
endif

# -----------------------------------------------------------------------
# Metal shader compilation
# -----------------------------------------------------------------------
ifeq (${METAL}, 1)

# Compile each .metal source to an .air intermediate.
# The -I flag lets the shader include metal/sim_kernels.h.
build/%.air: metal/%.metal metal/sim_kernels.h
	xcrun -sdk macosx metal -std=metal3.0 -O2 -I metal -c $< -o $@

# Link .air files into a single .metallib library placed in bin/.
${METAL_LIB}: ${METAL_AIRS}
	xcrun -sdk macosx metallib $^ -o $@

endif

# -----------------------------------------------------------------------
.PHONY:
clean:
	rm -rf bin/main bin/cubff* build/*
