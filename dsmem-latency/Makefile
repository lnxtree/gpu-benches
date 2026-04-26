# dsmem-latency/Makefile
# Requires: CUDA 12.0+, Hopper GPU (H100 / H200)
#
# Distributed SMEM (cluster group) requires sm_90a.
# The 'a' suffix enables the full Hopper ISA including:
#   - ld.shared::cluster / st.shared::cluster  (used here for pointer chasing)
#   - TMA (cp.async.bulk.tensor)
#   - WGMMA
#
# Usage:
#   make                   # build
#   ./cuda-dsmem-lat       # run
#   make clean             # remove binary

NVCC      := $(shell which nvcc)
CUDA_HOME := $(shell dirname $(shell dirname $(NVCC)))

# Auto-detect compute capability from nvidia-smi, append 'a' for Hopper+
SM_RAW := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader \
           2>/dev/null | head -1 | tr -d '.')
SM     := $(shell sm=$(SM_RAW); [ "$${sm}" -ge 90 ] 2>/dev/null \
           && echo "$${sm}a" || echo "$${sm}")

NVCCFLAGS := \
    -std=c++20 \
    -O3 \
    -arch=sm_$(SM) \
    --compiler-options="-O2 -pipe -Wall" \
    -I$(CUDA_HOME)/include \
    -L$(CUDA_HOME)/lib64 \
    -Wno-deprecated-gpu-targets

.PHONY: all clean

all: cuda-dsmem-lat

cuda-dsmem-lat: main.cu ../MeasurementSeries.hpp ../gpu-error.h
	@echo "Building for sm_$(SM) ..."
	$(NVCC) $(NVCCFLAGS) -o $@ main.cu

clean:
	rm -f cuda-dsmem-lat
