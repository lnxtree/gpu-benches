// dsmem-bandwidth/main.cu
// Distributed Shared Memory (DSMEM) Cross-SM Bandwidth Benchmark
//
// On Hopper, a Thread Block Cluster groups multiple blocks into one CGA
// (Cooperative Group Array). Blocks within a cluster are co-scheduled on
// SMs in the same GPC and can directly read/write each other's shared memory
// via `ld.shared::cluster` / `st.shared::cluster` PTX instructions.
//
// This benchmark measures the bandwidth of that cross-SM SMEM fabric.
//
// Sweeps:
//   1. Occupancy sweep  – vary threads/block, fixed SMEM=64kB
//   2. SMEM size sweep  – vary SMEM/block,   fixed threads=1024
//   3. Distance sweep   – vary neighbor distance in cluster ring, CS=4 & CS=8
//
// Requires: CUDA 12.0+, Hopper GPU (H100/H200, sm_90a)
//
// Build:
//   make cuda-dsmem-bw
//
// Run:
//   ./cuda-dsmem-bw

#include "../MeasurementSeries.hpp"
#include "../gpu-error.h"
#include <cooperative_groups.h>
#include <iomanip>
#include <iostream>
#include <vector>

namespace cg = cooperative_groups;
using namespace std;

// ─── globals ──────────────────────────────────────────────────────────────────
static double *d_sink = nullptr;   // write target – prevents dead-code elimination
static volatile bool secretlyFalse = false;

// ─── kernels ──────────────────────────────────────────────────────────────────

// Read bandwidth kernel.
//
// Each block reads the full SMEM of one remote block in the cluster (selected
// by `distance` hops in the cluster ring).
//
// Pattern: streaming reduction with 4 independent FP64 accumulators to break
// the loop-carried dependency chain and allow the hardware to keep multiple
// ld.shared::cluster requests in flight simultaneously.
template <int CLUSTER_SIZE>
__global__ void __cluster_dims__(CLUSTER_SIZE, 1, 1)
dsmem_read_kernel(double *__restrict__ sink,
                  int     smem_n,    // doubles per block in shared mem
                  int     iters,     // how many times to re-read remote SMEM
                  int     distance,  // which cluster neighbor to read from
                  bool    sf)        // secretlyFalse – keeps acc alive
{
    cg::cluster_group cluster = cg::this_cluster();
    extern __shared__ double smem[];

    const int tid         = (int)threadIdx.x;
    const int bdim        = (int)blockDim.x;
    const int my_rank     = (int)cluster.block_rank();
    const int remote_rank = (my_rank + distance) % CLUSTER_SIZE;

    // Populate local SMEM so the remote block has valid data to read
    for (int i = tid; i < smem_n; i += bdim)
        smem[i] = (double)(my_rank * 1000000 + i) * 0.001;

    cluster.sync();   // all blocks must finish init before anyone reads remotely

    // Map remote block's shared memory – generates ld.shared::cluster
    double *remote = cluster.map_shared_rank(smem, remote_rank);

    // 8 independent accumulators.
    //
    // For 64kB SMEM + 1024 threads: 8192 doubles / 1024 = 8 elements per thread.
    // The 8-wide inner loop handles all 8 in ONE step → all 8 loads are
    // simultaneously in-flight per thread with NO mid-iteration dependency.
    //
    // Previously a 4-wide loop needed 2 steps, creating a RAW chain:
    //   step1 loads → FP-add → step2 loads (step2 must wait for step1)
    // With 8-wide + 1 step, all loads are issued before any add completes.
    //
    // Cross-iteration dependency (a_k[N+1] depends on a_k[N]) is unavoidable
    // without register explosion, but the GPU hides it via warp switching.
    double a0=0,a1=0,a2=0,a3=0,a4=0,a5=0,a6=0,a7=0;

    for (int it = 0; it < iters; ++it) {
        int i = tid;
        // 8-wide unrolled – covers 8 stride-bdim elements simultaneously
        for (; i + 7 * bdim < smem_n; i += 8 * bdim) {
            a0 += remote[i            ];  a1 += remote[i +     bdim ];
            a2 += remote[i + 2 * bdim ];  a3 += remote[i + 3 * bdim ];
            a4 += remote[i + 4 * bdim ];  a5 += remote[i + 5 * bdim ];
            a6 += remote[i + 6 * bdim ];  a7 += remote[i + 7 * bdim ];
        }
        // 4-wide tail for smem sizes not divisible by 8*bdim
        for (; i + 3 * bdim < smem_n; i += 4 * bdim) {
            a0 += remote[i            ];  a1 += remote[i +     bdim ];
            a2 += remote[i + 2 * bdim ];  a3 += remote[i + 3 * bdim ];
        }
        // Scalar tail
        for (; i < smem_n; i += bdim) a0 += remote[i];
    }

    // *** Critical: wait for all cluster blocks to finish reading ***
    // Without this, a block that finishes iters slightly earlier can exit and
    // have its SMEM reclaimed by the SM while a peer block is still reading it
    // via cluster.map_shared_rank → unspecified launch failure.
    cluster.sync();

    if (sf) sink[blockIdx.x * bdim + tid] = a0+a1+a2+a3+a4+a5+a6+a7;
}

// Write bandwidth kernel.
//
// Each block writes registers → one remote block's SMEM.
// In the distance-1 ring: block B → block (B+1)'s SMEM.
// No two blocks write to the same SMEM region → no races, no inter-iteration
// sync required. Generates st.shared::cluster.
template <int CLUSTER_SIZE>
__global__ void __cluster_dims__(CLUSTER_SIZE, 1, 1)
dsmem_write_kernel(double *__restrict__ sink,
                   int     smem_n,
                   int     iters,
                   int     distance,
                   bool    sf)
{
    cg::cluster_group cluster = cg::this_cluster();
    extern __shared__ double smem[];

    const int tid         = (int)threadIdx.x;
    const int bdim        = (int)blockDim.x;
    const int my_rank     = (int)cluster.block_rank();
    const int remote_rank = (my_rank + distance) % CLUSTER_SIZE;

    for (int i = tid; i < smem_n; i += bdim)
        smem[i] = 0.0;

    cluster.sync();   // ensure all SMEM initialised before writes begin

    double *remote = cluster.map_shared_rank(smem, remote_rank);

    // Pure streaming write – each store is independent → max write throughput
    // Use (double)it to avoid int overflow in it * smem_n at large iters.
    for (int it = 0; it < iters; ++it) {
        double base = (double)it;
        for (int i = tid; i < smem_n; i += bdim)
            remote[i] = base + (double)i * 0.001;
    }

    // *** Critical: same rationale as read kernel – must sync before any block
    // exits so that peer blocks still writing to our SMEM see it as live. ***
    cluster.sync();

    // Touch own SMEM (written by neighbor) to create a visible side effect
    if (sf) sink[blockIdx.x * bdim + tid] = smem[tid % smem_n];
}

// ─── measurement helper ───────────────────────────────────────────────────────
// Returns aggregate bandwidth in GB/s (all SM pairs combined).
template <typename LaunchFn>
double measure_bw(LaunchFn launch, long long bytes_per_launch)
{
    // Clear any stale error state from a previous (failed) launch before we
    // check the result of this one.
    cudaGetLastError();

    // Warmup
    launch();
    GPU_ERROR(cudaDeviceSynchronize());

    MeasurementSeries t;
    cudaEvent_t ev0, ev1;
    GPU_ERROR(cudaEventCreate(&ev0));
    GPU_ERROR(cudaEventCreate(&ev1));

    for (int r = 0; r < 31; ++r) {
        GPU_ERROR(cudaEventRecord(ev0));
        launch();
        GPU_ERROR(cudaEventRecord(ev1));
        GPU_ERROR(cudaEventSynchronize(ev1));
        float ms = 0.f;
        GPU_ERROR(cudaEventElapsedTime(&ms, ev0, ev1));
        t.add((double)ms * 1e-3);
    }

    GPU_ERROR(cudaEventDestroy(ev0));
    GPU_ERROR(cudaEventDestroy(ev1));

    return (double)bytes_per_launch / t.median() * 1e-9;
}

// ─── per-configuration driver ─────────────────────────────────────────────────
struct DevInfo {
    int    num_sms;
    int    max_thr_per_sm;
    size_t max_smem_per_block;
};

// Template-instantiated measurement for a fixed CLUSTER_SIZE.
// Called from the dispatch switch below.
template <int CS>
void do_measure(const DevInfo &di,
                int    block_size,
                size_t smem_bytes,
                int    distance,
                int    iters)
{
    const int smem_n      = (int)(smem_bytes / sizeof(double));
    const int num_clusters = di.num_sms / CS;
    if (num_clusters == 0) return;
    const int num_blocks  = num_clusters * CS;   // == di.num_sms (1 block per SM)

    // Occupancy: 1 block per SM, block_size threads
    const double occ = (double)block_size / di.max_thr_per_sm * 100.0;

    GPU_ERROR(cudaFuncSetAttribute(dsmem_read_kernel<CS>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes));
    GPU_ERROR(cudaFuncSetAttribute(dsmem_write_kernel<CS>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes));

    // bytes transferred per kernel launch:
    // every block reads/writes smem_bytes from/to a remote block, iters times
    long long bytes = (long long)num_blocks * (long long)smem_bytes * iters;

    double read_bw = measure_bw([&]() {
        dsmem_read_kernel<CS><<<num_blocks, block_size, smem_bytes>>>(
            d_sink, smem_n, iters, distance, (bool)secretlyFalse);
    }, bytes);

    double write_bw = measure_bw([&]() {
        dsmem_write_kernel<CS><<<num_blocks, block_size, smem_bytes>>>(
            d_sink, smem_n, iters, distance, (bool)secretlyFalse);
    }, bytes);

    cout << fixed
         << setw(4)  << CS                         // cluster size
         << "  dist=" << distance
         << "  " << setw(5) << block_size          // threads/block
         << "  " << setw(8) << (long long)num_blocks * block_size  // total threads
         << "  " << setprecision(1) << setw(5) << occ << "%"
         << "  " << smem_bytes / 1024 << "kB"
         << "   |  "
         << setprecision(0)
         << setw(8) << read_bw  << "   "
         << setw(8) << write_bw
         << "\n";
    cout.flush();
}

// Dispatch on runtime cluster size → compile-time template argument
void run_config(int cs, const DevInfo &di,
                int block_size, size_t smem_bytes, int distance, int iters)
{
    if (block_size > di.max_thr_per_sm)              return;
    if (smem_bytes > di.max_smem_per_block)          return;
    // smem_n must be a multiple of block_size (clean strided access)
    if ((smem_bytes / sizeof(double)) % (size_t)block_size != 0) return;

    switch (cs) {
        case 2: do_measure<2>(di, block_size, smem_bytes, distance, iters); break;
        case 4: do_measure<4>(di, block_size, smem_bytes, distance, iters); break;
        case 8: do_measure<8>(di, block_size, smem_bytes, distance, iters); break;
        default: break;
    }
}

// ─── main ─────────────────────────────────────────────────────────────────────
int main()
{
    int dev;
    cudaDeviceProp prop;
    GPU_ERROR(cudaGetDevice(&dev));
    GPU_ERROR(cudaGetDeviceProperties(&prop, dev));

    if (prop.major < 9) {
        cerr << "ERROR: Hopper GPU (sm_90+) required for Distributed SMEM.\n"
             << "  Detected: " << prop.name
             << " (sm_" << prop.major << prop.minor << ")\n";
        return 1;
    }

    DevInfo di;
    di.num_sms            = prop.multiProcessorCount;
    di.max_thr_per_sm     = prop.maxThreadsPerMultiProcessor;
    di.max_smem_per_block = prop.sharedMemPerBlockOptin;   // Hopper: up to 228 kB

    const int    iters          = 200;   // inner iterations per kernel launch
    const size_t fixed_smem     = 64 * 1024;   // 64 kB for occupancy sweep
    const int    fixed_threads  = 1024;         // for SMEM-size sweep

    cout << "=== Distributed SMEM Cross-SM Bandwidth Benchmark ===\n"
         << "GPU : " << prop.name << "\n"
         << "SMs : " << di.num_sms << "\n"
         << "Max threads/SM  : " << di.max_thr_per_sm << "\n"
         << "Max SMEM/block  : " << di.max_smem_per_block / 1024 << " kB\n"
         << "Inner iterations: " << iters << "\n\n";

    GPU_ERROR(cudaMalloc(&d_sink,
        (size_t)di.num_sms * 1024 * sizeof(double)));
    GPU_ERROR(cudaMemset(d_sink, 0,
        (size_t)di.num_sms * 1024 * sizeof(double)));

    // ─── header helper ───────────────────────────────────────────────────────
    auto print_header = []() {
        cout << "  CS  dist  block   tot_thr    occ   smem  |"
             << "  read(GB/s)  write(GB/s)\n"
             << string(68, '-') << "\n";
    };

    // ── Sweep 1: occupancy sweep ─────────────────────────────────────────────
    // Fixed SMEM = 64 kB, sweep threads/block from 64 to 1024 in steps of 64.
    // Shows how many warps are needed to saturate the cross-SM SMEM fabric.
    cout << "── Sweep 1: occupancy sweep (SMEM = " << fixed_smem / 1024
         << " kB/block) ─────────────────\n";
    print_header();

    for (int cs : {2, 4, 8}) {
        for (int t = 64; t <= 1024; t += 64) {
            run_config(cs, di, t, fixed_smem, /*distance=*/1, iters);
        }
        cout << "\n";
    }

    // ── Sweep 2: SMEM size sweep ─────────────────────────────────────────────
    // Fixed block = 1024 threads (max occupancy), sweep SMEM per block.
    // Shows whether larger SMEM allocation changes cross-SM bandwidth
    // (e.g., reveals any traffic-shaping effects in the GPC interconnect).
    cout << "── Sweep 2: SMEM size sweep (block = " << fixed_threads
         << " threads) ─────────────────────\n";
    print_header();

    const vector<int> smem_kbs = {4, 8, 16, 32, 64, 96, 128, 160, 192, 228};
    for (int cs : {2, 4, 8}) {
        for (int kb : smem_kbs) {
            run_config(cs, di, fixed_threads, (size_t)kb * 1024,
                       /*distance=*/1, iters);
        }
        cout << "\n";
    }

    // ── Sweep 3: distance sweep ──────────────────────────────────────────────
    // For CS=4 and CS=8, vary which ring-neighbor to read from.
    // distance=1: adjacent SM
    // distance=2: skip one SM
    // distance=N/2: read from the "opposite" SM in the ring
    //
    // On Hopper, SMs within a GPC share an interconnect; topology may cause
    // bandwidth to vary with distance (e.g., if SMs are arranged in pairs).
    cout << "── Sweep 3: distance sweep (block = " << fixed_threads
         << ", SMEM = " << fixed_smem / 1024
         << " kB) ────────────────\n";
    print_header();

    for (int cs : {4, 8}) {
        for (int dist = 1; dist < cs; ++dist) {
            run_config(cs, di, fixed_threads, fixed_smem, dist, iters);
        }
        cout << "\n";
    }

    GPU_ERROR(cudaFree(d_sink));
    return 0;
}
