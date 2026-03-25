#!/usr/bin/env python3
import argparse
import time
import csv
import os
from datetime import datetime
import numpy as np
import os
import sys

# Windows Python 3.8+ strict DLL loading fix
if os.name == 'nt':
    cuda_path = os.environ.get('CUDA_PATH')
    if cuda_path:
        bin_path = os.path.join(cuda_path, 'bin')
        if os.path.exists(bin_path):
            try:
                os.add_dll_directory(bin_path)
                print(f"DEBUG: Added {bin_path} to DLL directories.")
            except AttributeError:
                pass
    else:
        print("WARNING: CUDA_PATH environment variable not found. Is the CUDA Toolkit installed?")

# ==============================================================================
# CUDA BACKEND
# ==============================================================================
def run_cuda(n, iterations, dtype):
    try:
        import cupy as cp
    except ImportError:
        print("ERROR: CuPy is not installed. Run: pip install cupy-cuda12x (or match your CUDA version)")
        exit(1)

    dt = np.float32 if dtype == 'float32' else np.float64
    bytes_per_element = 4 if dtype == 'float32' else 8
    memory_mb = ((n * n) + 2 * n) * bytes_per_element / (1024 ** 2)

    try:
        device_name = f"CUDA GPU (Device {cp.cuda.Device(0).id})"
    except:
        device_name = "CUDA GPU"

    print("Generating matrix data on CPU and transferring to GPU...")
    try:
        # BYPASS cuRAND: Generate on CPU (NumPy), transfer to GPU (CuPy)
        A_cpu = np.random.rand(n, n).astype(dt)
        b_cpu = np.random.rand(n).astype(dt)
        A = cp.asarray(A_cpu)
        b = cp.asarray(b_cpu)
    except cp.cuda.memory.OutOfMemoryError:
        print("ERROR: Out of GPU Memory. Try a smaller matrix size (-n).")
        exit(1)
    except Exception as e:
        print(f"CUDA Error during allocation: {e}")
        exit(1)

    print("Warming up CUDA context...")
    cp.linalg.solve(A, b)
    cp.cuda.Stream.null.synchronize()

    # FLOPS for LU decomposition + solve: (2/3)*N^3 + 2*N^2
    flops_per_run = (2.0 / 3.0) * (n ** 3) + 2.0 * (n ** 2)
    times = []

    print("Running CUDA iterations...")
    for _ in range(iterations):
        # Slightly alter 'b' to prevent caching (CPU to GPU transfer)
        b_cpu = np.random.rand(n).astype(dt)
        b = cp.asarray(b_cpu)
        cp.cuda.Stream.null.synchronize()

        start_time = time.perf_counter()
        
        # cuSOLVER handles the LU decomposition and solve
        _ = cp.linalg.solve(A, b)
        
        cp.cuda.Stream.null.synchronize()
        end_time = time.perf_counter()
        
        times.append(end_time - start_time)

    return process_results(n, memory_mb, times, flops_per_run, device_name)

# ==============================================================================
# OPENCL BACKEND
# ==============================================================================
def run_opencl(n, iterations, dtype):
    try:
        import pyopencl as cl
    except ImportError:
        print("ERROR: pyopencl is not installed. Run: pip install pyopencl")
        exit(1)

    # Ensure N is a multiple of tile size (16)
    tile_size = 16
    if n % tile_size != 0:
        n = (n // tile_size) * tile_size
        print(f"Note: Adjusted Matrix Size to {n} for OpenCL tile size compatibility.")

    dt = np.float32 if dtype == 'float32' else np.float64
    bytes_per_element = 4 if dtype == 'float32' else 8
    memory_mb = (3 * (n * n)) * bytes_per_element / (1024 ** 2)

    platforms = cl.get_platforms()
    device = None
    for platform in platforms:
        devices = platform.get_devices(device_type=cl.device_type.GPU)
        if devices:
            device = devices[0]
            break
    
    if not device:
        print("WARNING: No OpenCL GPU found. Falling back to default.")
        device = None
        for platform in platforms:
            devices = platform.get_devices()
            if devices:
                device = devices[0]
                break

        if device is None:
            raise RuntimeError("No OpenCL devices found on any platform!")
        
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx)
    device_name = device.name

    print("Generating OpenCL matrix data in RAM...")
    h_A = np.random.rand(n, n).astype(dt)
    h_B = np.random.rand(n, n).astype(dt)
    h_C = np.zeros((n, n), dtype=dt)

    mf = cl.mem_flags
    try:
        d_A = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h_A)
        d_B = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h_B)
        d_C = cl.Buffer(ctx, mf.WRITE_ONLY, h_C.nbytes)
    except cl.MemoryError:
        print("ERROR: Out of GPU Memory. Try a smaller matrix size (-n).")
        exit(1)

    c_type = "float" if dtype == "float32" else "double"
    pragma = "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n" if dtype == "float64" else ""
    
    kernel_code = f"""
    {pragma}
    #define TS {tile_size}
    __kernel void gemm(__global const {c_type}* A, __global const {c_type}* B, __global {c_type}* C, const int N) {{
        int row = get_global_id(0);
        int col = get_global_id(1);
        int local_row = get_local_id(0);
        int local_col = get_local_id(1);
        
        __local {c_type} Asub[TS][TS];
        __local {c_type} Bsub[TS][TS];
        {c_type} sum = 0.0;
        int num_tiles = N / TS;
        
        for (int t = 0; t < num_tiles; t++) {{
            Asub[local_row][local_col] = A[row * N + (t * TS + local_col)];
            Bsub[local_row][local_col] = B[(t * TS + local_row) * N + col];
            barrier(CLK_LOCAL_MEM_FENCE);
            for (int k = 0; k < TS; k++) {{
                sum += Asub[local_row][k] * Bsub[k][local_col];
            }}
            barrier(CLK_LOCAL_MEM_FENCE);
        }}
        C[row * N + col] = sum;
    }}
    """

    print("Compiling OpenCL kernel & Warming up...")
    prg = cl.Program(ctx, kernel_code).build()
    gemm = prg.gemm

    gemm(queue, (n, n), (tile_size, tile_size), d_A, d_B, d_C, np.int32(n))
    queue.finish()

    # FLOPS for matrix multiplication: 2 * N^3
    flops_per_run = 2.0 * (n ** 3)
    times = []

    print("Running OpenCL iterations...")
    for _ in range(iterations):
        start_time = time.perf_counter()
        gemm(queue, (n, n), (tile_size, tile_size), d_A, d_B, d_C, np.int32(n))
        queue.finish()
        end_time = time.perf_counter()
        times.append(end_time - start_time)

    return process_results(n, memory_mb, times, flops_per_run, device_name)

# ==============================================================================
# HELPER & MAIN
# ==============================================================================
def process_results(n, memory_mb, times, flops_per_run, device_name):
    avg_time = sum(times) / len(times)
    best_time = min(times)
    avg_tflops = (flops_per_run / avg_time) / 1e12
    peak_tflops = (flops_per_run / best_time) / 1e12
    
    return {
        'actual_n': n,
        'device_name': device_name,
        'memory_mb': memory_mb,
        'avg_time': avg_time,
        'best_time': best_time,
        'avg_tflops': avg_tflops,
        'peak_tflops': peak_tflops
    }

def main():
    parser = argparse.ArgumentParser(description="Unified GPU Benchmark (CUDA & OpenCL)")
    parser.add_argument("-b", "--backend", choices=["cuda", "opencl"], required=True, 
                        help="Choose the compute backend to run.")
    parser.add_argument("-n", "--size", type=int, default=8192, 
                        help="Matrix size N. Default: 8192")
    parser.add_argument("-i", "--iterations", type=int, default=10, 
                        help="Number of iterations. Default: 10")
    parser.add_argument("-d", "--dtype", choices=["float32", "float64"], default="float32", 
                        help="Data precision. Default: float32")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Path to CSV file to append results.")
    
    args = parser.parse_args()

    print("==================================================")
    print(f"      GPU COMPUTE BENCHMARK ({args.backend.upper()})")
    print("==================================================")
    print(f"Target Dimension (N) : {args.size}")
    print(f"Data Type            : {args.dtype}")
    print(f"Iterations           : {args.iterations}")
    print("--------------------------------------------------")

    # Route to appropriate backend
    if args.backend == "cuda":
        res = run_cuda(args.size, args.iterations, args.dtype)
    else:
        res = run_opencl(args.size, args.iterations, args.dtype)

    print("--------------------------------------------------")
    print(f"Device               : {res['device_name']}")
    print(f"Actual Matrix Size   : {res['actual_n']}")
    print(f"Est. VRAM Usage      : {res['memory_mb']:.2f} MB")
    print(f"Average Time         : {res['avg_time']:.6f} s")
    print(f"Best Time            : {res['best_time']:.6f} s")
    print(f"Avg Performance      : {res['avg_tflops']:.4f} TFLOPS")
    print(f"Peak Performance     : {res['peak_tflops']:.4f} TFLOPS")
    print("==================================================")

    if args.output:
        file_exists = os.path.isfile(args.output)
        try:
            with open(args.output, mode='a', newline='') as csvfile:
                fieldnames = ['timestamp', 'backend', 'device_name', 'matrix_size', 'dtype', 
                              'iterations', 'est_vram_mb', 'avg_time_s', 'best_time_s', 
                              'avg_tflops', 'peak_tflops']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                if not file_exists:
                    writer.writeheader()

                writer.writerow({
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'backend': args.backend,
                    'device_name': res['device_name'],
                    'matrix_size': res['actual_n'],
                    'dtype': args.dtype,
                    'iterations': args.iterations,
                    'est_vram_mb': round(res['memory_mb'], 2),
                    'avg_time_s': round(res['avg_time'], 6),
                    'best_time_s': round(res['best_time'], 6),
                    'avg_tflops': round(res['avg_tflops'], 4),
                    'peak_tflops': round(res['peak_tflops'], 4)
                })
            print(f">>> Successfully appended results to {args.output}")
        except Exception as e:
            print(f">>> ERROR: Could not write to CSV. {e}")

if __name__ == "__main__":
    main()