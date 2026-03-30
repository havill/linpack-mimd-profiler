#!/usr/bin/env python3
import argparse
import time
import csv
import os
from datetime import datetime, timezone
import numpy as np
import os
import sys
import threading
import time

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

try:
    import pynvml
    HAS_NVML = True
except ImportError:
    HAS_NVML = False

class PowerMonitor(threading.Thread):
    """Background thread to poll the GPU power sensor via NVML."""
    def __init__(self, device_index=0):
        super().__init__()
        self.keep_running = True
        self.readings = []
        self.valid = False
        if HAS_NVML:
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
                self.valid = True
            except Exception:
                pass # Fails silently if no NVIDIA GPU is found
        
    def run(self):
        if not self.valid: return
        while self.keep_running:
            try:
                power_mw = pynvml.nvmlDeviceGetPowerUsage(self.handle)
                self.readings.append(power_mw / 1000.0) # Convert to Watts
            except Exception:
                pass
            time.sleep(0.01) # Sample every 10ms
            
    def stop(self):
        self.keep_running = False
        if self.valid:
            try: pynvml.nvmlShutdown() 
            except: pass
        if not self.readings: 
            return 0.0, 0.0
        avg_power = sum(self.readings) / len(self.readings)
        peak_power = max(self.readings)
        return avg_power, peak_power

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
        # Query the CUDA runtime for the exact hardware name
        props = cp.cuda.runtime.getDeviceProperties(0)
        device_name = props['name']
        
        # Depending on the CuPy version, it might return bytes instead of a string
        if isinstance(device_name, bytes):
            device_name = device_name.decode('utf-8')
    except Exception as e:
        print(f"DEBUG: Could not get exact CUDA device name: {e}")
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

    monitor = PowerMonitor()
    monitor.start()

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

    avg_power, peak_power = monitor.stop()

    return process_results(
        n=n,
        memory_mb=memory_mb,
        times=times,
        flops_per_run=flops_per_run,
        device_name=device_name,
        avg_power=avg_power,
        peak_power=peak_power
    )

# ==============================================================================
# HPL-AI BACKEND (Mixed Precision / Iterative Refinement)
# ==============================================================================
def run_hpl_ai(n, iterations, dtype):
    try:
        import cupy as cp
        from cupyx.scipy.linalg import lu_factor, lu_solve
    except ImportError:
        print("ERROR: CuPy is not installed. Run: pip install cupy-cuda12x (or match your CUDA version)")
        exit(1)

    # For HPL-AI, we force the "high precision" to FP32 and the "low precision" to FP16
    # (If the user asks for FP64, we can respect that for the high precision layer)
    high_dt = cp.float64 if dtype == 'float64' else cp.float32
    low_dt = cp.float16
    
    bytes_per_element = 8 if dtype == 'float64' else 4
    # Memory estimation includes the high-precision matrix, low-precision matrix, and vectors
    memory_mb = ((n * n * bytes_per_element) + (n * n * 2)) / (1024 ** 2)

    try:
        props = cp.cuda.runtime.getDeviceProperties(0)
        device_name = props['name']
        if isinstance(device_name, bytes):
            device_name = device_name.decode('utf-8')
    except Exception:
        device_name = "CUDA GPU"

    print("Generating HPL-AI matrices and warming up Tensor Cores...")
    try:
        A_high = cp.random.rand(n, n, dtype=high_dt)
        b_high = cp.random.rand(n, dtype=high_dt)
        x_init = cp.zeros_like(b_high)
    except cp.cuda.memory.OutOfMemoryError:
        print("ERROR: Out of GPU Memory. Try a smaller matrix size (-n).")
        exit(1)

    # FLOPS for standard LU decomposition + solve: (2/3)*N^3 + 2*N^2
    # We use the standard formula to compare apples-to-apples with standard FP32
    flops_per_run = (2.0 / 3.0) * (n ** 3) + 2.0 * (n ** 2)
    times = []

    # Warmup
    A_low = A_high.astype(low_dt)
    lu_and_piv = lu_factor(A_low)
    cp.cuda.Stream.null.synchronize()

    monitor = PowerMonitor()
    monitor.start()

    print(f"Running HPL-AI iterations (High: {dtype}, Low: FP16)...")
    for _ in range(iterations):
        # Reset our guess
        x = x_init.copy()
        
        # Slightly alter 'b' to prevent aggressive caching
        b_high = cp.random.rand(n, dtype=high_dt)
        cp.cuda.Stream.null.synchronize()

        start_time = time.perf_counter()
        
        # Tensor Core Phase: LU factorization in low precision
        A_low = A_high.astype(low_dt)
        lu_and_piv = lu_factor(A_low)
        
        # Iterative Refinement Phase
        tolerance = 1e-8 if dtype == 'float32' else 1e-12
        for i in range(50): # Max 50 iterations
            residual = b_high - cp.dot(A_high, x)
            if cp.linalg.norm(residual) < tolerance:
                break
            
            correction = lu_solve(lu_and_piv, residual.astype(low_dt))
            x = x + correction.astype(high_dt)
            
        cp.cuda.Stream.null.synchronize()
        end_time = time.perf_counter()
        
        times.append(end_time - start_time)

    avg_power, peak_power = monitor.stop()
    print(f"Average Power: {avg_power:.2f} W, Peak Power: {peak_power:.2f} W")

    return process_results(
        n=n,
        memory_mb=memory_mb,
        times=times,
        flops_per_run=flops_per_run,
        device_name=device_name,
        avg_power=avg_power,
        peak_power=peak_power
    )

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

    monitor = PowerMonitor()
    monitor.start()
 
    print("Running OpenCL iterations...")
    for _ in range(iterations):
        start_time = time.perf_counter()
        gemm(queue, (n, n), (tile_size, tile_size), d_A, d_B, d_C, np.int32(n))
        queue.finish()
        end_time = time.perf_counter()
        times.append(end_time - start_time)

    avg_power, peak_power = monitor.stop()

    return process_results(
        n=n,
        memory_mb=memory_mb,
        times=times,
        flops_per_run=flops_per_run,
        device_name=device_name,
        avg_power=avg_power,
        peak_power=peak_power
    )

# ==============================================================================

def process_results(n, memory_mb, times, flops_per_run, device_name, avg_power=0.0, peak_power=0.0):
    avg_time = sum(times) / len(times)
    best_time = min(times)
    
    # Calculate TFLOPS
    avg_tflops = (flops_per_run / avg_time) / 1e12
    peak_tflops = (flops_per_run / best_time) / 1e12
    
    # Calculate Energy Efficiency (GFLOPS per Watt)
    gflops_per_watt = 0.0
    if avg_tflops > 0 and avg_power > 0:
        gflops_per_watt = (avg_tflops * 1000) / avg_power
    
    return {
        'actual_n': n,
        'device_name': device_name,
        'memory_mb': memory_mb,
        'avg_time': avg_time,
        'best_time': best_time,
        'avg_tflops': avg_tflops,
        'peak_tflops': peak_tflops,
        
        'avg_power_w': round(avg_power, 2) if avg_power > 0 else "N/A",
        'peak_power_w': round(peak_power, 2) if peak_power > 0 else "N/A",
        'efficiency_gflops_w': round(gflops_per_watt, 2) if gflops_per_watt > 0 else "N/A"
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

    # Capture Local time for console & UTC ISO8601 for CSV
    start_time_local = datetime.now().astimezone()
    start_time_local_str = f"{start_time_local.strftime('%Y-%m-%d %H:%M:%S')} {start_time_local.tzname()}"
    start_time_utc_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Route to appropriate backend
    if args.backend == "cuda":
        res = run_cuda(args.size, args.iterations, args.dtype)
    else:
        res = run_opencl(args.size, args.iterations, args.dtype)

    # Capture End Time (Local for console)
    end_time_local = datetime.now().astimezone()
    end_time_local_str = f"{end_time_local.strftime('%Y-%m-%d %H:%M:%S')} {end_time_local.tzname()}"

    print("--------------------------------------------------")
    print(f"Start Time           : {start_time_local_str}")
    print(f"End Time             : {end_time_local_str}")
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
                fieldnames = [
                    "Start_Time_UTC", "GPU_Model", "Backend", "Size", "Iterations", "Dtype", 
                    "Latency_ms", "TFLOPS", "Avg_Power_W", "Peak_Power_W", "Efficiency_GFLOPS_W"
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                if not file_exists:
                    writer.writeheader()

                writer.writerow({
                    'Start_Time_UTC': start_time_utc_iso,
                    'GPU_Model': res['device_name'],
                    'Backend': args.backend.upper(),
                    'Size': res['actual_n'],
                    'Iterations': args.iterations,
                    'Dtype': args.dtype,
                    'Latency_ms': round(res['avg_time'] * 1000, 2),
                    'TFLOPS': round(res['avg_tflops'], 4),
                    'Avg_Power_W': res['avg_power_w'],
                    'Peak_Power_W': res['peak_power_w'],
                    'Efficiency_GFLOPS_W': res['efficiency_gflops_w']
                })
            print(f">>> Successfully appended results to {args.output}")
        except Exception as e:
            print(f">>> ERROR: Could not write to CSV. {e}")

if __name__ == "__main__":
    main()