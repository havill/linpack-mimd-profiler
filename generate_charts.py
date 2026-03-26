import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

def generate_charts(csv_file="benchmark_results.csv", filter_backend=None, filter_dtype=None, filter_gpu=None, interactive=False, export_formats=["png"]):
    # Supported formats for matplotlib/seaborn
    supported_formats = {"eps", "jpeg", "jpg", "pdf", "pgf", "png", "ps", "raw", "rgba", "svg", "svgz", "tif", "tiff", "webp"}
    filtered_formats = [fmt.lower() for fmt in export_formats if fmt.lower() in supported_formats]
    
    if not filtered_formats:
        print("❌ No valid export formats specified. Defaulting to 'png'.")
        filtered_formats = ["png"]
    export_formats = filtered_formats

    # Resolve paths
    csv_file = os.path.abspath(csv_file)
    out_dir = os.path.dirname(csv_file)
    
    print(f"--- 📊 Generating Charts from {csv_file} ---")
    
    if not os.path.exists(csv_file):
        print(f"❌ Error: Could not find '{csv_file}'.")
        return

    # ==========================================
    # LENIENT DATA LOADING (Line-by-Line Parsing)
    # ==========================================
    parsed_data = []
    try:
        import csv
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row: 
                    continue # Skip empty lines
                
                # Skip header rows (even if they accidentally got injected mid-file)
                if str(row[0]).strip().lower() in ['backend', 'start_time_utc', 'gpu_model']:
                    continue
                
                # Dynamically map the data based on how many columns exist in this specific row!
                num_cols = len(row)
                if num_cols == 9:
                    # Oldest format (9 cols): Backend, Size, Iterations, Dtype, Latency, TFLOPS, Avg Pwr, Peak Pwr, Eff
                    parsed_data.append({
                        'Start_Time_UTC': 'Unknown',
                        'GPU_Model': 'Unknown GPU',
                        'Backend': row[0], 'Size': row[1], 'Iterations': row[2], 'Dtype': row[3],
                        'Latency_ms': row[4], 'TFLOPS': row[5], 'Avg_Power_W': row[6], 
                        'Peak_Power_W': row[7], 'Efficiency_GFLOPS_W': row[8]
                    })
                elif num_cols == 10:
                    # Mid format (10 cols): Start_Time_UTC, Backend, Size...
                    parsed_data.append({
                        'Start_Time_UTC': row[0],
                        'GPU_Model': 'Unknown GPU',
                        'Backend': row[1], 'Size': row[2], 'Iterations': row[3], 'Dtype': row[4],
                        'Latency_ms': row[5], 'TFLOPS': row[6], 'Avg_Power_W': row[7], 
                        'Peak_Power_W': row[8], 'Efficiency_GFLOPS_W': row[9]
                    })
                elif num_cols >= 11:
                    # Newest format (11 cols): Start_Time_UTC, GPU_Model, Backend, Size...
                    parsed_data.append({
                        'Start_Time_UTC': row[0],
                        'GPU_Model': row[1],
                        'Backend': row[2], 'Size': row[3], 'Iterations': row[4], 'Dtype': row[5],
                        'Latency_ms': row[6], 'TFLOPS': row[7], 'Avg_Power_W': row[8], 
                        'Peak_Power_W': row[9], 'Efficiency_GFLOPS_W': row[10]
                    })
        
        if not parsed_data:
            print("❌ Error: No valid data found in the CSV file.")
            return
            
        # Convert the cleanly parsed list directly into a Pandas DataFrame
        df = pd.DataFrame(parsed_data)

    except Exception as e:
        print(f"❌ Error: Failed to parse the CSV file. ({e})")
        return

    # ==========================================
    # DATA TYPE CONVERSION & CLEANING
    # ==========================================
    # Force the numerical columns back into floats. "N/A" strings become NaN, which we fill with 0.
    numeric_cols = ['Latency_ms', 'TFLOPS', 'Avg_Power_W', 'Peak_Power_W', 'Efficiency_GFLOPS_W']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Convert Size to string for better categorical plotting on the X-axis
    df['Size'] = df['Size'].astype(str)

    # ==========================================
    # 3. APPLY COMMAND-LINE FILTERS
    # ==========================================
    if filter_backend:
        df = df[df['Backend'].str.lower() == filter_backend.lower()]
    if filter_dtype:
        df = df[df['Dtype'].str.lower() == filter_dtype.lower()]
    if filter_gpu:
        df = df[df['GPU_Model'].str.contains(filter_gpu, case=False, na=False)]

    if df.empty:
        print("❌ Error: No data left to plot after applying filters.")
        return

    sns.set_theme(style="whitegrid")

    # Determine title suffix based on GPU filtering
    unique_gpus = df['GPU_Model'].nunique()
    title_suffix = f" ({df['GPU_Model'].iloc[0]})" if unique_gpus == 1 and df['GPU_Model'].iloc[0] != "Unknown GPU" else ""

    # ==========================================