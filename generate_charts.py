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
    # BACKWARDS COMPATIBLE DATA LOADING & VALIDATION
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
    # APPLY FILTERS
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
    # CHART 1: Performance Scaling (With Raw Data Dots)
    # ==========================================
    print("📈 Generating Performance Scaling Chart...")
    plt.figure(figsize=(12, 6))
    
    chart1 = sns.catplot(
        data=df, kind="bar",
        x="Size", y="TFLOPS", hue="Backend", col="Dtype",
        height=5, aspect=1.0, palette="viridis", sharey=False, alpha=0.6, capsize=.1
    )
    
    chart1.map_dataframe(
        sns.stripplot, x="Size", y="TFLOPS", hue="Backend",
        dodge=True, palette="dark:black", alpha=0.7, size=4, jitter=True
    )

    chart1.set_axis_labels("Matrix Size (N)", "Performance (TFLOPS)")
    chart1.fig.suptitle(f"GPU Compute Performance{title_suffix}", y=1.05)
                
    for fmt in export_formats:
        out_file = os.path.join(out_dir, f"chart_performance_tflops.{fmt}")
        chart1.savefig(out_file, dpi=300, bbox_inches="tight")
        print(f"   ✅ Saved: {out_file}")
    plt.close('all')

    # ==========================================
    # CHART 2: Energy Efficiency (With Raw Data Dots)
    # ==========================================
    print("🔋 Generating Energy Efficiency Chart...")
    eff_df = df[df['Efficiency_GFLOPS_W'] > 0]
    if not eff_df.empty:
        plt.figure(figsize=(10, 6))
        
        sns.barplot(
            data=eff_df, x="Size", y="Efficiency_GFLOPS_W", hue="Backend", 
            palette="magma", alpha=0.6, capsize=.1
        )
        
        sns.stripplot(
            data=eff_df, x="Size", y="Efficiency_GFLOPS_W", hue="Backend",
            dodge=True, color="black", alpha=0.7, size=4, jitter=True, legend=False
        )
        
        plt.title(f"Hardware Energy Efficiency Scaling{title_suffix}", fontsize=14)
        plt.ylabel("Efficiency (GFLOPS / Watt)", fontsize=12)
        plt.xlabel("Matrix Size (N)", fontsize=12)
        plt.tight_layout()
        
        for fmt in export_formats:
            out_file = os.path.join(out_dir, f"chart_energy_efficiency.{fmt}")
            plt.savefig(out_file, dpi=300)
            print(f"   ✅ Saved: {out_file}")
    else:
        print("   ⚠️ Skipped Energy chart (No valid power/efficiency data found).")
    plt.close('all')

    # ==========================================
    # CHART 3: Compute Latency 
    # ==========================================
    print("⏱️ Generating Latency Chart...")
    plt.figure(figsize=(10, 6))
    
    sns.lineplot(
        data=df, x="Size", y="Latency_ms", hue="Backend", style="Dtype", 
        markers=True, dashes=False, palette="Set1", linewidth=2.5, markersize=8, errorbar='sd'
    )
    
    plt.title(f"Compute Latency Scaling (Log Scale){title_suffix}", fontsize=14)
    plt.ylabel("Latency (ms) - Log Scale", fontsize=12)
    plt.xlabel("Matrix Size (N)", fontsize=12)
    plt.yscale('log')
    plt.legend(title='Backend & Dtype')
    plt.tight_layout()
    
    for fmt in export_formats:
        out_file = os.path.join(out_dir, f"chart_latency.{fmt}")
        plt.savefig(out_file, dpi=300)
        print(f"   ✅ Saved: {out_file}")
    plt.close('all')

    # ==========================================
    # CHART 4: Power Consumption
    # ==========================================
    print("⚡ Generating Power Profile Chart...")
    pwr_df = df[(df['Avg_Power_W'] > 0) & (df['Peak_Power_W'] > 0)].copy()
    if not pwr_df.empty:
        melted_pwr = pwr_df.melt(id_vars=['Size', 'Backend', 'Dtype'], 
                                 value_vars=['Avg_Power_W', 'Peak_Power_W'],
                                 var_name='Power_Type', value_name='Watts')
        
        plt.figure(figsize=(12, 6))
        chart4 = sns.catplot(
            data=melted_pwr, kind="bar",
            x="Size", y="Watts", hue="Power_Type", col="Backend", row="Dtype",
            height=4, aspect=1.2, palette="coolwarm", alpha=0.8
        )
        
        chart4.map_dataframe(
            sns.stripplot, x="Size", y="Watts", hue="Power_Type",
            dodge=True, palette="dark:black", alpha=0.5, size=3, jitter=True
        )

        chart4.set_axis_labels("Matrix Size (N)", "Power Consumption (Watts)")
        chart4.fig.suptitle(f"Average vs Peak Power Consumption{title_suffix}", y=1.05)
        
        for fmt in export_formats:
            out_file = os.path.join(out_dir, f"chart_power_profile.{fmt}")
            chart4.savefig(out_file, dpi=300, bbox_inches="tight")
            print(f"   ✅ Saved: {out_file}")
    else:
        print("   ⚠️ Skipped Power chart (No valid power data found).")
    plt.close('all')

    # ==========================================
    # CHART 5: Interactive Charts (Plotly)
    # ==========================================
    if interactive:
        try:
            import plotly.express as px
            print("🖱️ Generating Interactive Bubble Chart...")
            
            df['Plot_Power'] = df['Avg_Power_W'].apply(lambda x: x if x > 0 else 1)
            
            # Added GPU_Model to hover_data!
            fig = px.scatter(
                df, x="Size", y="TFLOPS", color="Backend", symbol="Dtype",
                size="Plot_Power", hover_data=["GPU_Model", "Start_Time_UTC", "Latency_ms", "Efficiency_GFLOPS_W", "Peak_Power_W"],
                title="Interactive Performance Overview (Hover for details)",
                labels={"Size": "Matrix Size (N)", "TFLOPS": "Performance (TFLOPS)"},
                size_max=30
            )
            out_file = os.path.join(out_dir, "chart_interactive_overview.html")
            fig.write_html(out_file)
            print(f"   ✅ Saved: {out_file}")
        except ImportError:
            print("   ⚠️ Plotly not installed. Skipping interactive chart.")

    print("\n🎉 All charts generated successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate charts from GPU Benchmark CSV data.")
    parser.add_argument("-f", "--file", type=str, default="benchmark_results.csv")
    parser.add_argument("-b", "--backend", type=str, default=None)
    parser.add_argument("-d", "--dtype", type=str, default=None)
    parser.add_argument("-g", "--gpu", type=str, default=None, help="Filter to a specific GPU Model substring")
    parser.add_argument("-i", "--interactive", action="store_true")
    parser.add_argument("-x", "--export-formats", type=str, default="png")
    
    args = parser.parse_args()
    export_formats = [fmt.strip() for fmt in args.export_formats.split(",") if fmt.strip()]
    
    generate_charts(
        csv_file=args.file, filter_backend=args.backend, filter_dtype=args.dtype, 
        filter_gpu=args.gpu, interactive=args.interactive, export_formats=export_formats
    )