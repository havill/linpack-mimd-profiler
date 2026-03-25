import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

def generate_charts(csv_file="benchmark_results.csv", filter_backend=None, filter_dtype=None, interactive=False, export_formats=["png"]):
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
        print("Please run the benchmark first to generate the CSV, or specify the correct path with -f.")
        return

    # ==========================================
    # 1. ROBUST DATA LOADING & VALIDATION
    # ==========================================
    try:
        df = pd.read_csv(csv_file)
    except pd.errors.EmptyDataError:
        print("❌ Error: The CSV file is completely empty.")
        return
    except pd.errors.ParserError:
        print("❌ Error: The CSV file is malformed and could not be parsed. Check for broken lines.")
        return
    except Exception as e:
        print(f"❌ Error: Failed to read the CSV file. ({e})")
        return

    # Core columns required to run at all
    required_cols = ['Backend', 'Size', 'Dtype', 'Latency_ms', 'TFLOPS']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"❌ Error: The CSV is missing required columns: {missing_cols}")
        print("This file does not appear to be a valid GPU Benchmark result file.")
        return

    # Optional columns (Power metrics) - Inject as 0 if missing (e.g., from older benchmark runs)
    optional_cols = ['Avg_Power_W', 'Peak_Power_W', 'Efficiency_GFLOPS_W']
    for col in optional_cols:
        if col not in df.columns:
            df[col] = 0.0

    # ==========================================
    # 2. DATA CLEANING & TYPE CONVERSION
    # ==========================================
    numeric_cols = ['Latency_ms', 'TFLOPS', 'Avg_Power_W', 'Peak_Power_W', 'Efficiency_GFLOPS_W']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Convert Size to string for better categorical plotting on X-axis
    df['Size'] = df['Size'].astype(str)

    # ==========================================
    # 3. APPLY FILTERS
    # ==========================================
    if filter_backend:
        df = df[df['Backend'].str.lower() == filter_backend.lower()]
    if filter_dtype:
        df = df[df['Dtype'].str.lower() == filter_dtype.lower()]

    if df.empty:
        print("❌ Error: No data left to plot after applying filters. Check your -b or -d arguments.")
        return

    # Set the visual style
    sns.set_theme(style="whitegrid")

    # ==========================================
    # CHART 1: Performance Scaling (TFLOPS)
    # ==========================================
    print("📈 Generating Performance Scaling Chart...")
    plt.figure(figsize=(12, 6))
    chart1 = sns.catplot(
        data=df, kind="bar",
        x="Size", y="TFLOPS", hue="Backend", col="Dtype",
        height=5, aspect=1.0, palette="viridis", sharey=False
    )
    chart1.set_axis_labels("Matrix Size (N)", "Performance (TFLOPS)")
    chart1.fig.suptitle("GPU Compute Performance by Matrix Size & Precision", y=1.05)
    
    # Add numerical labels on top of bars
    for ax in chart1.axes.flat:
        for p in ax.patches:
            height = p.get_height()
            if height > 0:
                ax.annotate(f'{height:.2f}', (p.get_x() + p.get_width() / 2., height),
                            ha='center', va='bottom', fontsize=8, color='black', xytext=(0, 2), textcoords='offset points')
                
    for fmt in export_formats:
        out_file = os.path.join(out_dir, f"chart_performance_tflops.{fmt}")
        chart1.savefig(out_file, dpi=300, bbox_inches="tight")
        print(f"   ✅ Saved: {out_file}")
    plt.close('all')

    # ==========================================
    # CHART 2: Energy Efficiency (GFLOPS/W)
    # ==========================================
    print("🔋 Generating Energy Efficiency Chart...")
    eff_df = df[df['Efficiency_GFLOPS_W'] > 0]
    if not eff_df.empty:
        plt.figure(figsize=(10, 6))
        chart2 = sns.barplot(
            data=eff_df, 
            x="Size", y="Efficiency_GFLOPS_W", hue="Backend", 
            palette="magma"
        )
        plt.title("Hardware Energy Efficiency Scaling", fontsize=14)
        plt.ylabel("Efficiency (GFLOPS / Watt)", fontsize=12)
        plt.xlabel("Matrix Size (N)", fontsize=12)
        plt.legend(title='Backend')
        plt.tight_layout()
        
        for fmt in export_formats:
            out_file = os.path.join(out_dir, f"chart_energy_efficiency.{fmt}")
            plt.savefig(out_file, dpi=300)
            print(f"   ✅ Saved: {out_file}")
    else:
        print("   ⚠️ Skipped Energy chart (No valid power/efficiency data found in CSV).")
    plt.close('all')

    # ==========================================
    # CHART 3: Compute Latency 
    # ==========================================
    print("⏱️ Generating Latency Chart...")
    plt.figure(figsize=(10, 6))
    chart3 = sns.lineplot(
        data=df,
        x="Size", y="Latency_ms", hue="Backend", style="Dtype", 
        markers=True, dashes=False, palette="Set1", linewidth=2.5, markersize=10
    )
    plt.title("Compute Latency Scaling (Log Scale)", fontsize=14)
    plt.ylabel("Latency (ms) - Log Scale", fontsize=12)
    plt.xlabel("Matrix Size (N)", fontsize=12)
    plt.yscale('log') # Log scale because O(N^3) complexity explodes the time
    plt.legend(title='Backend & Dtype')
    plt.tight_layout()
    
    for fmt in export_formats:
        out_file = os.path.join(out_dir, f"chart_latency.{fmt}")
        plt.savefig(out_file, dpi=300)
        print(f"   ✅ Saved: {out_file}")
    plt.close('all')

    # ==========================================
    # CHART 4: Power Consumption (Avg vs Peak)
    # ==========================================
    print("⚡ Generating Power Profile Chart...")
    pwr_df = df[(df['Avg_Power_W'] > 0) & (df['Peak_Power_W'] > 0)].copy()
    if not pwr_df.empty:
        # Melt the dataframe to put Avg and Peak in the same column for side-by-side plotting
        melted_pwr = pwr_df.melt(id_vars=['Size', 'Backend', 'Dtype'], 
                                 value_vars=['Avg_Power_W', 'Peak_Power_W'],
                                 var_name='Power_Type', value_name='Watts')
        
        plt.figure(figsize=(12, 6))
        chart4 = sns.catplot(
            data=melted_pwr, kind="bar",
            x="Size", y="Watts", hue="Power_Type", col="Backend", row="Dtype",
            height=4, aspect=1.2, palette="coolwarm"
        )
        chart4.set_axis_labels("Matrix Size (N)", "Power Consumption (Watts)")
        chart4.fig.suptitle("Average vs Peak Power Consumption", y=1.05)
        
        for fmt in export_formats:
            out_file = os.path.join(out_dir, f"chart_power_profile.{fmt}")
            chart4.savefig(out_file, dpi=300, bbox_inches="tight")
            print(f"   ✅ Saved: {out_file}")
    else:
        print("   ⚠️ Skipped Power chart (No valid power data found in CSV).")
    plt.close('all')

    # ==========================================
    # CHART 5: Interactive Charts (Plotly)
    # ==========================================
    if interactive:
        try:
            import plotly.express as px
            print("🖱️ Generating Interactive Bubble Chart...")
            
            # Bubble chart: X=Size, Y=TFLOPS, Bubble Size=Power, Color=Backend
            # Create a safe column for bubble size (cannot be 0)
            df['Plot_Power'] = df['Avg_Power_W'].apply(lambda x: x if x > 0 else 1)
            
            fig = px.scatter(
                df, x="Size", y="TFLOPS", color="Backend", symbol="Dtype",
                size="Plot_Power", hover_data=["Latency_ms", "Efficiency_GFLOPS_W", "Peak_Power_W"],
                title="Interactive Performance Overview (Bubble size = Avg Power)",
                labels={"Size": "Matrix Size (N)", "TFLOPS": "Performance (TFLOPS)"},
                size_max=30
            )
            out_file = os.path.join(out_dir, "chart_interactive_overview.html")
            fig.write_html(out_file)
            print(f"   ✅ Saved: {out_file}")
        except ImportError:
            print("   ⚠️ Plotly not installed. Skipping interactive chart. (Run: pip install plotly)")

    print("\n🎉 All charts generated successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate charts from GPU Benchmark CSV data.")
    parser.add_argument(
        "-f", "--file", 
        type=str, 
        default="benchmark_results.csv", 
        help="Path to the master CSV file (defaults to 'benchmark_results.csv' in current dir)."
    )
    parser.add_argument(
        "-b", "--backend",
        type=str,
        default=None,
        help="Filter charts to a specific backend (e.g., CUDA or OPENCL)."
    )
    parser.add_argument(
        "-d", "--dtype",
        type=str,
        default=None,
        help="Filter charts to a specific data type (e.g., float32 or float64)."
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Generate interactive HTML charts (requires Plotly)."
    )
    parser.add_argument(
        "-x", "--export-formats",
        type=str,
        default="png",
        help="Comma-separated list of export formats (e.g., png,pdf,svg)."
    )
    
    args = parser.parse_args()
    export_formats = [fmt.strip() for fmt in args.export_formats.split(",") if fmt.strip()]
    
    generate_charts(
        csv_file=args.file,
        filter_backend=args.backend,
        filter_dtype=args.dtype,
        interactive=args.interactive,
        export_formats=export_formats
    )