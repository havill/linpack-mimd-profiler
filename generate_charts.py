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
        return

    # ==========================================
    # 1. ROBUST DATA LOADING & VALIDATION
    # ==========================================
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"❌ Error: Failed to read the CSV file. ({e})")
        return

    required_cols = ['Backend', 'Size', 'Dtype', 'Latency_ms', 'TFLOPS']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ Error: The CSV is missing required columns: {missing_cols}")
        return

    optional_cols = ['Avg_Power_W', 'Peak_Power_W', 'Efficiency_GFLOPS_W']
    for col in optional_cols:
        if col not in df.columns:
            df[col] = 0.0

    numeric_cols = ['Latency_ms', 'TFLOPS', 'Avg_Power_W', 'Peak_Power_W', 'Efficiency_GFLOPS_W']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    df['Size'] = df['Size'].astype(str)

    if filter_backend:
        df = df[df['Backend'].str.lower() == filter_backend.lower()]
    if filter_dtype:
        df = df[df['Dtype'].str.lower() == filter_dtype.lower()]

    if df.empty:
        print("❌ Error: No data left to plot after applying filters.")
        return

    sns.set_theme(style="whitegrid")

    # ==========================================
    # CHART 1: Performance Scaling (With Raw Data Dots)
    # ==========================================
    print("📈 Generating Performance Scaling Chart...")
    plt.figure(figsize=(12, 6))
    
    # Draw the averaged bars (slightly transparent so we can see the dots)
    chart1 = sns.catplot(
        data=df, kind="bar",
        x="Size", y="TFLOPS", hue="Backend", col="Dtype",
        height=5, aspect=1.0, palette="viridis", sharey=False, alpha=0.6, capsize=.1
    )
    
    # OVERLAY: Draw every single CSV row as a dot on top of the bars!
    chart1.map_dataframe(
        sns.stripplot, x="Size", y="TFLOPS", hue="Backend",
        dodge=True, palette="dark:black", alpha=0.7, size=4, jitter=True
    )

    chart1.set_axis_labels("Matrix Size (N)", "Performance (TFLOPS)")
    chart1.fig.suptitle("GPU Compute Performance (Bars=Avg, Dots=Individual Runs)", y=1.05)
                
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
        
        # Averaged bars
        sns.barplot(
            data=eff_df, x="Size", y="Efficiency_GFLOPS_W", hue="Backend", 
            palette="magma", alpha=0.6, capsize=.1
        )
        
        # OVERLAY: Individual data points
        sns.stripplot(
            data=eff_df, x="Size", y="Efficiency_GFLOPS_W", hue="Backend",
            dodge=True, color="black", alpha=0.7, size=4, jitter=True, legend=False
        )
        
        plt.title("Hardware Energy Efficiency Scaling (Bars=Avg, Dots=Individual Runs)", fontsize=14)
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
    
    # Line plot automatically shows variance bounds if multiple runs exist
    sns.lineplot(
        data=df, x="Size", y="Latency_ms", hue="Backend", style="Dtype", 
        markers=True, dashes=False, palette="Set1", linewidth=2.5, markersize=8, errorbar='sd'
    )
    
    plt.title("Compute Latency Scaling (Log Scale with Standard Deviation)", fontsize=14)
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
        
        # Overlay data dots here too!
        chart4.map_dataframe(
            sns.stripplot, x="Size", y="Watts", hue="Power_Type",
            dodge=True, palette="dark:black", alpha=0.5, size=3, jitter=True
        )

        chart4.set_axis_labels("Matrix Size (N)", "Power Consumption (Watts)")
        chart4.fig.suptitle("Average vs Peak Power Consumption", y=1.05)
        
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
            
            # The interactive chart natively shows EVERY row! 
            fig = px.scatter(
                df, x="Size", y="TFLOPS", color="Backend", symbol="Dtype",
                size="Plot_Power", hover_data=["Start_Time_UTC", "Latency_ms", "Efficiency_GFLOPS_W", "Peak_Power_W"],
                title="Interactive Performance Overview (Hover to see specific run data!)",
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
    parser.add_argument("-i", "--interactive", action="store_true")
    parser.add_argument("-x", "--export-formats", type=str, default="png")
    
    args = parser.parse_args()
    export_formats = [fmt.strip() for fmt in args.export_formats.split(",") if fmt.strip()]
    
    generate_charts(
        csv_file=args.file, filter_backend=args.backend, filter_dtype=args.dtype,
        interactive=args.interactive, export_formats=export_formats
    )