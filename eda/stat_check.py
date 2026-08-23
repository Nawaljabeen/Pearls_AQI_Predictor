from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns


def analyze_sensor_bias(df: pd.DataFrame, output_dir: Path):
    """Quantifies statistical distribution divergence and environmental humidity drift

    between StateAir (reference-grade) and Clarity (low-cost optical) sensors.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Filter non-interpolated raw values per source
    stateair = df[
        (df["source"] == "dead") & (df["pm25_interpolated"] == False)
    ]["pm25"].dropna()
    clarity = df[(df["source"] == "live") & (df["pm25_interpolated"] == False)][
        "pm25"
    ].dropna()

    # 2. Summary Statistics Table
    metrics = {
        "Metric": [
            "Sample Count",
            "Mean",
            "Std Dev",
            "Median",
            "IQR",
            "95th Percentile",
        ],
        "StateAir (Reference)": [
            len(stateair),
            stateair.mean(),
            stateair.std(),
            stateair.median(),
            stateair.quantile(0.75) - stateair.quantile(0.25),
            stateair.quantile(0.95),
        ],
        "Clarity (Optical)": [
            len(clarity),
            clarity.mean(),
            clarity.std(),
            clarity.median(),
            clarity.quantile(0.75) - clarity.quantile(0.25),
            clarity.quantile(0.95),
        ],
    }
    stats_summary = pd.DataFrame(metrics)

    print("\n" + "=" * 60)
    print("SENSOR SOURCE STATISTICAL COMPARISON")
    print("=" * 60)
    print(stats_summary.to_string(index=False))

    # 3. Two-Sample Kolmogorov-Smirnov Test (Distribution Similarity)
    ks_stat, p_value = stats.ks_2samp(stateair, clarity)
    print(f"\nKS-Test Statistic: {ks_stat:.4f} | p-value: {p_value:.4e}")
    if p_value < 0.05:
        print(
            "⚠️  SIGNIFICANT DISTRIBUTION DRIFT DETECTED: Models trained on StateAir "
            "will mis-estimate targets measured by Clarity."
        )

    # 4. Plotting Density Overlay and Humidity Interactions
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Density Plot
    sns.kdeplot(
        stateair,
        ax=axes[0],
        label="StateAir (Reference)",
        color="#1f77b4",
        fill=True,
        alpha=0.3,
    )
    sns.kdeplot(
        clarity,
        ax=axes[0],
        label="Clarity (Optical)",
        color="#2ca02c",
        fill=True,
        alpha=0.3,
    )
    axes[0].set_title("PM2.5 Density Overlay (StateAir vs Clarity)")
    axes[0].set_xlabel("PM2.5 (µg/m³)")
    axes[0].set_xlim(0, 400)
    axes[0].legend()

    # Humidity Interaction Check (Optical sensors over-read at >70% RH)
    if "relative_humidity_2m" in df.columns:
        df_valid = df[df["pm25_interpolated"] == False].copy()
        df_valid["humidity_bin"] = pd.cut(
            df_valid["relative_humidity_2m"],
            bins=[0, 40, 65, 80, 100],
            labels=["<40% (Dry)", "40-65% (Norm)", "65-80% (High)", ">80% (Extreme)"],
        )
        sns.boxplot(
            data=df_valid,
            x="humidity_bin",
            y="pm25",
            hue="source",
            ax=axes[1],
            palette={"dead": "#1f77b4", "live": "#2ca02c"},
            showfliers=False,
        )
        axes[1].set_title("Humidity Sensitivity Check (Optical Drift)")
        axes[1].set_xlabel("Relative Humidity Bin")
        axes[1].set_ylabel("PM2.5 (µg/m³)")

    plt.tight_layout()
    plot_path = output_dir / "08_sensor_bias_analysis.png"
    plt.savefig(plot_path, dpi=120)
    plt.close()
    print(f"\nSaved bias visualization to: {plot_path}")


# Add to run_eda() inside eda_lahore.py:
# analyze_sensor_bias(df, OUTPUT_DIR)