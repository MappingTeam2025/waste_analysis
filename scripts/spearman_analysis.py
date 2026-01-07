# ------------------------------------------------------------
# 1. Import Libraries
# ------------------------------------------------------------
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ------------------------------------------------------------
# 2. Load Data (file must be in same folder)
# ------------------------------------------------------------
df = pd.read_excel("LL_final_transects.xlsx")

# ------------------------------------------------------------
# 3. Create results folder
# ------------------------------------------------------------
output_folder = "results_spearman_visuals"
os.makedirs(output_folder, exist_ok=True)

# ------------------------------------------------------------
# 4. Helper Function: Spearman Test
# ------------------------------------------------------------
def spearman_test(data, x, y):
    rho, p = spearmanr(data[x], data[y])
    return rho, p

# ------------------------------------------------------------
# 5. Get communes
# ------------------------------------------------------------
communes = df["Commune"].unique()

# ------------------------------------------------------------
# 6. Prepare results storage
# ------------------------------------------------------------
results = []

# ------------------------------------------------------------
# 7. LOOP THROUGH COMMUNES
# ------------------------------------------------------------
for commune in communes:
    sub = df[df["Commune"] == commune]

    # ============================================================
    # INQUIRY 1: Bin Density vs. Waste Volume
    # x = Trash Bins per 200m
    # y = Waste Volume
    # ============================================================
    x1 = "Trash Bins per 200m"
    y1 = "Waste Volume"

    rho1, p1 = spearman_test(sub, x1, y1)

    results.append({
        "Commune": commune,
        "Inquiry": "Bin Density → Waste Volume",
        "Predictor": x1,
        "Outcome": y1,
        "Spearman_rho": rho1,
        "p_value": p1
    })

    # --- Visualization A1: Scatter + LOWESS ---
    sns.set(style="whitegrid")
    plt.figure(figsize=(7,5))
    sns.regplot(data=sub, x=x1, y=y1, lowess=True, scatter_kws={'alpha':0.6})
    plt.title(f"{commune}: Bin Density vs. Waste Volume")
    plt.tight_layout()
    plt.savefig(f"{output_folder}/{commune}_scatter_bins_vs_waste.png", dpi=300)
    plt.close()


    # ============================================================
    # INQUIRY 2: Burning per 200m vs. Waste Volume
    # x = Burning per 200m
    # y = Waste Volume
    # ============================================================
    x2 = "Burning per 200m"
    y2 = "Waste Volume"

    rho2, p2 = spearman_test(sub, x2, y2)

    results.append({
        "Commune": commune,
        "Inquiry": "Burning Intensity → Waste Volume",
        "Predictor": x2,
        "Outcome": y2,
        "Spearman_rho": rho2,
        "p_value": p2
    })

    # --- Visualization A2: Scatter + LOWESS ---
    plt.figure(figsize=(7,5))
    sns.regplot(data=sub, x=x2, y=y2, lowess=True, scatter_kws={'alpha':0.6})
    plt.title(f"{commune}: Burning vs. Waste Volume")
    plt.tight_layout()
    plt.savefig(f"{output_folder}/{commune}_scatter_burning_vs_waste.png", dpi=300)
    plt.close()


    # ============================================================
    # INQUIRY 3: Building Density vs. Multiple Waste Indicators
    # x = Building Density per 200m
    # y = multiple outcomes
    # ============================================================
    x3 = "Building Density per 200m"

    outcomes3 = {
        "Waste Volume": "Waste Volume",
        "Open Dumping per 200m": "Open Dumping per 200m",
        "Burning per 200m": "Burning per 200m",
        "Waste Diversity": "Waste Diversity",
        "Waste Disposition": "Waste Disposition"
    }

    # --- Run Spearman for each outcome ---
    for label, col in outcomes3.items():
        rho3, p3 = spearman_test(sub, x3, col)

        results.append({
            "Commune": commune,
            "Inquiry": "Building Density → Waste Indicators",
            "Predictor": x3,
            "Outcome": col,
            "Spearman_rho": rho3,
            "p_value": p3
        })

        # --- Visualization B: Scatterplots ---
        plt.figure(figsize=(7,5))
        sns.regplot(data=sub, x=x3, y=col, lowess=True, scatter_kws={'alpha':0.6})
        plt.title(f"{commune}: {x3} vs. {col}")
        plt.tight_layout()
        plt.savefig(f"{output_folder}/{commune}_scatter_building_vs_{col.replace(' ','_')}.png", dpi=300)
        plt.close()

    # --- Visualization C: Correlation Heatmap ---
    heatmap_data = sub[list(outcomes3.values()) + [x3]].corr(method="spearman")
    plt.figure(figsize=(8,6))
    sns.heatmap(heatmap_data, annot=True, cmap="coolwarm", center=0)
    plt.title(f"{commune}: Spearman Correlation Matrix")
    plt.tight_layout()
    plt.savefig(f"{output_folder}/{commune}_correlation_heatmap.png", dpi=300)
    plt.close()


# ------------------------------------------------------------
# 8. Save results to CSV
# ------------------------------------------------------------
results_df = pd.DataFrame(results)
results_df.to_csv(f"{output_folder}/spearman_results_all_communes.csv", index=False)

print("All Spearman tests and visualizations saved in:", output_folder)
