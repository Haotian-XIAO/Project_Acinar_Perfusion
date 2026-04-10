import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compare_activation_levels(
    csv_ref,
    csv_new,
    output_dir,
    ref_label="stretch_0.0",
    new_label="stretch_0.2",
    eps=1e-12,
):
    """
    Compare cell perfusion/activation levels between two cases.

    Parameters
    ----------
    csv_ref : str
        Path to reference CSV.
    csv_new : str
        Path to new CSV.
    output_dir : str
        Folder to save outputs.
    ref_label : str
        Label for reference case.
    new_label : str
        Label for new case.
    eps : float
        Small value to avoid division by zero.

    Returns
    -------
    df : pandas.DataFrame
        Merged comparison table.
    summary : dict
        Global summary statistics.
    """

    os.makedirs(output_dir, exist_ok=True)

    # ===== read =====
    df_ref = pd.read_csv(csv_ref)
    df_new = pd.read_csv(csv_new)

    required_cols = ["cell_id", "center_x", "center_y", "perfusion_score"]
    for col in required_cols:
        if col not in df_ref.columns:
            raise ValueError(f"Missing column '{col}' in reference csv")
        if col not in df_new.columns:
            raise ValueError(f"Missing column '{col}' in new csv")

    # ===== rename scores =====
    df_ref = df_ref.rename(columns={"perfusion_score": f"score_{ref_label}"})
    df_new = df_new.rename(columns={"perfusion_score": f"score_{new_label}"})

    # keep one set of centers from ref
    df_new = df_new.drop(columns=["center_x", "center_y"], errors="ignore")

    # ===== merge by cell_id =====
    df = pd.merge(df_ref, df_new, on="cell_id", how="inner")

    s0 = df[f"score_{ref_label}"].to_numpy()
    s1 = df[f"score_{new_label}"].to_numpy()

    # ===== compute changes =====
    df["delta_score"] = s1 - s0

    # relative change wrt ref
    df["percent_change"] = np.where(
        np.abs(s0) > eps,
        100.0 * (s1 - s0) / s0,
        np.nan
    )

    # fold change
    df["fold_change"] = np.where(
        np.abs(s0) > eps,
        s1 / s0,
        np.nan
    )

    # qualitative label
    df["trend"] = np.where(
        df["delta_score"] > 0, "up",
        np.where(df["delta_score"] < 0, "down", "unchanged")
    )

    # sort by strongest increase / decrease
    df_up = df[df["delta_score"] > 0].sort_values("delta_score", ascending=False)
    df_down = df[df["delta_score"] < 0].sort_values("delta_score", ascending=True)

    # ===== summary =====
    summary = {
        "n_cells_common": int(len(df)),
        "n_up": int((df["delta_score"] > 0).sum()),
        "n_down": int((df["delta_score"] < 0).sum()),
        "n_unchanged": int((df["delta_score"] == 0).sum()),
        "mean_ref": float(np.mean(s0)),
        "mean_new": float(np.mean(s1)),
        "mean_delta": float(np.mean(df["delta_score"])),
        "median_percent_change": float(np.nanmedian(df["percent_change"])),
        "mean_percent_change": float(np.nanmean(df["percent_change"])),
        "max_increase_cell_id": int(df_up.iloc[0]["cell_id"]) if len(df_up) > 0 else None,
        "max_increase_value": float(df_up.iloc[0]["delta_score"]) if len(df_up) > 0 else None,
        "max_decrease_cell_id": int(df_down.iloc[0]["cell_id"]) if len(df_down) > 0 else None,
        "max_decrease_value": float(df_down.iloc[0]["delta_score"]) if len(df_down) > 0 else None,
    }

    # ===== save comparison table =====
    comparison_csv = os.path.join(output_dir, f"compare_{ref_label}_vs_{new_label}.csv")
    df.to_csv(comparison_csv, index=False)

    up_csv = os.path.join(output_dir, f"cells_up_{ref_label}_vs_{new_label}.csv")
    down_csv = os.path.join(output_dir, f"cells_down_{ref_label}_vs_{new_label}.csv")
    df_up.to_csv(up_csv, index=False)
    df_down.to_csv(down_csv, index=False)

    # ===== print summary =====
    print("=== COMPARISON SUMMARY ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    # ===== Figure 1: scatter ref vs new =====
    plt.figure(figsize=(6, 6))
    plt.scatter(df[f"score_{ref_label}"], df[f"score_{new_label}"], s=20)
    xy_min = min(df[f"score_{ref_label}"].min(), df[f"score_{new_label}"].min())
    xy_max = max(df[f"score_{ref_label}"].max(), df[f"score_{new_label}"].max())
    plt.plot([xy_min, xy_max], [xy_min, xy_max], "--")
    plt.xlabel(ref_label)
    plt.ylabel(new_label)
    plt.title("Cell activation/perfusion: reference vs new")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"scatter_{ref_label}_vs_{new_label}.png"), dpi=300)
    plt.close()

    # ===== Figure 2: histogram of percent change =====
    plt.figure(figsize=(7, 5))
    valid_percent = df["percent_change"].replace([np.inf, -np.inf], np.nan).dropna()
    plt.hist(valid_percent, bins=30)
    plt.xlabel("Percent change (%)")
    plt.ylabel("Number of cells")
    plt.title(f"Percent change: {ref_label} -> {new_label}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"hist_percent_change_{ref_label}_vs_{new_label}.png"), dpi=300)
    plt.close()

    # ===== Figure 3: spatial map of delta =====
    plt.figure(figsize=(7, 6))
    sc = plt.scatter(
        df["center_x"],
        df["center_y"],
        c=df["delta_score"],
        s=45
    )
    plt.colorbar(sc, label="delta score")
    plt.gca().set_aspect("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Spatial delta map: {new_label} - {ref_label}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"spatial_delta_{ref_label}_vs_{new_label}.png"), dpi=300)
    plt.close()

    # ===== Figure 4: spatial map of percent change =====
    plt.figure(figsize=(7, 6))
    percent_vals = df["percent_change"].replace([np.inf, -np.inf], np.nan)
    absmax = np.nanmax(np.abs(percent_vals.to_numpy()))
    sc = plt.scatter(
        df["center_x"],
        df["center_y"],
        c=percent_vals,
        s=45,
        cmap="bwr",
        vmin=-absmax,
        vmax=absmax
    )
    plt.colorbar(sc, label="percent change (%)")
    plt.gca().set_aspect("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Spatial percent change map: {new_label} vs {ref_label}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"spatial_percent_{ref_label}_vs_{new_label}.png"), dpi=300)
    plt.close()

    return df, summary


if __name__ == "__main__":
    base_dir = "/Users/xiao/PhD/Project_Acinar_Perfusion/results"

    csv_ref = os.path.join(base_dir, "stretch_0.0_data.csv")
    csv_new = os.path.join(base_dir, "p_f_0.3_data.csv")

    output_dir = os.path.join(base_dir, "comparison_stretch_0.0_vs_p_f_0.3_data")

    df_compare, summary = compare_activation_levels(
        csv_ref=csv_ref,
        csv_new=csv_new,
        output_dir=output_dir,
        ref_label="stretch_0.0",
        new_label="p_f_0.3",
    )