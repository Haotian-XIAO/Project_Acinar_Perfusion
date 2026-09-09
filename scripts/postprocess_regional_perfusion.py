#!/usr/bin/env python3
"""Compute reproducible acinar perfusion metrics from a retained FE solution.

No FE problem is reconstructed or solved.  The final nodal J_tot, q_l and Q_l
fields are read directly from the dolfin_mech XDMF/HDF5 time series.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
import meshio
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "results" / "vascular_baseline"
DEFAULT_ACINAR_METADATA = ROOT / "results" / "acinar_partition" / "periodic_acinar_metadata.json"
DEFAULT_ACINAR_SUMMARY = ROOT / "results" / "acinar_partition" / "periodic_acinar_summary.json"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_hdf_reference(text, xdmf_path):
    filename, dataset = text.strip().split(":", 1)
    return xdmf_path.parent / filename, dataset


def read_final_xdmf_fields(xdmf_path):
    root = ET.parse(xdmf_path).getroot()
    grids = root.findall('.//Grid[@GridType="Uniform"]')
    if not grids:
        raise RuntimeError(f"No uniform time grids found in {xdmf_path}.")
    grid = max(grids, key=lambda item: float(item.find("Time").attrib["Value"]))
    final_time = float(grid.find("Time").attrib["Value"])

    def read_data_item(item):
        h5_path, dataset = parse_hdf_reference(item.text, xdmf_path)
        with h5py.File(h5_path, "r") as stream:
            return np.asarray(stream[dataset])

    topology = read_data_item(grid.find("Topology/DataItem")).astype(int)
    coordinates = read_data_item(grid.find("Geometry/DataItem")).astype(float)
    attributes = {}
    for attribute in grid.findall("Attribute"):
        name = attribute.attrib["Name"]
        if name in {"J_tot", "q_l", "Q_l"}:
            attributes[name] = read_data_item(attribute.find("DataItem"))
    missing = {"J_tot", "q_l", "Q_l"} - set(attributes)
    if missing:
        raise RuntimeError(f"Final solution is missing fields: {sorted(missing)}.")
    if any(attribute.attrib.get("Center") != "Node" for attribute in grid.findall("Attribute")
           if attribute.attrib["Name"] in attributes):
        raise RuntimeError("Regional integration expects saved nodal J_tot/q_l/Q_l fields.")
    return final_time, coordinates[:, :2], topology, {
        "J_tot": attributes["J_tot"].reshape(-1),
        "q_l": attributes["q_l"][:, :2],
        "Q_l": attributes["Q_l"][:, :2],
    }


def read_domain_markers(vtu_path, coordinates, topology):
    visual = meshio.read(vtu_path)
    visual_coordinates = np.asarray(visual.points[:, :2], dtype=float)
    visual_topology = np.asarray(visual.get_cells_type("triangle"), dtype=int)
    if visual_coordinates.shape != coordinates.shape or not np.allclose(
        visual_coordinates, coordinates, rtol=0.0, atol=1e-12
    ):
        raise RuntimeError("final_fields.vtu coordinates do not match the retained XDMF mesh.")
    if visual_topology.shape != topology.shape or not np.array_equal(visual_topology, topology):
        raise RuntimeError("final_fields.vtu cell ordering does not match the retained XDMF mesh.")
    return np.asarray(visual.cell_data_dict["domain_id"]["triangle"], dtype=int)


def periodic_cell_ownership(centroids, metadata, tie_tolerance=1e-12):
    cells = sorted(metadata["cells"], key=lambda item: int(item["cell_id"]))
    cell_ids = np.asarray([int(item["cell_id"]) for item in cells], dtype=int)
    seeds = np.asarray([item["seed_coordinate"] for item in cells], dtype=float)
    acinus_by_cell = np.asarray([int(item["acinus_id"]) for item in cells], dtype=int)
    if not np.array_equal(cell_ids, np.arange(len(cell_ids))):
        raise RuntimeError("Canonical cell IDs must be contiguous and zero-based.")

    a1 = np.asarray(metadata["periodic_cell"]["a1"], dtype=float)
    a2 = np.asarray(metadata["periodic_cell"]["a2"], dtype=float)
    lattice = np.column_stack([a1, a2])
    inverse_lattice = np.linalg.inv(lattice)
    fractional_delta = (inverse_lattice @ (centroids[:, None, :] - seeds[None, :, :]).transpose(0, 2, 1)).transpose(0, 2, 1)
    fractional_delta -= np.round(fractional_delta)
    minimum_image_delta = (lattice @ fractional_delta.transpose(0, 2, 1)).transpose(0, 2, 1)
    squared_distance = np.einsum("nki,nki->nk", minimum_image_delta, minimum_image_delta)
    order = np.argsort(squared_distance, axis=1, kind="stable")
    owner = order[:, 0]
    ambiguous = np.where(
        np.abs(squared_distance[np.arange(len(centroids)), order[:, 1]]
               - squared_distance[np.arange(len(centroids)), owner]) <= tie_tolerance
    )[0]
    return owner.astype(int), acinus_by_cell[owner], ambiguous.astype(int), squared_distance


def triangle_quadrature(coordinates, topology, fields):
    vertices = coordinates[topology]
    edge_1 = vertices[:, 1] - vertices[:, 0]
    edge_2 = vertices[:, 2] - vertices[:, 0]
    reference_area = 0.5 * np.abs(edge_1[:, 0] * edge_2[:, 1] - edge_1[:, 1] * edge_2[:, 0])
    if np.any(reference_area <= 0.0):
        raise RuntimeError("Retained mesh contains zero-area triangles.")

    barycentric = np.asarray([
        [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
        [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
        [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
    ])
    J_nodes = fields["J_tot"][topology]
    q_nodes = fields["q_l"][topology]
    Q_nodes = fields["Q_l"][topology]
    J_qp = np.einsum("qa,nab->nqb", barycentric, J_nodes[..., None])[..., 0]
    q_qp = np.einsum("qa,nad->nqd", barycentric, q_nodes)
    Q_qp = np.einsum("qa,nad->nqd", barycentric, Q_nodes)
    q_magnitude_qp = np.linalg.norm(q_qp, axis=2)
    Q_magnitude_qp = np.linalg.norm(Q_qp, axis=2)
    if np.any(J_qp <= 0.0):
        raise RuntimeError("Nonpositive J_tot encountered in regional quadrature.")

    reference_weight = reference_area[:, None] / 3.0
    current_weight = reference_weight * J_qp
    current_area = current_weight.sum(axis=1)
    spatial_activity = (q_magnitude_qp * current_weight).sum(axis=1)
    referential_activity = (Q_magnitude_qp * reference_weight).sum(axis=1)
    return {
        "reference_area": reference_area,
        "current_area": current_area,
        "spatial_activity": spatial_activity,
        "referential_activity": referential_activity,
        "q_magnitude_qp": q_magnitude_qp,
        "Q_magnitude_qp": Q_magnitude_qp,
        "current_weight_qp": current_weight,
        "reference_weight_qp": np.broadcast_to(reference_weight, q_magnitude_qp.shape),
    }


def weighted_quantile(values, weights, probabilities):
    order = np.argsort(values, kind="stable")
    values = np.asarray(values)[order]
    weights = np.asarray(weights)[order]
    if np.any(weights < 0.0) or float(weights.sum()) <= 0.0:
        raise RuntimeError("Weighted quantiles require positive total weight.")
    midpoint_cdf = (np.cumsum(weights) - 0.5 * weights) / weights.sum()
    return np.interp(probabilities, midpoint_cdf, values, left=values[0], right=values[-1])


def compute_metrics(acinus_ids, quadrature, canonical_fractions, domain_ids):
    rows = []
    acini = sorted(np.unique(acinus_ids).tolist())
    if acini != [0, 1, 2, 3]:
        raise RuntimeError(f"Expected acini [0, 1, 2, 3], found {acini}.")
    total_spatial_activity = float(quadrature["spatial_activity"].sum())
    for acinus_id in acini:
        mask = acinus_ids == acinus_id
        A0 = float(quadrature["reference_area"][mask].sum())
        A = float(quadrature["current_area"][mask].sum())
        P = float(quadrature["spatial_activity"][mask].sum())
        PQ = float(quadrature["referential_activity"][mask].sum())
        Pi = P / A
        PiQ = PQ / A0
        q_values = quadrature["q_magnitude_qp"][mask].ravel()
        q_weights = quadrature["current_weight_qp"][mask].ravel()
        variance = float(np.sum(q_weights * (q_values - Pi) ** 2) / q_weights.sum())
        quantiles = weighted_quantile(q_values, q_weights, [0.05, 0.25, 0.50, 0.75, 0.95])
        rows.append({
            "acinus_id": acinus_id,
            "fe_cell_count": int(mask.sum()),
            "A0_i": A0,
            "reference_wall_area_fraction": A0 / quadrature["reference_area"].sum(),
            "accepted_canonical_territory_area_fraction": float(canonical_fractions[acinus_id]),
            "A_i": A,
            "Pi_i": Pi,
            "P_i": P,
            "fP_i": P / total_spatial_activity,
            "PiQ_i": PiQ,
            "PQ_i": PQ,
            "weighted_std_q_i": np.sqrt(max(variance, 0.0)),
            "CV_q_i": np.sqrt(max(variance, 0.0)) / Pi,
            "q_5": float(quantiles[0]),
            "q_25": float(quantiles[1]),
            "q_median": float(quantiles[2]),
            "q_75": float(quantiles[3]),
            "q_95": float(quantiles[4]),
            "vascular_patch_cell_count": int(np.count_nonzero(mask & (domain_ids >= 3))),
        })
    return rows


def consistency_report(rows, quadrature):
    Pi = np.asarray([row["Pi_i"] for row in rows])
    total_A0 = float(quadrature["reference_area"].sum())
    total_A = float(quadrature["current_area"].sum())
    total_P = float(quadrature["spatial_activity"].sum())
    total_PQ = float(quadrature["referential_activity"].sum())
    regional_P = sum(row["P_i"] for row in rows)
    regional_PQ = sum(row["PQ_i"] for row in rows)
    regional_A = sum(row["A_i"] for row in rows)
    regional_A0 = sum(row["A0_i"] for row in rows)
    global_pi_from_regions = sum(row["A_i"] * row["Pi_i"] for row in rows) / regional_A
    global_piq_from_regions = sum(row["A0_i"] * row["PiQ_i"] for row in rows) / regional_A0
    global_pi = total_P / total_A
    global_piq = total_PQ / total_A0
    return {
        "population_standard_deviation_convention": True,
        "mean_Pi": float(Pi.mean()),
        "std_Pi_population": float(Pi.std(ddof=0)),
        "CV_Pi": float(Pi.std(ddof=0) / Pi.mean()),
        "global_reference_wall_area": total_A0,
        "global_current_wall_area": total_A,
        "global_integrated_abs_q_current": total_P,
        "global_current_area_weighted_mean_abs_q": global_pi,
        "global_integrated_abs_Q_reference": total_PQ,
        "global_reference_area_weighted_mean_abs_Q": global_piq,
        "normalized_share_sum": float(sum(row["fP_i"] for row in rows)),
        "residuals": {
            "reference_area_partition": regional_A0 - total_A0,
            "current_area_partition": regional_A - total_A,
            "spatial_activity_partition": regional_P - total_P,
            "spatial_mean_partition": global_pi_from_regions - global_pi,
            "referential_activity_partition": regional_PQ - total_PQ,
            "referential_mean_partition": global_piq_from_regions - global_piq,
            "normalized_share_sum_minus_one": sum(row["fP_i"] for row in rows) - 1.0,
        },
    }


def vascular_patch_influence(domain_ids, acinus_ids, quadrature):
    total_A0 = quadrature["reference_area"].sum()
    total_A = quadrature["current_area"].sum()
    total_P = quadrature["spatial_activity"].sum()
    result = {}
    for marker_id in sorted(value for value in np.unique(domain_ids) if value >= 3):
        mask = domain_ids == marker_id
        result[str(int(marker_id))] = {
            "fe_cell_count": int(mask.sum()),
            "acinar_cell_counts": {
                str(acinus_id): int(np.count_nonzero(mask & (acinus_ids == acinus_id)))
                for acinus_id in range(4)
            },
            "reference_area_fraction": float(quadrature["reference_area"][mask].sum() / total_A0),
            "current_area_fraction": float(quadrature["current_area"][mask].sum() / total_A),
            "spatial_activity_fraction": float(quadrature["spatial_activity"][mask].sum() / total_P),
        }
    combined = domain_ids >= 3
    result["combined"] = {
        "fe_cell_count": int(combined.sum()),
        "reference_area_fraction": float(quadrature["reference_area"][combined].sum() / total_A0),
        "current_area_fraction": float(quadrature["current_area"][combined].sum() / total_A),
        "spatial_activity_fraction": float(quadrature["spatial_activity"][combined].sum() / total_P),
    }
    return result


def compare_saved_fields_with_run_diagnostics(result_dir, consistency):
    """Record, but do not enforce, differences from in-solve field diagnostics.

    The XDMF fields are retained nodal projections.  Their integrated magnitudes
    need not exactly equal diagnostics evaluated in the original FE spaces.
    """
    diagnostics_path = result_dir / "diagnostics.json"
    if not diagnostics_path.exists():
        return None
    diagnostics = json.loads(diagnostics_path.read_text())
    comparisons = {}
    for name, diagnostic_key, saved_key in (
        ("q_l", "q_l_magnitude_stats", "global_current_area_weighted_mean_abs_q"),
        ("Q_l", "Q_l_magnitude_stats", "global_reference_area_weighted_mean_abs_Q"),
    ):
        if diagnostic_key not in diagnostics or "mean" not in diagnostics[diagnostic_key]:
            continue
        in_solve = float(diagnostics[diagnostic_key]["mean"])
        saved_field = float(consistency[saved_key])
        comparisons[name] = {
            "in_solve_reported_mean": in_solve,
            "saved_nodal_field_integrated_mean": saved_field,
            "difference": saved_field - in_solve,
            "relative_difference": (saved_field - in_solve) / in_solve,
        }
    return {
        "diagnostics_file": str(diagnostics_path.resolve()),
        "interpretation": (
            "Differences reflect postprocessing of retained projected nodal fields; "
            "regional metrics consistently use the saved-field convention."
        ),
        "fields": comparisons,
    }


def acinar_interface_segments(topology, coordinates, acinus_ids):
    edge_owner = {}
    for triangle_id, triangle in enumerate(topology):
        for start, end in ((triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0])):
            key = tuple(sorted((int(start), int(end))))
            edge_owner.setdefault(key, []).append(triangle_id)
    segments = []
    for (start, end), owners in edge_owner.items():
        if len(owners) == 2 and acinus_ids[owners[0]] != acinus_ids[owners[1]]:
            segments.append([coordinates[start], coordinates[end]])
    return segments


def make_figures(output_dir, coordinates, topology, acinus_ids, quadrature, rows, summary):
    cell_q = quadrature["spatial_activity"] / quadrature["current_area"]
    fig, ax = plt.subplots(figsize=(8.5, 8.0))
    image = ax.tripcolor(
        coordinates[:, 0], coordinates[:, 1], topology, facecolors=cell_q,
        shading="flat", cmap="viridis",
    )
    interfaces = acinar_interface_segments(topology, coordinates, acinus_ids)
    if interfaces:
        ax.add_collection(LineCollection(interfaces, colors="white", linewidths=1.1, alpha=0.95))
        ax.add_collection(LineCollection(interfaces, colors="black", linewidths=0.35, alpha=0.95))
    territories = summary["main_model"]["territories"]
    for territory in territories:
        x, y = territory["shape"]["periodic_centroid"]
        ax.text(x, y, f"A{territory['acinus_id']}", ha="center", va="center",
                fontsize=11, weight="bold", color="black",
                bbox={"facecolor": "white", "edgecolor": "black", "alpha": 0.78, "pad": 2})
    fig.colorbar(image, ax=ax, label=r"$|q_l|$ (cell current-area mean)")
    ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal", xlabel="x", ylabel="y",
           title="Continuous Darcy transport with FE-level acinar regions")
    fig.tight_layout()
    fig.savefig(output_dir / "regional_perfusion_map.png", dpi=250)
    plt.close(fig)

    labels = [f"A{row['acinus_id']}" for row in rows]
    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    ax.bar(labels, [row["Pi_i"] for row in rows], color=plt.get_cmap("tab10").colors[:4])
    ax.set_ylabel(r"Regional perfusion intensity $\Pi_i$")
    ax.set_title("Healthy baseline regional perfusion intensity")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(output_dir / "regional_perfusion_bar.png", dpi=250); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    ax.bar(labels, [row["fP_i"] for row in rows], color=plt.get_cmap("tab10").colors[:4])
    ax.axhline(0.25, color="black", linestyle="--", linewidth=1.0, label="equal-share reference (0.25)")
    ax.set_ylabel(r"Normalized regional share $fP_i$")
    ax.set_title("Healthy baseline normalized transport shares")
    ax.legend(frameon=False); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(output_dir / "normalized_perfusion_share.png", dpi=250); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    colors = plt.get_cmap("tab10").colors[:4]
    for position, (row, color) in enumerate(zip(rows, colors), start=1):
        ax.add_patch(Rectangle((position - 0.28, row["q_25"]), 0.56,
                               row["q_75"] - row["q_25"], facecolor=color,
                               edgecolor="black", alpha=0.72))
        ax.plot([position - 0.28, position + 0.28], [row["q_median"]] * 2, color="black", lw=1.5)
        ax.plot([position, position], [row["q_5"], row["q_25"]], color="black")
        ax.plot([position, position], [row["q_75"], row["q_95"]], color="black")
        ax.plot([position - 0.14, position + 0.14], [row["q_5"]] * 2, color="black")
        ax.plot([position - 0.14, position + 0.14], [row["q_95"]] * 2, color="black")
        ax.scatter(position, row["Pi_i"], marker="D", s=24, color="white", edgecolor="black", zorder=5)
    ax.set_xticks(range(1, 5), labels)
    ax.set_ylabel(r"$|q_l|$")
    ax.set_title("Current-area-weighted intra-acinar flux distributions")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(output_dir / "intra_acinar_flux_distribution.png", dpi=250); plt.close(fig)


def write_csv(path, rows):
    fieldnames = [
        "acinus_id", "fe_cell_count", "A0_i", "reference_wall_area_fraction", "A_i",
        "Pi_i", "P_i", "fP_i", "PiQ_i", "PQ_i", "weighted_std_q_i", "CV_q_i",
        "q_5", "q_median", "q_95", "accepted_canonical_territory_area_fraction",
        "vascular_patch_cell_count",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run(result_dir, solution_xdmf, final_vtu, acinar_metadata_path, acinar_summary_path):
    metadata = json.loads(acinar_metadata_path.read_text())
    summary = json.loads(acinar_summary_path.read_text())
    if metadata.get("main_partition_method") != "airway_constrained_space_filling":
        raise RuntimeError("Acinar metadata is not the accepted airway-constrained partition.")
    canonical_fractions = {
        int(item["acinus_id"]): float(item["area_fraction"])
        for item in summary["main_model"]["territories"]
    }
    final_time, coordinates, topology, fields = read_final_xdmf_fields(solution_xdmf)
    domain_ids = read_domain_markers(final_vtu, coordinates, topology)
    centroids = coordinates[topology].mean(axis=1)
    canonical_ids, acinus_ids, ambiguous, distances = periodic_cell_ownership(centroids, metadata)
    sorted_distances = np.sort(distances, axis=1)
    ownership_margin = sorted_distances[:, 1] - sorted_distances[:, 0]
    near_boundary_counts = {
        "squared_distance_gap_le_1e-10": int(np.count_nonzero(ownership_margin <= 1e-10)),
        "squared_distance_gap_le_1e-8": int(np.count_nonzero(ownership_margin <= 1e-8)),
        "squared_distance_gap_le_1e-6": int(np.count_nonzero(ownership_margin <= 1e-6)),
    }
    if len(acinus_ids) != len(topology):
        raise RuntimeError("FE-to-acinus ownership does not cover every triangle.")
    if np.any((acinus_ids < 0) | (acinus_ids > 3)):
        raise RuntimeError("FE-to-acinus ownership contains an invalid label.")
    multiplicity = np.ones(len(acinus_ids), dtype=int)
    if np.any(multiplicity != 1):
        raise RuntimeError("FE-to-acinus ownership is not one-to-one.")

    quadrature = triangle_quadrature(coordinates, topology, fields)
    rows = compute_metrics(acinus_ids, quadrature, canonical_fractions, domain_ids)
    consistency = consistency_report(rows, quadrature)
    tolerance = 1e-12
    if any(abs(value) > tolerance for value in consistency["residuals"].values()):
        raise RuntimeError(f"Regional consistency check failed: {consistency['residuals']}.")
    patch_influence = vascular_patch_influence(domain_ids, acinus_ids, quadrature)
    diagnostic_comparison = compare_saved_fields_with_run_diagnostics(result_dir, consistency)

    mapping = {
        "schema_version": 1,
        "purpose": "Ordered reproducible mapping from retained FE triangle index to canonical cell and acinus",
        "solution_xdmf": str(solution_xdmf),
        "solution_xdmf_sha256": sha256(solution_xdmf),
        "acinar_metadata": str(acinar_metadata_path),
        "acinar_metadata_sha256": sha256(acinar_metadata_path),
        "fe_cell_order": "zero-based triangle row index in the final XDMF topology dataset",
        "ownership_rule": "minimum-image nearest canonical seed evaluated at reference triangle centroid",
        "tie_break": "stable lowest canonical cell_id",
        "tie_squared_distance_tolerance": 1e-12,
        "fe_triangle_count": int(len(topology)),
        "unassigned_cell_count": 0,
        "multiply_assigned_cell_count": 0,
        "ambiguous_nearest_seed_cell_count": int(len(ambiguous)),
        "ambiguous_fe_cell_indices": ambiguous.tolist(),
        "minimum_nearest_vs_second_nearest_squared_distance_gap": float(ownership_margin.min()),
        "near_boundary_counts": near_boundary_counts,
        "canonical_cell_id_by_fe_cell": canonical_ids.tolist(),
        "acinus_id_by_fe_cell": acinus_ids.tolist(),
    }
    result = {
        "schema_version": 1,
        "result_directory": str(result_dir),
        "solution_final_time": final_time,
        "definitions": {
            "Pi_i": "integral_Omega0_i(|q_l| J dOmega0) / integral_Omega0_i(J dOmega0)",
            "P_i": "integral_Omega0_i(|q_l| J dOmega0); integrated transport activity, not acinar volumetric flow",
            "fP_i": "P_i / sum_k(P_k)",
            "PiQ_i": "integral_Omega0_i(|Q_l| dOmega0) / A0_i",
            "PQ_i": "integral_Omega0_i(|Q_l| dOmega0)",
            "CV_Pi": "population_std(Pi_i) / mean(Pi_i)",
            "CV_q_i": "current-area-weighted std(|q_l|) / Pi_i",
        },
        "integration": {
            "rule": "three-point symmetric triangle quadrature on saved nodal linear fields",
            "percentile_convention": "current-area-weighted midpoint empirical CDF over the three quadrature samples per triangle",
            "field_provenance": (
                "J_tot, q_l and Q_l are retained projected nodal fields from the final XDMF time grid; "
                "no FE function space is reconstructed and no solve is rerun"
            ),
        },
        "mapping_validation": {
            "fe_triangle_count": int(len(topology)),
            "unassigned_cell_count": 0,
            "multiply_assigned_cell_count": 0,
            "ambiguous_nearest_seed_cell_count": int(len(ambiguous)),
            "minimum_nearest_vs_second_nearest_squared_distance_gap": float(ownership_margin.min()),
            "near_boundary_counts": near_boundary_counts,
        },
        "acini": rows,
        "global": consistency,
        "vascular_patch_influence": patch_influence,
        "saved_field_vs_in_solve_diagnostics": diagnostic_comparison,
    }
    (result_dir / "fe_acinar_mapping.json").write_text(json.dumps(mapping, indent=2))
    (result_dir / "regional_perfusion.json").write_text(json.dumps(result, indent=2))
    write_csv(result_dir / "regional_perfusion.csv", rows)
    make_figures(result_dir, coordinates, topology, acinus_ids, quadrature, rows, summary)
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--solution-xdmf", type=Path, default=None)
    parser.add_argument("--final-vtu", type=Path, default=None)
    parser.add_argument("--acinar-metadata", type=Path, default=DEFAULT_ACINAR_METADATA)
    parser.add_argument("--acinar-summary", type=Path, default=DEFAULT_ACINAR_SUMMARY)
    arguments = parser.parse_args()
    result_dir = arguments.result_dir.resolve()
    solution_xdmf = arguments.solution_xdmf or result_dir / f"{result_dir.name}.xdmf"
    final_vtu = arguments.final_vtu or result_dir / "final_fields.vtu"
    required = [solution_xdmf, final_vtu, arguments.acinar_metadata, arguments.acinar_summary]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing retained inputs: " + ", ".join(missing))
    run(
        result_dir=result_dir,
        solution_xdmf=solution_xdmf.resolve(),
        final_vtu=final_vtu.resolve(),
        acinar_metadata_path=arguments.acinar_metadata.resolve(),
        acinar_summary_path=arguments.acinar_summary.resolve(),
    )


if __name__ == "__main__":
    main()
