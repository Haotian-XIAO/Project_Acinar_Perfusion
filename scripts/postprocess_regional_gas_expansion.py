#!/usr/bin/env python3
"""Regional gas-pore expansion and regional-pg validation postprocessing.

This reads retained XDMF/HDF5 fields.  It does not reconstruct or rerun the FE
problem.  Gas pores are measured from their oriented physical gas-boundary
edges after periodic unwrapping about the owning canonical Voronoi seed.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

import postprocess_regional_perfusion as regional


ROOT = Path(__file__).resolve().parents[1]
VALIDATION_ROOT = ROOT / "results" / "regional_pg_validation"
ACINAR_METADATA = ROOT / "results" / "acinar_partition" / "periodic_acinar_metadata.json"
BASELINE_DIR = ROOT / "results" / "vascular_baseline"
BASELINE_XDMF = BASELINE_DIR / "vascular_baseline.xdmf"
CASES = {
    "regional_zero": VALIDATION_ROOT / "regional_zero",
    "global_uniform_002": VALIDATION_ROOT / "global_uniform_002",
    "regional_uniform_002": VALIDATION_ROOT / "regional_uniform_002",
    "A2_only_002": VALIDATION_ROOT / "A2_only_002",
}


def parse_hdf_reference(text, xdmf_path):
    filename, dataset = text.strip().split(":", 1)
    return xdmf_path.parent / filename, dataset


def read_final_fields(xdmf_path):
    document = ET.parse(xdmf_path).getroot()
    grids = document.findall('.//Grid[@GridType="Uniform"]')
    if not grids:
        raise RuntimeError(f"No time grids in {xdmf_path}.")
    grid = max(grids, key=lambda item: float(item.find("Time").attrib["Value"]))

    def read(item):
        h5_path, dataset = parse_hdf_reference(item.text, xdmf_path)
        with h5py.File(h5_path, "r") as stream:
            return np.asarray(stream[dataset])

    coordinates = read(grid.find("Geometry/DataItem"))[:, :2]
    topology = read(grid.find("Topology/DataItem")).astype(int)
    wanted = {"U_tot", "J_tot", "pl_tot", "q_l", "Q_l"}
    fields = {}
    for attribute in grid.findall("Attribute"):
        name = attribute.attrib["Name"]
        if name in wanted and attribute.attrib.get("Center") == "Node":
            fields[name] = read(attribute.find("DataItem"))
    missing = wanted - set(fields)
    if missing:
        raise RuntimeError(f"Missing retained fields in {xdmf_path}: {sorted(missing)}")
    fields["U_tot"] = fields["U_tot"][:, :2]
    fields["q_l"] = fields["q_l"][:, :2]
    fields["Q_l"] = fields["Q_l"][:, :2]
    fields["J_tot"] = fields["J_tot"].reshape(-1)
    fields["pl_tot"] = fields["pl_tot"].reshape(-1)
    return coordinates, topology, fields


def oriented_boundary_edges(coordinates, topology):
    """Return boundary edges oriented with the solid triangle on their left."""
    incidence = {}
    for triangle_id, triangle in enumerate(topology):
        points = coordinates[triangle]
        signed_twice_area = float(np.cross(points[1] - points[0], points[2] - points[0]))
        if signed_twice_area == 0.0:
            raise RuntimeError(f"Zero-area triangle {triangle_id}.")
        directed = [(triangle[0], triangle[1]), (triangle[1], triangle[2]),
                    (triangle[2], triangle[0])]
        if signed_twice_area < 0.0:
            directed = [(end, start) for start, end in directed]
        for start, end in directed:
            key = tuple(sorted((int(start), int(end))))
            incidence.setdefault(key, []).append((int(start), int(end), triangle_id))
    invalid = [key for key, owners in incidence.items() if len(owners) > 2]
    if invalid:
        raise RuntimeError(f"Non-manifold FE edges encountered: {len(invalid)}.")
    return [owners[0] for owners in incidence.values() if len(owners) == 1]


def gas_edges(coordinates, topology, boundary_tolerance=1e-8):
    xmin, ymin = coordinates.min(axis=0)
    xmax, ymax = coordinates.max(axis=0)
    gas, outer = [], []
    for start, end, triangle_id in oriented_boundary_edges(coordinates, topology):
        midpoint = 0.5 * (coordinates[start] + coordinates[end])
        on_outer_cut = (
            abs(midpoint[0] - xmin) <= boundary_tolerance
            or abs(midpoint[0] - xmax) <= boundary_tolerance
            or abs(midpoint[1] - ymin) <= boundary_tolerance
            or abs(midpoint[1] - ymax) <= boundary_tolerance
        )
        (outer if on_outer_cut else gas).append((start, end, triangle_id))
    return gas, outer


def deformation_gradient_from_config(case_dir):
    config = json.loads((case_dir / "run_config.json").read_text())
    gradient = np.asarray(config.get("macroscopic_displacement_gradient", [[0, 0], [0, 0]]), dtype=float)
    if gradient.shape != (2, 2):
        raise RuntimeError("macroscopic_displacement_gradient must be 2x2.")
    return np.eye(2) + gradient


def regional_gas_areas(coordinates, topology, displacement, metadata, Fbar):
    edges, outer = gas_edges(coordinates, topology)
    midpoints = np.asarray([0.5 * (coordinates[a] + coordinates[b]) for a, b, _ in edges])
    cell_ids, acinus_ids, ambiguous, distances = regional.periodic_cell_ownership(midpoints, metadata)
    if len(ambiguous):
        raise RuntimeError(f"Ambiguous gas-edge ownership for {len(ambiguous)} edges.")
    cells = sorted(metadata["cells"], key=lambda item: int(item["cell_id"]))
    seeds = np.asarray([item["seed_coordinate"] for item in cells], dtype=float)
    cell_acini = np.asarray([int(item["acinus_id"]) for item in cells], dtype=int)
    lattice = np.column_stack([
        np.asarray(metadata["periodic_cell"]["a1"], dtype=float),
        np.asarray(metadata["periodic_cell"]["a2"], dtype=float),
    ])
    inverse_lattice = np.linalg.inv(lattice)
    bounds = np.asarray(metadata["periodic_cell"]["bounds"], dtype=float)
    origin = bounds[:2]
    macro_gradient = Fbar - np.eye(2)
    perturbation = displacement - (macro_gradient @ (coordinates - origin).T).T
    fractional_nodes = (inverse_lattice @ (coordinates - origin).T).T
    wrapped_nodes = fractional_nodes - np.floor(fractional_nodes + 1e-10)
    periodic_groups = {}
    for node_id, wrapped in enumerate(wrapped_nodes):
        key = tuple(np.round(wrapped, 10))
        periodic_groups.setdefault(key, []).append(node_id)
    canonical_perturbation = perturbation.copy()
    for node_ids in periodic_groups.values():
        canonical_perturbation[node_ids] = perturbation[node_ids].mean(axis=0)
    reference_signed = np.zeros(len(cells), dtype=float)
    current_signed = np.zeros(len(cells), dtype=float)
    reference_closure = np.zeros((len(cells), 2), dtype=float)
    current_closure = np.zeros((len(cells), 2), dtype=float)

    def unwrap(point, seed):
        fractional = inverse_lattice @ (point - seed)
        fractional -= np.round(fractional)
        return seed + lattice @ fractional

    for (start, end, _), cell_id in zip(edges, cell_ids):
        seed = seeds[cell_id]
        start_ref = unwrap(coordinates[start], seed)
        end_ref = unwrap(coordinates[end], seed)
        start_cur = origin + Fbar @ (start_ref - origin) + canonical_perturbation[start]
        end_cur = origin + Fbar @ (end_ref - origin) + canonical_perturbation[end]
        reference_signed[cell_id] += 0.5 * float(np.cross(start_ref, end_ref))
        current_signed[cell_id] += 0.5 * float(np.cross(start_cur, end_cur))
        reference_closure[cell_id] += end_ref - start_ref
        current_closure[cell_id] += end_cur - start_cur

    closure_ref = np.linalg.norm(reference_closure, axis=1)
    closure_cur = np.linalg.norm(current_closure, axis=1)
    if closure_ref.max() > 1e-8 or closure_cur.max() > 1e-8:
        raise RuntimeError(
            f"Periodic gas-pore loops did not close: reference={closure_ref.max():.3e}, "
            f"current={closure_cur.max():.3e}."
        )
    reference_cell_area = np.abs(reference_signed)
    current_cell_area = np.abs(current_signed)
    if np.any(reference_cell_area <= 0.0) or np.any(current_cell_area <= 0.0):
        raise RuntimeError("A canonical gas pore has nonpositive polygon area.")
    rows = []
    periodic_deformed_coordinates = (
        origin + (Fbar @ (coordinates - origin).T).T + canonical_perturbation
    )
    deformed_triangles = periodic_deformed_coordinates[topology]
    geometric_wall_area = float(
        0.5 * np.abs(np.cross(
            deformed_triangles[:, 1] - deformed_triangles[:, 0],
            deformed_triangles[:, 2] - deformed_triangles[:, 0],
        )).sum()
    )
    for acinus_id in range(4):
        mask = cell_acini == acinus_id
        rows.append({
            "acinus_id": acinus_id,
            "canonical_pore_count": int(mask.sum()),
            "reference_mesh_gas_area": float(reference_cell_area[mask].sum()),
            "current_gas_area": float(current_cell_area[mask].sum()),
            "gas_boundary_facet_count": int(np.count_nonzero(acinus_ids == acinus_id)),
        })
    return rows, {
        "gas_boundary_facet_count": len(edges),
        "outer_periodic_cut_facet_count": len(outer),
        "ambiguous_edge_count": len(ambiguous),
        "max_reference_loop_closure_residual": float(closure_ref.max()),
        "max_current_loop_closure_residual": float(closure_cur.max()),
        "periodic_displacement_canonicalization": (
            "U_tilde = U_tot - (Fbar-I)(X-origin) was averaged over periodically "
            "equivalent boundary nodes before deforming unwrapped pore loops"
        ),
        "reference_gas_area": float(reference_cell_area.sum()),
        "current_gas_area": float(current_cell_area.sum()),
        "periodically_canonicalized_geometric_wall_area": geometric_wall_area,
        "per_cell_reference_area": reference_cell_area.tolist(),
        "per_cell_current_area": current_cell_area.tolist(),
    }


def regional_mechanics(coordinates, topology, fields, metadata):
    centroids = coordinates[topology].mean(axis=1)
    _, acinus_ids, ambiguous, _ = regional.periodic_cell_ownership(centroids, metadata)
    if len(ambiguous):
        raise RuntimeError(f"Ambiguous FE-cell ownership for {len(ambiguous)} cells.")
    vertices = coordinates[topology]
    area = 0.5 * np.abs(np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0]))
    barycentric = np.asarray([
        [2 / 3, 1 / 6, 1 / 6], [1 / 6, 2 / 3, 1 / 6], [1 / 6, 1 / 6, 2 / 3]
    ])
    J_qp = np.einsum("qa,na->nq", barycentric, fields["J_tot"][topology])
    U_qp = np.einsum("qa,nad->nqd", barycentric, fields["U_tot"][topology])
    U_mag = np.linalg.norm(U_qp, axis=2)
    reference_weight = area[:, None] / 3.0
    current_weight = reference_weight * J_qp
    rows = []
    for acinus_id in range(4):
        mask = acinus_ids == acinus_id
        A0 = float(area[mask].sum())
        A = float(current_weight[mask].sum())
        rows.append({
            "acinus_id": acinus_id,
            "reference_wall_area": A0,
            "current_wall_area": A,
            "mean_J": A / A0,
            "min_J": float(J_qp[mask].min()),
            "max_J": float(J_qp[mask].max()),
            "mean_displacement_magnitude": float(
                np.sum(U_mag[mask] * current_weight[mask]) / current_weight[mask].sum()
            ),
            "max_displacement_magnitude": float(U_mag[mask].max()),
        })
    return rows, acinus_ids


def read_final_void_qoi(qoi_path):
    if not qoi_path.exists():
        return None
    lines = [line for line in qoi_path.read_text().splitlines() if line.strip()]
    names = lines[0].lstrip("#").split()
    values = [float(value) for value in lines[-1].split()]
    return dict(zip(names, values)).get("vf")


def process_case(case_name, case_dir, solution_xdmf, reference_current_areas, write_output=True):
    metadata = json.loads(ACINAR_METADATA.read_text())
    coordinates, topology, fields = read_final_fields(solution_xdmf)
    Fbar = deformation_gradient_from_config(case_dir)
    gas_rows, gas_validation = regional_gas_areas(
        coordinates, topology, fields["U_tot"], metadata, Fbar
    )
    mechanics_rows, acinus_ids = regional_mechanics(coordinates, topology, fields, metadata)
    for gas_row, mechanics_row in zip(gas_rows, mechanics_rows):
        acinus_id = gas_row["acinus_id"]
        reference = float(reference_current_areas[acinus_id])
        current = gas_row["current_gas_area"]
        gas_row.update({
            "accepted_pg0_reference_gas_area": reference,
            "Delta_Ag_i": current - reference,
            "E_i": (current - reference) / reference,
        })
        gas_row.update({key: value for key, value in mechanics_row.items() if key != "acinus_id"})
    delta_total = sum(row["Delta_Ag_i"] for row in gas_rows)
    for row in gas_rows:
        row["fE_i"] = row["Delta_Ag_i"] / delta_total if delta_total > 1e-14 else None

    qoi_basename = solution_xdmf.with_suffix("").name + "-qois.dat"
    qoi_void = read_final_void_qoi(case_dir / qoi_basename)
    gas_validation["global_void_qoi"] = qoi_void
    gas_validation["current_gas_area_minus_global_void_qoi"] = (
        gas_validation["current_gas_area"] - qoi_void if qoi_void is not None else None
    )
    result = {
        "case_name": case_name,
        "solution_xdmf": str(solution_xdmf.resolve()),
        "definitions": {
            "A_g_i": "sum of periodically unwrapped polygon areas of canonical gas pores owned by acinus i",
            "Delta_Ag_i": "A_g_i(case) - A_g_i(accepted pg=0 healthy reference)",
            "E_i": "Delta_Ag_i / A_g_i(accepted pg=0 healthy reference)",
            "interpretation": "regional alveolar gas-area expansion surrogate; not airflow or ventilation rate",
        },
        "periodic_area_method": (
            "Oriented exterior gas edges are classified by periodic nearest canonical seed. "
            "Each edge endpoint is minimum-image unwrapped about that seed; its deformed image "
            "uses x+U plus Fbar times the lattice shift. Polygon area is the oriented boundary integral."
        ),
        "validation": gas_validation,
        "acini": gas_rows,
        "global": {
            "mean_displacement": float(np.linalg.norm(fields["U_tot"], axis=1).mean()),
            "max_displacement": float(np.linalg.norm(fields["U_tot"], axis=1).max()),
            "min_J": float(fields["J_tot"].min()),
            "mean_J": float(fields["J_tot"].mean()),
            "max_J": float(fields["J_tot"].max()),
            "global_current_wall_area": float(sum(row["current_wall_area"] for row in gas_rows)),
            "periodically_canonicalized_geometric_wall_area": gas_validation[
                "periodically_canonicalized_geometric_wall_area"
            ],
        },
    }
    if write_output:
        (case_dir / "regional_gas_expansion.json").write_text(json.dumps(result, indent=2))
    return result, coordinates, topology, fields, acinus_ids


def normalized_error(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    difference = b - a
    denominator = max(float(np.linalg.norm(a.ravel())), float(np.linalg.norm(b.ravel())), 1e-30)
    return {
        "absolute_l2": float(np.linalg.norm(difference.ravel())),
        "relative_l2": float(np.linalg.norm(difference.ravel()) / denominator),
        "max_absolute": float(np.max(np.abs(difference))),
    }


def compare_case_fields(case_a, xdmf_a, case_b, xdmf_b):
    xa, ta, fa = read_final_fields(xdmf_a)
    xb, tb, fb = read_final_fields(xdmf_b)
    if not np.array_equal(ta, tb) or not np.allclose(xa, xb, rtol=0, atol=1e-14):
        raise RuntimeError("Compared cases do not share identical reference meshes.")
    return {name: normalized_error(fa[name], fb[name]) for name in ("U_tot", "pl_tot", "q_l", "J_tot")}


def pyvista_grid(coordinates, topology, fields):
    points = np.column_stack([coordinates, np.zeros(len(coordinates))])
    cells = np.column_stack([np.full(len(topology), 3), topology]).astype(np.int64).ravel()
    grid = pv.UnstructuredGrid(cells, np.full(len(topology), pv.CellType.TRIANGLE, np.uint8), points)
    for name, values in fields.items():
        values = np.asarray(values)
        if values.ndim == 2 and values.shape[1] == 2:
            values = np.column_stack([values, np.zeros(len(values))])
        grid.point_data[name] = values
    return grid


def finish(plotter, path):
    plotter.view_xy(); plotter.camera.parallel_projection = True; plotter.reset_camera()
    plotter.screenshot(path); plotter.close()


def scalar_figure(grid, scalar, title, scalar_title, path, cmap="viridis"):
    plotter = pv.Plotter(off_screen=True, window_size=(1450, 1150))
    plotter.set_background("white"); plotter.add_text(title, color="black", font_size=13)
    plotter.add_mesh(grid, scalars=scalar, cmap=cmap, show_edges=False,
                     scalar_bar_args={"title": scalar_title})
    finish(plotter, path)


def acinar_interface_polydata(coordinates, topology, acinus_ids):
    owners = {}
    for triangle_id, triangle in enumerate(topology):
        for start, end in ((triangle[0], triangle[1]), (triangle[1], triangle[2]),
                           (triangle[2], triangle[0])):
            owners.setdefault(tuple(sorted((int(start), int(end)))), []).append(triangle_id)
    interfaces = [edge for edge, adjacent in owners.items()
                  if len(adjacent) == 2 and acinus_ids[adjacent[0]] != acinus_ids[adjacent[1]]]
    if not interfaces:
        return None
    points = np.column_stack([coordinates, np.full(len(coordinates), 0.003)])
    lines = np.asarray([[2, start, end] for start, end in interfaces], dtype=np.int64).ravel()
    return pv.PolyData(points, lines=lines)


def make_A2_figures(case_dir, coordinates, topology, fields, acinus_ids, expansion):
    deformed = coordinates + fields["U_tot"]
    plot_fields = dict(fields)
    plot_fields["displacement_magnitude"] = np.linalg.norm(fields["U_tot"], axis=1)
    plot_fields["q_l_magnitude"] = np.linalg.norm(fields["q_l"], axis=1)
    deformed_grid = pyvista_grid(deformed, topology, plot_fields)
    reference_grid = pyvista_grid(coordinates, topology, {})
    reference_edges = reference_grid.extract_feature_edges(
        boundary_edges=True, feature_edges=False, manifold_edges=False, non_manifold_edges=False)
    deformed_edges = deformed_grid.extract_feature_edges(
        boundary_edges=True, feature_edges=False, manifold_edges=False, non_manifold_edges=False)
    plotter = pv.Plotter(off_screen=True, window_size=(1450, 1150)); plotter.set_background("white")
    plotter.add_text("A2-only gas pressure: reference/deformed geometry", color="black", font_size=13)
    plotter.add_mesh(reference_edges, color="#777777", line_width=2, opacity=.65, label="reference")
    plotter.add_mesh(deformed_edges, color="#d62728", line_width=2, opacity=.85, label="deformed")
    plotter.add_legend(bcolor="white"); finish(plotter, case_dir / "A2_only_deformation_overlay.png")
    scalar_figure(deformed_grid, "displacement_magnitude", "A2-only gas pressure: displacement", "|u|",
                  case_dir / "A2_only_displacement.png", "magma")
    scalar_figure(deformed_grid, "J_tot", "A2-only gas pressure: Jacobian", "J",
                  case_dir / "A2_only_J_field.png", "coolwarm")
    scalar_figure(deformed_grid, "pl_tot", "A2-only gas pressure: liquid pressure", "p_l",
                  case_dir / "A2_only_pressure_field.png", "coolwarm")
    scalar_figure(deformed_grid, "q_l_magnitude", "A2-only gas pressure: spatial Darcy flux", "|q_l|",
                  case_dir / "A2_only_flux_magnitude.png")

    cell_labels = np.asarray(acinus_ids, dtype=float)
    labeled = deformed_grid.copy(); labeled.cell_data["acinus_id"] = cell_labels
    plotter = pv.Plotter(off_screen=True, window_size=(1450, 1150)); plotter.set_background("white")
    plotter.add_text("A2-only gas pressure: regional perfusion", color="black", font_size=13)
    plotter.add_mesh(labeled, scalars="q_l_magnitude", cmap="viridis", show_edges=False,
                     scalar_bar_args={"title": "|q_l|"})
    interfaces = acinar_interface_polydata(deformed, topology, acinus_ids)
    if interfaces is not None:
        plotter.add_mesh(interfaces, color="white", line_width=4, opacity=.95)
        plotter.add_mesh(interfaces, color="black", line_width=2, opacity=.95)
    centers = deformed[topology].mean(axis=1)
    label_points = np.asarray([[*centers[acinus_ids == i].mean(axis=0), .003] for i in range(4)])
    plotter.add_point_labels(label_points, [f"A{i}" for i in range(4)], font_size=15,
                             text_color="black", shape_color="white", shape_opacity=.8)
    finish(plotter, case_dir / "A2_only_regional_perfusion.png")

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    values = [row["E_i"] for row in expansion["acini"]]
    ax.bar([f"A{i}" for i in range(4)], values, color=plt.get_cmap("tab10").colors[:4])
    ax.axhline(0, color="black", lw=.8); ax.set_ylabel(r"Regional gas-area expansion $E_i$")
    ax.set_title("A2-only prescribed gas pressure (model units: 0.02)")
    ax.grid(axis="y", alpha=.25); fig.tight_layout()
    fig.savefig(case_dir / "A2_only_regional_expansion.png", dpi=250); plt.close(fig)


def case_summary(case_name, case_dir, expansion):
    perfusion = json.loads((case_dir / "regional_perfusion.json").read_text())
    pressure = json.loads((case_dir / "patch_pressures.json").read_text())
    diagnostics = json.loads((case_dir / "diagnostics.json").read_text())
    balance = json.loads((case_dir / "mass_balance.json").read_text())
    config = json.loads((case_dir / "run_config.json").read_text())
    p_by_id = {int(row["acinus_id"]): row for row in perfusion["acini"]}
    e_by_id = {int(row["acinus_id"]): row for row in expansion["acini"]}
    row = {
        "case_name": case_name,
        "convergence": bool(diagnostics["nonlinear_converged"]),
        "mass_balance_residual": float(balance["mass_balance_residual"]),
        "CV_Pi": float(perfusion["global"]["CV_Pi"]),
        "R_hyd": float(pressure["R_hyd"]),
        "mean_displacement": expansion["global"]["mean_displacement"],
        "max_displacement": expansion["global"]["max_displacement"],
        "global_current_wall_area": expansion["global"]["global_current_wall_area"],
        "min_J": expansion["global"]["min_J"],
        "mean_J": expansion["global"]["mean_J"],
        "max_J": expansion["global"]["max_J"],
        "ptilde_l_volume_average": diagnostics["ptilde_l_volume_average"],
        "lambda_p": diagnostics["lambda_p"],
    }
    mode = config.get("gas_pressure_mode", "global")
    row["global_pg_mode"] = mode == "global"
    row["regional_pg_mode"] = mode == "regional"
    pressures = config.get("regional_gas_pressures", [0.0] * 4)
    if mode == "global":
        pressures = [config.get("global_gas_pressure", 0.0)] * 4
    for i in range(4):
        row[f"pg{i}"] = float(pressures[i])
        row[f"Pi{i}"] = p_by_id[i]["Pi_i"]
        row[f"fP{i}"] = p_by_id[i]["fP_i"]
        row[f"Ag{i}"] = e_by_id[i]["current_gas_area"]
        row[f"DeltaAg{i}"] = e_by_id[i]["Delta_Ag_i"]
        row[f"E{i}"] = e_by_id[i]["E_i"]
        row[f"mean_J{i}"] = e_by_id[i]["mean_J"]
    return row


def comparison_metrics(row_a, row_b, field_errors):
    result = {"field_errors": field_errors, "scalar_relative_differences": {}}
    for key in [*(f"Pi{i}" for i in range(4)), *(f"fP{i}" for i in range(4)),
                "CV_Pi", "R_hyd", "global_current_wall_area", "mean_displacement",
                "max_displacement", "min_J", "mean_J", "max_J"]:
        a, b = float(row_a[key]), float(row_b[key])
        result["scalar_relative_differences"][key] = {
            "absolute": b - a,
            "relative": (b - a) / max(abs(a), abs(b), 1e-30),
        }
    return result


def uniform_comparison_figure(case_b_dir, case_c_dir, row_b, row_c, field_errors, path):
    xb, tb, fb = read_final_fields(case_b_dir / "global_uniform_002.xdmf")
    xc, tc, fc = read_final_fields(case_c_dir / "regional_uniform_002.xdmf")
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    lo = min(fb["J_tot"].min(), fc["J_tot"].min()); hi = max(fb["J_tot"].max(), fc["J_tot"].max())
    for ax, title, x, t, J in zip(axes[:2], ["Global operator", "Four regional operators"],
                                 [xb, xc], [tb, tc], [fb["J_tot"], fc["J_tot"]]):
        image = ax.tripcolor(x[:, 0], x[:, 1], t, J, shading="gouraud", cmap="coolwarm", vmin=lo, vmax=hi)
        ax.set(title=title, aspect="equal", xlim=(0, 1), ylim=(0, 1)); figure.colorbar(image, ax=ax, label="J")
    axes[2].axis("off")
    lines = ["Global vs regional-uniform equivalence", "", "Relative L2 field errors:"]
    lines += [f"{key}: {value['relative_l2']:.3e}" for key, value in field_errors.items()]
    lines += ["", "Selected scalar relative errors:",
              f"CV_Pi: {(row_c['CV_Pi']-row_b['CV_Pi'])/row_b['CV_Pi']:.3e}",
              f"R_hyd: {(row_c['R_hyd']-row_b['R_hyd'])/row_b['R_hyd']:.3e}"]
    axes[2].text(.03, .95, "\n".join(lines), va="top", family="monospace", fontsize=10)
    figure.tight_layout(); figure.savefig(path, dpi=250); plt.close(figure)


def write_summary(rows, zero_regression, equivalence):
    report = {
        "interpretation": {
            "pg_i": "prescribed regional alveolar/gas gauge pressure in model units",
            "E_i": "regional alveolar gas-area expansion surrogate; not ventilation rate",
            "Pi_i": "predicted regional perfusion intensity under prescribed vascular flow",
        },
        "cases": rows,
        "zero_regression_vs_accepted_baseline": zero_regression,
        "global_uniform_vs_regional_uniform_equivalence": equivalence,
    }
    VALIDATION_ROOT.mkdir(parents=True, exist_ok=True)
    (VALIDATION_ROOT / "regional_pg_validation.json").write_text(json.dumps(report, indent=2))
    keys = list(rows[0])
    with (VALIDATION_ROOT / "regional_pg_validation.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)
    return report


def main():
    metadata = json.loads(ACINAR_METADATA.read_text())
    bx, bt, bf = read_final_fields(BASELINE_XDMF)
    baseline_gas, _ = regional_gas_areas(bx, bt, bf["U_tot"], metadata, np.eye(2))
    reference = {row["acinus_id"]: row["current_gas_area"] for row in baseline_gas}
    processed = {}
    rows = []
    for name, case_dir in CASES.items():
        xdmf = case_dir / f"{name}.xdmf"
        result, x, t, fields, labels = process_case(name, case_dir, xdmf, reference)
        processed[name] = (result, x, t, fields, labels)
        rows.append(case_summary(name, case_dir, result))
    by_name = {row["case_name"]: row for row in rows}
    zero_fields = compare_case_fields("baseline", BASELINE_XDMF, "regional_zero",
                                      CASES["regional_zero"] / "regional_zero.xdmf")
    zero_regression = comparison_metrics(
        case_summary("baseline", BASELINE_DIR,
                     process_case("baseline", BASELINE_DIR, BASELINE_XDMF, reference,
                                  write_output=False)[0]),
        by_name["regional_zero"], zero_fields)
    uniform_fields = compare_case_fields(
        "global_uniform_002", CASES["global_uniform_002"] / "global_uniform_002.xdmf",
        "regional_uniform_002", CASES["regional_uniform_002"] / "regional_uniform_002.xdmf")
    equivalence = comparison_metrics(by_name["global_uniform_002"],
                                     by_name["regional_uniform_002"], uniform_fields)
    make_A2_figures(CASES["A2_only_002"], *processed["A2_only_002"][1:],
                    processed["A2_only_002"][0])
    uniform_comparison_figure(
        CASES["global_uniform_002"], CASES["regional_uniform_002"],
        by_name["global_uniform_002"], by_name["regional_uniform_002"], uniform_fields,
        VALIDATION_ROOT / "uniform_global_vs_regional_comparison.png")
    write_summary(rows, zero_regression, equivalence)


if __name__ == "__main__":
    main()
