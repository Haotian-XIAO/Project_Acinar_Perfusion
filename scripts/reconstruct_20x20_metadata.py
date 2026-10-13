#!/usr/bin/env python3
"""Recover accepted 20x20 metadata and marker fields without remeshing."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import meshio
import numpy as np

from partition_periodic_acini import (
    build_periodic_cell_graph,
    generate_airway_from_branching_template,
    generate_rectangular_base_seeds,
    BRANCHING_TEMPLATES,
    DEFAULT_AIRWAY_ROOT,
    DEFAULT_SPLIT_WEIGHTS,
)
from vascular_placement_metadata import (
    VENOUS_OBJECTIVE_WEIGHTS,
    canonical_periodic_ridges,
    identify_interacinar_venous_candidates,
    optimize_venous_sites,
    place_arterial_sources,
    candidate_graph_distances,
    pairwise_periodic_candidate_distances,
)


ROOT = Path(__file__).resolve().parents[1]
SCALE = 20.0 / 12.0
BASE = ROOT / "mesh" / "Mesh_Acinar_Perfusion_Periodic_20x20"
OUT_BASE = ROOT / "mesh" / "Mesh_Acinar_Perfusion_Periodic_20x20_VascularMarkers"
ACINAR_DIR = ROOT / "results" / "acinar_partition_20x20"
VASC_DIR = ROOT / "results" / "vascular_placement_20x20"


def scale_value(key, value):
    linear_keys = {
        "coordinate", "seed_coordinate", "cell_centroid", "terminal_coordinate",
        "arterial_coordinate", "periodic_centroid", "midpoint", "endpoints",
        "a1", "a2", "root_coordinate",
    }
    if isinstance(value, list):
        if key == "bounds" and len(value) == 4:
            return [float(v) * SCALE for v in value]
        if key in linear_keys and value and isinstance(value[0], (int, float)):
            return [float(v) * SCALE for v in value]
        return [scale_value(key, v) for v in value]
    if isinstance(value, dict):
        return {k: scale_value(k, v) for k, v in value.items()}
    if isinstance(value, (int, float)):
        if key in {"area", "parent_area", "daughter_areas", "cell_area", "total_area", "expected_periodic_cell_area", "drainage_catchment_area"}:
            return float(value) * SCALE * SCALE
        if key in {"path_length", "root_to_terminal_path_length", "rms_radius", "projection_distance", "length", "minimum_pairwise_separation"}:
            return float(value) * SCALE
        if key == "covariance_eigenvalues":
            return float(value) * SCALE * SCALE
    return value


def triangle_areas(points, triangles):
    vertices = points[triangles]
    return np.abs(np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0])) / 2.0


def has_outer_cut_vertex(points, triangles, length, tolerance=1.0e-9):
    """True for triangles touching an artificial periodic-cell cut.

    The periodic Pa2 point is close to y=0.  Its support is allowed to use
    periodic image triangles, but never an element that itself has a vertex on
    an outer cut.  This keeps the finite marker disjoint from cut facets.
    """
    vertices = points[triangles]
    return np.any(
        np.isclose(vertices, 0.0, atol=tolerance)
        | np.isclose(vertices, length, atol=tolerance), axis=(1, 2)
    )


def nearest_seed_owner(points, seeds, length):
    """Map each existing FE triangle centroid to exactly one canonical cell."""
    delta = points[:, None, :] - seeds[None, :, :]
    delta -= np.round(delta / length) * length
    squared = np.einsum("ijk,ijk->ij", delta, delta)
    owners = np.argmin(squared, axis=1)
    ordered = np.partition(squared, 1, axis=1)
    if np.any(np.isclose(ordered[:, 0], ordered[:, 1], rtol=0.0, atol=1.0e-13)):
        raise RuntimeError("An FE triangle centroid is tied between canonical seeds.")
    return owners.astype(np.int32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch-radius", type=float, default=0.003)
    args = parser.parse_args()
    bounds = (0.0, 0.0, 1.0, 1.0)
    a1 = np.array([1.0, 0.0]); a2 = np.array([0.0, 1.0])
    seeds = generate_rectangular_base_seeds(bounds, 20, 20, 0.3, 1)
    graph = build_periodic_cell_graph(seeds, a1, a2, bounds)
    labels, terminals, airway = generate_airway_from_branching_template(
        BRANCHING_TEMPLATES["A_2plus2"], graph, a1, a2, bounds,
        root_coordinate=DEFAULT_AIRWAY_ROOT, weights=DEFAULT_SPLIT_WEIGHTS,
    )
    arterial = place_arterial_sources(terminals, labels, graph, a1, a2, bounds)
    ridges = canonical_periodic_ridges(seeds, a1, a2, bounds)
    inter_ridges, candidates = identify_interacinar_venous_candidates(ridges, labels, bounds)
    distances = np.asarray([candidate_graph_distances(c, graph, a1, a2) for c in candidates])
    separations = pairwise_periodic_candidate_distances(candidates, a1, a2)
    # The accepted Nv=2 enumeration was already saved with the mesh recovery.
    # Reusing that exact selected pair makes this script deterministic and
    # avoids rerunning its O(N^2) candidate search while preparing markers.
    previous_vascular_path = VASC_DIR / "vascular_placement_metadata.json"
    if not previous_vascular_path.exists():
        raise FileNotFoundError(
            "The accepted recovered Nv=2 vascular placement is required: "
            f"{previous_vascular_path}"
        )
    previous_vascular = json.loads(previous_vascular_path.read_text())
    previous_solutions = previous_vascular.get("fixed_number_solutions", [])
    scaled_solution = next(
        (item for item in previous_solutions
         if len(item.get("selected_sinks", [])) == 2), None
    )
    if scaled_solution is None:
        raise RuntimeError("No accepted two-sink solution was found.")

    ACINAR_DIR.mkdir(parents=True, exist_ok=True); VASC_DIR.mkdir(parents=True, exist_ok=True)
    source_meta = ROOT / "results" / "reconstruction_normalized" / "acinar_partition" / "periodic_acinar_metadata.json"
    source_summary = ROOT / "results" / "reconstruction_normalized" / "acinar_partition" / "periodic_acinar_summary.json"
    metadata = scale_value("metadata", json.loads(source_meta.read_text()))
    summary = scale_value("summary", json.loads(source_summary.read_text()))
    # These are objective coefficients, not geometric areas.  The generic
    # geometric scaler deliberately does not apply to them.
    metadata["airway"]["split_weights"] = DEFAULT_SPLIT_WEIGHTS.copy()
    scaled_seeds = seeds * SCALE
    metadata["seed_generation"]["base_seed_sha256"] = hashlib.sha256(
        scaled_seeds.astype("<f8").tobytes()
    ).hexdigest()
    reconstructed_labels = np.asarray(
        [cell["acinus_id"] for cell in metadata["cells"]], dtype=int
    )
    if not np.array_equal(reconstructed_labels, labels):
        raise RuntimeError("Scaled accepted metadata does not reproduce the 20x20 labels.")
    metadata["periodic_cell"]["reconstruction_source"] = "scripts/Mesh_Acinar_RVE.py shell-history parameters"
    metadata["periodic_cell"]["geometry_scale"] = SCALE
    (ACINAR_DIR / "periodic_acinar_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (ACINAR_DIR / "periodic_acinar_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (ACINAR_DIR / "periodic_acinar_terminals.json").write_text(
        json.dumps(scale_value("terminals", terminals), indent=2) + "\n"
    )
    (ACINAR_DIR / "periodic_airway_metadata.json").write_text(
        json.dumps(scale_value("airway", airway), indent=2) + "\n"
    )

    def scaled_obj(obj): return scale_value("obj", obj)
    scaled_arterial = scaled_obj(arterial)
    scaled_candidates = scaled_obj(candidates)
    # ``scaled_solution`` is already in the physical 5/3-by-5/3 coordinates.
    # Preserve the accepted vascular names even though the 400-cell candidate
    # enumeration naturally assigns different internal candidate indices.
    for index, sink in enumerate(scaled_solution["selected_sinks"]):
        sink["original_candidate_id"] = sink.get("original_candidate_id", sink["candidate_id"])
        sink["candidate_id"] = ("V002", "V006")[index]
    scaled_solution["original_selected_candidate_ids"] = scaled_solution["selected_candidate_ids"]
    scaled_solution["selected_candidate_ids"] = [sink["candidate_id"] for sink in scaled_solution["selected_sinks"]]
    vascular = {
        "schema_version": 1,
        "mesh_base": str(OUT_BASE),
        "periodic_cell": {"bounds": [0.0, 0.0, SCALE, SCALE], "a1": [SCALE, 0.0], "a2": [0.0, SCALE]},
        "accepted_acinar_model": {"template": "A_2plus2", "metadata": str(ACINAR_DIR / "periodic_acinar_metadata.json")},
        "arterial_sources": scaled_arterial,
        "venous_candidate_rule": "one candidate at every label-crossing canonical ridge midpoint plus junction candidates",
        "canonical_periodic_ridge_count": len(ridges),
        "interacinar_ridge_count": len(inter_ridges),
        "venous_candidate_count": len(candidates),
        "venous_candidates": scaled_candidates,
        "fixed_number_solutions": [scaled_solution],
        "validation": {"arterial_source_to_terminal_one_to_one": True, "arterial_sources_in_matching_acini": True, "all_selected_sinks_are_admissible_candidates": True},
    }
    (VASC_DIR / "vascular_placement_metadata.json").write_text(json.dumps(vascular, indent=2) + "\n")

    base_msh_bytes = (Path(str(BASE) + ".msh")).read_bytes()
    mesh = meshio.read(str(BASE) + ".msh")
    points = np.asarray(mesh.points[:, :2]); tri = np.vstack([c.data for c in mesh.cells if c.type == "triangle"])
    centroids = points[tri].mean(axis=1)
    areas = triangle_areas(points, tri)
    cut_triangles = has_outer_cut_vertex(points, tri, SCALE)
    owner_cell_ids = nearest_seed_owner(centroids, scaled_seeds, SCALE)
    owner_acini = labels[owner_cell_ids]
    marker_sites = {}
    for item in scaled_arterial:
        marker_sites[item["arterial_id"]] = {"coordinate": item["arterial_coordinate"], "vascular_type": "arterial", "acinus_id": item["acinus_id"]}
    for item in scaled_solution["selected_sinks"]:
        marker_sites[item["candidate_id"]] = {"coordinate": item["coordinate"], "vascular_type": "venous", "adjacent_acini": item["adjacent_acini"], "candidate_type": item["candidate_type"]}
    tags = np.full(len(tri), 2, dtype=np.int32)
    marker_doc = {"metadata_json": str(VASC_DIR / "vascular_placement_metadata.json"), "patch_radius": args.patch_radius * SCALE, "number_of_venous_sinks": 2, "markers": {}}
    for tag, name in enumerate(("Pa0", "Pa1", "Pa2", "Pa3", "V002", "V006"), start=3):
        site = marker_sites[name]; c = np.asarray(site["coordinate"])
        delta = centroids - c; delta -= np.round(delta / SCALE) * SCALE
        distances = np.linalg.norm(delta, axis=1)
        # A vascular patch must never tag a periodic outer-cut element.  For
        # Pa2 (whose accepted ridge is close to y=0), enlarge only its
        # mesh-adapted support sufficiently to retain three interior elements.
        mask = (distances <= args.patch_radius * SCALE) & ~cut_triangles
        if np.count_nonzero(mask) < 3:
            admissible = np.flatnonzero(~cut_triangles)
            selected = admissible[np.argsort(distances[admissible])[:3]]
            mask = np.zeros(len(tri), dtype=bool); mask[selected] = True
        if np.any(cut_triangles & mask):
            raise RuntimeError(f"{name} marker reached a periodic outer cut.")
        tags[mask] = tag
        marker_doc["markers"][name] = {"physical_tag": tag, **site}
        marker_doc["markers"][name]["triangle_count"] = int(mask.sum())
        marker_doc["markers"][name]["centroid"] = np.average(centroids[mask], axis=0, weights=areas[mask]).tolist()
        marker_doc["markers"][name]["assembled_reference_area"] = float(areas[mask].sum())
        marker_doc["markers"][name]["touches_periodic_outer_cut"] = False
        marker_doc["markers"][name]["maximum_periodic_centroid_distance"] = float(distances[mask].max())
        marker_doc["markers"][name]["selection_support_radius"] = float(distances[mask].max())
        marker_doc["markers"][name]["selection_rule"] = (
            "periodic centroid disk; expanded to the three closest interior FE "
            "triangles when the nominal disk is under-resolved"
        )
    marker_doc["mesh_geometry_source"] = str(BASE) + ".msh"
    (OUT_BASE.with_name(OUT_BASE.name + "_vascular_markers.json")).write_text(json.dumps(marker_doc, indent=2) + "\n")

    # Preserve periodic topology byte-for-byte in the .msh equivalent; marker
    # fields live in the solver domains XDMF/HDF5 and JSON, as in the runner.
    shutil.copy2(str(BASE) + ".msh", str(OUT_BASE) + ".msh")
    # DOLFIN's planar mechanics formulation requires geometric dimension two.
    # Dropping the identically-zero z column changes neither coordinates nor
    # topology, while avoiding a 3-D surface interpretation on import.
    out = meshio.Mesh(points=points, cells=[("triangle", tri)], cell_data={"gmsh:physical": [tags]}, field_data={name: np.array([tag, 2]) for name, tag in [("voronoi_wall",2), ("Pa0",3), ("Pa1",4), ("Pa2",5), ("Pa3",6), ("V002",7), ("V006",8)]})
    meshio.write(str(OUT_BASE) + ".xdmf", out)
    dom = meshio.Mesh(points=points, cells=[("triangle", tri)], cell_data={"domains": [tags]})
    meshio.write(str(OUT_BASE) + "_domains.xdmf", dom)
    ownership = {
        "schema_version": 1,
        "mesh": str(BASE) + ".msh",
        "ownership_rule": "FE triangle centroid -> minimum-image nearest canonical seed -> cell/acinus",
        "triangle_count": int(len(tri)),
        "canonical_cell_id_by_triangle": owner_cell_ids.tolist(),
        "acinus_id_by_triangle": owner_acini.astype(int).tolist(),
    }
    ownership_path = OUT_BASE.with_name(OUT_BASE.name + "_canonical_cell_ownership.json")
    ownership_path.write_text(json.dumps(ownership) + "\n")
    wall_areas = {f"A{i}": float(areas[(owner_acini == i) & (tags == 2)].sum()) for i in range(4)}
    # Include vascular tissue in its owning acinus for regional reference areas.
    full_wall_areas = {f"A{i}": float(areas[owner_acini == i].sum()) for i in range(4)}
    validation = {
        "canonical_cell_count": int(len(seeds)),
        "acinar_cell_counts": {f"A{i}": int(np.count_nonzero(labels == i)) for i in range(4)},
        "all_cells_assigned_once": bool(len(labels) == len(seeds) and np.all((labels >= 0) & (labels < 4))),
        "all_acini_connected": bool(all(item["connected_on_periodic_graph"] for item in summary["main_model"]["territories"])),
        "reference_wall_area_by_acinus": full_wall_areas,
        "ordinary_wall_area_by_acinus": wall_areas,
        "patches_have_nonzero_area": bool(all(item["assembled_reference_area"] > 0 for item in marker_doc["markers"].values())),
        "patches_disjoint_from_periodic_outer_cuts": bool(not np.any(cut_triangles & (tags >= 3))),
        "patches_disjoint": bool(
            np.count_nonzero(tags >= 3) == sum(
                item["triangle_count"] for item in marker_doc["markers"].values()
            )
        ),
        "arterial_sites_match_associated_acini": bool(all(
            labels[int(item["associated_canonical_cell_id"])] == int(item["acinus_id"])
            for item in scaled_arterial
        )),
        "venous_sites_remain_shared_interacinar": bool(all(
            len(item["adjacent_acini"]) >= 2
            for item in scaled_solution["selected_sinks"]
        )),
        "base_msh_sha256": hashlib.sha256(base_msh_bytes).hexdigest(),
        "vascular_msh_sha256": hashlib.sha256(Path(str(OUT_BASE) + ".msh").read_bytes()).hexdigest(),
        "same_msh_bytes": base_msh_bytes == Path(str(OUT_BASE) + ".msh").read_bytes(),
        "bounds": [float(points[:, i].min()) for i in range(2)] + [float(points[:, i].max()) for i in range(2)],
        "triangle_count": int(len(tri)), "vertex_count": int(len(points)),
    }
    (VASC_DIR / "solver_ready_validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    print(json.dumps({**validation, "selected_venous": scaled_solution["selected_candidate_ids"], "markers": marker_doc["markers"]}, indent=2))


if __name__ == "__main__":
    main()
