"""Full-coefficient checkpoint of the converged mixed solution, and its reload.

``write()`` stores, in one HDF5 file, the mesh, the cell domain markers and the
mixed solution ``sol_func`` with all its coefficients, plus a JSON sidecar with
what is needed to rebuild the function space: the UFL element, the subsolution
order and the periodic map (method and tolerance).  After writing, the file is
read back into the solver's own space and compared with the in-memory vector;
any difference raises.

``load()`` rebuilds mesh, periodic mixed space and solution in a fresh process.

Command line (verification against a run's saved fields):
    python scripts/solution_checkpoint.py <run_dir>/<name>_checkpoint.h5 [--xdmf <run_dir>/<name>.xdmf]
reloads the checkpoint and, with --xdmf, compares the vertex values of the
reloaded p_l_perturbation and U_tilde with the final time grid of the XDMF
output (written from the same solution).
"""
from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import dolfin
import numpy as np
import ufl

SCHEMA_VERSION = 1


def sidecar_path(h5_path):
    return Path(h5_path).with_suffix(".json")


def subsolution_l2_squared(solution, mesh):
    """Integral of inner(s, s) over the mesh for each subsolution (covers every coefficient, e.g. P2 midpoints)."""
    measure = dolfin.dx(domain=mesh, metadata={"quadrature_degree": 6})
    return [float(dolfin.assemble(dolfin.inner(part, part) * measure)) for part in dolfin.split(solution)]


def write(problem, h5_path, domains, periodic_method, extra=None):
    """Write mesh, domains and the mixed solution; verify the read-back; return the metadata."""
    h5_path = Path(h5_path)
    mesh = problem.mesh
    with dolfin.HDF5File(mesh.mpi_comm(), str(h5_path), "w") as stream:
        stream.write(mesh, "/mesh")
        stream.write(domains, "/domains")
        stream.write(problem.sol_func, "/solution")
    reread = dolfin.Function(problem.sol_fs)
    with dolfin.HDF5File(mesh.mpi_comm(), str(h5_path), "r") as stream:
        stream.read(reread, "/solution")
    difference = float(np.abs(reread.vector().get_local() - problem.sol_func.vector().get_local()).max())
    if difference != 0.0:
        raise RuntimeError(f"Checkpoint read-back differs from the in-memory solution by {difference}.")
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "h5_file": h5_path.name,
        "datasets": {"mesh": "/mesh", "domains": "/domains", "solution": "/solution"},
        "element": repr(problem.sol_fe),
        "subsolutions": [{"name": subsol.name, "element": repr(subsol.fe)} for subsol in problem.subsols],
        "mixed_space_dim": int(problem.sol_fs.dim()),
        "periodic_map": periodic_method,
        "readback_max_abs_difference": difference,
        "subsolution_l2_squared": subsolution_l2_squared(problem.sol_func, mesh),
        **(extra or {}),
    }
    sidecar_path(h5_path).write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def _element(text):
    namespace = {name: getattr(ufl, name) for name in dir(ufl) if not name.startswith("_")}
    element = eval(text, namespace)  # trusted: written by write() above
    if repr(element) != text:
        raise RuntimeError(f"Element rebuilt from the checkpoint does not match: {element!r} != {text}")
    return element


def load(h5_path):
    """Return (mesh, domains, space, solution, metadata) rebuilt from a checkpoint."""
    import periodic_map  # same directory
    import dolfin_mech as dmech

    h5_path = Path(h5_path)
    metadata = json.loads(sidecar_path(h5_path).read_text())
    mesh = dolfin.Mesh()
    with dolfin.HDF5File(mesh.mpi_comm(), str(h5_path), "r") as stream:
        stream.read(mesh, "/mesh", False)
        domains = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim())
        stream.read(domains, "/domains")
    coordinates = mesh.coordinates()[:, :2]
    bbox = [coordinates[:, 0].min(), coordinates[:, 0].max(), coordinates[:, 1].min(), coordinates[:, 1].max()]
    vertices = np.array([[bbox[0], bbox[2]], [bbox[1], bbox[2]], [bbox[1], bbox[3]], [bbox[0], bbox[3]]])
    method = metadata["periodic_map"]
    if method["legacy_periodic_tolerance"]:
        constrained = dmech.PeriodicSubDomain(2, bbox, vertices)
    else:
        constrained = periodic_map.ExactPeriodicSubDomain(2, bbox, vertices, tol=method["periodic_tolerance"])
    space = dolfin.FunctionSpace(mesh, _element(metadata["element"]), constrained_domain=constrained)
    if space.dim() != metadata["mixed_space_dim"]:
        raise RuntimeError(f"Rebuilt space has {space.dim()} dofs, checkpoint recorded {metadata['mixed_space_dim']}.")
    solution = dolfin.Function(space)
    with dolfin.HDF5File(mesh.mpi_comm(), str(h5_path), "r") as stream:
        stream.read(solution, "/solution")
    return mesh, domains, space, solution, metadata


def subsolution(solution, metadata, name):
    """Deep copy of one named subsolution."""
    names = [entry["name"] for entry in metadata["subsolutions"]]
    return solution.split(deepcopy=True)[names.index(name)]


def final_xdmf_vertex_values(xdmf_path, names):
    root = ET.parse(xdmf_path).getroot()
    grid = max(root.findall('.//Grid[@GridType="Uniform"]'), key=lambda item: float(item.find("Time").attrib["Value"]))
    import h5py

    values = {}
    for attribute in grid.findall("Attribute"):
        if attribute.attrib["Name"] in names and attribute.attrib.get("Center") == "Node":
            filename, dataset = attribute.find("DataItem").text.strip().split(":", 1)
            with h5py.File(Path(xdmf_path).parent / filename, "r") as stream:
                values[attribute.attrib["Name"]] = np.asarray(stream[dataset])
    missing = set(names) - set(values)
    if missing:
        raise RuntimeError(f"Final XDMF grid lacks nodal fields {sorted(missing)}.")
    return float(grid.find("Time").attrib["Value"]), values


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--xdmf", type=Path, help="run XDMF output to compare vertex values with")
    args = parser.parse_args()
    mesh, domains, space, solution, metadata = load(args.checkpoint)
    report = {"checkpoint": str(args.checkpoint), "mixed_space_dim": space.dim(), "periodic_map": metadata["periodic_map"]}
    recorded = metadata["subsolution_l2_squared"]
    reloaded_norms = subsolution_l2_squared(solution, mesh)
    report["subsolution_l2_squared_relative_difference"] = {
        entry["name"]: (abs(new - old) / old if old != 0.0 else abs(new))
        for entry, old, new in zip(metadata["subsolutions"], recorded, reloaded_norms)}
    if args.xdmf is not None:
        final_time, saved = final_xdmf_vertex_values(args.xdmf, ["p_l_perturbation", "U_tilde"])
        report["xdmf_final_time"] = final_time
        for name in ("p_l_perturbation", "U_tilde"):
            reloaded = subsolution(solution, metadata, name).compute_vertex_values(mesh)
            reloaded = reloaded.reshape(-1, mesh.num_vertices()).T if name == "U_tilde" else reloaded.reshape(-1, 1)
            reference = saved[name].reshape(mesh.num_vertices(), -1)[:, :reloaded.shape[1]]
            report[f"{name}_max_abs_difference_vs_xdmf"] = float(np.abs(reloaded - reference).max())
            report[f"{name}_max_abs"] = float(np.abs(reference).max())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
