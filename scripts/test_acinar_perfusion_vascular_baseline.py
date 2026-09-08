#!/usr/bin/env python3
"""Healthy prescribed-flux baseline on the validated vascular-marker mesh.

The implementation deliberately keeps one ``DarcyFlowOperator`` and adds six
project-local source-only residual operators.  It does not modify dolfin_mech.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import dolfin
import dolfin_mech as dmech
import meshio
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MESH_BASE = ROOT / "mesh" / "Mesh_Acinar_Perfusion_VascularMarkers"
MARKER_JSON = Path(str(MESH_BASE) + "_vascular_markers.json")
RESULTS = ROOT / "results" / "vascular_baseline"
RESULT_BASE = RESULTS / "vascular_baseline"

Q_TOTAL = 1.0
TARGET_FLOWS = {
    "Pa0": 0.25,
    "Pa1": 0.25,
    "Pa2": 0.25,
    "Pa3": 0.25,
    "V002": 0.50,
    "V006": 0.50,
}
MATERIAL_PARAMETERS = {
    "alpha": 0.16,
    "gamma": 0.5,
    "c1": 0.2,
    "c2": 0.4,
    "kappa": 1.0,
    "eta": 1e-5,
}
FLOW_PARAMETERS = {
    "k_wall": 1.0,
    "use_kozeny_carman": False,
}
POROSITY_PARAMETERS = {"type": "constant", "val": 0.3}
STEP_PARAMETERS = {
    "n_steps": 1,
    "Deltat": 0.1,
    "dt_ini": 0.005,
    "dt_min": 1e-5,
    "dt_max": 0.02,
}
SOLVER_PARAMETERS = {
    "sol_tol": "1e-6 for every mixed subsolution",
    "n_iter_max": 32,
    "linear_solver_type": "dolfin",
    "linear_solver_name": "umfpack",
    "relax_type": "constant",
}


class VascularSourceOnlyOperator(dmech.Operator):
    """One time-varying volumetric source/sink contribution, without Darcy conduction."""

    def __init__(self, p_test, measure, theta_ini, theta_fin, residual_sign):
        self.measure = measure
        self.residual_sign = float(residual_sign)
        self.tv_theta = dmech.TimeVaryingConstant(
            val_ini=float(theta_ini), val_fin=float(theta_fin)
        )
        self.res_form = self.residual_sign * self.tv_theta.val * p_test * measure

    def set_value_at_t_step(self, t_step):
        self.tv_theta.set_value_at_t_step(t_step)


def load_mesh_and_markers():
    required = [
        Path(str(MESH_BASE) + suffix)
        for suffix in (".msh", ".xdmf", ".h5", "_domains.xdmf", "_domains.h5")
    ] + [MARKER_JSON]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing validated vascular mesh files: " + ", ".join(missing))

    mesh = dolfin.Mesh()
    with dolfin.XDMFFile(str(MESH_BASE) + ".xdmf") as infile:
        infile.read(mesh)

    mvc = dolfin.MeshValueCollection("size_t", mesh, mesh.topology().dim())
    with dolfin.XDMFFile(str(MESH_BASE) + "_domains.xdmf") as infile:
        infile.read(mvc, "domains")
    domains = dolfin.MeshFunction("size_t", mesh, mvc)

    marker_document = json.loads(MARKER_JSON.read_text())
    markers = marker_document["markers"]
    if set(markers) != set(TARGET_FLOWS):
        raise RuntimeError(
            f"Expected markers {sorted(TARGET_FLOWS)}, found {sorted(markers)}."
        )
    expected_tags = {"Pa0": 3, "Pa1": 4, "Pa2": 5, "Pa3": 6, "V002": 7, "V006": 8}
    actual_tags = {key: int(value["physical_tag"]) for key, value in markers.items()}
    if actual_tags != expected_tags:
        raise RuntimeError(f"Unexpected vascular marker tags: {actual_tags}.")
    return mesh, domains, marker_document


def make_boundary_markers(mesh):
    coordinates = mesh.coordinates()
    xmin, ymin = coordinates.min(axis=0)
    xmax, ymax = coordinates.max(axis=0)
    tol = 1e-8
    boundaries = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)
    for marker_id, axis, value in ((1, 0, xmin), (2, 0, xmax), (3, 1, ymin), (4, 1, ymax)):
        dolfin.CompiledSubDomain(
            f"near(x[{axis}], side, tol) && on_boundary", side=float(value), tol=tol
        ).mark(boundaries, marker_id)
    points = dolfin.MeshFunction("size_t", mesh, 0)
    points.set_all(0)
    vertices = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
    return boundaries, points, vertices, [xmin, xmax, ymin, ymax]


def assemble_patch_loading(mesh, domains, marker_document):
    dx = dolfin.Measure("dx", domain=mesh, subdomain_data=domains)
    rows = {}
    for vascular_id, marker in marker_document["markers"].items():
        tag = int(marker["physical_tag"])
        area = float(dolfin.assemble(dolfin.Constant(1.0) * dx(tag)))
        if area <= 0.0:
            raise RuntimeError(f"Vascular marker {vascular_id} has zero assembled area.")
        prescribed_q = TARGET_FLOWS[vascular_id]
        theta = prescribed_q / area
        rows[vascular_id] = {
            "physical_tag": tag,
            "vascular_type": marker["vascular_type"],
            "assembled_reference_area": area,
            "prescribed_integrated_Q": prescribed_q,
            "Theta": theta,
            "recovered_Theta_times_area": theta * area,
        }

    arterial = [key for key in rows if rows[key]["vascular_type"] == "arterial"]
    venous = [key for key in rows if rows[key]["vascular_type"] == "venous"]
    q_in = sum(rows[key]["recovered_Theta_times_area"] for key in arterial)
    q_out = sum(rows[key]["recovered_Theta_times_area"] for key in venous)
    residual = q_in - q_out
    report = {
        "Q_total": Q_TOTAL,
        "patches": rows,
        "Q_in": q_in,
        "Q_out": q_out,
        "mass_balance_residual": residual,
        "required_tolerance": 1e-12,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "mass_balance.json").write_text(json.dumps(report, indent=2))

    print("vascular patch loading:")
    for key, row in rows.items():
        print(
            f"  {key}: marker={row['physical_tag']} area={row['assembled_reference_area']:.16e} "
            f"Q={row['prescribed_integrated_Q']:.16e} Theta={row['Theta']:.16e} "
            f"Theta*Area={row['recovered_Theta_times_area']:.16e}"
        )
    print(f"Q_in={q_in:.16e} Q_out={q_out:.16e} residual={residual:.16e}")
    if abs(residual) >= 1e-12:
        raise RuntimeError("Source/sink mass balance failed; refusing to start solve.")
    return dx, report


def scalar_cell_values(function, mesh):
    values = np.empty(mesh.num_cells(), dtype=float)
    local = function.vector().get_local()
    dofmap = function.function_space().dofmap()
    for cell_index in range(mesh.num_cells()):
        values[cell_index] = local[dofmap.cell_dofs(cell_index)[0]]
    return values


def project_scalar_cell_data(expr, mesh):
    space = dolfin.FunctionSpace(mesh, "DG", 0)
    return scalar_cell_values(dolfin.project(expr, space), mesh)


def project_vector_cell_data(expr, mesh):
    return np.column_stack(
        [project_scalar_cell_data(expr[component], mesh) for component in range(mesh.geometry().dim())]
    )


def write_final_vtu(mesh, domains, problem, darcy_operator):
    points = np.column_stack([mesh.coordinates(), np.zeros(mesh.num_vertices())])
    triangles = np.asarray(mesh.cells(), dtype=int)
    p_l = project_scalar_cell_data(darcy_operator.pl_tot, mesh)
    p_tilde = project_scalar_cell_data(problem.pl_perturbation_subsol.subfunc, mesh)
    q_l = project_vector_cell_data(darcy_operator.q_l, mesh)
    Q_l = project_vector_cell_data(darcy_operator.Q_l, mesh)
    displacement = project_vector_cell_data(problem.U_tot, mesh)
    phis = project_scalar_cell_data(darcy_operator.Phis_expr, mesh)
    phif = project_scalar_cell_data(darcy_operator.Phif_expr, mesh)
    k_xx = project_scalar_cell_data(darcy_operator.k_l_intr[0, 0], mesh)
    k_xy = project_scalar_cell_data(darcy_operator.k_l_intr[0, 1], mesh)
    k_yy = project_scalar_cell_data(darcy_operator.k_l_intr[1, 1], mesh)
    q_l_3d = np.column_stack([q_l, np.zeros(mesh.num_cells())])
    Q_l_3d = np.column_stack([Q_l, np.zeros(mesh.num_cells())])
    displacement_3d = np.column_stack([displacement, np.zeros(mesh.num_cells())])
    output = meshio.Mesh(
        points=points,
        cells=[("triangle", triangles)],
        cell_data={
            "domain_id": [np.asarray(domains.array(), dtype=int)],
            "p_l": [p_l],
            "ptilde_l": [p_tilde],
            "q_l": [q_l_3d],
            "q_l_magnitude": [np.linalg.norm(q_l, axis=1)],
            "Q_l": [Q_l_3d],
            "Q_l_magnitude": [np.linalg.norm(Q_l, axis=1)],
            "displacement": [displacement_3d],
            "Phis": [phis],
            "Phif": [phif],
            "k_l_intr_xx": [k_xx],
            "k_l_intr_xy": [k_xy],
            "k_l_intr_yy": [k_yy],
        },
    )
    meshio.write(RESULTS / "final_fields.vtu", output)


def periodic_pressure_jump(mesh, pressure_function, bbox):
    coordinates = mesh.coordinates()
    xmin, xmax, ymin, ymax = bbox
    values = pressure_function.compute_vertex_values(mesh)
    tol = 1e-8

    def compare(axis, lo, hi, transverse_axis):
        lo_ids = np.where(np.abs(coordinates[:, axis] - lo) < tol)[0]
        hi_ids = np.where(np.abs(coordinates[:, axis] - hi) < tol)[0]
        lo_groups = {}
        hi_groups = {}
        for index in lo_ids:
            key = round(float(coordinates[index, transverse_axis]), 10)
            lo_groups.setdefault(key, []).append(float(values[index]))
        for index in hi_ids:
            key = round(float(coordinates[index, transverse_axis]), 10)
            hi_groups.setdefault(key, []).append(float(values[index]))
        jumps = []
        for key in sorted(set(lo_groups) & set(hi_groups)):
            jumps.append(abs(float(np.mean(lo_groups[key]) - np.mean(hi_groups[key]))))
        return max(jumps, default=0.0), len(jumps)

    lr_jump, lr_pairs = compare(0, xmin, xmax, 1)
    bt_jump, bt_pairs = compare(1, ymin, ymax, 0)
    return {
        "left_right_max_abs_jump": lr_jump,
        "left_right_matched_vertices": lr_pairs,
        "bottom_top_max_abs_jump": bt_jump,
        "bottom_top_matched_vertices": bt_pairs,
        "max_abs_jump": max(lr_jump, bt_jump),
    }


def parse_increment_table(path):
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        if not line.startswith("|") or "k_step" in line:
            continue
        fields = [item.strip() for item in line.strip("|").split("|")]
        if len(fields) == 7 and fields[0].isdigit():
            rows.append({
                "k_step": int(fields[0]),
                "k_t": int(fields[1]),
                "dt": float(fields[2]),
                "t": float(fields[3]),
                "t_step": float(fields[4]),
                "nonlinear_iterations": int(fields[5]),
                "success": fields[6] == "True",
            })
    return rows


def run():
    RESULTS.mkdir(parents=True, exist_ok=True)
    mesh, domains, marker_document = load_mesh_and_markers()
    dx, mass_report = assemble_patch_loading(mesh, domains, marker_document)
    boundaries, points, vertices, bbox = make_boundary_markers(mesh)

    material = {
        "skel": {"parameters": copy.deepcopy(MATERIAL_PARAMETERS), "scaling": "no"},
        "bulk": {"parameters": copy.deepcopy(MATERIAL_PARAMETERS), "scaling": "no"},
        "pore": {"parameters": copy.deepcopy(MATERIAL_PARAMETERS), "scaling": "no"},
    }
    problem = dmech.MicroPoroFlowHyperelasticityProblem(
        mesh=mesh,
        vertices=vertices,
        domains_mf=domains,
        boundaries_mf=boundaries,
        points_mf=points,
        displacement_perturbation_degree=2,
        quadrature_degree=6,
        bcs="pbc",
        porosity_init_val=POROSITY_PARAMETERS["val"],
        flow_params=copy.deepcopy(FLOW_PARAMETERS),
        skel_behavior=material["skel"],
        bulk_behavior=material["bulk"],
        pore_behavior=material["pore"],
    )

    permeability_space = dolfin.FunctionSpace(mesh, "DG", 0)
    permeability_scalar = dolfin.Function(permeability_space)
    permeability_scalar.vector()[:] = FLOW_PARAMETERS["k_wall"]
    zero = dolfin.Constant(0.0)
    permeability = dolfin.as_matrix(((permeability_scalar, zero), (zero, permeability_scalar)))

    step_index = problem.add_step(
        Deltat=STEP_PARAMETERS["Deltat"],
        dt_ini=STEP_PARAMETERS["dt_ini"],
        dt_min=STEP_PARAMETERS["dt_min"],
        dt_max=STEP_PARAMETERS["dt_max"],
    )
    problem.add_surface_pressure_loading_operator(
        measure=problem.dS(0), P_ini=0.0, P_fin=0.0, k_step=step_index
    )
    for i in range(2):
        for j in range(2):
            problem.add_macroscopic_stretch_component_penalty_operator(
                i=i, j=j, U_bar_ij_ini=0.0, U_bar_ij_fin=0.0, pen_val=1e6,
                k_step=step_index,
            )
    problem.add_surface_area_operator(measure=problem.dS(0), k_step=step_index)
    problem.add_surface_tension_loading_operator(
        measure=problem.dS(0), gamma_ini=0.0, gamma_fin=0.0,
        tension_params={}, k_step=step_index,
    )

    # Exactly one Darcy operator: full-domain conductivity/coupling/outputs,
    # with its built-in single inlet/outlet slots intentionally disabled.
    darcy_operator = problem.add_Darcy_operator(
        kinematics=problem.kinematics,
        U=problem.displacement_perturbation_subsol.subfunc,
        U_test=problem.displacement_perturbation_subsol.dsubtest,
        X=problem.X,
        X_0=problem.X_0,
        grad_p_bar_ini=(0.0, 0.0),
        grad_p_bar_fin=(0.0, 0.0),
        pl_bar_ini=0.0,
        pl_bar_fin=0.0,
        Theta_in_ini=0.0,
        Theta_in_fin=0.0,
        Theta_out_ini=0.0,
        Theta_out_fin=0.0,
        k_l0=permeability,
        use_kozeny_carman=FLOW_PARAMETERS["use_kozeny_carman"],
        subdomain_id=None,
        inlet_id=None,
        outlet_id=None,
        k_step=step_index,
    )
    source_operators = []
    for vascular_id, row in mass_report["patches"].items():
        sign = -1.0 if row["vascular_type"] == "arterial" else 1.0
        source_operators.append(problem.add_operator(
            VascularSourceOnlyOperator(
                p_test=problem.pl_perturbation_subsol.dsubtest,
                measure=problem.get_subdomain_measure(row["physical_tag"]),
                theta_ini=0.0,
                theta_fin=row["Theta"],
                residual_sign=sign,
            ),
            k_step=step_index,
        ))
    if sum(hasattr(op, "K_l") for op in problem.steps[step_index].operators) != 1:
        raise RuntimeError("Expected exactly one global Darcy conductivity operator.")

    problem.add_deformed_solid_volume_qoi()
    problem.add_deformed_fluid_volume_qoi()
    problem.add_deformed_volume_qoi()
    problem.add_macroscopic_stretch_qois()
    problem.add_macroscopic_solid_stress_qois()
    problem.add_macroscopic_stress_qois()
    problem.add_fluid_pressure_qoi()
    problem.add_interfacial_surface_qois()
    problem.add_darcy_qois()
    problem.add_foi(
        expr=dolfin.sqrt(dolfin.inner(darcy_operator.q_l, darcy_operator.q_l)),
        fs=problem.sfoi_fs, name="q_l_magnitude", update_type="project",
    )
    problem.add_foi(
        expr=dolfin.sqrt(dolfin.inner(darcy_operator.Q_l, darcy_operator.Q_l)),
        fs=problem.sfoi_fs, name="Q_l_magnitude", update_type="project",
    )

    run_config = {
        "mesh_filebasename": str(MESH_BASE),
        "mesh_files": [str(Path(str(MESH_BASE) + suffix)) for suffix in (
            ".msh", ".xdmf", ".h5", "_domains.xdmf", "_domains.h5",
            "_vascular_markers.json",
        )],
        "marker_map": marker_document,
        "Q_total": Q_TOTAL,
        "target_flows": TARGET_FLOWS,
        "Theta": {key: row["Theta"] for key, row in mass_report["patches"].items()},
        "pbar_l": 0.0,
        "macro_pressure_gradient": [0.0, 0.0],
        "material_parameters": MATERIAL_PARAMETERS,
        "permeability_parameters": FLOW_PARAMETERS,
        "porosity_parameters": POROSITY_PARAMETERS,
        "step_parameters": STEP_PARAMETERS,
        "solver_parameters": SOLVER_PARAMETERS,
        "darcy_conductivity_operator_count": 1,
        "source_only_operator_count": len(source_operators),
    }
    (RESULTS / "run_config.json").write_text(json.dumps(run_config, indent=2))

    solver = dmech.NonlinearSolver(
        problem=problem,
        parameters={
            "sol_tol": [1e-6] * len(problem.subsols),
            "n_iter_max": SOLVER_PARAMETERS["n_iter_max"],
            "linear_solver_type": SOLVER_PARAMETERS["linear_solver_type"],
            "linear_solver_name": SOLVER_PARAMETERS["linear_solver_name"],
        },
        relax_type=SOLVER_PARAMETERS["relax_type"],
        write_iter=0,
    )
    integrator = dmech.TimeIntegrator(
        problem=problem,
        solver=solver,
        parameters={"n_iter_for_accel": 4, "n_iter_for_decel": 16,
                    "accel_coeff": 2, "decel_coeff": 2},
        print_out="stdout",
        print_sta=str(RESULTS / "increments"),
        write_qois=str(RESULT_BASE) + "-qois",
        write_sol=str(RESULT_BASE),
        write_vtus=0,
        write_vtus_with_preserved_connectivity=0,
    )
    success = bool(integrator.integrate())
    integrator.close()
    if not success:
        raise RuntimeError("Healthy vascular baseline integration failed.")

    problem.update_fois()
    total_area = float(dolfin.assemble(dolfin.Constant(1.0) * problem.dV))
    p_l_function = problem.get_foi("pl_tot").func
    q_l_function = problem.get_foi("q_l").func
    Q_l_function = problem.get_foi("Q_l").func
    p_l_values = p_l_function.vector().get_local()
    q_magnitude = problem.get_foi("q_l_magnitude").func
    Q_magnitude = problem.get_foi("Q_l_magnitude").func
    ptilde_mean = float(dolfin.assemble(problem.pl_perturbation_subsol.subfunc * problem.dV) / total_area)
    lambda_p = float(dolfin.assemble(
        problem.lambda_pl_perturbation_zero_mean_subsol.subfunc * problem.dV
    ) / total_area)

    def field_stats(magnitude_function, magnitude_expression):
        values = magnitude_function.vector().get_local()
        return {
            "min": float(values.min()),
            "max": float(values.max()),
            "mean": float(dolfin.assemble(magnitude_expression * problem.dV) / total_area),
        }

    pressure_stats = {
        "min": float(p_l_values.min()),
        "max": float(p_l_values.max()),
        "mean": float(dolfin.assemble(darcy_operator.pl_tot * problem.dV) / total_area),
    }
    q_stats = field_stats(q_magnitude, dolfin.sqrt(dolfin.inner(darcy_operator.q_l, darcy_operator.q_l)))
    Q_stats = field_stats(Q_magnitude, dolfin.sqrt(dolfin.inner(darcy_operator.Q_l, darcy_operator.Q_l)))

    patch_pressures = {}
    for vascular_id, row in mass_report["patches"].items():
        patch_pressure = float(
            dolfin.assemble(darcy_operator.pl_tot * dx(row["physical_tag"]))
            / row["assembled_reference_area"]
        )
        patch_pressures[vascular_id] = patch_pressure
    pa_eff = sum(TARGET_FLOWS[key] * patch_pressures[key] for key in TARGET_FLOWS if key.startswith("Pa")) / Q_TOTAL
    pv_eff = sum(TARGET_FLOWS[key] * patch_pressures[key] for key in TARGET_FLOWS if key.startswith("V")) / Q_TOTAL
    delta_p = pa_eff - pv_eff
    hydraulic_resistance = delta_p / Q_TOTAL
    pressure_report = {
        "patch_area_averaged_pressure": patch_pressures,
        "pa_eff": pa_eff,
        "pv_eff": pv_eff,
        "Delta_p_av": delta_p,
        "R_hyd": hydraulic_resistance,
        "Q_total": Q_TOTAL,
    }
    (RESULTS / "patch_pressures.json").write_text(json.dumps(pressure_report, indent=2))

    increments = parse_increment_table(RESULTS / "increments.sta")
    periodic_continuity = periodic_pressure_jump(
        mesh, problem.pl_perturbation_subsol.func, bbox
    )
    pressure_range = pressure_stats["max"] - pressure_stats["min"]
    periodic_continuity["max_jump_relative_to_pressure_range"] = (
        periodic_continuity["max_abs_jump"] / pressure_range if pressure_range > 0 else 0.0
    )
    diagnostics = {
        "nonlinear_converged": success,
        "load_steps": 1,
        "time_load_increments": len(increments),
        "total_nonlinear_iterations": sum(row["nonlinear_iterations"] for row in increments),
        "increments": increments,
        "pbar_l": 0.0,
        "ptilde_l_volume_average": ptilde_mean,
        "lambda_p": lambda_p,
        "pressure_stats": pressure_stats,
        "q_l_magnitude_stats": q_stats,
        "Q_l_magnitude_stats": Q_stats,
        "final_mass_balance_residual": mass_report["mass_balance_residual"],
        "periodic_pressure_continuity": periodic_continuity,
    }
    (RESULTS / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    write_final_vtu(mesh, domains, problem, darcy_operator)

    if abs(ptilde_mean) > 1e-8:
        raise RuntimeError(f"Pressure perturbation mean is not zero: {ptilde_mean}.")
    if abs(lambda_p) > 1e-8:
        raise RuntimeError(f"Pressure zero-mean multiplier is materially nonzero: {lambda_p}.")

    print(json.dumps({"diagnostics": diagnostics, "patch_pressures": pressure_report}, indent=2))


if __name__ == "__main__":
    os.chdir(ROOT)
    run()
