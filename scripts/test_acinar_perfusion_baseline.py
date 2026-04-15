################################################################################
###                                                                          ###
### Created by Haotian XIAO, 2024-2027                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
###                                                                          ###
### And Martin Genet, 2018-2025                                              ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

#################################################################### imports ###

import sys
import dolfin
import numpy
import myPythonLibrary as mypy
import dolfin_mech as dmech
import os
import numpy as np

## Function for Post-processing
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection


def postprocess_and_save_cell_perfusion(
    problem,
    json_path,
    output_dir,
    fig_name="voronoi_cell_perfusion_map.png",
    csv_name="voronoi_cell_perfusion_data.csv",
    json_name="voronoi_cell_perfusion_polygons.json",
    title="Voronoi cell perfusion map",
    cmap="viridis",
    show_centers=True,
    save_figure=True,
    save_csv=True,
    save_json=True,
    dpi=300,
):
    """
    Compute cell perfusion from metadata JSON, plot Voronoi cell perfusion map,
    and save figure / tabular data / polygon geometry.

    Parameters
    ----------
    problem : object
        Your problem object with method compute_cell_perfusion_from_json(json_path).
    json_path : str
        Path to cell metadata JSON.
    output_dir : str
        Directory where outputs will be saved.
    fig_name : str
        Output figure filename.
    csv_name : str
        Output CSV filename.
    json_name : str
        Output JSON filename.
    title : str
        Figure title.
    cmap : str
        Matplotlib colormap name.
    show_centers : bool
        Whether to plot cell centers.
    save_figure : bool
        Whether to save PNG figure.
    save_csv : bool
        Whether to save CSV table.
    save_json : bool
        Whether to save JSON geometry + values.
    dpi : int
        Figure resolution.

    Returns
    -------
    cell_scores : list of dict
    summary : dict
    """

    os.makedirs(output_dir, exist_ok=True)

    # ===== compute cell scores =====
    cell_scores, summary = problem.compute_cell_perfusion_from_json(json_path)

    print("=== CELL PERFUSION SUMMARY ===")
    for k, v in summary.items():
        print(k, ":", v)

    # ===== load cell metadata =====
    with open(json_path, "r") as f:
        cell_metadata = json.load(f)

    # score map: cell_id -> score
    score_map = {c["cell_id"]: c["score"] for c in cell_scores}

    # ===== build Voronoi polygons =====
    patches = []
    values = []
    centers_x = []
    centers_y = []
    polygon_data = []

    for cell in cell_metadata:
        cid = cell["cell_id"]

        if cid not in score_map:
            continue

        verts = cell["vertices"]
        cx = float(cell["center"][0])
        cy = float(cell["center"][1])
        score = float(score_map[cid])

        poly = Polygon(verts, closed=True)
        patches.append(poly)
        values.append(score)
        centers_x.append(cx)
        centers_y.append(cy)

        polygon_data.append({
            "cell_id": int(cid),
            "center_x": cx,
            "center_y": cy,
            "perfusion_score": score,
            "vertices": verts
        })

    # ===== plot =====
    fig, ax = plt.subplots(figsize=(8, 8))

    pc = PatchCollection(
        patches,
        cmap=cmap,
        edgecolor="black",
        linewidth=0.5
    )
    pc.set_array(values)
    ax.add_collection(pc)

    if show_centers:
        ax.scatter(
            centers_x,
            centers_y,
            s=8,
            c="white",
            edgecolors="black",
            linewidths=0.2
        )

    ax.autoscale()
    ax.set_aspect("equal")

    cbar = fig.colorbar(pc, ax=ax)
    cbar.set_label("cell perfusion score")

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    plt.tight_layout()

    # ===== save figure =====
    if save_figure:
        fig_path = os.path.join(output_dir, fig_name)
        plt.savefig(fig_path, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved to: {fig_path}")

    # ===== save csv =====
    if save_csv:
        df = pd.DataFrame({
            "cell_id": [d["cell_id"] for d in polygon_data],
            "center_x": [d["center_x"] for d in polygon_data],
            "center_y": [d["center_y"] for d in polygon_data],
            "perfusion_score": [d["perfusion_score"] for d in polygon_data],
        })
        csv_path = os.path.join(output_dir, csv_name)
        df.to_csv(csv_path, index=False)
        print(f"CSV saved to: {csv_path}")

    # ===== save json =====
    if save_json:
        json_out = {
            "summary": summary,
            "plot_meta": {
                "title": title,
                "cmap": cmap,
                "n_cells": len(polygon_data),
                "vmin": float(min(values)) if len(values) > 0 else None,
                "vmax": float(max(values)) if len(values) > 0 else None,
            },
            "cells": polygon_data
        }
        json_out_path = os.path.join(output_dir, json_name)
        with open(json_out_path, "w") as f:
            json.dump(json_out, f, indent=2)
        print(f"JSON saved to: {json_out_path}")

    plt.close(fig)

    return cell_scores, summary

##function for running the perfusion test case

def run_Acinar_Perfusion(
        dim=2,
        bcs="pbc",
        mesh_params={},
        mat_params={},
        flow_params={},
        step_params={},
        load_params={},
        porosity_params={},
        res_basename={},
        verbose=1):

    # ------------------------- Mesh ------------------------- #
    mesh = dolfin.Mesh()
    mesh_filebasename = mesh_params["mesh_filebasename"]

    with dolfin.XDMFFile(mesh_filebasename + ".xdmf") as infile:
        infile.read(mesh)

    coord = mesh.coordinates()
    xmax = max(coord[:,0]); xmin = min(coord[:,0])
    ymax = max(coord[:,1]); ymin = min(coord[:,1])
    if (dim==2):    
        bbox = [xmin, xmax, ymin, ymax]
        vertices = numpy.array([[xmin, ymin],
                                [xmax, ymin],
                                [xmax, ymax],
                                [xmin, ymax]])
        a1 = vertices[1,:]-vertices[0,:] # first vector generating periodicity
        a2 = vertices[3,:]-vertices[0,:] # second vector generating periodicity
        tol = 1e-8
        assert numpy.linalg.norm(vertices[2,:]-vertices[3,:] - a1) <= tol # check if UC vertices form indeed a parallelogram
        assert numpy.linalg.norm(vertices[2,:]-vertices[1,:] - a2) <= tol # check if UC vertices form indeed a parallelogram
    elif (dim==3):    
        zmax = max(coord[:,2]); zmin = min(coord[:,2])
        bbox = [xmin, xmax, ymin, ymax, zmin, zmax]
        vertices = numpy.array([[xmin, ymin, zmin],
                                [xmax, ymin, zmin],
                                [xmax, ymax, zmin],
                                [xmin, ymax, zmin],
                                [xmin, ymin, zmax],
                                [xmax, ymin, zmax],
                                [xmax, ymax, zmax],
                                [xmin, ymax, zmax]])

    ################################################## Subdomains & Measures ###

    tol = 1e-8
    xmin_sd = dolfin.CompiledSubDomain("near(x[0], x0, tol) && on_boundary", x0=xmin, tol=tol)
    xmax_sd = dolfin.CompiledSubDomain("near(x[0], x0, tol) && on_boundary", x0=xmax, tol=tol)
    ymin_sd = dolfin.CompiledSubDomain("near(x[1], x0, tol) && on_boundary", x0=ymin, tol=tol)
    ymax_sd = dolfin.CompiledSubDomain("near(x[1], x0, tol) && on_boundary", x0=ymax, tol=tol)
    if (dim==3): zmin_sd = dolfin.CompiledSubDomain("near(x[2], x0) && on_boundary", x0=zmin, tol=tol)
    if (dim==3): zmax_sd = dolfin.CompiledSubDomain("near(x[2], x0) && on_boundary", x0=zmax, tol=tol)

    # if (dim==2):
    #     sint_sd = dolfin.CompiledSubDomain("near(pow(x[0] - x0, 2) + pow(x[1] - y0, 2), pow(r0, 2), 1e-2) && on_boundary", x0=x0, y0=y0, r0=r0)
    # elif (dim==3):
    #     sint_sd = dolfin.CompiledSubDomain("near(pow(x[0] - x0, 2) + pow(x[1] - y0, 2) + pow(x[2] - z0, 2), pow(r0, 2), 1e-2) && on_boundary", x0=x0, y0=y0, z0=z0, r0=r0)

    xmin_id = 1
    xmax_id = 2
    ymin_id = 3
    ymax_id = 4
    if (dim==3): zmin_id = 5
    if (dim==3): zmax_id = 6
    # sint_id = 9

    boundaries_mf = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim()-1) # MG20180418: size_t looks like unsigned int, but more robust wrt architecture and os
    boundaries_mf.set_all(0)

    xmin_sd.mark(boundaries_mf, xmin_id)
    xmax_sd.mark(boundaries_mf, xmax_id)
    ymin_sd.mark(boundaries_mf, ymin_id)
    ymax_sd.mark(boundaries_mf, ymax_id)
    if (dim==3): zmin_sd.mark(boundaries_mf, zmin_id)
    if (dim==3): zmax_sd.mark(boundaries_mf, zmax_id)
    # sint_sd.mark(boundaries_mf, sint_id)

    if (verbose):
        xdmf_file_boundaries = dolfin.XDMFFile(res_basename+"-boundaries.xdmf")
        xdmf_file_boundaries.write(boundaries_mf)
        xdmf_file_boundaries.close()

    points_mf = dolfin.MeshFunction("size_t", mesh, 0)
    points_mf.set_all(0)

    # ---- read cell domains from *_domains.xdmf ----
    mvc_domains = dolfin.MeshValueCollection("size_t", mesh, mesh.topology().dim())
    with dolfin.XDMFFile(mesh_filebasename + "_domains.xdmf") as infile:
        infile.read(mvc_domains, "domains")

    domains_mf = dolfin.MeshFunction("size_t", mesh, mvc_domains)


    # ------------------- Porosity Init ---------------------- #
    poro_type = porosity_params.get("type", "constant")
    poro_val = porosity_params.get("val", 0.5)

    porosity_fun = None
    if poro_type == "function_constant":
        poro_fs = dolfin.FunctionSpace(mesh, 'DG', 0)
        porosity_fun = dolfin.Function(poro_fs)
        porosity_fun.vector()[:] = poro_val
        poro_val = None
    elif poro_type == "random":
        poro_fs = dolfin.FunctionSpace(mesh, 'DG', 0)
        porosity_fun = dolfin.Function(poro_fs)
        porosity_fun.vector()[:] = numpy.random.uniform(low=0.4, high=0.6, size=porosity_fun.vector().size())
        poro_val = None

    # ---------------------- Problem ------------------------- #
    problem = dmech.MicroPoroFlowHyperelasticityProblem(
        mesh=mesh,
        domains_mf=domains_mf,
        boundaries_mf=boundaries_mf,
        points_mf=points_mf,
        displacement_perturbation_degree=2,
        quadrature_degree = 6,
        bcs=bcs,
        porosity_init_val=poro_val,
        porosity_init_fun=porosity_fun,
        flow_params=flow_params,
        skel_behavior=mat_params["skel"],
        bulk_behavior=mat_params["bulk"],
        pore_behavior=mat_params["pore"])
    
    # -------------------- Time Step ------------------------- #
    n_steps = step_params.get("n_steps", 1)
    Deltat_lst = step_params.get("Deltat_lst", [step_params.get("Deltat", 1.)/n_steps]*n_steps)
    dt_ini_lst = step_params.get("dt_ini_lst", [step_params.get("dt_ini", 1.)/n_steps]*n_steps)
    dt_min_lst = step_params.get("dt_min_lst", [step_params.get("dt_min", 1.)/n_steps]*n_steps)
    dt_max_lst = step_params.get("dt_max_lst", [step_params.get("dt_max", 1.)/n_steps]*n_steps)

    load_params = {} if load_params is None else load_params

    load_params_solid  = load_params.get("solid", {})
    load_params_liquid = load_params.get("liquid", {})
    load_params_air    = load_params.get("air", {})

    # --- solid loading lists ---

    U_bar_ij_lst = [[None for i in range(dim)] for j in range(dim)]
    sigma_bar_ij_lst = [[None for i in range(dim)] for j in range(dim)]

    for i in range(dim):
        for j in range(dim):
            U_bar_ij_lst[i][j] = load_params_solid.get(
                f"U_bar_{i}{j}_lst",
                [load_params_solid.get(f"U_bar_{i}{j}", None) for k_step in range(n_steps)]
            )
            sigma_bar_ij_lst[i][j] = load_params_solid.get(
                f"sigma_bar_{i}{j}_lst",
                [load_params_solid.get(f"sigma_bar_{i}{j}", None) for k_step in range(n_steps)]
            )

    gamma_lst = load_params_solid.get(
        "gamma_lst",
        [(k_step+1) * load_params_solid.get("gamma", 0.0) / n_steps for k_step in range(n_steps)]
    )

    tension_params = load_params_solid.get("tension_params", {})

    # --- liquid loading lists ---

    pl_bar_ini_lst = load_params_liquid.get("pl_bar_ini_lst", [0.0] * n_steps)
    pl_bar_fin_lst = load_params_liquid.get("pl_bar_fin_lst", [0.0] * n_steps)

    grad_p_bar_x_ini_lst = load_params_liquid.get("grad_p_bar_x_ini_lst", [0.0] * n_steps)
    grad_p_bar_x_fin_lst = load_params_liquid.get("grad_p_bar_x_fin_lst", [0.0] * n_steps)

    grad_p_bar_y_ini_lst = load_params_liquid.get("grad_p_bar_y_ini_lst", [0.0] * n_steps)
    grad_p_bar_y_fin_lst = load_params_liquid.get("grad_p_bar_y_fin_lst", [0.0] * n_steps)

    Theta_in_ini_lst  = load_params_liquid.get("Theta_in_ini_lst",  [0.0] * n_steps)
    Theta_in_fin_lst  = load_params_liquid.get("Theta_in_fin_lst",  [0.0] * n_steps)
    Theta_out_ini_lst = load_params_liquid.get("Theta_out_ini_lst", [0.0] * n_steps)
    Theta_out_fin_lst = load_params_liquid.get("Theta_out_fin_lst", [0.0] * n_steps)

    # --- air loading lists ---

    pf_lst = load_params_air.get(
        "pf_lst",
        [(k_step+1) * load_params_air.get("pf", 0.0) / n_steps for k_step in range(n_steps)]
    )


    for k_step in range(n_steps):

        Deltat = Deltat_lst[k_step]
        dt_ini = dt_ini_lst[k_step]
        dt_min = dt_min_lst[k_step]
        dt_max = dt_max_lst[k_step]

        k_step = problem.add_step(
            Deltat=Deltat,
            dt_ini=dt_ini,
            dt_min=dt_min,
            dt_max=dt_max)

        #   air pressure loading

        pf = pf_lst[k_step]
        pf_old = pf_lst[k_step-1] if (k_step > 0) else 0.

        problem.add_surface_pressure_loading_operator(
            measure=problem.dS(0),
            P_ini=pf_old, P_fin=pf,
            k_step=k_step)

        for i in range(dim):
         for j in range (dim):
            U_bar_ij = U_bar_ij_lst[i][j][k_step]
            U_bar_ij_old = U_bar_ij_lst[i][j][k_step-1] if (k_step > 0) else 0.
            sigma_bar_ij = sigma_bar_ij_lst[i][j][k_step]
            sigma_bar_ij_old = sigma_bar_ij_lst[i][j][k_step-1] if (k_step > 0) else 0.
            assert ((U_bar_ij is not None) or (sigma_bar_ij is not None))
            if (U_bar_ij is not None):
                problem.add_macroscopic_stretch_component_penalty_operator(
                    i=i, j=j,
                    U_bar_ij_ini=U_bar_ij_old, U_bar_ij_fin=U_bar_ij,
                    pen_val=1e6,
                    k_step=k_step)
            elif (sigma_bar_ij is not None):
                problem.add_macroscopic_stress_component_constraint_operator(
                    i=i, j=j,
                    sigma_bar_ij_ini=sigma_bar_ij_old, sigma_bar_ij_fin=sigma_bar_ij,
                    pf_ini=pf_old, pf_fin=pf,
                    k_step=k_step)
        
        problem.add_surface_area_operator(
            measure=problem.dS(0),
            k_step=k_step)
        
        gamma = gamma_lst[k_step]
        gamma_old = gamma_lst[k_step-1] if (k_step > 0) else 0.
        problem.add_surface_tension_loading_operator(
            measure=problem.dS(0),
            gamma_ini=gamma_old, gamma_fin=gamma,
            tension_params=tension_params,
            k_step=k_step)
        
        # ---- flow loadings (pull from your manual arrays using k) ----
        pl_bar_ini = pl_bar_ini_lst[k_step]
        pl_bar_fin = pl_bar_fin_lst[k_step]

        grad_p_bar_ini = (grad_p_bar_x_ini_lst[k_step], grad_p_bar_y_ini_lst[k_step])
        grad_p_bar_fin = (grad_p_bar_x_fin_lst[k_step], grad_p_bar_y_fin_lst[k_step])

        Theta_in_ini  = Theta_in_ini_lst[k_step]
        Theta_in_fin  = Theta_in_fin_lst[k_step]
        Theta_out_ini = Theta_out_ini_lst[k_step]
        Theta_out_fin = Theta_out_fin_lst[k_step]


        k_l   = flow_params.get("k_l", dolfin.Constant(1.0) * dolfin.Identity(dim))


        problem.add_Darcy_operator(
            # --- kinematics / fields ---
            kinematics=problem.kinematics,
            U=problem.displacement_perturbation_subsol.subfunc,
            U_test=problem.displacement_perturbation_subsol.dsubtest,
            X=problem.X,
            X_0=problem.X_0,

            # --- macro loads ---
            grad_p_bar_ini=grad_p_bar_ini,
            grad_p_bar_fin=grad_p_bar_fin,
            pl_bar_ini=pl_bar_ini,
            pl_bar_fin=pl_bar_fin,
            Theta_in_ini=Theta_in_ini,   Theta_in_fin=Theta_in_fin,
            Theta_out_ini=Theta_out_ini, Theta_out_fin=Theta_out_fin,

            # --- material ---
            k_l0=k_l,   
            use_kozeny_carman=flow_params.get("use_kozeny_carman", False),  
            # --- ids ---
            subdomain_id=None,
            inlet_id=3,
            outlet_id=2,

            # --- step ---
            k_step=k_step,
        )

    # -------------------- Quantities of Interest ------------- #
    problem.add_deformed_solid_volume_qoi()
    problem.add_deformed_fluid_volume_qoi()
    problem.add_deformed_volume_qoi()
    problem.add_macroscopic_stretch_qois()
    problem.add_macroscopic_solid_stress_qois()
    problem.add_macroscopic_stress_qois()
    problem.add_fluid_pressure_qoi()
    problem.add_interfacial_surface_qois()
    problem.add_darcy_qois() 

    # -------------------- Solver & Integrator ---------------- #
    solver = dmech.NonlinearSolver(
    problem=problem,
    parameters={
        "sol_tol": [1e-6]*len(problem.subsols),
        "n_iter_max": 32,
        "linear_solver_type": "dolfin",
        "linear_solver_name": "umfpack",
    },
    relax_type="constant",
    write_iter=0)


    integrator = dmech.TimeIntegrator(
        problem=problem,
        solver=solver,
        parameters={
            "n_iter_for_accel": 4,
            "n_iter_for_decel": 16,
            "accel_coeff": 2,
            "decel_coeff": 2},
        print_out=1,  #res_basename*verbose,
        print_sta=res_basename*verbose,
        write_qois=res_basename+"-qois",
        write_sol=res_basename,
        write_vtus=0,
        write_vtus_with_preserved_connectivity=0)

    success = integrator.integrate()
    assert success, "Integration failed. Aborting."

    #post-process cell perfusion and save outputs#

    json_path = "/Users/xiao/PhD/Project_Acinar_Perfusion/mesh/Mesh_Acinar_Perfusion_2D_cell_metadata.json"
    output_dir = "/Users/xiao/PhD/Project_Acinar_Perfusion/results"
    case_name = "kubc"

    cell_scores, summary = postprocess_and_save_cell_perfusion(
        problem=problem,
        json_path=json_path,
        output_dir=output_dir,
        fig_name=f"{case_name}_map.png",
        csv_name=f"{case_name}_data.csv",
        json_name=f"{case_name}_polygons.json",
        title=f"Cell perfusion map - {case_name}",
    )



####################################################################### test ###



res_folder = sys.argv[0][:-3]
os.makedirs(res_folder, exist_ok=True)

mesh_filebasename = "mesh/Mesh_Acinar_Perfusion_2D"


# --------------------------------------------------
# material parameters
# --------------------------------------------------

mat_params = {
    "alpha": 0.16,
    "gamma": 0.5,
    "c1": 0.2,
    "c2": 0.4,
    "kappa": 1.0,
    "eta": 1e-5,
}


# --------------------------------------------------
# loading parameters
# no grad_p_bar, no stretch
# only inlet / outlet flux
# --------------------------------------------------

n_steps = 1

load_params = {
    "solid": {},
    "liquid": {},
    "air": {},
}

# solid loading = none
dim = 2
for i in range(dim):
    for j in range(dim):
        load_params["solid"]["sigma_bar_" + str(i) + str(j)] = 0.0

load_params["solid"]["sigma_bar_00"] = 0#0.15 #0.2

# air loading = none
load_params["air"]["pf"] = 0.0

# liquid loading
load_params["liquid"]["pl_bar_ini_lst"] = [0.0] * n_steps
load_params["liquid"]["pl_bar_fin_lst"] = [0.0] * n_steps

load_params["liquid"]["grad_p_bar_x_ini_lst"] = [0.0] * n_steps
load_params["liquid"]["grad_p_bar_x_fin_lst"] = [0.0] * n_steps
load_params["liquid"]["grad_p_bar_y_ini_lst"] = [0.0] * n_steps
load_params["liquid"]["grad_p_bar_y_fin_lst"] = [0.0] * n_steps

# inlet / outlet flux 
# important : keep conservative, i.e. inlet flux = outlet flux
load_params["liquid"]["Theta_in_ini_lst"]  = [0.0]
load_params["liquid"]["Theta_in_fin_lst"]  = [0.002]

load_params["liquid"]["Theta_out_ini_lst"] = [0.0]
load_params["liquid"]["Theta_out_fin_lst"] = [0.002]



# --------------------------------------------------
# run
# --------------------------------------------------
import time

t0 = time.perf_counter()

run_Acinar_Perfusion(
    dim=2,


    mesh_params={
        "dim": 2,
        "mesh_filebasename": mesh_filebasename,
    },

    mat_params={
        "skel": {"parameters": mat_params, "scaling": "no"},
        "bulk": {"parameters": mat_params, "scaling": "no"},
        "pore": {"parameters": mat_params, "scaling": "no"},
    },

    flow_params={
        "k_l": dolfin.Constant(((1, 0.0),
                                (0.0, 1))),
        "use_kozeny_carman": False,
    },

    porosity_params={
        "type": "constant",
        "val": 0.3,
    },

    bcs="kubc",

    step_params={
        "n_steps": 1,
        "Deltat_lst": [1e-1],
        "dt_ini_lst": [5e-3],
        "dt_min_lst": [1e-4],
        "dt_max_lst": [2e-2],
    },

    load_params=load_params,

    res_basename=os.path.join(res_folder, "run_with_airway_flux"),

    verbose=0,
)

t1 = time.perf_counter()
print(f"Elapsed time: {t1 - t0:.6f} s")