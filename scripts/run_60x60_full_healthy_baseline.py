#!/usr/bin/env python3
"""One zero-macro-strain, healthy-flow coupled solve on frozen 60x60 assets.

This is an adapter around the successful 20x20 formulation, not a new weak
form.  The established solver is dimensionless: k_wall=1 is the code's
baseline intrinsic-permeability coefficient and its pressure/flow units have
no documented SI calibration.  The prescribed physical 0.667 mL/min is
therefore represented by the explicit project convention Q_model=Q/(1
mL/min), while its SI value is retained in metadata.  No out-of-plane depth,
Pa conversion, or intrinsic k is invented here.
"""
from __future__ import annotations
import json, os, time
from pathlib import Path
import dolfin, meshio, numpy as np
import test_acinar_perfusion_vascular_baseline as base

ROOT=Path(__file__).resolve().parents[1]
# Keep the direct-UMFPACK attempt intact for provenance.  This retry changes
# only field-output projection: CG mass-matrix solves avoid the prohibitive
# direct-factorization memory cost before the first Newton increment.
RESULTS=ROOT/'results/physiological_60x60_full_healthy_baseline_iterative_foi'
MARKER_BASE=ROOT/'mesh/Mesh_Acinar_Perfusion_Physiological_60x60_vascular_markers'
SITES=ROOT/'mesh/Mesh_Acinar_Perfusion_Physiological_60x60_vascular_sites.json'
OWNERSHIP=ROOT/'mesh/Mesh_Acinar_Perfusion_Physiological_60x60_acinar_ownership.json'
Q_TOTAL=0.667
FLOWS={'Pa0':Q_TOTAL/4,'Pa1':Q_TOTAL/4,'Pa2':Q_TOTAL/4,'Pa3':Q_TOTAL/4,'V002':Q_TOTAL/2,'V006':Q_TOTAL/2}

def iterative_foi_projection(self):
    """L2-project a field of interest without UMFPACK factorization.

    FOIs do not enter the residual.  CG with Jacobi preconditioning retains
    the same variational projection while making the large 60x60 setup fit in
    memory; the nonlinear mixed-system solver is deliberately untouched.
    """
    dolfin.project(v=self.expr, V=self.fs, function=self.func,
                   form_compiler_parameters=self.form_compiler_parameters,
                   solver_type='cg', preconditioner_type='jacobi')

def load_60_mesh_and_markers():
    mesh=dolfin.Mesh()
    with dolfin.XDMFFile(str(MARKER_BASE)+'.xdmf') as f: f.read(mesh)
    mvc=dolfin.MeshValueCollection('size_t',mesh,mesh.topology().dim())
    with dolfin.XDMFFile(str(MARKER_BASE)+'_domains.xdmf') as f: f.read(mvc,'domains')
    domains=dolfin.MeshFunction('size_t',mesh,mvc)
    support=json.loads(SITES.read_text())
    markers={name:{'physical_tag':p['physical_tag'],'vascular_type':p['vascular_type'],
                   'reference_center_model':p['reference_center_model'],
                   'realized_wall_support_area_mm2':p['realized_wall_support_area_mm2']}
             for name,p in support['patches'].items()}
    if {k:int(v['physical_tag']) for k,v in markers.items()} != {'Pa0':3,'Pa1':4,'Pa2':5,'Pa3':6,'V002':7,'V006':8}:
        raise RuntimeError('Frozen 60x60 marker-tag integrity failure')
    return mesh,domains,{'markers':markers,'support_source':str(SITES)}

def global_gas_boundary_only(mesh,boundaries):
    # Global p_g=0 uses the original exterior measure; regional gas tagging is
    # irrelevant to this loading and is intentionally not reconstructed.
    return boundaries,dolfin.Measure('exterior_facet',domain=mesh,subdomain_data=boundaries),{
        'status':'not_needed_for_global_zero_gas_pressure','global_gas_pressure':0.0}

def write_projected_metadata(mesh,domains,problem,darcy):
    """Called after convergence by the unmodified established solver path."""
    old_write(mesh,domains,problem,darcy)
    dV=problem.dV; area=float(dolfin.assemble(dolfin.Constant(1.)*dV))
    J=base.project_scalar_cell_data(problem.kinematics.J,mesh)
    U=base.project_vector_cell_data(problem.U_tot,mesh)
    macro=problem.macroscopic_stretch_subsol.subfunc.vector().get_local().reshape(-1)
    mechanics={'J_min':float(J.min()),'J_mean':float(np.dot(J, np.ones_like(J))/len(J)),'J_max':float(J.max()),
               'displacement_magnitude_min':float(np.linalg.norm(U,axis=1).min()),'displacement_magnitude_max':float(np.linalg.norm(U,axis=1).max()),
               'macroscopic_displacement_gradient_flat':macro.tolist(),'F_bar':[[1.,0.],[0.,1.]],
               'macro_zero_verified':bool(np.max(abs(macro))<1e-10),'local_displacement_free_and_nonzero':bool(np.linalg.norm(U,axis=1).max()>0)}
    dims={s.name:int(s.func.function_space().dim()) for s in problem.subsols}
    (RESULTS/'mechanical_qois.json').write_text(json.dumps(mechanics,indent=2)+'\n')
    (RESULTS/'space_report.json').write_text(json.dumps({'mixed_dofs':int(problem.sol_func.function_space().dim()),'subspace_dofs':dims,'mesh_vertices':mesh.num_vertices(),'mesh_cells':mesh.num_cells()},indent=2)+'\n')

def translated_ownership_files():
    own=json.loads(OWNERSHIP.read_text()); cut=np.load(str(ROOT/'mesh/Mesh_Acinar_Perfusion_Physiological_60x60_Directional_CutOptimized_directional_metadata.npz'))['periodic_cut_origin']; L=5/3
    cells=[]
    for x in own['cells']:
        cells.append({'cell_id':x['cell_id'],'seed_coordinate':np.mod(np.asarray(x['seed_reference_model'])-cut,L).tolist(),'acinus_id':int(x['acinus'][1:])})
    md={'periodic_cell':{'a1':[L,0.],'a2':[0.,L]},'cells':cells}
    summary={'main_model':{'territories':[{'acinus_id':i,'area_fraction':.25} for i in range(4)]}}
    (RESULTS/'translated_acinar_metadata.json').write_text(json.dumps(md,indent=2)+'\n')
    (RESULTS/'acinar_summary_for_postprocessing.json').write_text(json.dumps(summary,indent=2)+'\n')

def postprocess():
    import postprocess_regional_perfusion as regional
    translated_ownership_files()
    regional.run(RESULTS,RESULTS/'healthy_baseline_60x60.xdmf',RESULTS/'final_fields.vtu',RESULTS/'translated_acinar_metadata.json',RESULTS/'acinar_summary_for_postprocessing.json')

def main():
    allowed_precreated={'solver.log','runtime.log'}
    if RESULTS.exists() and any(p.name not in allowed_precreated for p in RESULTS.iterdir()): raise RuntimeError(f'Refusing to overwrite populated result directory: {RESULTS}')
    RESULTS.mkdir(parents=True,exist_ok=True)
    unit={'coordinate_unit':'model length; 1 model unit = 8.4 mm','pressure_unit':'undocumented code pressure unit; no Pa/mmHg conversion verified','source_coefficient_unit':'Q_model per model-area in Theta*p_test*dx','darcy_coefficient':{'symbol':'k_wall / k_l0','value':1.0,'interpretation':'dimensionless established baseline intrinsic-permeability coefficient','SI_intrinsic_permeability_claimed':False},'flow_mapping':{'Q_total_mL_per_min':Q_TOTAL,'Q_total_m3_per_s':Q_TOTAL*1e-6/60,'model_convention':'Q_model=Q_physical/(1 mL/min), preserving the existing normalized weak-form scale','out_of_plane_depth':'not specified; no 3D-to-2D thickness inferred'},'gas_pressure':{'value':0.0,'unit':'code pressure unit','provenance':'existing healthy baseline GLOBAL_GAS_PRESSURE'}}
    (RESULTS/'unit_audit.json').write_text(json.dumps(unit,indent=2)+'\n')
    base.MESH_BASE=MARKER_BASE; base.MARKER_JSON=SITES; base.RESULTS=RESULTS; base.RESULT_BASE=RESULTS/'healthy_baseline_60x60'; base.Q_TOTAL=Q_TOTAL; base.TARGET_FLOWS=dict(FLOWS); base.MACROSCOPIC_DISPLACEMENT_GRADIENT=((0.,0.),(0.,0.)); base.load_mesh_and_markers=load_60_mesh_and_markers; base.make_regional_gas_boundary_markers=global_gas_boundary_only
    base.dmech.FOI.update_project=iterative_foi_projection
    global old_write; old_write=base.write_final_vtu; base.write_final_vtu=write_projected_metadata
    start=time.perf_counter(); base.run(); wall=time.perf_counter()-start
    cfg=json.loads((RESULTS/'run_config.json').read_text()); cfg['physical_loading']=unit['flow_mapping'];cfg['wall_time_seconds']=wall;cfg['macro_condition']='U_bar=0; F_bar=I; periodic local displacement fluctuations free';(RESULTS/'run_config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    postprocess()
    print(json.dumps({'status':'complete','wall_time_seconds':wall,'results':str(RESULTS)},indent=2))
if __name__=='__main__': os.chdir(ROOT); main()
