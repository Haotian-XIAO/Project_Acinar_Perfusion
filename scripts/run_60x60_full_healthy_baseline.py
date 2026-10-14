#!/usr/bin/env python3
"""One zero-macro-strain, healthy-flow coupled solve on frozen 60x60 assets.

This is an adapter around the successful 20x20 formulation, not a new weak
form.  The established solver is dimensionless: k_wall=1 is the code's
baseline intrinsic-permeability coefficient and its pressure/flow units have
no documented SI calibration.  The prescribed physical 0.667 mL/min is
therefore represented by the explicit project convention Q_model=Q/(1
mL/min), while its SI value is retained in metadata.  No out-of-plane depth,
Pa conversion, or intrinsic k is invented here.

--geometry G20/G12 runs this script's own code path (loading adapter, final
output, metadata, regional post-processing) on a small registered geometry
with that geometry's established loading (Q_total = 1.0), for end-to-end
rehearsals before a G60 run.  The default is G60.

--alpha-pa2 A sets the Pa2 arterial suppression at unchanged total flow:
Q_Pa2 = (A/4) Q_total, Q_Pa0 = Q_Pa1 = Q_Pa3 = (4 - A)/12 Q_total, venous
sinks Q_total/2 each.  Without it the healthy equal split is used.
"""
from __future__ import annotations
import argparse, json, os, time
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
OWNERSHIP_NPZ=ROOT/'mesh/Mesh_Acinar_Perfusion_Physiological_60x60_acinar_ownership.npz'
CUT_METADATA=ROOT/'mesh/Mesh_Acinar_Perfusion_Physiological_60x60_Directional_CutOptimized_directional_metadata.npz'
# Partition label written into the translated metadata: the G60 ownership rule
# (whole canonical cells, quadrants by polygon centroid), not the G12/G20 one.
G60_PARTITION_METHOD='whole_canonical_cell_quadrant_by_polygon_centroid'
Q_TOTAL=0.667
FLOWS={'Pa0':Q_TOTAL/4,'Pa1':Q_TOTAL/4,'Pa2':Q_TOTAL/4,'Pa3':Q_TOTAL/4,'V002':Q_TOTAL/2,'V006':Q_TOTAL/2}
# Rehearsal geometries: registered in the vascular driver, run with their established loading.
REHEARSAL_Q_TOTAL=1.0
GEOMETRY='G60'
ALPHA_PA2=None

def healthy_flows(q_total):
    return {'Pa0':q_total/4,'Pa1':q_total/4,'Pa2':q_total/4,'Pa3':q_total/4,'V002':q_total/2,'V006':q_total/2}

def alpha_flows(q_total,alpha):
    """Pa2 suppression at fixed total flow; Pa0, Pa1 and Pa3 share the remainder equally."""
    if not 0.0<=alpha<=1.0: raise ValueError(f'alpha_Pa2 must lie in [0, 1], got {alpha}')
    compensated=(4.0-alpha)/12.0*q_total
    return {'Pa0':compensated,'Pa1':compensated,'Pa2':alpha/4.0*q_total,'Pa3':compensated,'V002':q_total/2,'V006':q_total/2}

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

def macroscopic_displacement_gradient(problem,dV,area):
    """Converged U_bar, evaluated from the mixed solution.

    U_bar is a constant ("R") tensor subsolution.  In the mixed space its
    ``subfunc`` is ``dolfin.split(sol_func)[k]``, a UFL ListTensor view without
    a vector, so each component is evaluated as its domain mean (exact for a
    constant).
    """
    U_bar=problem.macroscopic_stretch_subsol.subfunc
    dim=problem.mesh.geometry().dim()
    macro=np.array([[float(dolfin.assemble(U_bar[i,j]*dV))/area for j in range(dim)] for i in range(dim)])
    if not np.all(np.isfinite(macro)): raise RuntimeError(f'Non-finite macroscopic displacement gradient: {macro.tolist()}')
    return macro

def write_projected_metadata(mesh,domains,problem,darcy):
    """Called after convergence by the unmodified established solver path."""
    old_write(mesh,domains,problem,darcy)
    dV=problem.dV; area=float(dolfin.assemble(dolfin.Constant(1.)*dV))
    J=base.project_scalar_cell_data(problem.kinematics.J,mesh)
    U=base.project_vector_cell_data(problem.U_tot,mesh)
    macro=macroscopic_displacement_gradient(problem,dV,area)
    mechanics={'J_min':float(J.min()),'J_mean':float(np.dot(J, np.ones_like(J))/len(J)),'J_max':float(J.max()),
               'J_mean_convention':'unweighted mean of DG0 cell values','J_area_weighted_mean':float(dolfin.assemble(problem.kinematics.J*dV))/area,
               'displacement_magnitude_min':float(np.linalg.norm(U,axis=1).min()),'displacement_magnitude_max':float(np.linalg.norm(U,axis=1).max()),
               'macroscopic_displacement_gradient_flat':macro.reshape(-1).tolist(),'macroscopic_displacement_gradient_convention':'row-major [U_bar_00, U_bar_01, U_bar_10, U_bar_11], domain mean of the converged mixed-solution component',
               'F_bar':(np.eye(len(macro))+macro).tolist(),
               'macro_zero_verified':bool(np.max(abs(macro))<1e-10),'local_displacement_free_and_nonzero':bool(np.linalg.norm(U,axis=1).max()>0)}
    dims={s.name:int(s.func.function_space().dim()) for s in problem.subsols}
    (RESULTS/'mechanical_qois.json').write_text(json.dumps(mechanics,indent=2)+'\n')
    (RESULTS/'space_report.json').write_text(json.dumps({'mixed_dofs':int(problem.sol_func.function_space().dim()),'subspace_dofs':dims,'mesh_vertices':mesh.num_vertices(),'mesh_cells':mesh.num_cells()},indent=2)+'\n')

def periodic_mean(points,L):
    """Circular mean of points on the periodic square [0,L)^2 (label position only)."""
    angle=2*np.pi*np.asarray(points)/L
    return np.mod(np.arctan2(np.sin(angle).mean(axis=0),np.cos(angle).mean(axis=0))*L/(2*np.pi),L)

def translated_ownership_files():
    """G60 ownership in the mesh (cut) frame, in the metadata/summary format of the regional post-processing."""
    own=json.loads(OWNERSHIP.read_text()); npz=np.load(str(OWNERSHIP_NPZ)); cut_md=np.load(str(CUT_METADATA))
    cut=cut_md['periodic_cut_origin']; L=float(cut_md['domain_length'])
    if not np.array_equal(cut,npz['periodic_cut_origin']): raise RuntimeError(f'Periodic cut origin differs: {cut} (mesh metadata) vs {npz["periodic_cut_origin"]} (ownership npz)')
    if not np.isclose(L,5/3,rtol=0,atol=1e-12): raise RuntimeError(f'Unexpected G60 domain length {L}')
    cells_json=sorted(own['cells'],key=lambda x:int(x['cell_id']))
    labels=np.array([int(x['acinus'][1:]) for x in cells_json]); seeds=np.array([x['seed_reference_model'] for x in cells_json])
    if not (np.array_equal([int(x['cell_id']) for x in cells_json],npz['canonical_cell_id']) and np.array_equal(labels,npz['acinus_id']) and np.array_equal(seeds,npz['seed_reference'])):
        raise RuntimeError('G60 ownership JSON and NPZ disagree (cell ids, acinus labels or seeds)')
    cells=[]
    for x in cells_json:
        cells.append({'cell_id':x['cell_id'],'seed_coordinate':np.mod(np.asarray(x['seed_reference_model'])-cut,L).tolist(),'acinus_id':int(x['acinus'][1:])})
    md={'main_partition_method':G60_PARTITION_METHOD,'partition_source':{'ownership':base.repo_relative(OWNERSHIP),'assignment':own['assignment'],'coordinate_transform_for_mesh':own['coordinate_transform_for_mesh']},
        'periodic_cell':{'bounds':[0.,0.,L,L],'a1':[L,0.],'a2':[0.,L]},'cells':cells}
    area=npz['cell_area_model2']; centroids=np.mod(npz['centroid_reference']-cut,L)
    territories=[{'acinus_id':i,'number_of_cells':int(np.sum(labels==i)),'area':float(area[labels==i].sum()),'area_fraction':float(area[labels==i].sum()/area.sum()),
                  'shape':{'periodic_centroid':periodic_mean(centroids[labels==i],L).tolist(),'periodic_centroid_convention':'circular mean of translated canonical-cell centroids'}} for i in range(4)]
    summary={'main_model':{'method':G60_PARTITION_METHOD,'area_source':base.repo_relative(OWNERSHIP_NPZ)+':cell_area_model2','territories':territories}}
    (RESULTS/'translated_acinar_metadata.json').write_text(json.dumps(md,indent=2)+'\n')
    (RESULTS/'acinar_summary_for_postprocessing.json').write_text(json.dumps(summary,indent=2)+'\n')
    return RESULTS/'translated_acinar_metadata.json',RESULTS/'acinar_summary_for_postprocessing.json',(G60_PARTITION_METHOD,)

def postprocess():
    import postprocess_regional_perfusion as regional
    if GEOMETRY=='G60':
        metadata,summary,methods=translated_ownership_files()
    else:
        entry=base.GEOMETRIES[GEOMETRY]; metadata,summary,methods=entry['acinar_metadata'],entry['acinar_summary'],('airway_constrained_space_filling',)
    regional.run(RESULTS,Path(str(base.RESULT_BASE)+'.xdmf'),RESULTS/'final_fields.vtu',metadata,summary,accepted_partition_methods=methods)

def unit_audit():
    if GEOMETRY=='G60':
        return {'coordinate_unit':'model length; 1 model unit = 8.4 mm','pressure_unit':'undocumented code pressure unit; no Pa/mmHg conversion verified','source_coefficient_unit':'Q_model per model-area in Theta*p_test*dx','darcy_coefficient':{'symbol':'k_wall / k_l0','value':1.0,'interpretation':'dimensionless established baseline intrinsic-permeability coefficient','SI_intrinsic_permeability_claimed':False},'flow_mapping':{'Q_total_mL_per_min':Q_TOTAL,'Q_total_m3_per_s':Q_TOTAL*1e-6/60,'model_convention':'Q_model=Q_physical/(1 mL/min), preserving the existing normalized weak-form scale','out_of_plane_depth':'not specified; no 3D-to-2D thickness inferred'},'gas_pressure':{'value':0.0,'unit':'code pressure unit','provenance':'existing healthy baseline GLOBAL_GAS_PRESSURE'}}
    return {'geometry':GEOMETRY,'purpose':'rehearsal of the G60 script code path on a registered small geometry; no physical calibration','coordinate_unit':'model length of the registered geometry; no physical length claimed','pressure_unit':'undocumented code pressure unit; no Pa/mmHg conversion verified','darcy_coefficient':{'symbol':'k_wall / k_l0','value':1.0,'SI_intrinsic_permeability_claimed':False},'flow_mapping':{'Q_total_model':Q_TOTAL,'physical_unit':None,'model_convention':'established dimensionless loading of this geometry','out_of_plane_depth':'not specified'},'gas_pressure':{'value':0.0,'unit':'code pressure unit','provenance':'existing healthy baseline GLOBAL_GAS_PRESSURE'}}

def main(argv=None):
    global RESULTS,GEOMETRY,Q_TOTAL,FLOWS,ALPHA_PA2
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir',type=Path,default=RESULTS.relative_to(ROOT),help='output directory (repository-relative)')
    parser.add_argument('--geometry',choices=('G60','G20','G12'),default='G60',help='G60 (default) or a registered small geometry for an end-to-end rehearsal of this code path')
    parser.add_argument('--alpha-pa2',type=float,default=None,help='Pa2 arterial suppression alpha in [0, 1] at fixed Q_total (default: healthy equal split)')
    base.add_solver_option_arguments(parser,foi_default='iterative')
    args=parser.parse_args(argv)
    RESULTS=ROOT/args.results_dir; GEOMETRY=args.geometry; ALPHA_PA2=args.alpha_pa2
    if GEOMETRY!='G60':
        Q_TOTAL=REHEARSAL_Q_TOTAL; FLOWS=healthy_flows(Q_TOTAL)
    if ALPHA_PA2 is not None:
        FLOWS=alpha_flows(Q_TOTAL,ALPHA_PA2)
    case='healthy_baseline' if ALPHA_PA2 is None else f'alpha_Pa2_{ALPHA_PA2:g}'
    solver_options=base.solver_options_from_args(args,foi_default='iterative')
    # Files written by scripts/tools/start_run.sh and run_with_meta.py before this script starts.
    launcher_files={'STATUS','git_state.json','job.sh','job.pid','run.log','run_meta.json'}
    allowed_precreated={'solver.log','runtime.log'}|launcher_files
    if RESULTS.exists() and any(p.name not in allowed_precreated for p in RESULTS.iterdir()): raise RuntimeError(f'Refusing to overwrite populated result directory: {RESULTS}')
    RESULTS.mkdir(parents=True,exist_ok=True)
    unit=unit_audit()
    (RESULTS/'unit_audit.json').write_text(json.dumps(unit,indent=2)+'\n')
    if ALPHA_PA2 is not None:
        # Written before the solve so that an aborted run still records its loading case.
        experiment={'suppressed_arterial_source':'Pa2','alpha_Pa2':ALPHA_PA2,'Q_total':Q_TOTAL,'target_flows':FLOWS,'definition':'Q_Pa2=(alpha/4) Q_total; Q_Pa0=Q_Pa1=Q_Pa3=(4-alpha)/12 Q_total; venous sinks Q_total/2 each'}
        (RESULTS/'case_definition.json').write_text(json.dumps(experiment,indent=2)+'\n')
    if GEOMETRY=='G60':
        base.GEOMETRY_LABEL='G60'; base.ACINAR_METADATA_JSON=OWNERSHIP  # ownership is not read by the driver here (global gas pressure); recorded for identification
        base.MESH_BASE=MARKER_BASE; base.MARKER_JSON=SITES; base.RESULT_BASE=RESULTS/f'{case}_60x60'; base.load_mesh_and_markers=load_60_mesh_and_markers
    else:
        base.select_geometry(GEOMETRY); base.GEOMETRY_LABEL=GEOMETRY; base.RESULT_BASE=RESULTS/f'{case}_{GEOMETRY}'
    base.RESULTS=RESULTS; base.Q_TOTAL=Q_TOTAL; base.TARGET_FLOWS=dict(FLOWS); base.MACROSCOPIC_DISPLACEMENT_GRADIENT=((0.,0.),(0.,0.)); base.make_regional_gas_boundary_markers=global_gas_boundary_only
    if solver_options is None:
        base.dmech.FOI.update_project=iterative_foi_projection
    else:
        base.SOLVER_OPTIONS=solver_options  # FOI projection, linear solver and diagnostics handled by the driver
    global old_write; old_write=base.write_final_vtu; base.write_final_vtu=write_projected_metadata
    start=time.perf_counter(); base.run(); wall=time.perf_counter()-start
    cfg=json.loads((RESULTS/'run_config.json').read_text()); cfg['physical_loading']=unit['flow_mapping'];cfg['wall_time_seconds']=wall;cfg['macro_condition']='U_bar=0; F_bar=I; periodic local displacement fluctuations free'
    if ALPHA_PA2 is not None: cfg['experiment']=experiment
    (RESULTS/'run_config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    postprocess()
    print(json.dumps({'status':'complete','wall_time_seconds':wall,'results':str(RESULTS)},indent=2))
if __name__=='__main__': os.chdir(ROOT); main()
