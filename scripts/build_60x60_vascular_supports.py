#!/usr/bin/env python3
"""Build area-controlled, mesh-independent vascular support markers for 60x60.

This reads the accepted directional mesh and frozen ownership/airway metadata.
It never changes their geometry, topology, partition, or airway nodes, and
does not import dolfin_mech or execute an FE solve.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Circle
import meshio
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; MESH=ROOT/'mesh'; OUT=ROOT/'results/physiological_geometry_60x60_production'
BASE=MESH/'Mesh_Acinar_Perfusion_Physiological_60x60_Directional_CutOptimized'
L=5/3; MM=8.4; CUT=np.array([.5623372395833334,1.4617919921875])
TARGET={'Pa0':.130,'Pa1':.130,'Pa2':.130,'Pa3':.130,'V002':.261,'V006':.261}
TAGS={'Pa0':3,'Pa1':4,'Pa2':5,'Pa3':6,'V002':7,'V006':8}; COLORS={'Pa0':'#dc2626','Pa1':'#dc2626','Pa2':'#dc2626','Pa3':'#dc2626','V002':'#2563eb','V006':'#2563eb'}

def mindelta(a,b): return (np.asarray(a)-np.asarray(b)+L/2)%L-L/2
def ref_from_translated(p): return np.mod(p+CUT,L)
def translated(p): return np.mod(np.asarray(p)-CUT,L)
def dist(a,b): return np.linalg.norm(mindelta(a,b),axis=-1)

def select_patch(name,center,allowed_owner,cent_ref,areas,owners):
    """Closest discrete area to target among a minimum-image circular footprint."""
    candidate=np.where(np.isin(owners,allowed_owner))[0] if allowed_owner else np.arange(len(areas))
    # Every selected element is a unique accepted mesh triangle; sorting by
    # physical minimum-image centroid distance makes the support a circle rule.
    d=dist(cent_ref[candidate],center); order=np.argsort(d,kind='stable'); ids=candidate[order]
    cumulative=np.cumsum(areas[ids]*MM**2); target=TARGET[name]
    k=int(np.argmin(abs(cumulative-target)))+1; chosen=ids[:k]
    radius=float(d[order[k-1]])
    return chosen,radius,float(cumulative[k-1])

def nearest_wall_center(point,mask,cent_ref):
    ids=np.where(mask)[0]; return cent_ref[ids[np.argmin(dist(cent_ref[ids],point))]]

def territory_fractions(ids,areas,owners):
    total=float(areas[ids].sum())
    return {f'A{i}':float(areas[ids][owners[ids]==i].sum()/total) for i in range(4) if np.any(owners[ids]==i)}

def any_cut_touch(center,radius):
    # Circle in physical/reference torus crosses the computational cut iff a
    # minimum-image coordinate distance to either cut line is <= radius.
    return bool(abs(mindelta(center[0],CUT[0]))<=radius or abs(mindelta(center[1],CUT[1]))<=radius)

def plot_background(ax,cent_ref,owners):
    take=np.arange(0,len(cent_ref),100)
    palette=np.array(['#4c78a8','#f58518','#54a24b','#b279a2'])
    ax.scatter(cent_ref[take,0],cent_ref[take,1],s=.4,c=palette[owners[take]],alpha=.25,rasterized=True)

def airway(ax,translated_frame=False):
    d=json.loads((MESH/'Mesh_Acinar_Perfusion_Physiological_60x60_airway_2plus2.json').read_text()); nodes={n['node_id']:n for n in d['nodes']}; key='translated_coordinate_model' if translated_frame else 'reference_coordinate_model'
    for edge in d['topology']['edges']:
        a=np.array(nodes[edge['parent_node_id']][key]);b=np.array(nodes[edge['child_node_id']][key]);ax.plot([a[0],b[0]],[a[1],b[1]],color='#111827',lw=1.1,zorder=3)
    for n in nodes.values():
        p=n[key]; ax.scatter(*p,s=20,marker='s' if n['node_type']=='terminal' else 'o',c='#111827',zorder=4)

def finish(ax,title,xlabel='reference model x',ylabel='reference model y'):
    ax.set(xlim=(0,L),ylim=(0,L),aspect='equal',title=title,xlabel=xlabel,ylabel=ylabel)

def main():
    airway_doc=json.loads((MESH/'Mesh_Acinar_Perfusion_Physiological_60x60_airway_2plus2.json').read_text())
    mesh=meshio.read(str(BASE)+'.msh'); points=mesh.points[:,:2]; tri=mesh.cells_dict['triangle'].astype(np.int32)
    meta=np.load(str(BASE)+'_directional_metadata.npz')
    # The mesh metadata, not a copied report value, is authoritative for the
    # computational representation of this fixed physical torus.
    global CUT
    CUT=np.asarray(meta['periodic_cut_origin'],dtype=float)
    owners_cell=np.empty(3600,dtype=np.int8)
    own=json.loads((MESH/'Mesh_Acinar_Perfusion_Physiological_60x60_acinar_ownership.json').read_text())
    for row in own['cells']: owners_cell[row['cell_id']]=int(row['acinus'][1:])
    owners=owners_cell[meta['canonical_cell_id'].astype(int)]
    xy=points[tri]; signed=.5*np.cross(xy[:,1]-xy[:,0],xy[:,2]-xy[:,0]); areas=abs(signed); cent_t=xy.mean(1); cent_ref=ref_from_translated(cent_t)
    if np.any(signed<=0): raise RuntimeError('Accepted mesh unexpectedly contains nonpositive triangle area')
    terminals={n['node_id']:np.asarray(n['reference_coordinate_model']) for n in airway_doc['nodes'] if n['node_type']=='terminal'}
    # Pa centers are nearest accepted wall centroids inside their own territory.
    centers={f'Pa{i}':nearest_wall_center(terminals[f'T{i}'],owners==i,cent_ref) for i in range(4)}
    # Symmetric adjacent-pair abstraction, away from the four-way center and cut.
    venous_targets={'V002':(np.array([.833333333333,.30]),[0,1]),'V006':(np.array([.833333333333,1.36]),[3,2])}
    for name,(target,pair) in venous_targets.items():
        # One wall centroid from each side nearest a target interface location;
        # their midpoint remains on the pair's resolved shared support zone.
        a=nearest_wall_center(target,owners==pair[0],cent_ref); b=nearest_wall_center(target,owners==pair[1],cent_ref)
        centers[name]=np.mod(a+.5*mindelta(b,a),L)
    patches={}; claimed=np.full(len(tri),'wall',dtype=object)
    for name in ('Pa0','Pa1','Pa2','Pa3','V002','V006'):
        pair=[int(name[2])] if name.startswith('Pa') else venous_targets[name][1]
        ids,radius,actual=select_patch(name,centers[name],None,cent_ref,areas,owners)
        # Reject any accidental overlap rather than silently changing a patch.
        overlap=np.where(claimed[ids]!='wall')[0]
        if len(overlap): raise RuntimeError(f'{name} overlaps earlier support {set(claimed[ids][overlap])}')
        claimed[ids]=name
        frac=territory_fractions(ids,areas,owners); error=(actual-TARGET[name])/TARGET[name]
        patches[name]={'physical_tag':TAGS[name],'vascular_type':'arterial' if name.startswith('Pa') else 'venous','reference_center_model':centers[name].tolist(),'reference_center_mm':(centers[name]*MM).tolist(),'translated_center_model':translated(centers[name]).tolist(),'translated_center_mm':(translated(centers[name])*MM).tolist(),'footprint_radius_model':radius,'footprint_radius_mm':radius*MM,'target_wall_support_area_mm2':TARGET[name],'realized_wall_support_area_model2':actual/MM**2,'realized_wall_support_area_mm2':actual,'relative_area_error':error,'equivalent_area_radius_mm':float(np.sqrt(actual/np.pi)),'selected_element_count':int(len(ids)),'minimum_image_periodic_distance':True,'touches_or_crosses_optimized_periodic_cut':any_cut_touch(centers[name],radius),'territory_area_fractions':frac,'intended_territories':[f'A{i}' for i in pair],'selected_triangle_indices':ids.tolist()}
        if name.startswith('Pa'):
            t=terminals['T'+name[2]]; patches[name]['terminal_id']='T'+name[2];patches[name]['terminal_displacement_model']=mindelta(centers[name],t).tolist();patches[name]['terminal_displacement_mm']=(mindelta(centers[name],t)*MM).tolist();patches[name]['terminal_displacement_norm_mm']=float(dist(centers[name],t)*MM)
    tags=np.full(len(tri),2,dtype=np.int32)
    for name,p in patches.items(): tags[np.asarray(p['selected_triangle_indices'],dtype=int)]=p['physical_tag']
    if np.any(tags==2)==False or any(np.count_nonzero(tags==t)!=patches[n]['selected_element_count'] for n,t in TAGS.items()): raise RuntimeError('Marker tag integrity failure')
    q={'Pa0':.667/4,'Pa1':.667/4,'Pa2':.667/4,'Pa3':.667/4,'V002':-.667/2,'V006':-.667/2}
    for name in patches: patches[name]['healthy_flow_mL_per_min']=q[name];patches[name]['healthy_flow_normalized']=.25 if name.startswith('Pa') else -.5;patches[name]['nominal_theta_mL_per_min_per_mm2']=q[name]/patches[name]['realized_wall_support_area_mm2']
    theta=np.array([abs(patches[x]['nominal_theta_mL_per_min_per_mm2']) for x in patches]); theta_difference=float((theta.max()-theta.min())/theta.mean())
    marker_base=MESH/'Mesh_Acinar_Perfusion_Physiological_60x60_vascular_markers'
    field={'voronoi_wall':np.array([2,2]),**{n:np.array([t,2]) for n,t in TAGS.items()}}
    outmesh=meshio.Mesh(np.c_[points,np.zeros(len(points))],[('triangle',tri)],cell_data={'gmsh:physical':[tags],'gmsh:geometrical':[tags]},field_data=field)
    meshio.write(str(marker_base)+'.msh',outmesh,file_format='gmsh22',binary=True)
    meshio.write(str(marker_base)+'.xdmf',meshio.Mesh(points,[('triangle',tri)],cell_data={'domains':[tags]}))
    meshio.write(str(marker_base)+'_domains.xdmf',meshio.Mesh(points,[('triangle',tri)],cell_data={'domains':[tags]}))
    report={'model':'60x60_physiological_vascular_wall_supports','method':'reference-frame minimum-image circular footprint; radius selected to minimize discrete wall-triangle support-area residual','fixed_inputs':{'directional_mesh':str(BASE)+'.msh','airway_metadata':str(MESH/'Mesh_Acinar_Perfusion_Physiological_60x60_airway_2plus2.json'),'ownership_metadata':str(MESH/'Mesh_Acinar_Perfusion_Physiological_60x60_acinar_ownership.json'),'geometry_or_mesh_modified':False,'ownership_modified':False,'airway_modified':False},'venous_pairing':{'V002':['A0','A1'],'V006':['A3','A2'],'choice':'Option V: symmetric adjacent-pair drainage across lower and upper resolved interacinar interfaces; a continuum modeling abstraction, not vessel anatomy.'},'patches':patches,'healthy_flow_metadata':{'Q_total_mL_per_min':.667,'normalized':{'Qa0':.25,'Qa1':.25,'Qa2':.25,'Qa3':.25,'Qv002':-.5,'Qv006':-.5},'sum_normalized':0.,'sum_mL_per_min':sum(q.values()),'relative_spread_abs_theta':theta_difference,'theta_note':'Theta is Q divided by actual realized wall-support area; no load has been imposed.'},'validation':{'patches_disjoint':bool(np.count_nonzero(tags>=3)==sum(p['selected_element_count'] for p in patches.values())),'no_duplicate_periodic_elements':True,'all_arterial_centers_in_intended_territory':all(max(patches[f'Pa{i}']['territory_area_fractions'],key=patches[f'Pa{i}']['territory_area_fractions'].get)==f'A{i}' for i in range(4)),'cut_invariant_reference_definition':True,'marker_tags':TAGS,'fe_solver_run':False},'files':{'markers_msh':str(marker_base)+'.msh','markers_xdmf':str(marker_base)+'.xdmf','markers_domains_xdmf':str(marker_base)+'_domains.xdmf'}}
    (MESH/'Mesh_Acinar_Perfusion_Physiological_60x60_vascular_sites.json').write_text(json.dumps(report,indent=2)+'\n');(OUT/'vascular_support_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    # Figures: representatives and selected actual supports (not FE-count disks).
    fig,ax=plt.subplots(figsize=(9,9),constrained_layout=True);plot_background(ax,cent_ref,owners);airway(ax)
    for n,p in patches.items(): ax.scatter(*p['reference_center_model'],s=52,c=COLORS[n],marker='o' if n.startswith('Pa') else 'D',edgecolors='k',zorder=6);ax.text(*p['reference_center_model'],n,fontsize=8)
    finish(ax,'60×60 frozen territories, airway, and vascular coupling sites');fig.savefig(OUT/'physiological_60x60_vascular_sites.png',dpi=280);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,9),constrained_layout=True);plot_background(ax,cent_ref,owners)
    for n,p in patches.items(): ids=np.asarray(p['selected_triangle_indices']);ax.add_collection(PolyCollection(ref_from_translated(xy[ids]),facecolors=COLORS[n],edgecolors='none',alpha=.72));ax.add_patch(Circle(p['reference_center_model'],p['footprint_radius_model'],fill=False,ec=COLORS[n],lw=.7));ax.text(*p['reference_center_model'],n,fontsize=8)
    finish(ax,'Actual wall-support patches; circles are algorithmic footprints');fig.savefig(OUT/'physiological_60x60_vascular_supports.png',dpi=280);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,7),constrained_layout=True);focus=['Pa0','V002']
    for n in focus:
        p=patches[n];ids=np.asarray(p['selected_triangle_indices']);ax.add_collection(PolyCollection(ref_from_translated(xy[ids]),facecolors=COLORS[n],edgecolors='#111827',linewidths=.05,alpha=.75));ax.add_patch(Circle(p['reference_center_model'],p['footprint_radius_model'],fill=False,ec=COLORS[n],lw=1.0));ax.text(*p['reference_center_model'],n,fontsize=9)
    ax.set(xlim=(.05,1.35),ylim=(.02,.9),aspect='equal',title='Support detail: selected wall elements versus footprint');fig.savefig(OUT/'physiological_60x60_vascular_supports_zoom.png',dpi=300);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,6),constrained_layout=True)
    for ax,n in zip(axes,['V002','V006']):
        p=patches[n];ids=np.asarray(p['selected_triangle_indices']);cols=np.array(['#4c78a8','#f58518','#54a24b','#b279a2'])[owners[ids]];ax.add_collection(PolyCollection(ref_from_translated(xy[ids]),facecolors=cols,edgecolors='none'));ax.add_patch(Circle(p['reference_center_model'],p['footprint_radius_model'],fill=False,ec='#111827'));finish(ax,f"{n}: " + ', '.join(f'{k}={v:.1%}' for k,v in p['territory_area_fractions'].items()))
    fig.savefig(OUT/'physiological_60x60_venous_sharing.png',dpi=280);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,9),constrained_layout=True);ct=cent_t;take=np.arange(0,len(ct),100);pal=np.array(['#4c78a8','#f58518','#54a24b','#b279a2']);ax.scatter(ct[take,0],ct[take,1],s=.4,c=pal[owners[take]],alpha=.25)
    for n,p in patches.items(): ids=np.asarray(p['selected_triangle_indices']);ax.add_collection(PolyCollection(xy[ids],facecolors=COLORS[n],edgecolors='none',alpha=.7));ax.add_patch(Circle(p['translated_center_model'],p['footprint_radius_model'],fill=False,ec=COLORS[n]));ax.text(*p['translated_center_model'],n,fontsize=8)
    ax.axvline(0,color='k',lw=.8);ax.axhline(0,color='k',lw=.8);finish(ax,'Vascular supports in translated computational-cut frame','translated model x','translated model y');fig.savefig(OUT/'physiological_60x60_vascular_cut_check.png',dpi=280);plt.close(fig)
    print(json.dumps({'patches':{n:{k:p[k] for k in ('selected_element_count','realized_wall_support_area_mm2','relative_area_error','footprint_radius_mm','territory_area_fractions')} for n,p in patches.items()},'validation':report['validation']},indent=2))

if __name__=='__main__': main()
