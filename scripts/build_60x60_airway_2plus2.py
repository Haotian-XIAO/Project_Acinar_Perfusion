#!/usr/bin/env python3
"""Create metadata-only 2+2 airway morphology for the frozen 60x60 partition.

No geometry, mesh, cell ownership, material marker, or FE object is changed.
All physical nodes are first defined in the fixed reference torus frame.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np

import generate_directional_70x70_mesh as gm

ROOT = Path(__file__).resolve().parents[1]
MESH = ROOT / "mesh"
OUT = ROOT / "results/physiological_geometry_60x60_production"
L, UNIT_MM = 5.0 / 3.0, 8.4
SEED, DOI, OFFSET = 1, .3, .0010334905403467218
CUT = np.array([.5623372395833334, 1.4617919921875])
PALETTE = ["#4c78a8", "#f58518", "#54a24b", "#b279a2"]


def poly_area(poly):
    return gm.polygon_area(poly)


def centroid(poly):
    x, y = poly[:, 0], poly[:, 1]
    cross = x*np.roll(y, -1)-np.roll(x, -1)*y
    area = cross.sum()/2
    return np.array([((x+np.roll(x,-1))*cross).sum()/(6*area),
                     ((y+np.roll(y,-1))*cross).sum()/(6*area)])


def point_segment_distance(p, a, b):
    d = b-a
    t = np.clip(np.dot(p-a, d)/np.dot(d,d), 0., 1.)
    return float(np.linalg.norm(p-(a+t*d)))


def seg_cross(a, b, c, d):
    """Strict crossing only; shared tree vertices are excluded by caller."""
    def orient(p,q,r): return np.cross(q-p,r-p)
    x1,x2,x3,x4=orient(a,b,c),orient(a,b,d),orient(c,d,a),orient(c,d,b)
    return bool(x1*x2 < -1e-14 and x3*x4 < -1e-14)


def angle_deg(parent, first, second):
    u, v = first-parent, second-parent
    return float(math.degrees(math.acos(np.clip(np.dot(u,v)/np.linalg.norm(u)/np.linalg.norm(v),-1,1))))


def build_reference_cells():
    gm.GRID_N, gm.DOI, gm.RNG_SEED, gm.OFFSET_MODEL = 60, DOI, SEED, OFFSET
    base, cells = gm.build_cells(L, 60, cut_origin=(0.,0.))
    ownership = json.loads((MESH/"Mesh_Acinar_Perfusion_Physiological_60x60_acinar_ownership.json").read_text())
    labels = np.empty(3600, dtype=np.int8)
    for item in ownership["cells"]: labels[item["cell_id"]] = int(item["acinus"][1:])
    return base, cells, labels, ownership


def choose_terminals(cells, labels):
    # Purposeful modest offsets from quadrant centroids: all targets remain
    # >1.5 mm from the macro x-boundary and >1.2 mm from the optimized cut.
    desired = np.array([[.65,.42], [1.05,.42], [1.05,1.15], [.65,1.15]])
    paired = gm.pair_ridges(cells, L)
    boundary_edges = [[] for _ in range(4)]
    for records in paired["groups"].values():
        c0, c1 = records[0]["cell"], records[1]["cell"]
        if labels[c0] != labels[c1]:
            for cid, rec in ((c0,records[0]),(c1,records[1])):
                boundary_edges[int(labels[cid])].append((rec["a"],rec["b"]))
    ccent = np.array([np.mod(centroid(c["outer"]),L) for c in cells])
    carea = np.array([poly_area(c["outer"]) for c in cells])
    territory_centroids = np.array([(ccent[labels==i]*carea[labels==i,None]).sum(0)/carea[labels==i].sum() for i in range(4)])
    chosen=[]
    for i in range(4):
        candidates=np.where(labels==i)[0]
        # Pick a full-cell centroid near the intentional target, while making
        # the territory-boundary clearance a hard preference (>0.12 model).
        clear=np.array([min(point_segment_distance(ccent[c],a,b) for a,b in boundary_edges[i]) for c in candidates])
        eligible=candidates[clear>=.12]
        if not len(eligible): raise RuntimeError(f"No interior candidate for A{i}")
        cid=int(eligible[np.argmin(np.linalg.norm(ccent[eligible]-desired[i],axis=1))])
        p=ccent[cid]
        chosen.append({"node_id":f"T{i}","node_type":"terminal","terminal_id":i,"acinus_id":i,
                       "canonical_cell_id":cid,"reference_coordinate_model":p.tolist(),
                       "territory_centroid_reference_model":territory_centroids[i].tolist(),
                       "distance_to_territory_centroid_model":float(np.linalg.norm(p-territory_centroids[i])),
                       "minimum_distance_to_territory_boundary_model":float(min(point_segment_distance(p,a,b) for a,b in boundary_edges[i]))})
    return chosen, territory_centroids


def translated(p): return np.mod(np.asarray(p)-CUT,L)


def draw_territories(ax,cells,labels):
    for cid, cell in enumerate(cells):
        for sx in (-L,0.,L):
            for sy in (-L,0.,L):
                p=cell["outer"]+[sx,sy]
                if p[:,0].max()<0 or p[:,0].min()>L or p[:,1].max()<0 or p[:,1].min()>L: continue
                ax.add_collection(PolyCollection([p],facecolors=PALETTE[labels[cid]],edgecolors="#475569",linewidths=.08,alpha=.62))


def draw_airway(ax,nodes,edges,frame="reference"):
    coord=lambda n: np.asarray(n[f"{frame}_coordinate_model"])
    index={n["node_id"]:n for n in nodes}
    for parent, child in edges:
        a,b=coord(index[parent]),coord(index[child])
        ax.plot([a[0],b[0]],[a[1],b[1]],color="#111827",lw=2.0,zorder=4)
    for n in nodes:
        p=coord(n); style={"root":("*",150,"#111827"),"branch":("o",65,"#ffffff"),"terminal":("s",58,"#dc2626")}[n["node_type"]]
        ax.scatter(*p,marker=style[0],s=style[1],c=style[2],edgecolors="#111827",zorder=5)
        ax.annotate(n["node_id"],p,xytext=(4,4),textcoords="offset points",fontsize=9,weight="bold",zorder=6)


def main():
    base,cells,labels,ownership=build_reference_cells()
    terminals, territory_centroids=choose_terminals(cells,labels)
    # Compact planar scaffold. B0 serves lower pair; B1 upper pair, preserving
    # the repository's established B0:(T0,T1), B1:(T2,T3) pairing.
    core=[{"node_id":"root","node_type":"root","reference_coordinate_model":[.70,.8333333333333334]},
          {"node_id":"B0","node_type":"branch","reference_coordinate_model":[.85,.58]},
          {"node_id":"B1","node_type":"branch","reference_coordinate_model":[.85,1.09]}]
    nodes=core+terminals
    for n in nodes:
        ref=np.asarray(n["reference_coordinate_model"])
        n["reference_coordinate_mm"]=(ref*UNIT_MM).tolist()
        n["translated_coordinate_model"]=translated(ref).tolist()
        n["translated_coordinate_mm"]=(translated(ref)*UNIT_MM).tolist()
    edges=[("root","B0"),("B0","T0"),("B0","T1"),("root","B1"),("B1","T2"),("B1","T3")]
    index={n["node_id"]:n for n in nodes}
    branch=[]
    for parent,child in edges:
        a=np.asarray(index[parent]["reference_coordinate_model"]); b=np.asarray(index[child]["reference_coordinate_model"])
        branch.append({"parent_node_id":parent,"child_node_id":child,"length_model":float(np.linalg.norm(b-a)),"length_mm":float(np.linalg.norm(b-a)*UNIT_MM),"crosses_optimized_periodic_cut":bool((a[0]-CUT[0])*(b[0]-CUT[0])<0 or (a[1]-CUT[1])*(b[1]-CUT[1])<0)})
    crossings=[]
    for i,(a,b) in enumerate(edges):
        for c,d in edges[i+1:]:
            if {a,b}&{c,d}: continue
            if seg_cross(np.asarray(index[a]["reference_coordinate_model"]),np.asarray(index[b]["reference_coordinate_model"]),np.asarray(index[c]["reference_coordinate_model"]),np.asarray(index[d]["reference_coordinate_model"])): crossings.append([a,b,c,d])
    angles={"root_B0_B1_deg":angle_deg(np.asarray(index["root"]["reference_coordinate_model"]),np.asarray(index["B0"]["reference_coordinate_model"]),np.asarray(index["B1"]["reference_coordinate_model"])),
            "B0_T0_T1_deg":angle_deg(np.asarray(index["B0"]["reference_coordinate_model"]),np.asarray(index["T0"]["reference_coordinate_model"]),np.asarray(index["T1"]["reference_coordinate_model"])),
            "B1_T2_T3_deg":angle_deg(np.asarray(index["B1"]["reference_coordinate_model"]),np.asarray(index["T2"]["reference_coordinate_model"]),np.asarray(index["T3"]["reference_coordinate_model"]))}
    terminal_validation=[]
    for t in terminals:
        tid=t["terminal_id"]; p=np.asarray(t["reference_coordinate_model"])
        terminal_validation.append({"terminal":t["node_id"],"assigned_territory":f"A{tid}","canonical_cell_ownership":f"A{int(labels[t['canonical_cell_id']])}","inside_intended_territory":bool(labels[t["canonical_cell_id"]]==tid),"reference_coordinate_model":p.tolist(),"reference_coordinate_mm":(p*UNIT_MM).tolist(),"translated_coordinate_model":translated(p).tolist(),"translated_coordinate_mm":(translated(p)*UNIT_MM).tolist(),"distance_to_territory_centroid_model":t["distance_to_territory_centroid_model"],"distance_to_territory_centroid_mm":t["distance_to_territory_centroid_model"]*UNIT_MM,"minimum_distance_to_territory_boundary_model":t["minimum_distance_to_territory_boundary_model"],"minimum_distance_to_territory_boundary_mm":t["minimum_distance_to_territory_boundary_model"]*UNIT_MM})
    seed_hash=hashlib.sha256(np.ascontiguousarray(base).tobytes()).hexdigest()
    metadata={"model":"60x60_fixed_partition_airway_morphology_scaffold","representation":"metadata_and_figures_only; no airflow or FE coupling","affects_geometry":False,"affects_directional_mesh":False,"affects_acinar_ownership":False,"affects_governing_equations":False,"fe_solver_run":False,"reference_frame":{"description":"fixed physical periodic torus [0,L)x[0,L), independent of computational cut","domain_model":[0.,L,0.,L],"domain_mm":[0.,14.,0.,14.],"seed_sha256":seed_hash},"computational_cut":{"origin_reference_model":CUT.tolist(),"transform":"translated = mod(reference - cut_origin, L)","topology_and_terminal_territory_assignment_cut_invariant":True},"topology":{"root":"root","B0_children":["T0","T1"],"B1_children":["T2","T3"],"edges":[{"parent_node_id":a,"child_node_id":b} for a,b in edges]},"nodes":nodes,"terminal_validation":terminal_validation,"branch_metrics":{"branches":branch,"total_tree_length_model":sum(x["length_model"] for x in branch),"total_tree_length_mm":sum(x["length_mm"] for x in branch),"branching_angles_deg":angles,"branch_crossings":crossings,"any_branch_crossings":bool(crossings),"any_branch_crosses_optimized_periodic_cut":any(x["crosses_optimized_periodic_cut"] for x in branch)},"validation":{"all_terminals_inside_intended_territory":all(x["inside_intended_territory"] for x in terminal_validation),"terminal_to_territory_mapping":{"T0":"A0","T1":"A1","T2":"A2","T3":"A3"},"no_partition_reassignment":True,"no_geometry_or_mesh_regeneration":True}}
    output=MESH/"Mesh_Acinar_Perfusion_Physiological_60x60_airway_2plus2.json"
    output.write_text(json.dumps(metadata,indent=2)+"\n")
    # Review figures in reference and translated frames.
    fig,ax=plt.subplots(figsize=(9,9),constrained_layout=True);draw_territories(ax,cells,labels);draw_airway(ax,nodes,edges);ax.set(xlim=(0,L),ylim=(0,L),aspect="equal",title="60×60 frozen A0–A3 partition with 2+2 airway scaffold",xlabel="reference model x",ylabel="reference model y");fig.savefig(OUT/"physiological_60x60_partition_with_airway.png",dpi=280);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,8),constrained_layout=True);draw_airway(ax,nodes,edges);ax.set(xlim=(0,L),ylim=(0,L),aspect="equal",title="60×60 airway morphology scaffold: reference torus",xlabel="reference model x",ylabel="reference model y");fig.savefig(OUT/"physiological_60x60_airway_only.png",dpi=280);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,8),constrained_layout=True);draw_airway(ax,nodes,edges,"translated");ax.axvline(0,color="#dc2626",lw=1);ax.axhline(0,color="#dc2626",lw=1);ax.set(xlim=(0,L),ylim=(0,L),aspect="equal",title="Airway in optimized translated cut frame\nred axes: periodic cut",xlabel="translated model x",ylabel="translated model y");fig.savefig(OUT/"physiological_60x60_airway_cut_check.png",dpi=280);plt.close(fig)
    print(json.dumps({"metadata":str(output),"terminals":terminal_validation,"branch_metrics":metadata["branch_metrics"]},indent=2))

if __name__ == "__main__": main()
