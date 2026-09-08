# Project_Acinar_Perfusion

Code repository for acinar perfusion modeling.

Main goals:
- build a 2D acinar Voronoi perfusion model
- study intra-acinar perfusion heterogeneity
- prepare reduced relations for larger-scale models

## Periodic acinar territory labels

Generate the four-territory metadata partition and its figures with:

```bash
/opt/anaconda3/envs/dolfin-clean/bin/python scripts/partition_periodic_acini.py
```

The script uses the same canonical `12 x 12` base-seed parameters as the
accepted periodic Voronoi patch. It builds adjacency from Voronoi ridges in a
3 x 3 periodic seed cloud. The main model recursively generates a binary
airway tree and connected daughter territories together by optimizing area
balance, compactness, and branch length with toroidal distances. The former
prescribed-terminal, multi-source shortest-path partition is retained as a
reference method. Reusable JSON metadata are written under
`results/acinar_partition/`.

The same command generates central and 3 x 3 tiled figures with a
metadata-only airway tree overlay. Root, branch, and terminal nodes are stored
in `periodic_airway_metadata.json`; they are plot annotations and do not create
airway channels or alter the Voronoi wall geometry, FE mesh, material regions,
or governing equations.

The same unchanged binary split objective,
`J = 2 J_area + 2 J_compactness + 0.15 J_branch_length`, is also evaluated for
three metadata-only branching templates: the 2+2 baseline, a root
trifurcation with one downstream bifurcation, and a 2+3 tree. Their individual
figures and JSON summaries are written to
`results/acinar_partition/branching_templates/`; the presentation summary and
side-by-side comparison are `airway_space_filling_summary.png` and
`branching_template_comparison.png` in the parent results directory.

## Vascular placement metadata

Generate metadata-only arterial sources and shared interacinar venous-site
comparisons with:

```bash
/opt/anaconda3/envs/dolfin-clean/bin/python scripts/vascular_placement_metadata.py
```

Each terminal bronchiole is projected to its nearest admissible Voronoi wall
ridge within the corresponding acinus. Venous candidates are canonical
interacinar ridge midpoints and junctions, and fixed-size configurations for
`N_v = 1, 2, 3, 4` are optimized independently using periodic graph distances.
The JSON metadata and figures are written under `results/vascular_placement/`.
These objects do not create vascular channels, FE regions, boundary conditions,
or source terms.
