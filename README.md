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
3 x 3 periodic seed cloud, assigns whole canonical cells by weighted
multi-source shortest paths, and writes reusable JSON metadata under
`results/acinar_partition/`. The acinar IDs are post-processing labels only;
the script does not modify the mesh, finite-element subdomains, or solver.

The same command generates central and 3 x 3 tiled figures with a
metadata-only airway tree overlay. Root, branch, and terminal nodes are stored
in `periodic_airway_metadata.json`; they are plot annotations and do not create
airway channels or alter the Voronoi wall geometry.
