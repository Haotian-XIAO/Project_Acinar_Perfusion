`geometry.md5` fingerprints every geometry-defining input (all files under `mesh/` plus the acinar partition and vascular placement files under `results/`); these files are not in git. Verify from the repository root with `md5sum -c geometry_manifest/geometry.md5` where `md5sum` is available, or on any macOS with:

```
python3 -c 'import hashlib,os;[print(("OK " if hashlib.md5(open(p,"rb").read()).hexdigest()==h else "FAILED ") if os.path.exists(p) else "MISSING ",p) for h,p in (l.rstrip("\n").split("  ",1) for l in open("geometry_manifest/geometry.md5"))]'
```
