"""Exact periodic map and single-vertex pinpoint for the dolfin_mech MicroPoroFlow problem.

dolfin_mech builds its periodic constraint and its rigid-translation pinpoint
internally (``dmech.PeriodicSubDomain(dim, bbox, vertices)`` and
``dmech.PinpointSubDomain(coords=..., tol=1e-3)``).  The periodic sub-domain then
uses its default tolerance, 1e-3 x domain diagonal, both as DOLFIN's ``map_tol``
and in its ``near`` tests.  On meshes whose edge vertices are closer than that
(G60: about 4.5e-4) this ties vertices that are not periodic images and leaves
true periodic pairs untied; the 1e-3 pinpoint can fix more than one vertex.

``installed()`` replaces both classes on the ``dolfin_mech`` module while the
problem is constructed (dolfin_mech files are not modified).  The meshes have
exactly periodic nodes, so a tolerance of ``REL_TOL`` x domain diagonal (about
2e-10 for L = 5/3) is far above coordinate round-off (about 1e-15) and many
orders below the smallest edge-vertex spacing.  ``verify()`` checks the
resulting dof pairing of a built function space and raises on any defect.
"""
from __future__ import annotations

import contextlib

import dolfin
import dolfin_mech as dmech
import numpy as np

REL_TOL = 1e-10


def domain_tolerance(bbox):
    return REL_TOL * float(np.hypot(bbox[1] - bbox[0], bbox[3] - bbox[2]))


class ExactPeriodicSubDomain(dolfin.SubDomain):
    """Periodic map of an axis-aligned rectangle; same call signature as dmech.PeriodicSubDomain."""

    def __init__(self, dim, bbox, vertices=None, tol=None):
        if dim != 2:
            raise NotImplementedError("ExactPeriodicSubDomain supports dim = 2 only.")
        self.xmin, self.xmax, self.ymin, self.ymax = (float(value) for value in bbox[:4])
        corners = np.array([[self.xmin, self.ymin], [self.xmax, self.ymin], [self.xmax, self.ymax], [self.xmin, self.ymax]])
        self.tol = domain_tolerance(bbox) if tol is None else float(tol)
        if vertices is not None and np.abs(np.asarray(vertices, dtype=float) - corners).max() > self.tol:
            raise ValueError(f"Periodic cell vertices {np.asarray(vertices).tolist()} are not the bounding-box corners {corners.tolist()}.")
        self.Lx, self.Ly = self.xmax - self.xmin, self.ymax - self.ymin
        dolfin.SubDomain.__init__(self, map_tol=self.tol)

    def _near(self, value, target):
        return abs(value - target) < self.tol

    def inside(self, x, on_boundary):
        # Masters: left and bottom edges, without the bottom-right and top-left corners.
        left, bottom = self._near(x[0], self.xmin), self._near(x[1], self.ymin)
        right, top = self._near(x[0], self.xmax), self._near(x[1], self.ymax)
        return bool(on_boundary and (left or bottom) and not ((right and bottom) or (left and top)))

    def map(self, x, y):
        right, top = self._near(x[0], self.xmax), self._near(x[1], self.ymax)
        if right and top:
            y[0], y[1] = x[0] - self.Lx, x[1] - self.Ly
        elif right:
            y[0], y[1] = x[0] - self.Lx, x[1]
        elif top:
            y[0], y[1] = x[0], x[1] - self.Ly
        else:
            # Not a slave: map far outside the domain so that no master matches.
            y[0], y[1] = self.xmin - 1000.0 * self.Lx, self.ymin - 1000.0 * self.Ly


class ExactPinpointSubDomain(dolfin.SubDomain):
    """Single-vertex pinpoint; same call signature as dmech.PinpointSubDomain.

    The tolerance requested by the caller (dolfin_mech passes 1e-3) is recorded
    in ``requested_tol`` and replaced by ``PINPOINT_TOL``, set by ``installed()``
    from the domain size.
    """

    PINPOINT_TOL = None

    def __init__(self, coords, tol=None):
        if self.PINPOINT_TOL is None:
            raise RuntimeError("ExactPinpointSubDomain used outside periodic_map.installed().")
        self.coords = np.asarray(coords, dtype=float)
        self.requested_tol = tol
        self.tol = self.PINPOINT_TOL
        dolfin.SubDomain.__init__(self)

    def move(self, coords):
        self.coords[:] = coords

    def inside(self, x, on_boundary):
        return bool(np.linalg.norm(np.asarray(x)[:len(self.coords)] - self.coords) < self.tol)


@contextlib.contextmanager
def installed(mesh, legacy=False):
    """Use the exact periodic map and pinpoint while building a problem on ``mesh``.

    With ``legacy=True`` nothing is replaced (dolfin_mech defaults, for
    reproducing historical runs).  Yields a dict describing the method used.
    """
    coordinates = mesh.coordinates()[:, :2]
    bbox = [coordinates[:, 0].min(), coordinates[:, 0].max(), coordinates[:, 1].min(), coordinates[:, 1].max()]
    if legacy:
        yield {"method": "legacy dolfin_mech default (PeriodicSubDomain tol = 1e-3 x domain diagonal; PinpointSubDomain tol = 1e-3)",
               "legacy_periodic_tolerance": True,
               "periodic_tolerance": 1e-3 * float(np.hypot(bbox[1] - bbox[0], bbox[3] - bbox[2])),
               "pinpoint_tolerance": 1e-3}
        return
    tol = domain_tolerance(bbox)
    original = (dmech.PeriodicSubDomain, dmech.PinpointSubDomain)
    ExactPinpointSubDomain.PINPOINT_TOL = tol
    dmech.PeriodicSubDomain, dmech.PinpointSubDomain = ExactPeriodicSubDomain, ExactPinpointSubDomain
    try:
        yield {"method": "project ExactPeriodicSubDomain / ExactPinpointSubDomain (scripts/periodic_map.py)",
               "legacy_periodic_tolerance": False,
               "relative_tolerance": REL_TOL,
               "periodic_tolerance": tol,
               "pinpoint_tolerance": tol}
    finally:
        dmech.PeriodicSubDomain, dmech.PinpointSubDomain = original
        ExactPinpointSubDomain.PINPOINT_TOL = None


def exact_periodic_pairs(coordinates, tol=1e-12):
    """Vertex pairs (left, right) and (bottom, top) with identical transverse coordinates."""
    x = coordinates[:, :2]
    xmin, ymin = x.min(axis=0)
    xmax, ymax = x.max(axis=0)
    pairs = {}
    for name, axis, lo, hi in (("left_right", 0, xmin, xmax), ("bottom_top", 1, ymin, ymax)):
        other = 1 - axis
        lo_ids = np.where(np.abs(x[:, axis] - lo) < tol)[0]
        hi_ids = np.where(np.abs(x[:, axis] - hi) < tol)[0]
        hi_sorted = hi_ids[np.argsort(x[hi_ids, other])]
        hi_values = x[hi_sorted, other]
        found = []
        for index in lo_ids:
            k = np.searchsorted(hi_values, x[index, other] - tol)
            if k < len(hi_values) and abs(hi_values[k] - x[index, other]) < tol:
                found.append((index, hi_sorted[k]))
        pairs[name] = {"pairs": np.asarray(found, dtype=int).reshape(-1, 2), "lo_vertices": len(lo_ids), "hi_vertices": len(hi_ids)}
    return pairs, (xmax - xmin, ymax - ymin)


def vertex_dofs(mesh, space):
    """Dof of each mesh vertex in a CG space (vertex dofs come first in each cell)."""
    dofmap = space.dofmap()
    result = np.full(mesh.num_vertices(), -1, dtype=np.int64)
    for cell in dolfin.cells(mesh):
        result[cell.entities(0)] = dofmap.cell_dofs(cell.index())[:3]
    return result


def pairing_report(mesh, space):
    """Untied exact periodic pairs, wrong ties, corner groups and dof count of one CG (sub)space."""
    coordinates = mesh.coordinates()[:, :2]
    pairs, (Lx, Ly) = exact_periodic_pairs(coordinates)
    v2d = vertex_dofs(mesh, space)
    untied = {name: int(np.sum(v2d[entry["pairs"][:, 0]] != v2d[entry["pairs"][:, 1]])) if len(entry["pairs"]) else 0
              for name, entry in pairs.items()}
    groups = {}
    for vertex, dof in enumerate(v2d):
        groups.setdefault(int(dof), []).append(vertex)
    wrong, correct = 0, 0
    for vertices in groups.values():
        for other in vertices[1:]:
            delta = coordinates[other] - coordinates[vertices[0]]
            error = np.hypot(delta[0] - np.round(delta[0] / Lx) * Lx, delta[1] - np.round(delta[1] / Ly) * Ly)
            if error > 1e-10:
                wrong += 1
            else:
                correct += 1
    xmin, ymin = coordinates.min(axis=0)
    xmax, ymax = coordinates.max(axis=0)
    corner_ids = [int(i) for i in np.where((np.minimum(np.abs(coordinates[:, 0] - xmin), np.abs(coordinates[:, 0] - xmax)) < 1e-12)
                                           & (np.minimum(np.abs(coordinates[:, 1] - ymin), np.abs(coordinates[:, 1] - ymax)) < 1e-12))[0]]
    parent = list(range(len(coordinates)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for entry in pairs.values():
        for a, b in entry["pairs"]:
            parent[find(a)] = find(b)
    expected = len({find(i) for i in range(len(coordinates))})
    return {
        "exact_pairs": {name: int(len(entry["pairs"])) for name, entry in pairs.items()},
        "edge_vertices": {name: [entry["lo_vertices"], entry["hi_vertices"]] for name, entry in pairs.items()},
        "exact_pairs_not_sharing_dof": untied,
        "wrong_ties": wrong,
        "correct_ties": correct,
        "corner_vertices": corner_ids,
        "corner_dofs": sorted({int(v2d[i]) for i in corner_ids}),
        "distinct_vertex_dofs": len(groups),
        "expected_distinct_vertex_dofs": expected,
    }


def verify(problem, raise_on_error=True):
    """Pairing of the p_l and U_tilde subspaces of a built problem and the pinpoint size; raises on any defect."""
    mesh = problem.mesh
    report = {}
    for name, subsol, component in (("p_l_perturbation", problem.pl_perturbation_subsol, None),
                                     ("U_tilde_x", problem.displacement_perturbation_subsol, 0)):
        space = subsol.fs if component is None else subsol.fs.sub(component)
        report[name] = pairing_report(mesh, space)
    pinned = [len(constraint.bc.get_boundary_values()) for constraint in problem.constraints]
    report["pointwise_constraint_dofs"] = pinned
    report["mixed_space_dim"] = int(problem.sol_fs.dim())
    errors = []
    for name in ("p_l_perturbation", "U_tilde_x"):
        entry = report[name]
        if any(entry["exact_pairs_not_sharing_dof"].values()):
            errors.append(f"{name}: exact periodic pairs not tied {entry['exact_pairs_not_sharing_dof']}")
        if entry["wrong_ties"]:
            errors.append(f"{name}: {entry['wrong_ties']} ties between non-periodic vertices")
        if len(entry["corner_dofs"]) > 1:
            errors.append(f"{name}: corners in {len(entry['corner_dofs'])} dof groups")
        if entry["distinct_vertex_dofs"] != entry["expected_distinct_vertex_dofs"]:
            errors.append(f"{name}: {entry['distinct_vertex_dofs']} vertex dofs, expected {entry['expected_distinct_vertex_dofs']}")
    if pinned != [problem.dim]:
        errors.append(f"pointwise constraints fix {pinned} dofs, expected one vertex ({problem.dim} dofs)")
    report["errors"] = errors
    if errors and raise_on_error:
        raise RuntimeError("Periodic map verification failed: " + "; ".join(errors))
    return report
