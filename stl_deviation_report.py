#!/usr/bin/env python3
"""
stl_deviation_report.py
=======================

Compare a "subject" STL against a "reference" STL and produce an interactive
HTML deviation report, Geomagic-style.

What it does
------------
1. Loads both STL files with trimesh.
2. Coarse-aligns the subject to the reference using a PCA principal-axis
   search across all 24 proper axis-permutation-and-flip orientations.
3. Refines the alignment with a fast point-to-point ICP (Kabsch / SVD) using a
   KD-tree on a dense point sample of the reference surface.
4. Decimates the aligned subject to a target face count for display.
5. Computes a signed deviation at every displayed vertex -- magnitude is the
   point-to-nearest-reference-surface-sample distance; sign is
   sign((P - nearest) . n_ref), i.e. + for "material extra / outside" and
   - for "material missing / inside".
6. Builds an interactive Plotly 3D HTML report with:
      * Signed colour map (diverging red/blue with a neutral in-tolerance band)
      * Toggle to show only out-of-tolerance regions (|d| > tolerance)
      * Optional ghosted reference overlay
      * Area-weighted summary stats (mean, RMS, min, max, % in/out of tol)

Requirements
------------
  python >= 3.9
  pip install numpy scipy trimesh plotly fast-simplification rtree

Usage
-----
  python stl_deviation_report.py SUBJECT.stl REFERENCE.stl [options]
  python stl_deviation_report.py --batch input.csv [options]

Batch CSV format
----------------
  Required columns: subject_file, reference_file
  Optional columns: run_number, case, out, title
  Relative paths are resolved against ./scans next to the CSV if it
  exists, otherwise the CSV's own directory. Override with --batch-dir.

Common examples
---------------
  # Default: 130 um tolerance, auto output filename next to subject
  python stl_deviation_report.py part_subject.stl part_reference.stl

  # Tighter tolerance, custom output, skip ICP (trust the CAD origin)
  python stl_deviation_report.py s.stl r.stl --tolerance-um 50 \\
      --no-icp --out my_report.html

  # Quicker, lighter HTML (good for large meshes)
  python stl_deviation_report.py s.stl r.stl --display-faces 80000 \\
      --ref-cloud 500000

Run with -h for the full option list.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from itertools import permutations, product
from pathlib import Path

import numpy as np


# ----------------------------------------------------------------------------
# Dependency check with a helpful error
# ----------------------------------------------------------------------------
def _require(name, pip_name=None):
    try:
        return __import__(name)
    except ImportError:
        sys.stderr.write(
            f"\nERROR: Python package '{name}' is not installed.\n"
            f"Install with:\n    pip install {pip_name or name}\n\n"
        )
        sys.exit(1)


trimesh = _require("trimesh")
plotly  = _require("plotly")
scipy   = _require("scipy", "scipy")
_require("rtree")  # used by trimesh for spatial indices; not imported directly
import plotly.graph_objects as go           # noqa: E402
from plotly.subplots import make_subplots    # noqa: E402
from scipy.spatial import cKDTree            # noqa: E402

try:
    import fast_simplification
    HAVE_FAST_SIMPLIFY = True
except ImportError:
    HAVE_FAST_SIMPLIFY = False


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def _tic(msg, verbose=True):
    if verbose:
        print(f"[.] {msg} ...", flush=True)
    return time.time()


def _toc(t0, msg="", verbose=True):
    if verbose:
        print(f"[\u2713] {msg} ({time.time() - t0:.2f}s)", flush=True)


def _principal_frame(pts):
    c = pts.mean(axis=0)
    X = pts - c
    cov = (X.T @ X) / len(X)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    return c, vecs[:, order]


def _kabsch(A, B):
    """Rigid R, t such that R @ A.T + t ~ B.T (point-to-point, no scale)."""
    ca, cb = A.mean(axis=0), B.mean(axis=0)
    H = (A - ca).T @ (B - cb)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    return R, cb - R @ ca


def _simplify(vertices, faces, target_faces):
    """Return a decimated (vertices, faces) pair, or the originals if unneeded."""
    if len(faces) <= target_faces:
        return vertices, faces
    if not HAVE_FAST_SIMPLIFY:
        print("  (warning) fast_simplification not installed; rendering full mesh")
        return vertices, faces
    reduction = 1.0 - target_faces / len(faces)
    nv, nf = fast_simplification.simplify(
        vertices.astype(np.float32),
        faces.astype(np.int32),
        reduction,
    )
    return nv, nf


# ----------------------------------------------------------------------------
# Alignment
# ----------------------------------------------------------------------------
def coarse_align(subj_pts, ref_pts, ref_tree, rng_seed=0, verbose=True):
    """Search the 24 proper axis-permutation/flip orientations; return T0."""
    t = _tic("PCA coarse alignment search", verbose)
    c_s, V_s = _principal_frame(subj_pts)
    c_r, V_r = _principal_frame(ref_pts)
    rng = np.random.default_rng(rng_seed)
    idx = rng.choice(len(subj_pts), min(5000, len(subj_pts)), replace=False)

    best = None
    for perm in permutations(range(3)):
        P = V_s[:, list(perm)]
        for s in product([-1, 1], repeat=3):
            S = P * np.array(s)
            R = V_r @ S.T
            if np.linalg.det(R) < 0:
                continue
            tvec = c_r - R @ c_s
            P_try = (R @ subj_pts[idx].T).T + tvec
            d, _ = ref_tree.query(P_try, k=1, workers=-1)
            cost = float(np.median(d))
            if best is None or cost < best[0]:
                best = (cost, R, tvec)
    T = np.eye(4)
    T[:3, :3] = best[1]
    T[:3,  3] = best[2]
    if verbose:
        print(f"    best coarse median-NN = {best[0]*1000:.1f} \u00b5m")
    _toc(t, verbose=verbose)
    return T


def icp(subj_pts, ref_pts, ref_tree, T0, max_iter=60, tol=1e-5,
        trim_quantile=0.95, verbose=True):
    """Point-to-point ICP with outlier trimming."""
    t = _tic(f"ICP (max_iter={max_iter})", verbose)
    T = T0.copy()
    P = (T[:3, :3] @ subj_pts.T).T + T[:3, 3]
    prev_cost = None
    cost = float("nan")
    for it in range(max_iter):
        d, idx = ref_tree.query(P, k=1, workers=-1)
        keep = d <= np.quantile(d, trim_quantile)
        R, tv = _kabsch(P[keep], ref_pts[idx][keep])
        T_step = np.eye(4); T_step[:3, :3] = R; T_step[:3, 3] = tv
        T = T_step @ T
        P = (R @ P.T).T + tv
        cost = float(np.sqrt(np.mean(d[keep] ** 2)))
        if verbose and (it % 5 == 0 or it == max_iter - 1):
            print(f"    iter {it:>3d}  RMS={cost*1000:.2f} \u00b5m")
        if prev_cost is not None and abs(prev_cost - cost) < tol:
            if verbose:
                print(f"    converged at iter {it}")
            break
        prev_cost = cost
    _toc(t, f"final RMS cost={cost*1000:.2f} \u00b5m", verbose)
    return T, cost


# ----------------------------------------------------------------------------
# Deviation
# ----------------------------------------------------------------------------
def build_ref_cloud(ref, n_samples, verbose=True):
    t = _tic("Building reference surface cloud + KDTree", verbose)
    pts, face = trimesh.sample.sample_surface(ref, n_samples)
    pts = np.vstack([pts, ref.vertices])
    vert_face = np.zeros(len(ref.vertices), dtype=np.int64)
    fi = np.arange(len(ref.faces), dtype=np.int64)
    vert_face[ref.faces[:, 0]] = fi
    vert_face[ref.faces[:, 1]] = fi
    vert_face[ref.faces[:, 2]] = fi
    face = np.concatenate([face, vert_face])
    tree = cKDTree(pts)
    _toc(t, f"cloud size = {len(pts):,}", verbose)
    return pts, face, tree


def signed_deviation(points, ref, ref_cloud, ref_cloud_face, ref_tree,
                     verbose=True):
    t = _tic("Nearest-point query for each display vertex", verbose)
    dist, nn = ref_tree.query(points, k=1, workers=-1)
    _toc(t, f"mean|d|={dist.mean():.4f} mm, max|d|={dist.max():.4f} mm",
         verbose)

    t = _tic("Signing the distances with reference face normals", verbose)
    face_normals = ref.face_normals[ref_cloud_face[nn]]
    delta = points - ref_cloud[nn]
    sgn = np.sign(np.einsum("ij,ij->i", delta, face_normals))
    sgn[sgn == 0] = 1.0
    _toc(t, verbose=verbose)
    return sgn * dist


# ----------------------------------------------------------------------------
# Stats and figure
# ----------------------------------------------------------------------------
def area_weighted_stats(vertices, faces, signed, tolerance):
    face_areas = trimesh.Trimesh(
        vertices=vertices, faces=faces, process=False
    ).area_faces
    w = np.zeros(len(vertices))
    for k in range(3):
        np.add.at(w, faces[:, k], face_areas / 3.0)
    abs_d = np.abs(signed)

    def wpct(mask):
        return 100.0 * w[mask].sum() / w.sum()

    return {
        "total_surface_area_mm2": float(face_areas.sum()),
        "min_signed_mm":  float(signed.min()),
        "max_signed_mm":  float(signed.max()),
        "mean_abs_mm":    float(np.average(abs_d, weights=w)),
        "rms_mm":         float(np.sqrt(np.average(signed**2, weights=w))),
        "pct_in_tol":     float(wpct(abs_d <= tolerance)),
        "pct_over_hi":    float(wpct(signed >  tolerance)),
        "pct_under_lo":   float(wpct(signed < -tolerance)),
        "tolerance_mm":   float(tolerance),
    }


FIXED_SCALE_MM = 1.0  # global cmin/cmax for every panel: ±1.0 mm


def build_figure(vertices, faces, signed, tolerance, stats,
                 ref_vertices=None, ref_faces=None, title="Deviation report"):
    v, f = vertices, faces

    cmax = FIXED_SCALE_MM
    cmin = -cmax
    frac = tolerance / cmax
    c0 = 0.5 - frac / 2   # -tolerance boundary in the [0,1] colorscale
    c1 = 0.5 + frac / 2   # +tolerance boundary in the [0,1] colorscale
    eps = 1e-4

    def _lin(a, b, t):
        return a + (b - a) * t

    # Geomagic-style palette:
    #   deep blue -> blue -> cyan -> (hard step) -> solid GREEN in-tolerance
    #   (hard step) -> yellow -> orange -> deep red
    colorscale = [
        [0.00,                       "rgb(0,0,170)"],      # deep blue
        [_lin(0.0, c0, 0.33),        "rgb(0,90,230)"],     # blue
        [_lin(0.0, c0, 0.66),        "rgb(0,180,240)"],    # cyan
        [max(0.0, c0 - eps),         "rgb(100,220,230)"],  # light cyan, just below -tol
        [c0,                         "rgb(40,180,60)"],    # GREEN band starts (step)
        [c1,                         "rgb(40,180,60)"],    # GREEN band ends   (step)
        [min(1.0, c1 + eps),         "rgb(230,230,80)"],   # yellow-green, just above +tol
        [_lin(c1, 1.0, 0.33),        "rgb(255,200,0)"],    # yellow
        [_lin(c1, 1.0, 0.66),        "rgb(255,110,0)"],    # orange
        [1.00,                       "rgb(180,0,0)"],      # deep red
    ]

    hover_text = [
        f"signed dev: {s*1000:+.1f} \u00b5m<br>|dev|: {abs(s)*1000:.1f} \u00b5m"
        for s in signed
    ]

    mesh_full = go.Mesh3d(
        x=v[:, 0], y=v[:, 1], z=v[:, 2],
        i=f[:, 0], j=f[:, 1], k=f[:, 2],
        intensity=signed, intensitymode="vertex",
        colorscale=colorscale, cmin=cmin, cmax=cmax, showscale=True,
        colorbar=dict(title=dict(text="Signed deviation (mm)", side="right"),
                      tickformat=".3f"),
        lighting=dict(ambient=0.55, diffuse=0.7, specular=0.15,
                      roughness=0.8, fresnel=0.05),
        lightposition=dict(x=100, y=200, z=150),
        hoverinfo="text", text=hover_text, visible=True,
        name="Subject (colored)",
    )

    oot_mask = np.abs(signed) > tolerance
    signed_oot = np.where(oot_mask, signed, np.nan)
    mesh_oot = go.Mesh3d(
        x=v[:, 0], y=v[:, 1], z=v[:, 2],
        i=f[:, 0], j=f[:, 1], k=f[:, 2],
        intensity=signed_oot, intensitymode="vertex",
        colorscale=colorscale, cmin=cmin, cmax=cmax, showscale=True,
        colorbar=dict(title=dict(text="Signed deviation (mm)", side="right"),
                      tickformat=".3f"),
        lighting=dict(ambient=0.55, diffuse=0.7, specular=0.15),
        lightposition=dict(x=100, y=200, z=150),
        hoverinfo="text", text=hover_text, visible=False,
        name=f"Out of tolerance (|d|>{tolerance*1000:.0f} \u00b5m)",
    )

    traces = [mesh_full, mesh_oot]
    if ref_vertices is not None and ref_faces is not None:
        traces.append(go.Mesh3d(
            x=ref_vertices[:, 0], y=ref_vertices[:, 1], z=ref_vertices[:, 2],
            i=ref_faces[:, 0], j=ref_faces[:, 1], k=ref_faces[:, 2],
            color="rgb(200,200,200)", opacity=0.35,
            name="Reference (ghost)", hoverinfo="skip",
            showscale=False,
            lighting=dict(ambient=0.7, diffuse=0.4),
            visible=False,
        ))
        buttons = [
            dict(label="Full color map",
                 method="update", args=[{"visible": [True, False, False]}]),
            dict(label=f"Only |d|>{int(tolerance*1000)} \u00b5m",
                 method="update", args=[{"visible": [False, True, False]}]),
            dict(label="Full + reference ghost",
                 method="update", args=[{"visible": [True, False, True]}]),
            dict(label="OOT + reference ghost",
                 method="update", args=[{"visible": [False, True, True]}]),
        ]
    else:
        buttons = [
            dict(label="Full color map",
                 method="update", args=[{"visible": [True, False]}]),
            dict(label=f"Only |d|>{int(tolerance*1000)} \u00b5m",
                 method="update", args=[{"visible": [False, True]}]),
        ]

    fig = go.Figure(data=traces)
    subtitle = (
        f"Tolerance \u00b1{tolerance*1000:.0f} \u00b5m \u2014 "
        f"in-tol: <b>{stats['pct_in_tol']:.2f}%</b> \u00b7 "
        f"over-build (>+tol): <b>{stats['pct_over_hi']:.2f}%</b> \u00b7 "
        f"under-build (<\u2212tol): <b>{stats['pct_under_lo']:.2f}%</b><br>"
        f"min = {stats['min_signed_mm']*1000:+.1f} \u00b5m \u00b7 "
        f"max = {stats['max_signed_mm']*1000:+.1f} \u00b5m \u00b7 "
        f"mean |d| = {stats['mean_abs_mm']*1000:.1f} \u00b5m \u00b7 "
        f"RMS = {stats['rms_mm']*1000:.1f} \u00b5m"
    )
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b><br>"
                 f"<span style='font-size:12px;color:#555'>{subtitle}</span>",
            x=0.02, xanchor="left",
        ),
        scene=dict(
            xaxis=dict(title="X (mm)"),
            yaxis=dict(title="Y (mm)"),
            zaxis=dict(title="Z (mm)"),
            aspectmode="data",
            bgcolor="rgb(245,245,248)",
        ),
        margin=dict(l=0, r=0, t=70, b=0),
        updatemenus=[dict(
            type="buttons", direction="right",
            x=0.02, y=0.98, xanchor="left", yanchor="top",
            buttons=buttons,
        )],
    )
    return fig


# ----------------------------------------------------------------------------
# Main CLI
# ----------------------------------------------------------------------------
def compute_deviation(subject_path, reference_path, tolerance_um,
                      display_faces, ref_cloud_samples, icp_samples,
                      icp_max_iter, do_icp, include_reference_ghost,
                      verbose=True):
    """Run alignment + deviation for one (subject, reference) pair.

    Returns a dict suitable for build_figure / build_multi_figure.
    """
    tolerance_mm = tolerance_um / 1000.0

    t = _tic("Loading STL meshes", verbose)
    subj = trimesh.load(subject_path,   process=True, force="mesh")
    ref  = trimesh.load(reference_path, process=True, force="mesh")
    _toc(t, f"subj={len(subj.faces):,} faces, ref={len(ref.faces):,} faces",
         verbose)

    if do_icp:
        t = _tic("Sampling surfaces for ICP", verbose)
        subj_pts, _ = trimesh.sample.sample_surface(subj, icp_samples)
        ref_pts,  _ = trimesh.sample.sample_surface(ref,  icp_samples * 4)
        ref_tree_icp = cKDTree(ref_pts)
        _toc(t, f"subj={len(subj_pts):,}  ref={len(ref_pts):,}", verbose)

        T0 = coarse_align(subj_pts, ref_pts, ref_tree_icp, verbose=verbose)
        T_icp, icp_rms = icp(subj_pts, ref_pts, ref_tree_icp, T0,
                             max_iter=icp_max_iter, verbose=verbose)
        subj_aligned = subj.copy()
        subj_aligned.apply_transform(T_icp)
    else:
        subj_aligned = subj
        icp_rms = float("nan")

    t = _tic(f"Decimating subject for rendering (target={display_faces:,})",
             verbose)
    dv, df = _simplify(subj_aligned.vertices, subj_aligned.faces, display_faces)
    subj_disp = trimesh.Trimesh(vertices=dv, faces=df, process=False)
    _toc(t, f"display mesh: {len(subj_disp.faces):,} faces, "
            f"{len(subj_disp.vertices):,} verts", verbose)

    ref_cloud, ref_cloud_face, ref_tree = build_ref_cloud(
        ref, ref_cloud_samples, verbose=verbose
    )
    signed = signed_deviation(
        subj_disp.vertices, ref, ref_cloud, ref_cloud_face, ref_tree,
        verbose=verbose,
    )

    t = _tic("Computing area-weighted statistics", verbose)
    stats = area_weighted_stats(
        subj_disp.vertices, subj_disp.faces, signed, tolerance_mm
    )
    stats["icp_rms_mm"] = float(icp_rms)
    stats["ref_faces"] = int(len(ref.faces))
    stats["subj_faces_full"] = int(len(subj_aligned.faces))
    stats["disp_faces"] = int(len(subj_disp.faces))
    if verbose:
        print(json.dumps(stats, indent=2))
    _toc(t, verbose=verbose)

    ref_v, ref_f = (None, None)
    if include_reference_ghost:
        t = _tic("Decimating reference for ghost overlay", verbose)
        ref_v, ref_f = _simplify(ref.vertices, ref.faces,
                                 min(80000, display_faces))
        _toc(t, f"ghost: {len(ref_f):,} faces", verbose)

    return dict(
        vertices=subj_disp.vertices, faces=subj_disp.faces,
        signed=signed, tolerance_mm=tolerance_mm, stats=stats,
        ref_vertices=ref_v, ref_faces=ref_f,
        subject_path=subject_path, reference_path=reference_path,
    )


def run(subject_path, reference_path, out_html, tolerance_um,
        display_faces, ref_cloud_samples, icp_samples, icp_max_iter,
        do_icp, include_reference_ghost, title=None, verbose=True):
    # Single-pair reports use the same lightweight renderer as multi-panel
    # mode: one full-color mesh, no OOT toggle, no ghost overlay. This keeps
    # the HTML small enough to open in a browser without hanging.
    panel = compute_deviation(
        subject_path, reference_path,
        tolerance_um=tolerance_um,
        display_faces=display_faces,
        ref_cloud_samples=ref_cloud_samples,
        icp_samples=icp_samples,
        icp_max_iter=icp_max_iter,
        do_icp=do_icp,
        include_reference_ghost=False,
        verbose=verbose,
    )
    panel["label"] = f"{Path(subject_path).stem} vs {Path(reference_path).stem}"

    t = _tic("Writing HTML", verbose)
    fig = build_multi_figure(
        [panel],
        title=title or f"Deviation: {Path(subject_path).stem} vs "
                       f"{Path(reference_path).stem}",
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_html)) or ".",
                exist_ok=True)
    fig.write_html(out_html, include_plotlyjs="cdn", full_html=True)
    _toc(t, f"wrote {out_html}", verbose)

    with open(Path(out_html).with_suffix(".stats.json"), "w") as fh:
        json.dump(panel["stats"], fh, indent=2)

    return panel["stats"]


# ----------------------------------------------------------------------------
# Multi-panel (per-case) figure
# ----------------------------------------------------------------------------
def _panel_colorscale(tolerance, signed):
    cmax = FIXED_SCALE_MM
    cmin = -cmax
    frac = tolerance / cmax
    c0 = 0.5 - frac / 2
    c1 = 0.5 + frac / 2
    eps = 1e-4

    def _lin(a, b, t):
        return a + (b - a) * t

    colorscale = [
        [0.00,                "rgb(0,0,170)"],
        [_lin(0.0, c0, 0.33), "rgb(0,90,230)"],
        [_lin(0.0, c0, 0.66), "rgb(0,180,240)"],
        [max(0.0, c0 - eps),  "rgb(100,220,230)"],
        [c0,                  "rgb(40,180,60)"],
        [c1,                  "rgb(40,180,60)"],
        [min(1.0, c1 + eps),  "rgb(230,230,80)"],
        [_lin(c1, 1.0, 0.33), "rgb(255,200,0)"],
        [_lin(c1, 1.0, 0.66), "rgb(255,110,0)"],
        [1.00,                "rgb(180,0,0)"],
    ]
    return colorscale, cmin, cmax


def build_multi_figure(panels, title):
    """panels: list of dicts with keys label + the compute_deviation outputs.

    Renders one full-color mesh per panel side by side. Kept lightweight so the
    HTML stays small enough to open in a browser; OOT toggles and ghost
    overlays are intentionally omitted (use the single-panel report for those).
    """
    n = len(panels)
    specs = [[{"type": "scene"} for _ in range(n)]]
    subtitles = []
    for p in panels:
        s = p["stats"]
        subtitles.append(
            f"<b>{p['label']}</b><br>"
            f"<span style='font-size:11px;color:#555'>"
            f"in-tol {s['pct_in_tol']:.1f}% · "
            f"over {s['pct_over_hi']:.1f}% · "
            f"under {s['pct_under_lo']:.1f}% · "
            f"RMS {s['rms_mm']*1000:.1f} µm"
            f"</span>"
        )

    fig = make_subplots(rows=1, cols=n, specs=specs, subplot_titles=subtitles,
                        horizontal_spacing=0.04)

    for i, p in enumerate(panels):
        v, f = p["vertices"], p["faces"]
        signed = p["signed"]
        tol = p["tolerance_mm"]
        colorscale, cmin, cmax = _panel_colorscale(tol, signed)

        col_left = i / n
        col_right = (i + 1) / n
        cbar = dict(
            title=dict(text="dev (mm)", side="right"),
            tickformat=".3f",
            x=col_right - 0.01, xanchor="right",
            y=0.5, yanchor="middle",
            len=0.75, thickness=10,
        )

        hover_text = [
            f"{s*1000:+.1f} µm" for s in signed
        ]
        mesh_full = go.Mesh3d(
            x=v[:, 0], y=v[:, 1], z=v[:, 2],
            i=f[:, 0], j=f[:, 1], k=f[:, 2],
            intensity=signed, intensitymode="vertex",
            colorscale=colorscale, cmin=cmin, cmax=cmax,
            showscale=True, colorbar=cbar,
            lighting=dict(ambient=0.55, diffuse=0.7, specular=0.15,
                          roughness=0.8, fresnel=0.05),
            lightposition=dict(x=100, y=200, z=150),
            text=hover_text,
            hovertemplate="dev: %{text}<extra></extra>",
            name=p["label"],
        )

        fig.add_trace(mesh_full, row=1, col=i+1)

    tol_um = int(round(panels[0]["tolerance_mm"] * 1000))

    hidden_axis = dict(
        visible=False, showgrid=False, zeroline=False, showline=False,
        showticklabels=False, showbackground=False, title="",
    )
    scene_kwargs = dict(
        xaxis=hidden_axis,
        yaxis=hidden_axis,
        zaxis=hidden_axis,
        aspectmode="data",
        bgcolor="rgb(245,245,248)",
    )
    layout_updates = {
        ("scene" if i == 0 else f"scene{i+1}"): scene_kwargs
        for i in range(n)
    }

    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b><br>"
                 f"<span style='font-size:12px;color:#555'>"
                 f"Tolerance ±{tol_um} µm</span>",
            x=0.02, xanchor="left",
        ),
        margin=dict(l=0, r=0, t=110, b=0),
        showlegend=False,
        **layout_updates,
    )
    return fig


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Build a Geomagic-style deviation report for two STL files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("subject", nargs="?", default=None,
                   help="Path to the subject STL (omit if using --batch)")
    p.add_argument("reference", nargs="?", default=None,
                   help="Path to the reference STL (omit if using --batch)")
    p.add_argument("--batch", default=None,
                   help="Path to a CSV with columns: subject_file, "
                        "reference_file, and optionally run_number, case, "
                        "out, title. Each row produces one report.")
    p.add_argument("--batch-dir", default=None,
                   help="Directory to resolve relative paths in the batch "
                        "CSV against (default: directory of the CSV file)")
    p.add_argument("--out", "-o", default=None,
                   help="Output HTML path (default: <subject>_deviation.html). "
                        "Ignored in batch mode unless the CSV has no 'out' "
                        "column.")
    p.add_argument("--tolerance-um", type=float, default=130.0,
                   help="Tolerance band half-width in micrometres")
    p.add_argument("--display-faces", type=int, default=220_000,
                   help="Target face count for the rendered subject (smaller "
                        "= lighter HTML, less detail)")
    p.add_argument("--multi-display-faces", type=int, default=80_000,
                   help="Per-panel face count cap when rendering a multi-panel "
                        "(per-case) report. Lower keeps the combined HTML "
                        "openable in a browser.")
    p.add_argument("--ref-cloud", type=int, default=2_000_000,
                   help="Number of points to sample on reference surface for "
                        "the distance KDTree (denser = more accurate)")
    p.add_argument("--icp-samples", type=int, default=30_000,
                   help="Points sampled on subject surface for ICP")
    p.add_argument("--icp-iter", type=int, default=60,
                   help="Max ICP iterations")
    p.add_argument("--no-icp", action="store_true",
                   help="Skip ICP (and the PCA pre-alignment); trust input "
                        "coordinates as-is")
    p.add_argument("--no-ghost", action="store_true",
                   help=argparse.SUPPRESS)  # ghost overlay no longer rendered
    p.add_argument("--title", default=None, help="Report title")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress progress output")
    args = p.parse_args(argv)

    common_kwargs = dict(
        tolerance_um=args.tolerance_um,
        display_faces=args.display_faces,
        ref_cloud_samples=args.ref_cloud,
        icp_samples=args.icp_samples,
        icp_max_iter=args.icp_iter,
        do_icp=not args.no_icp,
        include_reference_ghost=not args.no_ghost,
        verbose=not args.quiet,
    )
    multi_display_faces = args.multi_display_faces

    if args.batch:
        if args.subject or args.reference:
            p.error("Cannot pass positional subject/reference with --batch")
        run_batch(
            batch_csv=args.batch,
            batch_dir=args.batch_dir,
            default_out=args.out,
            default_title=args.title,
            common_kwargs=common_kwargs,
            multi_display_faces=multi_display_faces,
        )
        return

    if not args.subject or not args.reference:
        p.error("subject and reference are required (or use --batch CSV)")

    out_html = args.out
    if out_html is None:
        stem = Path(args.subject).with_suffix("")
        out_html = f"{stem}_deviation.html"

    run(
        subject_path=args.subject,
        reference_path=args.reference,
        out_html=out_html,
        title=args.title,
        **common_kwargs,
    )


def run_batch(batch_csv, batch_dir, default_out, default_title, common_kwargs,
              multi_display_faces=80_000):
    csv_path = Path(batch_csv).resolve()
    if batch_dir:
        base_dir = Path(batch_dir).resolve()
    else:
        scans_dir = csv_path.parent / "scans"
        base_dir = scans_dir if scans_dir.is_dir() else csv_path.parent
    results_dir = csv_path.parent / "results"
    results_dir.mkdir(exist_ok=True)

    with open(csv_path, newline="") as fh:
        rows = list(csv.DictReader(fh))

    if not rows:
        print(f"[!] {csv_path} is empty; nothing to do.")
        return

    required = {"subject_file", "reference_file"}
    missing = required - set(rows[0].keys())
    if missing:
        sys.exit(f"ERROR: batch CSV missing required column(s): "
                 f"{', '.join(sorted(missing))}")

    def _resolve(path_str):
        p_ = Path(path_str)
        return p_ if p_.is_absolute() else (base_dir / p_)

    has_types = ("subject_type" in rows[0] and "reference_type" in rows[0])
    has_case = "case" in rows[0]
    multi_panel = has_types and has_case

    if multi_panel:
        multi_kwargs = dict(common_kwargs)
        # Multi-panel HTMLs must stay openable -- cap per-panel mesh size and
        # skip the reference ghost (would triple the trace count).
        multi_kwargs["display_faces"] = min(
            common_kwargs["display_faces"], multi_display_faces
        )
        multi_kwargs["include_reference_ghost"] = False
        _run_batch_per_case(rows, results_dir, default_out, default_title,
                            multi_kwargs, _resolve)
    else:
        _run_batch_per_row(rows, results_dir, default_out, default_title,
                           common_kwargs, _resolve)


def _run_batch_per_row(rows, results_dir, default_out, default_title,
                       common_kwargs, _resolve):
    n = len(rows)
    results = []
    for i, row in enumerate(rows, 1):
        subject = str(_resolve(row["subject_file"]))
        reference = str(_resolve(row["reference_file"]))
        run_number = (row.get("run_number") or str(i)).strip()
        case = (row.get("case") or "").strip()

        out_html = (row.get("out") or "").strip() or default_out
        if not out_html:
            subj_stem = Path(subject).stem
            ref_stem = Path(reference).stem
            tag = f"run{run_number}"
            if case:
                tag = f"{case}_{tag}"
            out_html = str(results_dir /
                           f"{tag}_{subj_stem}_vs_{ref_stem}_deviation.html")
        elif n > 1 and (row.get("out") or "").strip() == "":
            stem = Path(out_html).with_suffix("")
            out_html = f"{stem}_run{run_number}.html"

        title = (row.get("title") or "").strip() or default_title

        header = f"\n=== [{i}/{n}] run {run_number}"
        if case:
            header += f"  case {case}"
        header += f"\n    subject:   {subject}\n    reference: {reference}\n"
        header += f"    out:       {out_html}"
        print(header, flush=True)

        try:
            stats = run(
                subject_path=subject,
                reference_path=reference,
                out_html=out_html,
                title=title,
                **common_kwargs,
            )
            results.append((run_number, case, out_html, stats, None))
        except Exception as e:
            print(f"[!] run {run_number} FAILED: {e}", flush=True)
            results.append((run_number, case, out_html, None, str(e)))

    print("\n=== Batch summary ===")
    for run_number, case, out_html, stats, err in results:
        tag = f"run {run_number}" + (f" ({case})" if case else "")
        if err:
            print(f"  {tag}: FAILED - {err}")
        else:
            print(f"  {tag}: in-tol={stats['pct_in_tol']:.2f}%  "
                  f"RMS={stats['rms_mm']*1000:.1f} um  -> {out_html}")


PANEL_ORDER = [
    ("injected", "cad"),
    ("fit", "injected"),
    ("fit", "cad"),
]


def _panel_sort_key(row):
    stype = (row.get("subject_type") or "").strip().lower()
    rtype = (row.get("reference_type") or "").strip().lower()
    try:
        return (PANEL_ORDER.index((stype, rtype)), 0)
    except ValueError:
        # Unknown combination -- keep it after the known ones, in CSV order.
        return (len(PANEL_ORDER), 0)


def _run_batch_per_case(rows, results_dir, default_out, default_title,
                        common_kwargs, _resolve):
    """Group rows by `case` and emit one multi-panel HTML per case."""
    cases = {}            # case -> list of rows in CSV order
    case_order = []
    for row in rows:
        case = (row.get("case") or "").strip() or "(uncased)"
        if case not in cases:
            cases[case] = []
            case_order.append(case)
        cases[case].append(row)

    for case in cases:
        cases[case].sort(key=_panel_sort_key)

    summary = []
    for case in case_order:
        case_rows = cases[case]
        print(f"\n=== Case {case}: {len(case_rows)} panel(s) ===", flush=True)
        panels = []
        failures = []
        for row in case_rows:
            subject = str(_resolve(row["subject_file"]))
            reference = str(_resolve(row["reference_file"]))
            stype = (row.get("subject_type") or "subj").strip()
            rtype = (row.get("reference_type") or "ref").strip()
            label = f"{stype} vs {rtype}"

            print(f"\n--- {case}: {label} ---")
            print(f"    subject:   {subject}")
            print(f"    reference: {reference}")
            try:
                panel = compute_deviation(
                    subject_path=subject,
                    reference_path=reference,
                    **common_kwargs,
                )
                panel["label"] = label
                panel["subject_type"] = stype
                panel["reference_type"] = rtype
                panels.append(panel)
            except Exception as e:
                print(f"[!] panel '{label}' FAILED: {e}", flush=True)
                failures.append((label, str(e)))

        if not panels:
            summary.append((case, None, [(l, e) for l, e in failures]))
            continue

        out_html = default_out if (default_out and len(case_order) == 1) \
            else str(results_dir / f"{case}_deviation.html")
        title = default_title or f"Deviation report — case {case}"

        t = _tic(f"Writing combined HTML for case {case}",
                 common_kwargs.get("verbose", True))
        fig = build_multi_figure(panels, title=title)
        os.makedirs(os.path.dirname(os.path.abspath(out_html)) or ".",
                    exist_ok=True)
        fig.write_html(out_html, include_plotlyjs="cdn", full_html=True)
        _toc(t, f"wrote {out_html}",
             common_kwargs.get("verbose", True))

        case_stats = {
            "case": case,
            "panels": [
                {"label": p["label"],
                 "subject_type": p.get("subject_type"),
                 "reference_type": p.get("reference_type"),
                 "subject_path": p["subject_path"],
                 "reference_path": p["reference_path"],
                 "stats": p["stats"]}
                for p in panels
            ],
            "failures": [{"label": l, "error": e} for l, e in failures],
        }
        with open(Path(out_html).with_suffix(".stats.json"), "w") as fh:
            json.dump(case_stats, fh, indent=2)

        summary.append((case, out_html, [(l, e) for l, e in failures]))

    print("\n=== Batch summary ===")
    for case, out_html, failures in summary:
        if out_html is None:
            print(f"  case {case}: FAILED (no panels rendered)")
        else:
            print(f"  case {case}: -> {out_html}")
        for label, err in failures:
            print(f"      panel '{label}' FAILED: {err}")


if __name__ == "__main__":
    main()
