# Mesh Compare

Compare a "subject" STL mesh against a "reference" STL mesh and produce an
interactive HTML deviation report (Geomagic-style: signed color map with a
green in-tolerance band, hover-to-inspect, area-weighted statistics).

Two run modes:

1. **Single pair** — one subject vs. one reference, one full-color HTML.
2. **Batch from CSV** — many pairs in one run, optionally grouped by `case`
   into a multi-panel side-by-side HTML per case.

Both modes render one full-color mesh per scene with a fixed ±1.0 mm color
scale, so HTMLs stay light and panels are directly comparable across
cases.

---

## Install

Requires Python ≥ 3.9.

```
pip install numpy scipy trimesh plotly fast-simplification rtree
```

`fast-simplification` is optional but recommended; without it the renderer
ships the un-decimated mesh, which makes the HTML huge.

---

## Quick start

### Single pair

```
python stl_deviation_report.py SUBJECT.stl REFERENCE.stl
```

Writes `SUBJECT_deviation.html` and `SUBJECT_deviation.stats.json` next to the
subject. Open the HTML in a browser.

Common options:

```
--tolerance-um 130       Tolerance band half-width (default 130 µm)
--display-faces 220000   Render face budget for the subject
--ref-cloud 2000000      Sample density on reference for the distance KDTree
--no-icp                 Skip alignment (trust input coordinates)
--out FILE.html          Custom output path
--title "..."            Custom report title
```

### Batch from CSV

```
python stl_deviation_report.py --batch input.csv
```

CSV format (header row required):

| column           | required | purpose                                          |
| ---------------- | -------- | ------------------------------------------------ |
| `subject_file`   | yes      | Subject STL (relative or absolute path)          |
| `reference_file` | yes      | Reference STL                                    |
| `case`           | optional | Groups rows for multi-panel mode                 |
| `subject_type`   | optional | Panel label, e.g. `injected`, `fit`              |
| `reference_type` | optional | Panel label, e.g. `cad`, `injected`              |
| `run_number`     | optional | Used in default output filenames                 |
| `out`            | optional | Per-row output HTML path                         |
| `title`          | optional | Per-row report title                             |

Example `input.csv`:

```
run_number,case,subject_file,reference_file,subject_type,reference_type
1,387-094-590,387-094-590_injected.stl,387-094-590_cad.stl,injected,cad
2,387-094-590,387-094-590_fit.stl,387-094-590_injected.stl,fit,injected
3,387-094-590,387-094-590_fit.stl,387-094-590_cad.stl,fit,cad
4,393-213-293,393-213-293_injected.stl,393-213-293_cad.stl,injected,cad
5,393-213-293,393-213-293_fit.stl,393-213-293_injected.stl,fit,injected
6,393-213-293,393-213-293_fit.stl,393-213-293_cad.stl,fit,cad
```

#### Per-case multi-panel mode

When `case`, `subject_type`, and `reference_type` are all present, the script
groups rows by `case` and writes **one HTML per case** with each row rendered
as a side-by-side 3D panel. With the example CSV above you get:

```
results/387-094-590_deviation.html   # 3 panels: injected→cad, fit→injected, fit→cad
results/393-213-293_deviation.html   # 3 panels: same combinations
```

Multi-panel mode renders one full-color mesh per panel (no out-of-tolerance
toggle, no ghost overlay) so the combined HTML stays small enough to open in
a browser. Each panel's mesh is capped at `--multi-display-faces` (default
80,000) to keep file size manageable.

Within each case, panels are placed in a fixed order regardless of CSV row
order:

1. left — `injected` vs `cad`
2. center — `fit` vs `injected`
3. right — `fit` vs `cad`

Any rows whose `(subject_type, reference_type)` pair isn't in that list are
appended after the three known panels in CSV order.

#### Per-row mode

If the CSV is missing the `case`/`subject_type`/`reference_type` columns, each
row produces its own full-featured single-panel HTML in `results/`.

---

## File layout

The batch runner expects this layout:

```
project/
  input.csv
  scans/                  ← STL files referenced from input.csv go here
    387-094-590_cad.stl
    387-094-590_injected.stl
    ...
  results/                ← created automatically; HTML + JSON outputs land here
  stl_deviation_report.py
```

Relative paths in `input.csv` are resolved against `./scans/` (next to the
CSV) if it exists, otherwise the CSV's own directory. Override with
`--batch-dir DIR`.

---

## Output

- **`*.html`** — interactive Plotly report. One full-color mesh per scene;
  rotate, pan, and zoom with the mouse. The color scale is fixed at
  **±1.0 mm** for every panel and every report so colors are directly
  comparable across cases and views (edit `FIXED_SCALE_MM` at the top of the
  figure section in `stl_deviation_report.py` to change it; deviations
  beyond the range get clamped to the endpoint color).
- **`*.stats.json`** — area-weighted summary stats:
  - `pct_in_tol`, `pct_over_hi`, `pct_under_lo` (% of surface area)
  - `min_signed_mm`, `max_signed_mm`, `mean_abs_mm`, `rms_mm`
  - `tolerance_mm`, `icp_rms_mm`
  - mesh sizes (`ref_faces`, `subj_faces_full`, `disp_faces`)

For multi-panel cases, the JSON contains a `panels` array with the same stats
per panel.

---

## Tuning

| flag                          | when to change it                                                                |
| ----------------------------- | -------------------------------------------------------------------------------- |
| `--tolerance-um`              | Tighten/widen the green in-tolerance band                                        |
| `--display-faces`             | Smaller = lighter HTML, less surface detail (single-panel mode)                  |
| `--multi-display-faces`       | Per-panel cap in multi-panel mode. 120k–150k is a safe upper bound; >250k risks browser hang |
| `--ref-cloud`                 | More samples = more accurate distance field, slower preprocessing                |
| `--icp-samples`, `--icp-iter` | Increase if alignment is shaky                                                   |
| `--no-icp`                    | Skip alignment when CAD origins already match                                    |
| `--quiet`                     | Suppress per-step progress output                                                |

---

## How it works

1. Load both STLs with `trimesh`.
2. Coarse-align the subject to the reference: PCA principal-axis search
   across all 24 proper axis-permutation/flip orientations.
3. Refine alignment with point-to-point ICP (Kabsch/SVD) against a KD-tree on
   a dense surface sample of the reference.
4. Decimate the aligned subject to the display face budget.
5. For each displayed vertex, find the nearest point on the reference surface
   sample. Sign the distance using the reference face normal:
   `+` = material outside the reference, `−` = material inside.
6. Build the area-weighted stats and emit a Plotly HTML.

---

## Troubleshooting

- **Blank HTML / browser hangs.** File too big — bring `--display-faces` (or
  `--multi-display-faces`) down. Anything past a few hundred MB will struggle
  to render.
- **Alignment looks wrong.** Increase `--icp-samples`, `--icp-iter`, or both.
  If the meshes share a coordinate system already, try `--no-icp`.
- **`fast_simplification not installed` warning.** Install it
  (`pip install fast-simplification`) — without it the renderer ships the
  full mesh.
- **Missing `rtree`.** Install it (`pip install rtree`); trimesh uses it for
  spatial indices.
