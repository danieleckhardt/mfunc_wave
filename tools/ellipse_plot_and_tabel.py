import configparser
from pathlib import Path
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplot2tikz import save as tikz_save
from dataclasses import dataclass
from typing import List

# =============================================================================
# USER SETTINGS
# =============================================================================

EXAMPLE: int = 202  # 101, 201, 202, 301

BETAS: List[float] = [0.01, 0.1, 0.001]  # see the corresponding ini file
# 101: [0.4, 0.03, 0.01]
# 201: [0.01, 0.1, 0.001]
# 202: [0.01, 0.1, 0.001]
# 301: [1, 0.1, 0.01]

OUTPUT_FILENAME: str = "ellipse_steps.tex"

# Method subfolder the ell_info file was written to ("cheb" or "leja").
# The ellipse itself is method independent, but the output path is not.
METHOD: str = "cheb"

# For example 202: the ellipse computations to compare.
# Use ["ritz"] for examples 101/201/301 (no suffix in the filename there).
ELLIPSE_METHODS: List[str] = ["ritz", "enclosure"]

# =============================================================================
# DATA MODEL
# =============================================================================

@dataclass(frozen=True)
class EllipseRecord:
    beta: float
    radius_x: float
    radius_y: float
    center: float
    time_s: float
    prep_s: float = 0.0          # coefficient precomputation time (optional)
    ellipse_method: str = ""     # "ritz", "enclosure", or "" (legacy)

# =============================================================================
# CONFIG HANDLING
# =============================================================================

class Config:
    def __init__(self, example: int) -> None:
        self._cfg = configparser.ConfigParser()
        self._load(example)

    def _load(self, example: int) -> None:
        config_map = {
            101: "config/example101.ini",
            201: "config/example201.ini",
            202: "config/example202.ini",
            301: "config/example301.ini",
        }

        if example not in config_map:
            raise ValueError(f"Unknown example {example}")

        config_path = Path(config_map[example])

        if not config_path.exists():
            raise FileNotFoundError(f"Missing config file: {config_path}")

        self._cfg.read(config_path)

    def get(self, section: str, key: str, cast=str, default=None):
        try:
            return cast(self._cfg.get(section, key))
        except (configparser.NoOptionError,
                configparser.NoSectionError,
                ValueError):
            return default
# =============================================================================
# FILE NAME BUILDER
# =============================================================================

def build_error_filename(cfg: Config, beta: float,
                         ellipse_method: str = "") -> Path:
    h   = cfg.get("Space_Discretization", "initial_refinement", int, 0)
    deg = cfg.get("Space_Discretization", "fe_degree", int, 0)
    tau = cfg.get("Data", "tau", str, "0")
    name = f"h{h}_deg{deg}_tau{tau}_beta{beta}"
    if ellipse_method:
        name += f"_ell{ellipse_method}"
    name += "_ell_info"
    return Path(METHOD) / name

# =============================================================================
# PARSING
# =============================================================================

def parse_dataset(line1: str, line2: str, beta: float,
                  ellipse_method: str = "") -> EllipseRecord:
    parts = line1.strip().split(";")

    if len(parts) < 2:
        raise ValueError("Malformed dataset line.")

    radius_x = float(parts[0])
    radius_y = float(parts[1])
    center   = float(parts[-1])

    # Line 2 holds the ellipse search time, optionally followed by the
    # coefficient precomputation time, separated by ";". Older files contain
    # only the first value, so the second field is treated as optional.
    time_parts = [p for p in line2.strip().split(";") if p != ""]

    if not time_parts:
        raise ValueError("Malformed timing line.")

    time_s = float(time_parts[0])
    prep_s = float(time_parts[1]) if len(time_parts) > 1 else 0.0

    return EllipseRecord(beta, radius_x, radius_y, center, time_s, prep_s,
                         ellipse_method)


def read_file_without_ref(path: Path, beta: float,
                          ellipse_method: str = "") -> List[EllipseRecord]:
    if not path.exists():
        raise FileNotFoundError(f"Missing error file: {path}")

    with open(path, "r") as f:
        lines = f.readlines()

    if len(lines) % 2 != 0:
        raise ValueError(f"File {path} has an odd number of lines.")

    records: List[EllipseRecord] = []

    for i in range(0, len(lines), 2):
        records.append(parse_dataset(lines[i], lines[i + 1], beta,
                                     ellipse_method))

    return records

####
#### when ref ellipse is given
####

def parse_ellipse(raw: str) -> List[float]:
    return [float(x) for x in raw.strip("[]").split(",")]


def parse_points(raw: str, ref  = False ) -> List[List[float]]:
    raw = raw.strip("[]")
    if not raw:
        return []

    points = []
    for p in raw.split("),("):
        p = p.strip("()")
        points.append([float(x) for x in p.split(",")])
    return points


def parse_step_line(row: str):
    """
    A step line has one of three shapes:
        "[a,b,c]"                          -> initial ellipse
        "deg;rad"                          -> final step (no new points)
        "deg;rad;[a,b,c];[(x,y,f),...]"    -> full step
    Returns ("ellipse", value) or ("step", (deg, rad, ellipse, points))
    or ("degrad", (deg, rad)).
    """
    row = row.strip()

    if row.startswith("["):
        return "ellipse", parse_ellipse(row)

    parts = row.split(";", 3)

    if len(parts) >= 4:
        return "step", (
            int(parts[0]),
            float(parts[1]),
            parse_ellipse(parts[2]),
            parse_points(parts[3]),
        )

    if len(parts) == 2:
        return "degrad", (int(parts[0]), float(parts[1]))

    raise ValueError(f"Unrecognised step line: {row[:60]!r}")


def read_with_ref(path: Path):
    """
    Layout written by ExpoInt::store_info_ellipse:
        0..n-3 : step records (may be absent entirely)
        n-2    : detected eigenvalues, ";" separated
        n-1    : final ellipse "[a,b,c]"
    The number of step records varies and may be zero, so only the last two
    lines have a fixed meaning.
    """
    if not path.exists():
        raise FileNotFoundError(path)

    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]

    if len(lines) < 2:
        raise ValueError(f"{path} does not contain enough data")

    ref_ellipse = parse_ellipse(lines[-1])

    ref_points = []
    for p in lines[-2].split(";"):
        p = p.strip().strip("()")
        if p:
            ref_points.append([float(x) for x in p.split(",")])

    first_ellipse = None
    steps = []
    tail = None

    for ln in lines[:-2]:
        kind, value = parse_step_line(ln)
        if kind == "ellipse":
            first_ellipse = value
        elif kind == "step":
            steps.append(value)
        else:
            tail = value

    deg    = [s[0] for s in steps] + ([tail[0]] if tail else [])
    radius = [s[1] for s in steps] + ([tail[1]] if tail else [])
    ellipse = np.array([s[2] for s in steps], dtype=object)
    points  = [s[3] for s in steps]

    # Files without any recorded step contain no initial ellipse either;
    # fall back to the final one so that plotting still works.
    if first_ellipse is None:
        first_ellipse = ref_ellipse

    return (np.array(deg), np.array(radius), ellipse, points,
            ref_points, ref_ellipse, first_ellipse)

# =============================================================================
# LATEX TABLE GENERATION
# =============================================================================

def generate_latex_table(data: List[EllipseRecord]) -> str:
    methods_present = list(dict.fromkeys(r.ellipse_method for r in data))
    multi_method    = len(methods_present) > 1

    if multi_method:
        col_fmt = "c|" + "|".join("cccc" for _ in methods_present)
        header1 = "$\\beta$"
        for m in methods_present:
            label = m if m else "default"
            header1 += f" & \\multicolumn{{4}}{{c}}{{{label}}}"
        header1 += " \\\\"
        header2 = " & " + " & ".join(
            "$r_x$ & $r_y$ & $c$ & $t$/s" for _ in methods_present
        ) + " \\\\"
        lines = [
            "\\begin{table}[ht]\n", "\\centering\n",
            f"\\begin{{tabular}}{{{col_fmt}}}",
            "\\hline", header1, header2, "\\hline",
        ]
        betas_seen = list(dict.fromkeys(r.beta for r in data))
        for beta in betas_seen:
            row = f"{beta:.3g}"
            for m in methods_present:
                rec = next((r for r in data if r.beta == beta
                            and r.ellipse_method == m), None)
                if rec is None:
                    row += " & - & - & - & -"
                else:
                    row += (f" & {rec.radius_x:.2f} & {rec.radius_y:.2f}"
                            f" & {rec.center:.2f} & {rec.time_s:.3f}")
            lines.append(row + " \\\\")
        lines += ["\\hline", "\\end{tabular}"]
    else:
        lines = [
            "\\begin{table}[ht]\n", "\\centering\n",
            "\\begin{tabular}{c|cccc}",
            "\\hline",
            "$\\beta$ & $r_x$ & $r_y$ & $c$ & Time (s) \\\\",
            "\\hline",
        ]
        for r in data:
            lines.append(
                f"{r.beta:.3g} & {r.radius_x:.2f} & {r.radius_y:.2f}"
                f" & {r.center:.2f} & {r.time_s:.3f} \\\\")
        lines += ["\\hline", "\\end{tabular}"]

    return "\n".join(lines)

#
# with ref
#

def count_new_points(points: List[List[float]]) -> int:
    return sum(p[-1] == 1 for p in points)


def ellipse_changed(ellipse: np.ndarray, i: int) -> bool:
    if i == 0:
        return ellipse[0][1] != 0
    return not np.array_equal(ellipse[i], ellipse[i - 1])


def build_latex_table(
    deg: np.ndarray,
    radius: np.ndarray,
    ellipse: np.ndarray,
    points: List[List[List[float]]],
) -> str:
    headers = [
        r"\texttt{Degree}",
        r"\texttt{Radius}",
        r"\texttt{Points found}",
        r"\texttt{Ellipse changed}",
    ]

    lines = [
        r"\begin{table}[h!]",
        r"\centering",
        r"\begin{tabular}{|c||r|r|r|r|}",
        "Steps & " + " & ".join(headers) + r" \\",
        r"\hline",
    ]

    for i in range(len(deg)):
        is_last = i == len(deg) - 1

        row = [
            str(i),
            str(deg[i]),
            str(radius[i]),
            "-" if is_last else str(count_new_points(points[i])),
            "-" if is_last else ("Yes" if ellipse_changed(ellipse, i) else "No"),
        ]

        lines.append(" & ".join(row) + r" \\")

    lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\caption{Steps for finding ellipse}",
        rf"\label{{tab:findellipse{EXAMPLE}}}",
        r"\end{table}",
    ])

    return "\n".join(lines)

# =============================================================================
# PLOTTING
# =============================================================================

def plot_complex_plane_steps(
    points: List[List[List[float]]],
    ref_points: List[List[float]],
    ellipse: np.ndarray,
    ref_ellipse: np.ndarray,
    first_ellipse: np.ndarray,
    beta: float,
    steps=None,
    save_dir: Path | None = None,
    show_new_points_only: bool = False,
    figsize=(7, 7),   # unused, kept for compatibility
):

    if steps is None:
        steps = range(len(points) + 1)

    if save_dir is None:
        return

    save_dir.mkdir(parents=True, exist_ok=True)
    tex_path = save_dir / f"ellipses_{beta:.3g}.tikz"

    # -------------------------------------------------
    # Viridis colors (like matplotlib)
    # -------------------------------------------------
    cmap = plt.cm.viridis
    cols = cmap(np.linspace(0, 0.75, len(steps)))

    rgb_colors = []
    for c in cols:
        r = int(round(c[0] * 255))
        g = int(round(c[1] * 255))
        b = int(round(c[2] * 255))
        rgb_colors.append((r, g, b))

    # -------------------------------------------------
    # Collect all points for bounding box
    # -------------------------------------------------
    all_x, all_y = [], []

    def collect_xy(data):
        for x, y in data:
            all_x.append(x)
            all_y.append(y)

    if ref_points:
        collect_xy(ref_points)

    for step_pts in points:
        filtered = step_pts
        if show_new_points_only:
            filtered = [p for p in step_pts if p[-1] == 1]
        collect_xy([p[:2] for p in filtered])

    for e in ellipse:
        a, b, c = e
        all_x.extend([c - a, c + a])
        all_y.extend([-b, b])

    if ref_ellipse is not None:
        a, b, c = ref_ellipse
        all_x.extend([c - a, c + a])
        all_y.extend([-b, b])

    xmin, xmax = min(all_x), max(all_x)
    ymin, ymax = min(all_y), max(all_y)

    pad_x = 0.05 * (xmax - xmin)
    pad_y = 0.05 * (ymax - ymin)

    xmin -= pad_x
    xmax += pad_x
    ymin -= pad_y
    ymax += pad_y

    # -------------------------------------------------
    # Write PGFPlots file
    # -------------------------------------------------
    with open(tex_path, "w") as f:

        # Define colors
        for i, (r, g, b) in enumerate(rgb_colors):
            f.write(f"\\definecolor{{step{i}}}{{RGB}}{{{r},{g},{b}}}\n")
        f.write("\n")

        # Axis
        f.write("\\begin{axis}[\n")
        f.write("axis lines=middle,\n")
        f.write(f"xmin={xmin}, xmax={xmax},\n")
        f.write(f"ymin={ymin}, ymax={ymax},\n")
        f.write("xlabel={$\\Re(z)$},\n")
        f.write("ylabel={$\\Im(z)$},\n")
        f.write("legend style={font=\\small},\n")
        f.write("]\n\n")

        # -------------------------------------------------
        # Loop over steps
        # -------------------------------------------------
        for idx, i in enumerate(steps):

            linewidth = max(0.2, 1.0 - 0.1 * i)

            # --- Ellipse ---
            if i == 0:
                # Reference ellipse
                a, b, c = ref_ellipse
                f.write(rf"""
                % Reference ellipse
                \addplot [
                    domain=0:360,
                    samples=200,
                    dashed,
                    red,
                    line width={linewidth*1.}pt
                ] ({{{a}*cos(x)+{c}}}, {{{b}*sin(x)}});

                \addlegendentry{{$\mathcal{{E}}_{{\text{{best}}}}$}}
                """)

                # Step 0 ellipse
                a, b, c = first_ellipse
                f.write(f"""
                    % Step 0
                    \\addplot [
                        domain=0:360,
                        samples=200,
                        smooth,
                        draw=step{idx},
                        line width={linewidth}pt
                    ] ({{{a}*cos(x)+{c}}}, {{{b}*sin(x)}});
                    \\addlegendentry{{Step 0}}
                    """)
            else:
                a, b, c = ellipse[i - 1]
                f.write(f"""
                    % Step {i}
                    \\addplot [
                        domain=0:360,
                        samples=200,
                        smooth,
                        draw=step{idx},
                        line width={linewidth}pt
                    ] ({{{a}*cos(x)+{c}}}, {{{b}*sin(x)}});
                    \\addlegendentry{{Step {i}}}
                    """)

            # --- Points ---
        for idx, i in enumerate(steps):
            if i == 0:
                if ref_points:
                    f.write("\\addplot [only marks, mark=*, mark options={scale=0.75},  red]\n")
                    f.write("table{%\nx y\n")
                    for x, y in ref_points:
                        f.write(f"{x} {y}\n")
                    f.write("};\n\n")
                    f.write(f"""\\addlegendentry{{EV}}""")
            else:
                step_pts = points[i - 1]

                if show_new_points_only:
                    step_pts = [p for p in step_pts if p[-1] == 1]

                new_pts = [p[:2] for p in step_pts if p[-1] == 1]

                if new_pts:
                    f.write(
                        f"\\addplot [only marks, "
                        f"mark=x, "
                        f"mark options={{scale=1.1}}, "
                        f"draw=step{idx}, "
                        f"forget plot]\n"
                    )
                    f.write("table{%\nx y\n")
                    for x, y in new_pts:
                        f.write(f"{x} {y}\n")
                    f.write("};\n\n")

        f.write("\\end{axis}\n")

    print(f"PGFPlots figure written to {tex_path}")

def plot_complex_plane_png(
    points: List[List[List[float]]],
    ref_points: List[List[float]],
    ellipse: np.ndarray,
    ref_ellipse: np.ndarray,
    first_ellipse: np.ndarray,
    beta: float,
    steps=None,
    save_dir: Path | None = None,
    show_new_points_only: bool = True,
    figsize=(7, 7),
):
    if steps is None:
        steps = range(len(points) + 1)

    fig, ax = plt.subplots(figsize=figsize)

    # Axes lines
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)

    ax.set_xlabel(r"$\Re(z)$")
    ax.set_ylabel(r"$\Im(z)$")

    theta = np.linspace(0, 2 * np.pi, 400)
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(steps)))

    for idx, i in enumerate(steps):
        color = colors[idx]
        linewidth = max(0.6, 2.0 - 0.3 * i)

        # -------------------------------------------------
        # Ellipses
        # -------------------------------------------------
        if i == 0:
            # Reference ellipse
            a, b, c = ref_ellipse
            x = a * np.cos(theta) + c
            y = b * np.sin(theta)
            ax.plot(x, y, "--", color="red",
                    linewidth=linewidth * 1.1,
                    label="Ref")

            # Step 0
            a, b, c = first_ellipse
            x = a * np.cos(theta) + c
            y = b * np.sin(theta)
            ax.plot(x, y,
                    color=color,
                    linewidth=linewidth,
                    label="Step 0")
        else:
            a, b, c = ellipse[i - 1]
            x = a * np.cos(theta) + c
            y = b * np.sin(theta)
            ax.plot(x, y,
                    color=color,
                    linewidth=linewidth,
                    label=f"Step {i}")

    # -------------------------------------------------
    # Points
    # -------------------------------------------------
    for idx, i in enumerate(steps):
        color = colors[idx]
        if i == 0:
            if ref_points:
                pts = np.array(ref_points)
                ax.scatter(pts[:, 0], pts[:, 1],
                           color="red", s=15)
        else:
            step_pts = points[i - 1]
            if show_new_points_only:
                step_pts = [p for p in step_pts if p[-1] == 1]

            new_pts = [p[:2] for p in step_pts if p[-1] == 1]

            if new_pts:
                pts = np.array(new_pts)
                ax.scatter(pts[:, 0], pts[:, 1],
                           marker="x",
                           color=color,
                           s=50)

    ax.grid(True, linestyle=":", linewidth=0.6)
    ax.legend(fontsize=8)
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()

    # -------------------------------------------------
    # Save PNG
    # -------------------------------------------------
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        png_path = save_dir / f"ellipses_{beta:.3g}.png"
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        print(f"PNG figure saved to {png_path}")

    plt.close(fig)
# =============================================================================
# FILE WRITING
# =============================================================================

def write_latex_file(content: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write(content)
        f.write("\n")
        f.write(
            "\\caption{Ellipse parameters per $\\beta$: semi-axes $r_x$, $r_y$, "
            "center $c$, and the time spent determining the ellipse.}\n"
        )
        f.write("\\end{table}\n")

# =============================================================================
# MAIN
# =============================================================================
def _plot_ellipse_comparison(records: List[EllipseRecord],
                             output_dir: Path) -> None:
    import numpy as np
    methods_present = sorted(set(r.ellipse_method for r in records))
    betas_present   = sorted(set(r.beta for r in records))
    x     = np.arange(len(betas_present))
    width = 0.35 / max(len(methods_present), 1)
    for attr, ylabel, fname in [
        ("radius_x", r"$r_x$",               "cmp_radiusx"),
        ("radius_y", r"$r_y$",               "cmp_radiusy"),
        ("time_s",   r"$t_{\mathrm{ell}}$/s", "cmp_time_ell"),
    ]:
        fig, ax = plt.subplots(figsize=(max(5, 2*len(betas_present)), 4))
        for j, ell_m in enumerate(methods_present):
            vals = []
            for beta in betas_present:
                rec = next((r for r in records
                            if r.beta == beta and r.ellipse_method == ell_m), None)
                vals.append(getattr(rec, attr, float("nan")) if rec else float("nan"))
            off = (j - (len(methods_present)-1)/2) * width
            ax.bar(x+off, vals, width*0.9, label=ell_m or "default")
        ax.set_xticks(x); ax.set_xticklabels([str(b) for b in betas_present])
        ax.set_xlabel(r"$\beta$"); ax.set_ylabel(ylabel)
        ax.set_yscale("log"); ax.legend()
        ax.grid(True, axis="y", linestyle=":", linewidth=0.6)
        plt.tight_layout()
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / f"{fname}.png", dpi=300)
        plt.close(fig)
        print(f"Comparison bar chart: {output_dir}/{fname}.png")


def main():
    config = Config(EXAMPLE)
    example_id = config.get("Calculation_Mode", "example", str, "10")
    base_folder = Path(f"output/errors/Example{example_id}")
    output_dir  = Path(f"output/errors/Example{EXAMPLE}/tables_figures")
    output_path = output_dir / OUTPUT_FILENAME
    INFO        = config.get("Output", "info_ellipse", str)
    info        = (INFO == "true")
    is_202      = (EXAMPLE == 202)
    ell_loop    = ELLIPSE_METHODS if is_202 else [""]
    all_records: List[EllipseRecord] = []

    for ell_m in ell_loop:
        for beta in BETAS:
            error_file = base_folder / build_error_filename(config, beta, ell_m)
            if info:
                try:
                    deg, radius, ellipse, points, ref_points, ref_ellipse, first_ellipse = \
                        read_with_ref(error_file)
                except FileNotFoundError as exc:
                    print(f"  [skip] {exc}"); continue
                suffix = f"_{ell_m}" if ell_m else ""
                plot_complex_plane_steps(
                    points, ref_points, ellipse, ref_ellipse, first_ellipse,
                    beta, save_dir=output_dir / f"steps{suffix}")
                plot_complex_plane_png(
                    points, ref_points, ellipse, ref_ellipse, first_ellipse,
                    beta, save_dir=output_dir / f"steps{suffix}")
            else:
                try:
                    records = read_file_without_ref(error_file, beta, ell_m)
                except FileNotFoundError as exc:
                    print(f"  [skip] {exc}"); continue
                all_records.extend(records)

    if not info:
        all_records.sort(key=lambda r: (r.beta, r.ellipse_method))
        latex_table = generate_latex_table(all_records)
        write_latex_file(latex_table, output_path)
        print(f"LaTeX table written to {output_path}")
        if is_202:
            _plot_ellipse_comparison(all_records, output_dir)
if __name__ == "__main__":
    main()