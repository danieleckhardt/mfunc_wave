import configparser
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplot2tikz import save as tikz_save


# =============================================================================
# USER SETTINGS
# =============================================================================

example = 301         # 201, 202, 301
methods = ["cheb", "leja", "poly_krylov"]
# ["cheb", "leja", "poly_krylov"]
# For example 202: the ellipse computations to compare.
# For 201/301 leave as ["ritz"] -- those files carry no ellipse suffix,
# so the entry is ignored there anyway.
ellipse_methods = ["ritz"]
# ["ritz", "enclosure"]
betas = [0.01, 0.1, 1]   # Betalist, see the corresponding ini file
# 201: [0.01, 0.001]
# 202: [0.01, 0.1, 0.001]   (same problem, new ellipse computation)
# 301: [1, 0.1, 0.01]

# X-range of the "error vs. polynomial degree" plot.
#   None            -> automatic, i.e. all points determine the range
#   (xmin, xmax)    -> fixed range, e.g. (0, 250)
# No point is ever dropped: entries outside the fixed range are additionally
# drawn as open markers directly on the border, so runs with a very high degree
# (e.g. Leja without convergence, degree = max_degree) stay visible.
degree_xlim = (0, 120)


# ============================================================
# Configuration Utilities
# ============================================================

def load_config(example: int) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()

    config_map = {
        101: "config/example101.ini",
        201: "config/example201.ini",
        202: "config/example202.ini",
        301: "config/example301.ini",
    }

    if example not in config_map:
        raise ValueError(f"Unknown example {example}")

    cfg.read(config_map[example])
    return cfg


def geti(cfg, section, key, default=0):
    try:
        return cfg.getint(section, key)
    except Exception:
        return int(default)


def getf(cfg, section, key, default=0.0):
    try:
        return cfg.getfloat(section, key)
    except Exception:
        return float(default)


def gets(cfg, section, key, default=""):
    return cfg.get(section, key, fallback=default)


# ============================================================
# File Handling
# ============================================================

def format_float(value: float) -> str:
    """Remove trailing zeros (1.0 → 1, 0.100 → 0.1)."""
    return f"{value:g}"


def build_error_filename(
    cfg: configparser.ConfigParser,
    scheme: str,
    beta: float,
    ellipse_method: str = ""
) -> str:
    """Filename written by MatrixFunction::run().

    matrix_function.cpp appends "_ell<ritz|enclosure>" for cheb and leja.
    poly_krylov / rat_krylov do not use an ellipse -> no suffix.
    """
    if not scheme:
        raise ValueError("Scheme must be provided.")

    h = geti(cfg, 'Space_Discretization', 'initial_refinement')
    deg = geti(cfg, 'Space_Discretization', 'fe_degree')
    tau = getf(cfg, 'Data', 'tau')

    tau_str = format_float(tau)
    beta_str = format_float(beta)

    name = f"h{h}_deg{deg}_tau{tau_str}_beta{beta_str}"
    if ellipse_method and scheme not in ("poly_krylov", "rat_krylov"):
        name += f"_ell{ellipse_method}"
    return f"{scheme}/{name}"


def read_xy(file_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not file_path.exists():
        raise FileNotFoundError(f"Missing file: {file_path}")

    data = np.loadtxt(file_path, skiprows=1)
    # deal.II ConvergenceTable writes at least 3 columns (error, mv, time)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data[:, 0], data[:, 1], data[:, 2]


def read_degree(file_path: Path) -> np.ndarray:
    """Polynomial degree (4th column).

    matrix_function.cpp writes this column only for cheb and leja.  For all
    other methods the file still has three columns -> empty array.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Missing file: {file_path}")

    data = np.loadtxt(file_path, skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 4:
        return np.array([])
    return data[:, 3]


# ============================================================
# Plotting Utilities
# ============================================================

METHOD_LABELS = {
    "poly_krylov": "Polynomial Krylov",
    "rat_krylov":  "Rational Krylov",
    "cheb":        "Chebyshev",
    "leja":        "Leja",
}

ELLIPSE_LABELS = {
    "ritz":      "Ritz/Arnoldi",
    "enclosure": "Enclosure",
    "":          "",
}

# These methods do not use an ellipse at all: their result files carry no
# "_ell..." suffix, so they must be plotted exactly once and without a label.
ELLIPSE_INDEPENDENT = ("poly_krylov", "rat_krylov")

# Only these methods construct an interpolation/approximation polynomial, so
# only for them a "error vs. polynomial degree" plot is meaningful.
POLYNOMIAL_METHODS = ("cheb", "leja")


def setup_axis(ax, xlabel: str):
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(xlabel)
    ax.set_ylabel("error")
    ax.grid(True, which='both', linestyle=':')


def create_marker_map(methods: List[str]) -> Dict[str, str]:
    marker_list = ['s', '^', 'o', 'D', 'v', '*', 'x', '+']
    return {
        method: marker_list[i % len(marker_list)]
        for i, method in enumerate(methods)
    }


def add_series(ax, x, y, method: str, markers: Dict[str, str]):
    if np.any(x <= 0) or np.any(y <= 0):
        raise ValueError("Log-scale plotting requires strictly positive values.")

    label = METHOD_LABELS.get(method, method)

    ax.plot(
        x,
        y,
        marker=markers[method],
        linestyle='-',
        linewidth=1.2,
        markersize=4.0,
        label=label
    )


def mark_out_of_range(ax, series, xlim):
    """Redraw points outside a fixed x-range clamped onto the border.

    'series' holds (x, y, marker, color) per curve.  Points outside the range
    are marked as open symbols on the axis border, so that a manually chosen
    x-range does not make any run disappear.
    """
    x_min, x_max = xlim
    for x, y, marker, color in series:
        outside = (x < x_min) | (x > x_max)
        if not np.any(outside):
            continue
        ax.plot(
            np.clip(x[outside], x_min, x_max),
            y[outside],
            marker=marker,
            linestyle='none',
            markersize=6.0,
            markerfacecolor='none',
            markeredgecolor=color,
            clip_on=False,
        )


# ============================================================
# LaTeX Table Export
# ============================================================

def export_latex_table(
    output_path: Path,
    betas: List[float],
    methods: List[str],
    cfg: configparser.ConfigParser,
    base_folder: Path,
    ellipse_method: str = "",
):
    """
    Single LaTeX table containing ALL betas and ALL iterations.
    """

    # Read everything first
    data = {}
    max_len_per_beta = {}

    for beta in betas:
        beta_data = {}
        max_len = 0

        for method in methods:
            fn = build_error_filename(cfg, method, beta, ellipse_method)
            try:
                err, mv, time = read_xy(base_folder / fn)
            except FileNotFoundError as exc:
                print(f"  [skip] {exc}")
                err, mv, time = np.array([]), np.array([]), np.array([])
            beta_data[method] = (err, mv, time)
            max_len = max(max_len, len(err))

        data[beta] = beta_data
        max_len_per_beta[beta] = max_len

    with open(output_path, "w") as f:

        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")

        col_format = "c" + "|ccc" * len(methods)
        f.write(f"\\begin{{tabular}}{{{col_format}}}\n")
        f.write("\\hline\n")

        # Header row 1
        header1 = "$\\delta$ "
        for m in methods:
            label = METHOD_LABELS.get(m, m)
            header1 += f"& \\multicolumn{{3}}{{c}}{{{label}}} "
        header1 += "\\\\"
        f.write(header1)

        # Header row 2
        header2 = " "
        for _ in methods:
            header2 += "& err & mv & time (s) "
        header2 += "\\\\"

        f.write(header2)
        f.write("\\hline")

        # Write all rows
        for beta in betas:

            beta_str = format_float(beta)
            max_len = max_len_per_beta[beta]

            for i in range(1, max_len, 2):
                if i == 1:
                    row = f"{beta_str}  "
                else:
                    row = " "


                for method in methods:

                    err, mv, time = data[beta][method]

                    if i < len(err):
                        row += (
                            f"& {err[i]:.2e} "
                            f"& {int(mv[i])} "
                            f"& {time[i]:.2e} "
                        )
                    else:
                        row += "& - & - & - "

                row += "\\\\"
                f.write(row)
            f.write("\\hline\n")

        f.write("\\end{tabular}\n")
        ell_label  = ELLIPSE_LABELS.get(ellipse_method, ellipse_method)
        cap_suffix = f", ellipse: {ell_label}" if ell_label else ""
        f.write(
            "\\caption{All iterations for all $\\beta$ values" + cap_suffix + ". "
            "Error, matrix-vector multiplications (mv), and runtime (s).}\n"
        )
        f.write("\\end{table}\n")


# ============================================================
# Main Routine
# ============================================================

def create_legend_label(method: str, ellipse_method: str) -> str:
    m = METHOD_LABELS.get(method, method)
    e = ELLIPSE_LABELS.get(ellipse_method, ellipse_method)
    if e:
        return f"{m} ({e})"
    return m


def main():

    cfg = load_config(example)

    ex = gets(cfg, 'Calculation_Mode', 'example', str(example))
    base_folder = Path(f"output/errors/Example{ex}")
    output_dir = base_folder / "tables_figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    # For example 202 we iterate over ellipse_methods; for all others we run
    # once without an ellipse suffix (backwards compatible).
    # is_202 = (example == 202)
    ell_loop = ellipse_methods 

    markers = create_marker_map(methods)

    for beta in betas:

        fig_time, ax_time = plt.subplots()
        fig_mv, ax_mv = plt.subplots()
        fig_deg, ax_deg = plt.subplots()

        setup_axis(ax_time, r"time in $s$")
        setup_axis(ax_mv, "Matrix-Vector multiplications")
        setup_axis(ax_deg, "polynomial degree")
        # semi-logarithmic: exponential convergence in the degree becomes a line
        ax_deg.set_xscale('linear')
        has_degree_data = False
        degree_series = []   # (x, y, marker, color) for mark_out_of_range()

        for ell_m in ell_loop:
            for method in methods:
                # plot the ellipse independent methods only in the first pass
                if method in ELLIPSE_INDEPENDENT and ell_m != ell_loop[0]:
                    continue
                try:
                    fn = build_error_filename(cfg, method, beta, ell_m)
                    err, mv, time = read_xy(base_folder / fn)
                except FileNotFoundError as exc:
                    print(f"  [skip] {exc}")
                    continue

                label_ell = "" if method in ELLIPSE_INDEPENDENT else ell_m
                label = create_legend_label(method, label_ell)
                marker = markers[method]

                ax_time.plot(time, err,
                             marker=marker, linestyle="-", linewidth=1.2,
                             markersize=4.0, label=label)
                ax_mv.plot(mv, err,
                           marker=marker, linestyle="-", linewidth=1.2,
                           markersize=4.0, label=label)

                # error vs. polynomial degree -- cheb and leja only
                if method in POLYNOMIAL_METHODS:
                    deg = read_degree(base_folder / fn)
                    if deg.size:
                        # degree 0 marks a run without a valid degree
                        # (e.g. leja without convergence) -> skip
                        mask = deg > 0
                        if np.any(mask):
                            line, = ax_deg.plot(deg[mask], err[mask],
                                                marker=marker, linestyle="-", linewidth=1.2,
                                                markersize=4.0, label=label)
                            degree_series.append(
                                (deg[mask], err[mask], marker, line.get_color()))
                            has_degree_data = True

        ax_time.legend(loc='lower right', fontsize='small')
        ax_mv.legend(loc='lower right', fontsize='small')

        beta_str = format_float(beta)

        tikz_save(output_dir / f"time_error{beta_str}.tikz", figure=fig_time)
        tikz_save(output_dir / f"mv_error{beta_str}.tikz", figure=fig_mv)

        fig_time.savefig(output_dir / f"time_error{beta_str}.png", dpi=300)
        fig_mv.savefig(output_dir / f"mv_error{beta_str}.png", dpi=300)

        if has_degree_data:
            if degree_xlim is not None:
                ax_deg.set_xlim(*degree_xlim)
                # mark_out_of_range(ax_deg, degree_series, degree_xlim)
            ax_deg.legend(loc='lower left', fontsize='small')
            tikz_save(output_dir / f"degree_error{beta_str}.tikz", figure=fig_deg)
            fig_deg.savefig(output_dir / f"degree_error{beta_str}.png", dpi=300)

        plt.close(fig_time)
        plt.close(fig_mv)
        plt.close(fig_deg)

    # Comparison table: for example 202, one block per ellipse_method
    for ell_m in ell_loop:
        suffix = f"_ell{ell_m}" if ell_m else ""
        export_latex_table(
            output_dir / f"summary_table_triplets{suffix}.tex",
            betas,
            methods,
            cfg,
            base_folder,
            ellipse_method=ell_m,
        )

    print(f"Results written to: {output_dir}")


if __name__ == "__main__":
    main()