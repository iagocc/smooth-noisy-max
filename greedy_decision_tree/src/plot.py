from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


class Plot:
    _log_dir: Path = Path.cwd() / "experiments_log"
    _fig_dir: Path = Path.cwd() / "figures"

    def __init__(self) -> None:
        self._set_matplotlib_latex_style()

    def _set_size(self, width, fraction=1, subplots=(1, 1)):
        if width == "thesis":
            width_pt = 426.79135
        elif width == "beamer":
            width_pt = 307.28987
        else:
            width_pt = width

        fig_width_pt = width_pt * fraction
        inches_per_pt = 1 / 72.27

        golden_ratio = (5**0.5 - 1) / 2

        fig_width_in = fig_width_pt * inches_per_pt
        fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

        return (fig_width_in, fig_height_in)

    def _set_matplotlib_latex_style(self):
        tex_fonts = {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": "cm",
            "axes.labelsize": 10,
            "font.size": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }

        sns.set_style("whitegrid")
        plt.style.use("seaborn-v0_8-paper")
        plt.rcParams.update(tex_fonts)

    def draw(self, experiment_name: str, grid_col: str, grid_rows: str, ncols: int = 1, nrows: int = 1):
        log_path = self._log_dir / f"{experiment_name}.csv"
        df = pd.read_csv(log_path)
        budgets = df["eps"].unique()

        aspect = self._set_size(width="thesis", subplots=(ncols, nrows))
        g = sns.FacetGrid(
            df,
            row=grid_rows,
            col=grid_col,
            aspect=aspect[0] / aspect[1],
            hue="method",
            hue_kws={"marker": ["o", "s", "v", "d", "^", "D"]},
            palette="colorblind",
            legend_out=False,
            sharey=False,
        )
        g.map_dataframe(sns.lineplot, x="eps", y="acc", errorbar=None, estimator="mean")
        g.add_legend()
        g.set_titles(col_template="{col_name} dataset")
        g.set_axis_labels(x_var="budget", y_var="accuracy")
        g.set(xscale="log")
        g.set(xticks=budgets)
        g.set_xticklabels([str(b) for b in budgets])
        g.tight_layout()
        g.savefig(self._fig_dir / f"{experiment_name}_accuracy.pdf", format="pdf", bbox_inches="tight")
        plt.close()


Plot().draw("pdid3_rlnm_new_util", "dataset", "depth", ncols=4, nrows=2)
