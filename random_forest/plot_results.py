import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from forest import ExperimentType


def set_size(width, fraction=1, subplots=(1, 1)):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == "thesis":
        width_pt = 426.79135
    elif width == "beamer":
        width_pt = 307.28987
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)


def set_matplotlib_latex_style():
    tex_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "font.family": "serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 10,
        "font.size": 10,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    }

    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update(tex_fonts)


def plot_dataset_results(name: str, budgets: list[float], metrics: pd.DataFrame):
    fig, ax = plt.subplots(1, 1, figsize=set_size(width="thesis"))
    ax.set_xscale("log")
    ax.set_xticks(budgets)
    ax.xaxis.set_ticklabels([str(b) for b in budgets])
    sn.lineplot(
        data=metrics,
        x="budget",
        y="accuracy",
        hue="method",
        style="method",
        markers=True,
        palette="colorblind",
    )
    fig.savefig(f"result/{name}/plots/{name}_accuracy.pdf", format="pdf", bbox_inches="tight")

    fig, ax = plt.subplots(1, 1, figsize=set_size(width="thesis"))
    ax.set_xscale("log")
    ax.set_xticks(budgets)
    ax.xaxis.set_ticklabels([str(b) for b in budgets])
    sn.lineplot(
        data=metrics,
        x="budget",
        y="f1",
        hue="method",
        style="method",
        markers=True,
        palette="colorblind",
    )
    fig.savefig(f"result/{name}/plots/{name}_f1.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)


def make_results_table(results: pd.DataFrame) -> str:
    grouped = results.groupby(["dataset", "method"], as_index=False).agg(
        {"accuracy": ["mean", "std"], "f1": ["mean", "std"]}
    )
    return grouped.to_latex(index=False, multirow=True, escape=False)


set_matplotlib_latex_style()
sn.set_style("whitegrid")

datasets = ["adult", "mushroom", "nursery", "gamma", "pendigits", "wallsensor", "compass", "wine"]
# datasets = ["nursery"]
budgets = [0.01, 0.05, 0.1, 1, 2]

metrics = pd.DataFrame(columns=["dataset", "budget", "method", "exec", "accuracy", "f1"])

type_legend = {
    ExperimentType.DP: r"Exponential Mechanism",
    ExperimentType.PF: r"Permute-and-Flip",
    ExperimentType.SMOOTHED_DP: r"Sam Fletcher \textit{et al}.",
    ExperimentType.RLNM_LAPLACE: r"RLNM w/ Laplace",
    ExperimentType.RLNM_EXPONENTIAL: r"RLNM w/ Exp.",
    ExperimentType.LOCAL_DAMPENING: r"Local Dampening",
}

for ds in datasets:
    for eps in budgets:
        for method in ExperimentType:
            if method == ExperimentType.DEFAULT or method == ExperimentType.SMOOTHED_DP:
                continue
            file_path = f"result/{ds}/logs/metrics_{method.value}_{eps}.csv"
            df = pd.read_csv(file_path)
            for i, m in df.iterrows():
                info = {
                    "dataset": ds,
                    "budget": eps,
                    "method": type_legend[method],
                    "exec": i,
                    "accuracy": m["accuracy"],
                    "f1": m["f1"],
                }
                metrics = pd.concat([metrics, pd.DataFrame([info])], ignore_index=True)

    plot_dataset_results(ds, budgets, metrics[metrics["dataset"] == ds])

print(make_results_table(metrics[metrics["budget"] == 1]))
