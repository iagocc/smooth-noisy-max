import pandas as pd

from pathlib import Path


def clean_dataframe_column_names(df):
    cols = df.columns
    new_column_names = []

    for col in cols:
        new_col = (
            str(col).lstrip().rstrip().lower().replace(" ", "_").replace("-", "_").replace("class", "class_name")
        )  # strip beginning spaces, makes lowercase, add underscpre
        new_column_names.append(new_col)

    df.columns = new_column_names


def extract_cont_disc_features(data: pd.DataFrame):
    continuos_feat = set(data._get_numeric_data().columns)  # type: ignore
    discrete_feat = set(data.columns) - continuos_feat
    return continuos_feat, discrete_feat


def features_2dict(data, continuos_feat, discrete_feat):
    continuous = {f: (data[f].min(axis=0), data[f].max(axis=0)) for f in continuos_feat}
    discrete = {str(f): list(data[f].unique()) for f in discrete_feat}
    return continuous, discrete


def generate_dataset_summary(
    name: str, df: pd.DataFrame, continuos: list[str], discrete: list[str], classes: list[str], class_col: str
):
    info = {
        "continuos": ", ".join(continuos),
        "n_continuos": len(continuos),
        "discrete": ", ".join(discrete),
        "n_discrete": len(discrete),
        "classes": ", ".join(classes),
        "n_classes": len(classes),
    }

    output_dir = Path(f'result/{name}/')
    output_dir.mkdir(parents=True, exist_ok=True)

    info_df = pd.Series(info)
    info_df.to_csv(f"result/{name}/info.csv")

    desc = df.describe()
    desc.to_csv(f"result/{name}/describe.csv", index=False)

    ax_dist = df[class_col].value_counts().plot(kind="bar")
    ax_dist.get_figure().savefig(f"result/{name}/class_dist.pdf")
