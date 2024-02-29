import pandas as pd
import numpy as np

from pathlib import Path

datasets = ["hepth", "patent", "income"]
percentiles = [50, 90, 99]

path = Path("results/")

for ds in datasets:
    x = np.load(Path(f"data/1D/{ds.upper()}.n4096.npy").resolve()).astype(np.float64)

    for p in percentiles:
        print(f"Dataset={ds} p={p} val={np.percentile(x, p)}")
        df1 = pd.read_table((path / f"err_{ds}_{p}.dat").resolve())
        df2 = pd.read_csv(
            (path / "local_dampening/" / f"err_{ds}_ld_{p}.dat").resolve(),
            delimiter=" ",
        )
        del df2["eps"]
        df2.columns = ["err_ld2"]
        df3 = pd.concat([df1, df2], axis=1)
        df3.to_csv((path / f"final_{ds}_{p}.dat").resolve(), index=False, sep="\t")
