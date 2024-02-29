from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

import pandas as pd
import warnings


@dataclass
class TreeLogger:
    experiment_name: str
    verbose: bool
    _log_dir: Path = Path.cwd() / "experiments_log"
    _df: pd.DataFrame = field(init=False)

    def log(self, dataset: str, method: str, trial: int, dm: str, sens: Optional[float],eps: Optional[float], depth: Optional[int], acc: float, train_time_ns: int) -> None:
        filepath = Path(self._log_dir / f"{self.experiment_name}.csv")

        if not hasattr(self, "_df"):
            if filepath.exists():
                warnings.warn("Be careful the log file already exists, the results will be appended.")
                self._df = pd.read_csv(filepath, index_col=None)
            else:
                self._df = pd.DataFrame(columns=["dataset", "method", "trial", "dm", "eps", "depth", "acc", "date"])

        logitem = {
            "dataset": dataset,
            "method": method,
            "trial": trial,
            "dm": dm,
            "sens": sens,
            "eps": eps,
            "depth": depth,
            "acc": acc,
            "elapsed_ns": train_time_ns,
            "date": datetime.now(),
        }

        if self.verbose:
            print(f"[{self.experiment_name}]\t{logitem}")

        self._df = pd.concat([self._df, pd.DataFrame.from_dict([logitem])])

    def save(self) -> None:
        self._df.to_csv(Path(self._log_dir / f"{self.experiment_name}.csv"), index=None)
