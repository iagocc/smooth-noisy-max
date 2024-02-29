import time


class ExperimentLogger:
    def __init__(self) -> None:
        self.start_time = time.time()

    def gen(self, metrics: dict):
        metrics["time"] = time.time() - self.start_time
        return metrics
