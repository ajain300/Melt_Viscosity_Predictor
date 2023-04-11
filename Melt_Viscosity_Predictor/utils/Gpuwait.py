import GPUtil
from datetime import datetime
import time

class GpuWait:
    """
    Keyword arguments:
        patience: The number of seconds between when we search GPUs
        max_wait: The max number of seconds to wait
        max_load: Maximum current relative load for a GPU to be considered available. GPUs with a load larger than max_load is not returned.
    """
    def __init__(self, patience, max_wait, max_load) -> None:
        self.patience = patience
        self.max_wait = max_wait
        self.max_load = max_load
        self.min_load = 0.06
        self.wait = True
    def __enter__(self):
        max_load = max(self.max_load, self.min_load)
        start = datetime.now()
        dt_string = start.strftime("%d/%m/%Y %H:%M:%S")
        print(f"\nGPU wait started at {dt_string}", flush=True)
        while self.wait:
            availableGPUs = GPUtil.getAvailable(
                order="first",
                limit=1,
                maxLoad=max_load,
                maxMemory=max_load,
                includeNan=False,
                excludeID=[],
                excludeUUID=[],
            )
            now = datetime.now()
            dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
            if availableGPUs:
                self.wait = False
                print(f"GPU checked at {dt_string}. Free.", flush=True)
            else:
                print(f"GPU checked at {dt_string}. Still busy.", flush=True)
                time.sleep(self.patience)
            time_delta = (now - start).total_seconds()
            if time_delta > self.max_wait:
                print(
                    f"Function terminated. GPU has not been free for {time_delta} seconds while the max_wait was set to {self.max_wait}.",
                    flush=True,
                )
                break
        if not self.wait:
            self.exec_start = datetime.now()
            print(
                f'Executable started at {self.exec_start.strftime("%d/%m/%Y %H:%M:%S")}.',
                flush=True,
            )
        return self.wait
    def __exit__(self, type, value, traceback):
        if not self.wait:
            exec_end = datetime.now()
            print(
                f'Executable finished at {exec_end.strftime("%d/%m/%Y %H:%M:%S")}.',
                flush=True,
            )
            time_delta = (exec_end - self.exec_start).total_seconds()
            print(f"Executable took {time_delta} seconds to run.", flush=True)