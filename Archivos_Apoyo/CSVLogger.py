# Archivos_Apoyo/csv_logger.py
import os, csv, multiprocessing as mp
from datetime import datetime

class CSVLogger:
    def __init__(self, base_dir="./logs_lift_leg", timestamp=None, only_workers=False):
        self.timestamp  = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_dir   = base_dir
        os.makedirs(base_dir, exist_ok=True)
        self.only_workers = bool(only_workers)
        self._is_main  = (mp.current_process().name == "MainProcess")
        self._silenced = (self.only_workers and self._is_main)
        self._pid = os.getpid()

    def _path(self, category:str)->str:
        return os.path.join(self.base_dir, f"{category}_{self.timestamp}.pid{self._pid}.csv")

    def write(self, category:str, row:dict):
        if self._silenced: 
            return
        path = self._path(category)
        new_file = not os.path.exists(path)
        # asegúrate de usar siempre las mismas claves/orden por categoría si quieres archivos limpios
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if new_file:
                writer.writeheader()
            writer.writerow(row)

    # pickle-safe
    def __getstate__(self):
        state = self.__dict__.copy()
        # re-calcular en el hijo
        state["_is_main"]  = False
        state["_silenced"] = False
        state["_pid"] = None
        return state
    def __setstate__(self, state):
        import os, multiprocessing as mp
        self.__dict__.update(state)
        self._is_main  = (mp.current_process().name == "MainProcess")
        self._pid      = os.getpid()
        self._silenced = (self.only_workers and self._is_main)
        os.makedirs(self.base_dir, exist_ok=True)
