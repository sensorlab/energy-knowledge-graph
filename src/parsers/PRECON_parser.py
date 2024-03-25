import pandas as pd
import os
from tqdm import tqdm
from pathlib import Path
from src.helper import save_to_pickle


def parse_PRECON(data_path: Path, save_path: Path):
    data = {}
    for f in tqdm(os.listdir(data_path)):
        if f.endswith(".csv") and "House" in f:
            name = "PRECON_" + f.split(".")[0][5:]
            df = pd.read_csv(data_path/f)
            df["Date_Time"] = pd.to_datetime(df["Date_Time"])
            df = df.set_index("Date_Time")
            df = pd.DataFrame(df["Usage_kW"])
            df.rename(columns={"Usage_kW": "aggregate"}, inplace=True)
            df = df.resample("1min").ffill(limit=2).dropna()
            df = df*1000
            data[name] = {"aggregate" : df}
            
    save_to_pickle(data, save_path)