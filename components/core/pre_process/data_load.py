import pandas as pd
import core
from pathlib import Path
from omegaconf import DictConfig

def load_data_csv_to_df(data_path: Path):

    df = pd.read_csv(str(data_path))

    return df