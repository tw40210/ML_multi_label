import pandas as pd
import core
from pathlib import Path
from omegaconf import DictConfig
from core.pre_process.data_load import load_data_csv_to_df
from sklearn.model_selection import train_test_split

class DataProcessor:
    def __init__(self, config: DictConfig):
        self.config = config

    def load_data(self, data_path: Path, file_type:str = "csv"):
        if file_type=="csv":
            data = load_data_csv_to_df(data_path)
        else:
            raise NotImplementedError(f"file_type: {file_type} is not supported.")

        return data

    def split_train_test_features_labels(self, data:pd.DataFrame):
        data_cfg = self.config.data_pre_process
        if data_cfg.include_id:
            data = data[data.columns[1:]]

        if data_cfg.num_feature_columns is not None:
            x, y = data[data.columns[:data_cfg.num_feature_columns]], data[data.columns[data_cfg.num_feature_columns:]]
        elif data_cfg.num_label_columns is not None:
            x, y = data[data.columns[:-data_cfg.num_label_columns]], data[data.columns[-data_cfg.num_label_columns:]]
        else:
            raise ValueError(f"num_feature_columns and num_label_columns can't both be None.")

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=data_cfg.test_size, shuffle=data_cfg.shuffle)

        return x_train, x_test, y_train, y_test

    def split_features_labels(self, ):
        pass

