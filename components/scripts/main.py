import hydra
import core
from omegaconf import DictConfig, OmegaConf
from core.pre_process.data_processor import DataProcessor
from core.models.XG_model import XGBoostModel
from pathlib import Path
from core.evaluation.evaluator import Evaluator

@hydra.main(config_path=core.CONFIG_PATH, config_name="main_config")
def main(config: DictConfig):

    data_processor = DataProcessor(config)
    model = XGBoostModel()
    evaluator = Evaluator()

    data = data_processor.load_data(Path(core.DATA_PATH)/"train.csv")
    x_train, x_test, y_train, y_test = data_processor.split_train_test_features_labels(data)

    model.train(x_train, x_test, y_train, y_test)
    y_pred = model.predict(x_test)
    acc = evaluator.get_binary_accuracy(y_test,y_pred )


    print(f"All acc: {acc}")

if __name__ == '__main__':
    main()