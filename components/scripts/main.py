import hydra
import core
from omegaconf import DictConfig, OmegaConf
@hydra.main(config_path=core.CONFIG_PATH, config_name="main_config")
def main(config: DictConfig):
    print("Hello!")

if __name__ == '__main__':
    main()