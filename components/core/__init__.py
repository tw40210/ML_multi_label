import datetime
import os

HOME_DIR = os.path.expanduser("~")
REPO_PATH = __file__[:__file__.find("components")]
CURRENT_TIME = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

CONFIG_PATH = os.path.join(REPO_PATH, "components/core/config")
DATA_PATH = os.path.join(REPO_PATH, "data")
