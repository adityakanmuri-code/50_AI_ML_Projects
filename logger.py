import logging
import os
from datetime import datetime
from config.configuration import Config

config = Config()

BASE_DIR = config.get("project", "base_dir")
PROJECT_NAME = config.get("project", "name")

LOG_DIR_NAME = config.get("logging", "log_dir")
LOG_LEVEL = config.get("logging", "log_level")
LOG_FORMAT = config.get("logging", "log_format")

log_dir = os.path.join(BASE_DIR, PROJECT_NAME, LOG_DIR_NAME)
os.makedirs(log_dir, exist_ok=True)

LOG_FILE = f"{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(log_dir, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    force=True
)