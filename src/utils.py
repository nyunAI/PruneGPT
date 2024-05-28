import os
import logging
import json

class LoggingManager:
    @staticmethod
    def ensure_directory_exists(directory: str):
        if not os.path.exists(directory):
            os.makedirs(directory)

    @staticmethod
    def setup_logging(log_file: str):
        logging.basicConfig(filename=log_file, level=logging.INFO, 
                            format='%(asctime)s:%(levelname)s:%(message)s')

    @staticmethod
    def log_info(message: str):
        logging.info(message)


class ConfigManager:
    def __init__(self, config_path):
        with open(config_path, 'r') as file:
            self.config = json.load(file)

    def get(self, key, default=None):
        return self.config.get(key, default)
        