# data_manager.py

import logging

logger = logging.getLogger(__name__)

class DataManager:
    def __init__(self, data_file):
        self.data_file = data_file
        self.data = self.load_data()

    def load_data(self):
        data = {}
        try:
            with open(self.data_file, 'r', encoding='utf-8') as file:
                for line in file:
                    if ':' in line:
                        key, value = line.strip().split(':', 1)
                        data[key.strip()] = value.strip()
            logger.info(f"Data loaded from {self.data_file}")
        except Exception as e:
            logger.exception(f"Error loading data: {e}")
        return data

    def get_keys(self):
        return list(self.data.keys())

    def get_value(self, key):
        return self.data.get(key, "Key not found.")
