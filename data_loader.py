from datasets import load_dataset
import os

class DataLoader:
    def __init__(self, data_name, cache_dir="cache"):
        self.data_name = data_name
        self.cache_dir = cache_dir
        self.dataset = None
        
    def load_dataset(self):
        if self.dataset is None:
            # Ensure cache directory exists
            os.makedirs(self.cache_dir, exist_ok=True)
            # Load and keep dataset in memory
            self.dataset = load_dataset(
                self.data_name,
                cache_dir=self.cache_dir,
                keep_in_memory=True
            )
        return self.dataset 