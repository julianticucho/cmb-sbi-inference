from src.simulator.config import PATHS
import os

def create_directories():
    for path in PATHS.values():
        if not os.path.exists(path):
            os.makedirs(path)