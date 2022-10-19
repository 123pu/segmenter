# -*- coding: utf-8 -*-

import yaml
from pathlib import Path
import os


def load_config():
    config_path = os.path.join(os.path.join(Path(__file__).parent, 'segmenter.yaml'))
    return yaml.load(
        open(config_path, "r"), Loader=yaml.FullLoader
    )
