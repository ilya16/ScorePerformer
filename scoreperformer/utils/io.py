import json
import os


def load_json(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    else:
        return {}


def dump_json(data, file_path):
    with open(file_path, 'w') as f:
        return json.dump(data, f)
