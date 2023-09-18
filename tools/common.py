import json
import os


def write_json(path, data, default=None):
    with open(path, "w") as f:
        json.dump(data, f, default=default)


def read_json(path):
    with open(path) as f:
        return json.load(f)

