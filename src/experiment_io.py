"""Small file and plotting helpers shared by the course experiments."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    def encode(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(type(value).__name__)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, default=encode,
                                     allow_nan=False) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_plot(path, title, xlabel="", ylabel=""):
    ax = plt.gca()
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=125, bbox_inches="tight")
    plt.close()
