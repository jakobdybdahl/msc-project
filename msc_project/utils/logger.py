# adapted from https://github.com/openai/spinningup/blob/master/spinup/utils/logx.py

import atexit
import json
import os

import numpy as np


class Logger(object):
    def __init__(self, output_dir, fname="results.csv") -> None:
        assert os.path.exists(output_dir)

        self.output_dir = output_dir

        self.output_file = open(os.path.join(self.output_dir, fname), "w")
        atexit.register(self.output_file.close)

        self.first_row = True
        self.log_headers = []
        self.log_current_row = {}

    def save_config(self, config):
        """
        Saves config to running directory.

        Args:
            config: Dictonary containing key-value pairs. Assmumes whole dictonary is
            serializable by json.dump()
        """
        file = self.output_dir + "/config.json"
        with open(str(file), "w") as f:
            json.dump(config, f, indent=2)

    def log(self, key, val):
        if self.first_row:
            self.log_headers.append(key)
        else:
            assert key in self.log_headers, f"Key, {key} introduced which wasn't introduced in the first run."

        assert key not in self.log_current_row, f"{key} as already been set in this iteration"

        self.log_current_row[key] = val

    def dump(self):
        """
        Writes all the diagnotics to the output file
        """

        vals = []

        key_lens = [len(key) for key in self.log_headers]
        max_key_len = max(15, max(key_lens))
        keystr = "%" + "%d" % max_key_len
        fmt = "| " + keystr + "s | %15s |"
        n_slashes = 22 + max_key_len

        print("-" * n_slashes)

        for key in self.log_headers:
            val = self.log_current_row.get(key, "")
            valstr = "%8.3g" % val if hasattr(val, "__float__") else val
            print(fmt % (key, valstr))
            vals.append(val)

        print("-" * n_slashes, flush=True)

        if self.output_file is not None:
            if self.first_row:
                self.output_file.write(",".join(self.log_headers) + "\n")
            self.output_file.write(",".join(map(str, vals)) + "\n")
            self.output_file.flush()
        self.log_current_row.clear()
        self.first_row = False


class EpochLogger(Logger):
    def __init__(self, output_dir, fname="results.csv") -> None:
        super().__init__(output_dir, fname)

        self.epoch_dict = {}

    def store(self, **kwargs):
        for k, v in kwargs.items():
            if not (k in self.epoch_dict.keys()):
                self.epoch_dict[k] = []
            self.epoch_dict[k].append(v)

    def log(self, key, val=None, with_min_and_max=False, average_only=False):
        if val is not None:
            super().log(key, val)
        else:
            v = self.epoch_dict[key]
            vals = np.concatenate(v) if isinstance(v[0], np.ndarray) and len(v[0].shape) > 0 else v
            stats = statistics(vals, with_min_and_max=with_min_and_max)
            super().log(key if average_only else "average_" + key, stats[0])
            if not (average_only):
                super().log("std_" + key, stats[1])
            if with_min_and_max:
                super().log("max_" + key, stats[3])
                super().log("min_" + key, stats[2])
        self.epoch_dict[key] = []


def statistics(x, with_min_and_max=False):
    x = np.array(x, dtype=np.float32)

    n = len(x)
    mean = x.mean()

    sum_sq = np.sum((x - mean) ** 2)
    std = np.sqrt(sum_sq / n)

    if with_min_and_max:
        minimum = np.min(x) if len(x) > 0 else np.inf
        maximum = np.max(x) if len(x) > 0 else -np.inf
        return mean, std, minimum, maximum

    return mean, std
