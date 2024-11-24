import re
import h5py
import math
import keras
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Literal

from settings import INPUT_DATA_PATH, LABEL_DATA_PATH


class ECGSequence(keras.utils.Sequence):
    @staticmethod
    def get_n_batches(hdf5_files, hdf5_dset, batch_size, drop=0):
        n_batches = 0
        batch_ranges = [0]
        for hdf5_file in hdf5_files:
            with h5py.File(hdf5_file, "r") as file:
                curr_batch_size = math.floor((file[hdf5_dset].shape[0] - drop) / batch_size)
                n_batches += curr_batch_size
                batch_ranges.append(batch_ranges[-1] + curr_batch_size)

        return batch_ranges, n_batches

    @classmethod
    def get_train_and_val(cls, hdf5_files, csv_files, hdf5_dset="tracings",
                          batch_size=8, val_split=0.02, drop=0, shuffle=True, **kwargs):
        n_batches = cls.get_n_batches(hdf5_files, hdf5_dset, batch_size, drop)[1]
        n_train = math.ceil(n_batches * (1 - val_split))
        train_seq = cls(hdf5_files, csv_files, hdf5_dset, batch_size,
                        drop=drop, shuffle=shuffle, end_batch=n_train, **kwargs)
        valid_seq = cls(hdf5_files, csv_files, hdf5_dset, batch_size,
                        drop=drop, shuffle=shuffle, start_batch=n_train, **kwargs)
        return train_seq, valid_seq

    def __init__(self, hdf5_files, csv_files=None, hdf5_dset="tracings", batch_size=8,
                 start_batch=0, end_batch=None, drop=0, shuffle=True, **kwargs):
        super().__init__(**kwargs)

        self.predict_mode = False
        if csv_files is None:
            self.predict_mode = True

        self.csvs = [] if self.predict_mode else \
            [pd.read_csv(file).values for file in csv_files]

        self.hdf5_dset = hdf5_dset
        self.hdf5s = [h5py.File(file, "r") for file in hdf5_files]
        self.ranges = self.get_n_batches(hdf5_files, hdf5_dset, batch_size, drop)[0]

        self.batch_size = batch_size
        self.start_batch = start_batch
        self.end_batch = end_batch if end_batch is not None else (
            self.get_n_batches(hdf5_files, hdf5_dset, batch_size, drop)[1])

        self.shuffle = shuffle
        self.shuffle_indices = np.arange(self.__len__())

    @property
    def n_classes(self):
        return self.csvs[0].shape[1]

    def on_epoch_end(self):
        np.random.shuffle(self.shuffle_indices)

    def __getitem__(self, item):
        if self.shuffle:
            item = self.shuffle_indices[item]

        file_index, batch_index = 0, 0
        for i in range(1, len(self.ranges)):
            if (self.start_batch + item) < self.ranges[i]:
                file_index = i - 1
                batch_index = item + self.start_batch - self.ranges[i - 1]
                break

        start = batch_index * self.batch_size
        end = start + self.batch_size

        if self.predict_mode:
            return np.array(self.hdf5s[file_index][self.hdf5_dset][start:end, :, :])
        else:
            return (np.array(self.hdf5s[file_index][self.hdf5_dset][start:end, :, :]),
                    np.array(self.csvs[file_index][start:end]))

    def __len__(self):
        return self.end_batch - self.start_batch

    def __del__(self):
        for file in self.hdf5s:
            file.close()


def _align_entries(indices: np.ndarray, data: np.ndarray, num_classes=6, drop=0):
    ydict = {}
    for item in data:
        ydict[item[0]] = item

    result = np.empty(shape=(len(indices) - drop, num_classes), dtype=object)
    for index, value in enumerate(indices):
        if index == len(indices) - drop: break
        result[index] = ydict[str(value)][4: 4 + num_classes] == "True"

    return result


def generate_label_file(input_file: str, output_file: str,
                        num_classes=6, drop=0, verbose=True):
    y = pd.read_csv(LABEL_DATA_PATH.parent / "exams.csv", dtype=object).values
    with h5py.File(INPUT_DATA_PATH / f"{input_file}", "r+") as file:
        x, ids = file["tracings"], file["exam_id"]
        y_curr = _align_entries(ids, y, num_classes, drop)

        if verbose:
            print(
                f"{f' FILE: {input_file} -> {output_file} '.center(60, '*')}\n"
                f"{f'X SHAPE: {x.shape}'.center(60, ' ')}\n"
                f"{f'Y SHAPE: {y_curr.shape} I SHAPE: {ids.shape}'.center(60, ' ')}"
            )

        # Save labels
        pd.DataFrame(y_curr).astype(int).to_csv(
            LABEL_DATA_PATH / output_file,
            sep=",", encoding="utf-8", index=False, header=True
        )


def generate_label_files(**kwargs):
    for file in get_files("input"):
        generate_label_file(file.name, f"{file.stem}.csv", **kwargs)


def get_files(data_type: Literal["input", "label"]):
    lookup = {
        "input": {
            "suffix": "hdf5",
            "path": INPUT_DATA_PATH,
        },
        "label": {
            "suffix": "csv",
            "path": LABEL_DATA_PATH,
        }
    }

    pattern = re.compile(rf"^exams_part\d+\.{lookup[data_type]['suffix']}$")
    files: list[Path] = [file for file in lookup[data_type]['path'].iterdir()
                         if pattern.match(file.name)]

    return sorted(files)
