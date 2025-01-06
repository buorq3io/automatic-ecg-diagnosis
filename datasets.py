import re
import h5py
import math
import keras
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Literal, Iterable
from helpers import ResourcePath, ArrhythmiaType, CardiogramLead


class CardiogramSequence(keras.utils.Sequence):
    @staticmethod
    def get_batch_info(hdf5_files: Iterable[Path], hdf5_dset: str,
                       batch_size: int, drop_last: bool):
        n_batches = 0
        batch_ranges = [0]

        drop = 1 if drop_last else 0
        for hdf5_file in hdf5_files:
            with h5py.File(hdf5_file, "r") as file:
                curr_batch_size = math.ceil(file[hdf5_dset].shape[0] / batch_size) - drop
                n_batches += curr_batch_size
                batch_ranges.append(batch_ranges[-1] + curr_batch_size)

        return batch_ranges, n_batches

    @classmethod
    def get_train_and_val(cls, hdf5_files: Iterable[Path], csv_files: Iterable[Path],
                          hdf5_dset: str = "tracings", batch_size: int = 32,
                          val_split: float = 0.02, drop_last: bool = True, **kwargs):
        n_batches = cls.get_batch_info(hdf5_files, hdf5_dset, batch_size, drop_last)[1]
        n_train = math.ceil(n_batches * (1 - val_split))
        train_seq = cls(hdf5_files, csv_files, hdf5_dset, batch_size,
                        drop_last=drop_last, end_batch=n_train, **kwargs)
        valid_seq = cls(hdf5_files, csv_files, hdf5_dset, batch_size,
                        drop_last=drop_last, start_batch=n_train, **kwargs)
        return train_seq, valid_seq

    def __init__(self, hdf5_files: Iterable[Path], csv_files: Iterable[Path] = None,
                 hdf5_dset: str = "tracings", batch_size: int = 32, start_batch: int = 0,
                 end_batch: int = None, drop_last: bool = True, shuffle: bool = True,
                 leads: Iterable[CardiogramLead] = tuple(CardiogramLead),
                 types: Iterable[ArrhythmiaType] = tuple(ArrhythmiaType), **kwargs):
        super().__init__(**kwargs)

        self.predict_mode = False
        if csv_files is None:
            self.predict_mode = True

        self.csvs = [] if self.predict_mode else \
            [pd.read_csv(file).loc[:, types].values for file in csv_files]

        self.hdf5_dset = hdf5_dset
        self.hdf5s = [h5py.File(file, "r") for file in hdf5_files]
        self.ranges = self.get_batch_info(hdf5_files, hdf5_dset, batch_size, drop_last)[0]

        self.batch_size = batch_size
        self.start_batch = start_batch
        self.end_batch = end_batch if end_batch is not None else (
            self.get_batch_info(hdf5_files, hdf5_dset, batch_size, drop_last)[1])

        self.leads = leads
        self.shuffle = shuffle
        self.shuffle_indices = np.arange(self.__len__())

    @property
    def n_classes(self):
        return self.csvs[0].shape[1]

    def on_epoch_end(self):
        np.random.shuffle(self.shuffle_indices)

    def __getitem__(self, item: int):
        if self.shuffle:
            item = self.shuffle_indices[item]

        file_index, batch_index = 0, 0
        for i in range(1, len(self.ranges)):
            if (self.start_batch + item) < self.ranges[i]:
                file_index = i - 1
                batch_index = item + self.start_batch - self.ranges[i - 1]
                break

        start = batch_index * self.batch_size
        end = min(start + self.batch_size,
                  self.hdf5s[file_index][self.hdf5_dset].shape[0])

        if self.predict_mode:
            return np.array(self.hdf5s[file_index][self.hdf5_dset][start:end, :, self.leads])
        else:
            return (np.array(self.hdf5s[file_index][self.hdf5_dset][start:end, :, self.leads]),
                    np.array(self.csvs[file_index][start:end]))

    def __len__(self):
        return self.end_batch - self.start_batch

    def __del__(self):
        for file in self.hdf5s:
            file.close()


def generate_label_file(input_file: str, output_file: str):
    with h5py.File(ResourcePath.TRAIN_INPUTS / f"{input_file}", "r") as file:
        x, ids = file["tracings"], file["exam_id"]

        y = pd.read_csv(ResourcePath.TRAIN_LABELS.parent / "exams.csv", dtype=object)
        y = (y.astype({"exam_id": int})).set_index("exam_id").reindex(ids)
        with pd.option_context('future.no_silent_downcasting', True):
            y.replace({"True": 1, "False": 0}, inplace=True)

        y.to_csv(ResourcePath.TRAIN_LABELS / output_file, index=False)


def generate_label_files():
    for file in get_files("input"):
        generate_label_file(file.name, f"{file.stem}.csv")


def get_files(data_type: Literal["input", "label"]):
    lookup = {
        "input": {
            "suffix": "hdf5",
            "path": ResourcePath.TRAIN_INPUTS,
        },
        "label": {
            "suffix": "csv",
            "path": ResourcePath.TRAIN_LABELS,
        }
    }

    pattern = re.compile(rf"^exams_part\d+\.{lookup[data_type]['suffix']}$")
    files: list[Path] = [file for file in lookup[data_type]['path'].iterdir()
                         if pattern.match(file.name)]

    return sorted(files)
