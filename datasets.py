import re
import h5py
import math
import keras
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Literal, Sequence
from helpers import ResourcePath, ArrhythmiaType, CardiogramLead


class CardiogramSequence(keras.utils.Sequence):
    @staticmethod
    def get_batch_info(hdf5_files: Sequence[Path], hdf5_dset: str,
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
    def get_train_and_val(cls, hdf5_files: Sequence[Path], csv_files: Sequence[Path],
                          hdf5_dset: str = "tracings", batch_size: int = 32, seed: int = 42,
                          val_split: float = 0.02, drop_last: bool = True, **kwargs):
        n_batches = cls.get_batch_info(hdf5_files, hdf5_dset, batch_size, drop_last)[1]
        mask = np.random.default_rng(seed).permutation(n_batches)

        n_train = math.ceil(n_batches * (1 - val_split))
        train_seq = cls(hdf5_files, csv_files, hdf5_dset, batch_size,
                        drop_last=drop_last, seed=seed, mask=mask[:n_train], **kwargs)
        valid_seq = cls(hdf5_files, csv_files, hdf5_dset, batch_size,
                        drop_last=drop_last, seed=seed, mask=mask[n_train:], **kwargs)
        return train_seq, valid_seq

    def __init__(self, hdf5_files: Sequence[Path], csv_files: Sequence[Path] = None,
                 hdf5_dset: str = "tracings", batch_size: int = 32, shuffle: bool = True,
                 drop_last: bool = True, mask: Sequence[int] = None, seed: int = 42,
                 leads: Sequence[CardiogramLead] = tuple(CardiogramLead),
                 types: Sequence[ArrhythmiaType] = tuple(ArrhythmiaType), **kwargs):
        super().__init__(**kwargs)

        self.leads = leads
        self.shuffle = shuffle
        self.n_classes = len(types)
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)

        self.predict_mode = True if csv_files is None else False
        self.csvs = [] if self.predict_mode else \
            [pd.read_csv(file).loc[:, types].values for file in csv_files]

        self.hdf5_dset = hdf5_dset
        self.hdf5s = [h5py.File(file, "r") for file in hdf5_files]

        self.__global_ranges, global_len = self.get_batch_info(
            hdf5_files, hdf5_dset, batch_size, drop_last)

        if mask is not None and (np.array(mask) >= global_len).any():
            raise ValueError("The provided mask contains indices "
                             "that exceed the available batch range.")
        self.indices = np.array(mask if mask is not None else range(global_len))

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.indices)
    
    def __len__(self):
        return len(self.indices)

    def __del__(self):
        for file in self.hdf5s:
            file.close()

    def __getitem__(self, item: int):
        item = self.indices[item]

        file_index, batch_index = 0, 0
        for i in range(1, len(self.__global_ranges)):
            if item < self.__global_ranges[i]:
                file_index, batch_index = i - 1, item - self.__global_ranges[i - 1]
                break

        start = batch_index * self.batch_size
        end = min(start + self.batch_size,
                  self.hdf5s[file_index][self.hdf5_dset].shape[0])

        if self.predict_mode:
            return np.array(self.hdf5s[file_index][self.hdf5_dset][start:end, :, self.leads])
        else:
            return (np.array(self.hdf5s[file_index][self.hdf5_dset][start:end, :, self.leads]),
                    np.array(self.csvs[file_index][start:end]))


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
