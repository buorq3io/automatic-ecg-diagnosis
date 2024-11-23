import h5py
import math
import keras
import numpy as np
import pandas as pd


class ECGSequence(keras.utils.Sequence):
    @staticmethod
    def get_n_batches(hdf5_files, hdf5_dset, batch_size):
        n_batches = 0
        batch_ranges = [0]
        for hdf5_file in hdf5_files:
            with h5py.File(hdf5_file, "r") as file:
                curr_batch_size = math.floor(file[hdf5_dset].shape[0] / batch_size)
                n_batches += curr_batch_size
                batch_ranges.append(batch_ranges[-1] + curr_batch_size)

        return batch_ranges, n_batches

    @classmethod
    def get_train_and_val(cls, hdf5_files, csv_files, hdf5_dset="tracings",
                          batch_size=8, val_split=0.02, **kwargs):
        n_batches = cls.get_n_batches(hdf5_files, hdf5_dset, batch_size)[1]
        n_train = math.ceil(n_batches * (1 - val_split))
        train_seq = cls(hdf5_files, csv_files, hdf5_dset,
                        batch_size, end_batch=n_train, **kwargs)
        valid_seq = cls(hdf5_files, csv_files, hdf5_dset,
                        batch_size, start_batch=n_train, **kwargs)
        return train_seq, valid_seq

    def __init__(self, hdf5_files, csv_files=None, hdf5_dset="tracings",
                 batch_size=8, start_batch=0, end_batch=None, **kwargs):
        super().__init__(**kwargs)

        self.predict_mode = False
        if csv_files is None:
            self.predict_mode = True

        self.csvs = [] if self.predict_mode else \
            [pd.read_csv(file).values for file in csv_files]

        self.hdf5_dset = hdf5_dset
        self.hdf5s = [h5py.File(file, "r") for file in hdf5_files]
        self.ranges = self.get_n_batches(hdf5_files, hdf5_dset, batch_size)[0]

        self.batch_size = batch_size
        self.start_batch = start_batch
        self.end_batch = end_batch if end_batch is not None else (
            self.get_n_batches(hdf5_files, hdf5_dset, batch_size)[1])

    @property
    def n_classes(self):
        return self.csvs[0].shape[1]

    def __getitem__(self, item):
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
