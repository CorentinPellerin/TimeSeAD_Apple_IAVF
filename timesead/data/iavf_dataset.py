import functools
import os
import hashlib
from typing import Tuple, Optional, Union, Callable, Dict, Any, List
import tempfile
import subprocess
import shutil
import logging

import numpy as np
import pandas as pd
import torch
from random import sample, seed as set_random_seed

from timesead.data.dataset import BaseTSDataset
from timesead.data.preprocessing import minmax_scaler
from timesead.utils.metadata import DATA_DIRECTORY
from timesead.data.preprocessing.common import save_statistics

_logger = logging.getLogger(__name__)

NORMAL_TS_FOLDERS = [
    "i.O"
]
ANOMALOUS_TS_FOLDERS = [
    "n.i.O"
]


def preprocess_apple_data(out_dir: str, normal_pds: List[pd.DataFrame], anomalous_pds: List[str] = None, ):
    data = None          
    timesteps = None  
    for current_data in normal_pds:
        if data is None:
            data = current_data
            timesteps = list(current_data.index)
        else:
            if list(current_data.columns) != list(data.columns):
                raise ValueError(
                    f"Dataframe has not the same columns as the others. Others have {list(data.columns)}. While current has {list(current_data.columns)}. "
                )
            #'''
            current_timesteps = list(current_data.index) 
            if len(current_timesteps) > len(timesteps):
                if current_timesteps[:len(timesteps)] != timesteps:
                    raise ValueError(f"Dataframe has not the same timesteps as the others.")
                timesteps = current_timesteps
            elif len(current_timesteps) < len(timesteps):
                if timesteps[:len(current_timesteps)] != current_timesteps:
                    raise ValueError(f"Dataframe has not the same timesteps as the others.")
            else:
                if timesteps != current_timesteps:
                    raise ValueError(f"Dataframe has not the same timesteps as the others.")
            #'''

            data = pd.concat([data, current_data], axis=0, ignore_index=True)

    os.makedirs(out_dir)
    stats_file = os.path.join(out_dir, f'apple_training_stats.npz')
    save_statistics(data, stats_file)


class IAVFDataset(BaseTSDataset):
    SEQUENCE_WISE_AD = True

    def __init__(self, dataset_path: str = os.path.join(DATA_DIRECTORY, 'IAVFDataset'), 
                 training: bool = True, standardize: Union[bool, Callable[[pd.DataFrame, Dict], pd.DataFrame]] = True,
                 download: bool = False, preprocess: bool = True):
        """
        :param dataset_path: Folder from which to load the dataset.
        :param training: Whether to load the training or the test set.
        :param standardize: Can be either a bool that decides whether to apply the dataset-dependent default
            standardization or a function with signature (dataframe, stats) -> dataframe, where stats is a dictionary of
            common statistics on the training dataset (i.e., mean, std, median, etc. for each feature)
        :param download: Whether to download the dataset if it doesn't exist.
        :param preprocess: Whether to setup the dataset for experiments.
        """

        self.dataset_path = dataset_path
        self.data_path = os.path.join(dataset_path, 'data_statistics')
        self.training = training

        if not self._check_exists():
            raise RuntimeError('Dataset not found.')

        set_random_seed(12345)
        self.training_files = [
            os.path.join(dataset_path, folder, file)
            for folder in NORMAL_TS_FOLDERS
            for file in sorted(os.listdir(os.path.join(dataset_path, folder)))
            if file.endswith(".dat")
        ]
        self.training_files = sorted(sample(self.training_files, int(len(self.training_files) * 0.5)))
        self.training_split_uid = hashlib.sha256(";".join(self.training_files).encode()).hexdigest()
        self.training_pds = [
            pd.read_csv(file, delimiter="\t", index_col='time') for file in sorted(self.training_files)
        ]
        self.training_files_basenames = [os.path.basename(f) for f in self.training_files]
        print("[DEBUG] Fichiers de training sélectionnés :", [os.path.basename(f) for f in self.training_files])

        self.test_files = [
            os.path.join(dataset_path, folder, file)
            for folder in NORMAL_TS_FOLDERS
            for file in sorted(os.listdir(os.path.join(dataset_path, folder)))
            if file.endswith(".dat") and os.path.join(dataset_path, folder, file) not in self.training_files
        ]
        self.test_targets = [0 for _ in range(len(self.test_files))]
        self.test_files += [
            os.path.join(dataset_path, folder, file)
            for folder in ANOMALOUS_TS_FOLDERS
            for file in sorted(os.listdir(os.path.join(dataset_path, folder)))
            if file.endswith(".dat") and os.path.join(dataset_path, folder, file) not in self.training_files
        ]
        self.test_targets += [
            1 for folder in ANOMALOUS_TS_FOLDERS
            for file in sorted(os.listdir(os.path.join(dataset_path, folder)))
            if file.endswith(".dat") and os.path.join(dataset_path, folder, file) not in self.training_files
        ]
        self.test_pds = [
            pd.read_csv(file, delimiter="\t", index_col='time') for file in self.test_files
        ]
        self.test_files_basenames = [os.path.basename(f) for f in self.test_files]
        print("[DEBUG] Fichiers de test sélectionnés :", [os.path.basename(f) for f in self.test_files])

        if not self._check_preprocessed():
            if not preprocess:
                raise RuntimeError('Dataset needs to be processed for proper working. Pass preprocess=True to setup the'
                                   ' dataset.')

            _logger.info("Processed data files not found! Running pre-processing now. This might take several minutes.")
            preprocess_apple_data(
                os.path.join(self.data_path, self.training_split_uid), 
                normal_pds=self.training_pds, anomalous_pds=None,
            )

        self.train_lengths = [frame.shape[0] for frame in self.training_pds]
        self.test_lengths = [frame.shape[0] for frame in self.test_pds]

        # self.training_pds = None
        # self.test_pds = None

        self.inputs = None
        self.targets = None

        if callable(standardize):
            with np.load(os.path.join(self.data_path, self.training_split_uid, f'apple_training_stats.npz')) as d:
                stats = dict(d)
            self.standardize_fn = functools.partial(standardize, stats=stats)
        elif standardize:
            with np.load(os.path.join(self.data_path, self.training_split_uid, f'apple_training_stats.npz')) as d:
                stats = dict(d)
            self.standardize_fn = functools.partial(minmax_scaler, stats=stats)
        else:
            self.standardize_fn = None

    def load_data(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:

        frames = self.training_pds if self.training else self.test_pds

        inputs, targets = [], []
        print(f"[DEBUG - load_data] {'TRAINING' if self.training else 'TESTING'} mode : {len(frames)} fichiers à traiter")

        for i, data in enumerate(frames):
            if self.training:
                target = np.zeros(len(data), dtype=np.int64)
            else:
                wltc_index = int(self.test_files_basenames[i].split("_")[1].split(".dat")[0])
                if wltc_index == 0:
                    raise ValueError("In that case, we need to do (wltc_index + 1) below, otherwise 0 could be falsely be used as normal")
                target = np.asarray((-wltc_index if self.test_targets[i] <= 0 else wltc_index)) 
                target = target.astype(np.int64).repeat(len(data))

            if self.standardize_fn is not None:
                data = self.standardize_fn(data)

            data = data.fillna(method='ffill')  # forward fill (replace nan with entry of previous row)
            data = data.apply(lambda col: col.fillna(0.0), axis=0)
            data = data.astype(np.float32)

            input = data.to_numpy()

            inputs.append(input)
            targets.append(target)

            # Log détaillé
            # print(f"  · Fichier {i}: {basename} | shape: {input_array.shape} | target unique: {np.unique(target)}")

        return inputs, targets

    def __getitem__(self, item: int) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        if not (0 <= item < len(self)):
            raise KeyError('Out of bounds')

        if self.inputs is None or self.targets is None:
            self.inputs, self.targets = self.load_data()

        return (torch.as_tensor(self.inputs[item]),), (torch.as_tensor(self.targets[item]),)

    def __len__(self) -> Optional[int]:
        return len(self.train_lengths) if self.training else len(self.test_lengths)

    @property
    def seq_len(self) -> List[int]:
        if self.training:
            return self.train_lengths
        else:
            return self.test_lengths

    @property
    def num_features(self) -> int:
        return 8

    @staticmethod
    def get_default_pipeline() -> Dict[str, Dict[str, Any]]:
        # return {
        #     'subsample': {'class': 'SubsampleTransform', 'args': {'subsampling_factor': 5, 'aggregation': 'mean'}},
        #     'cache': {'class': 'CacheTransform', 'args': {}}
        # }
        return {}

    @staticmethod
    def get_feature_names():
        return [str(i) for i in range(1, 9)]

    def _check_exists(self) -> bool:
        # Only checks if the `data` folder exists
        data_folder_path = os.path.join(self.dataset_path)
        if not os.path.isdir(data_folder_path):
            return False
        return True

    def _check_preprocessed(self) -> bool:
        # Only checks if the `processed` folder exsits
        if not os.path.isdir(os.path.join(self.data_path, self.training_split_uid)):
            return False
        return True

