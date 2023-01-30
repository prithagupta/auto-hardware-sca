import logging
import os
import sys
from abc import ABCMeta

import h5py
import numpy as np

from autohsca.dataset_reader.dataset_reader import DatasetReader
from ..constants import ASCAD_DESYNC0, ASCAD_DATASETS
from ..utils import check_file_exists


class ASCADDatasetReader(DatasetReader, metaclass=ABCMeta):
    def __init__(self, dataset_type=ASCAD_DESYNC0, load_key=False, load_metadata=False, **kwargs):
        super(ASCADDatasetReader, self).__init__(dataset_folder='ASCAD', **kwargs)
        self.logger = logging.getLogger(ASCADDatasetReader.__name__)
        self.load_metadata = load_metadata
        self.load_key = load_key
        assert (dataset_type in ASCAD_DATASETS), "Dataset Type {} does not exist, only following types" \
                                                 "exists {}".format(dataset_type, ASCAD_DATASETS)
        self.database_file = os.path.join(self.dirname, dataset_type, "{}.h5".format(dataset_type))

        if self.load_key:
            self.key_file = os.path.join(self.dirname, dataset_type, "key.npy")
        else:
            self.key_file = ''
        if 'variable' in dataset_type:
            self.key_string = '00112233445566778899AABBCCDDEEFF'
        else:
            self.key_string = '4DFBE0F27221FE10A78D4ADC8E490469'
        self.logger.info("Dataset File Path {}".format(self.database_file))
        check_file_exists(self.database_file)
        self.attack_byte = 2
        self.__load_dataset__()

    def __load_dataset__(self):
        try:
            in_file = h5py.File(self.database_file, "r")
        except:
            self.logger.error(
                f"Error: can't open HDF5 file {self.database_file} for reading (it might be malformed) ...")
            sys.exit(-1)
        # Load profiling traces
        self.X_profiling = np.array(in_file['Profiling_traces/traces'], dtype=np.float32)
        self.X_attack = np.array(in_file['Attack_traces/traces'], dtype=np.float32)
        self.Y_profiling = np.array(in_file['Profiling_traces/labels'], dtype=np.int)
        self.Y_attack = np.array(in_file['Attack_traces/labels'], dtype=np.int)

        # Load metadata
        if self.load_metadata:
            self.profiling_metadata = in_file['Profiling_traces/metadata']
            self.attack_metadata = in_file['Attack_traces/metadata']
            self.plaintext_profiling = self.profiling_metadata['plaintext']
            self.plaintext_attack = self.attack_metadata['plaintext']
            self.profiling_keys = self.profiling_metadata['key']
            self.attack_keys = self.attack_metadata['key']
            if "ciphertext" in self.profiling_metadata.dtype.fields and "ciphertext" in self.attack_metadata.dtype.fields:
                self.ciphertext_profiling = self.profiling_metadata['ciphertext']
                self.ciphertext_attack = self.attack_metadata['ciphertext']
            else:
                self.ciphertext_profiling, self.ciphertext_attack = None, None
        else:
            self.profiling_metadata = None
            self.attack_metadata = None

        if self.load_key:
            self.key = np.load(self.key_file)
            self.logger.info("Key {}".format(self.key))

        else:
            self.key = None

        in_file.close()
    def get_plaintext_ciphertext(self):
        return self.get_plaintext()

    def get_plaintext(self):
        return self.plaintext_profiling, self.plaintext_attack

    def get_ciphertext(self):
        return self.ciphertext_profiling, self.ciphertext_attack

    def get_metadata(self):
        return self.profiling_metadata, self.attack_metadata

    def get_key(self):
        return self.key

    def get_keys(self):
        return self.profiling_keys, self.attack_keys

    def get_train_test_dataset(self):
        return super(ASCADDatasetReader, self).get_train_test_dataset()