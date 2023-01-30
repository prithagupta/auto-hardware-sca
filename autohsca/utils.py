import inspect
import logging
import multiprocessing
import os
import os.path
import random
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from tensorflow.core.protobuf.config_pb2 import ConfigProto
from tensorflow.python.client.session import Session

from autohsca.constants import *

__all__ = ['setup_random_seed', 'create_dir_recursively', 'setup_logging', 'mean_rank_metric', 'standardize_features',
           'get_absolute_path', 'create_directory_safely', 'get_trained_models_path', 'get_results_path',
           'get_datasets_path', 'get_model_parameters_count']



def check_file_exists(file_path):
    file_path = os.path.normpath(file_path)
    if not os.path.exists(file_path):
        print("Error: provided file path '%s' does not exist!" % file_path)
        sys.exit(-1)
    return

def print_dictionary(dictionary):
    output = "\n"
    for key, value in dictionary.items():
        output = output + str(key) + " => " + str(value).strip() + "\n"
    return output



def setup_random_seed(seed=1234):
    # logger.info('Seed value: {}'.format(seed))
    logger = logging.getLogger("Setup Logging")
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["KERAS_BACKEND"] = "tensorflow"
    devices = tf.config.experimental.list_physical_devices('GPU')
    logger.info("Devices {}".format(devices))
    n_gpus = len(devices)
    logger.info("Number of GPUS {}".format(n_gpus))
    if n_gpus == 0:
        config = ConfigProto(
            intra_op_parallelism_threads=1,
            inter_op_parallelism_threads=1,
            allow_soft_placement=True,
            log_device_placement=False,
            device_count={"CPU": multiprocessing.cpu_count() - 2},
        )
    else:
        config = ConfigProto(
            allow_soft_placement=True,
            log_device_placement=True,
            intra_op_parallelism_threads=2,
            inter_op_parallelism_threads=2,
        )
        config.gpu_options.allow_growth = True
    sess = Session(config=config)
    K.set_session(sess)


def setup_logging(log_path=None, level=logging.INFO):
    """Function setup as many logging for the experiments"""
    if log_path is None:
        dirname = os.path.dirname(
            os.path.abspath(inspect.getfile(inspect.currentframe()))
        )
        dirname = os.path.dirname(dirname)
        log_path = os.path.join(dirname, "experiments", "logs", "logs.log")
        create_dir_recursively(log_path, True)
    logging.basicConfig(
        filename=log_path,
        filemode='w',
        level=level,
        format="%(asctime)s %(name)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("SetupLogger")
    logger.info("log file path: {}".format(log_path))
    return logger



def create_dir_recursively(path, is_file_path=False):
    if is_file_path:
        path = os.path.dirname(path)
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def create_directory_safely(path, is_file_path=False):
    try:
        if is_file_path:
            path = os.path.dirname(path)
        if not os.path.exists(path):
            os.mkdir(path)
    except Exception as e:
        print(str(e))


def standardize_features(x_train, x_test, standardize='minmax'):
    if standardize == 'minmax':
        standardize = MinMaxScaler()
    elif standardize == 'standard':
        standardize = StandardScaler()
    else:
        standardize = RobustScaler()
    x_train = standardize.fit_transform(x_train)
    x_test = standardize.transform(x_test)
    return x_train, x_test


def mean_rank_metric(y_true, y_pred):
    y_true = np.argmax(y_true, axis=1).astype(np.int32)
    scores_df = pd.DataFrame(data=y_pred)
    final_ranks = scores_df.rank(ascending=False, axis=1)
    final_ranks = final_ranks.to_numpy(dtype='int32')
    predicted_ranks = np.zeros(shape=(y_true.shape[0]))
    for itr in range(y_true.shape[0]):
        true_label = y_true[itr]
        predicted_ranks[itr] = final_ranks[itr, true_label]
    return np.mean(predicted_ranks)



def get_absolute_path():
    dirname = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    absolute_path = os.path.dirname(dirname)
    return absolute_path

def get_trained_models_path():
    absolute_path = get_absolute_path()
    trained_models_path = os.path.join(absolute_path, TRAINED_MODELS)
    # trained_models_path = os.path.join(absolute_path, "models", subfolder, folder)
    create_directory_safely(trained_models_path)
    return trained_models_path


def get_results_path(folder=RESULTS):
    absolute_path = get_absolute_path()
    results_path = os.path.join(absolute_path, folder)
    create_directory_safely(results_path)
    return results_path


def get_datasets_path():
    absolute_path = get_absolute_path()
    datasets_path = os.path.join(absolute_path, "deepscapy", "datasets")
    return datasets_path


def get_model_parameters_count(model):
    stringlist = []
    trainable_params = 0
    non_trainable_params = 0
    total_params = 0

    model.summary(print_fn=lambda x: stringlist.append(x))
    conv_layers = []
    dense_layers = []
    for line in stringlist:
        if 'Total params:' in line:
            total_params = int(line.split('Total params: ')[1].replace(',', ''))
        if 'Trainable params: ' in line:
            trainable_params = int(line.split('Trainable params: ')[1].replace(',', ''))
        if 'Non-trainable params: ' in line:
            non_trainable_params = int(line.split('Non-trainable params: ')[1].replace(',', ''))
        if 'Conv' in line:
            conv_layers.append(line)
        if 'Dense' in line:
            dense_layers.append(line)
    n_conv_layers = len(conv_layers)
    n_dense_layers = len(dense_layers)
    return trainable_params, non_trainable_params, total_params, n_conv_layers, n_dense_layers


