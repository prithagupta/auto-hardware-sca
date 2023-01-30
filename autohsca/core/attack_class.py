import logging
import os
from abc import ABCMeta, abstractmethod

import h5py
import matplotlib.pyplot as plot
import numpy as np
from keras.models import load_model
from keras.utils import to_categorical
from sklearn.utils import check_random_state

from autohsca.constants import *
from autohsca.utils import get_results_path, create_directory_safely, get_trained_models_path, \
    get_model_parameters_count


class AttackModel(metaclass=ABCMeta):
    def __init__(self, AES_SBOX, model_name, num_attacks, dataset_type, real_key, byte, plaintext_ciphertext=None,
                 num_classes=256, seed=None, shuffle=True, **kwargs):
        self.AES_SBOX = AES_SBOX
        self.model_name = model_name
        self.num_attacks = num_attacks
        self.dataset_type = dataset_type
        self.real_key = real_key
        self.byte = byte
        self.plaintext_ciphertext = plaintext_ciphertext
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.random_state = check_random_state(seed=seed)
        self.logger = logging.getLogger(AttackModel.__name__)
        # Initialize result variables
        self.trainable_params = 0
        self.non_trainable_params = 0
        self.total_params = 0
        self.n_conv_layers = 0
        self.n_dense_layers = 0
        self.model_scores = None
        self.model_accuracy = 0
        self.guessing_entropies = None
        self.guessing_entropy_final = 0
        self.model_qte_traces = 0
        self.model_qte_traces_3 = 0
        self.model = None

    def attack(self, X_attack, Y_attack):
        # Load model for attack
        self.model = self._load_attack_model_()
        self.model.summary(print_fn=self.logger.info)
        for tuner_type in TUNER_TYPES:
            if tuner_type in self.model_name:
                self.logger.info(f"Tuner Type is {tuner_type}")
                break
        self.trainable_params, self.non_trainable_params, self.total_params, self.n_conv_layers, self.n_dense_layers \
            = get_model_parameters_count(self.model)
        self.logger.info(f'Trainable params for model  {self.model_name} = {self.trainable_params}')
        self.logger.info(f'Non-trainable params for model {self.model_name} = {self.non_trainable_params}')
        self.logger.info(f'Total params for model {self.model_name} = {self.total_params}')
        self.logger.info(f'Convolutional layers for model {self.model_name} = {self.n_conv_layers}')
        self.logger.info(f'Dense layers for model {self.model_name} = {self.n_dense_layers}')
        X_attack = X_attack.reshape((X_attack.shape[0], X_attack.shape[1], 1))
        Y_attack = to_categorical(Y_attack, num_classes=self.num_classes)
        self.logger.info('*****************************************************************************')
        self.logger.info(f'Performing attack using model {self.model_name}')
        predictions = self.model.predict(X_attack)
        model_accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(Y_attack, axis=1))
        self.model_scores = predictions
        self.model_accuracy = model_accuracy

        guess_entropy_evol = self._perform_attacks_(predictions=predictions, plain_cipher=self.plaintext_ciphertext)
        self.logger.info(f"Guess Entropy {guess_entropy_evol}")
        self.guessing_entropies = guess_entropy_evol
        self.guessing_entropy_final = self.guessing_entropies[-1]
        self.logger.info(f'Ranks = {self.guessing_entropies}')
        self.logger.info(f'Guessing Entropy Final {self.guessing_entropy_final}')
        def get_value(guess_entropy_evol, n):
            if not np.any(guess_entropy_evol <= n):
                return Y_attack.shape[0]
            else:
                return np.argmax(guess_entropy_evol <= n) + 1

        self.model_qte_traces = get_value(guess_entropy_evol, 0.0)
        self.model_qte_traces_3 = get_value(guess_entropy_evol, 2.0)
        self.logger.info(f'Model {self.model_name} QTE for GE smaller that 1: {self.model_qte_traces}')
        self.logger.info(f'Model {self.model_name} QTE for GE smaller that 3: {self.model_qte_traces_3}')

        self._store_results()

    def attack_from_scores(self, scores, model):
        _, _, total_params = get_model_parameters_count(model)
        model_min_complete_mean_rank = np.zeros(self.n_folds)
        for fold_id in range(self.n_folds):
            predictions = scores[fold_id * self.num_traces:(fold_id + 1) * self.num_traces]
            plain_cipher_fold = self.plaintext_ciphertext[fold_id * self.num_traces:(fold_id + 1) * self.num_traces]
            offset_fold = self.offset[fold_id * self.num_traces:(fold_id + 1) * self.num_traces]
            avg_rank = self._perform_attacks_(predictions=predictions, plain_cipher_fold=plain_cipher_fold,
                                              offset_fold=offset_fold)
            model_min_complete_mean_rank[fold_id] = rank = np.amin(avg_rank)
            self.logger.info('Fold {}: min complete mean rank for model = {}'.format(fold_id, rank))
        return model_min_complete_mean_rank, total_params

    def _rk_key(self, rank_array, key):
        if np.any(np.isnan(rank_array)):
            rank = 125
        else:
            ranking = np.argsort(np.argsort(rank_array)[::-1])
            rank = ranking[key]
        return rank


    @abstractmethod
    def _perform_attacks_(self, predictions, plain_cipher):
        pass


    def _plot_model_attack_results(self, model_results_dir_path):
        plot.rcParams["figure.figsize"] = (15, 10)
        plot.ylim(-5, 200)
        plot.grid(True)
        plot.plot(self.guessing_entropies, '-')
        plot.xlabel('Number of Traces', size=30)
        plot.ylabel('Guessing Entropy', size=30)
        plot.xticks(fontsize=30)
        plot.yticks(fontsize=30)
        plot.savefig(os.path.join(model_results_dir_path, f'{self.model_name}_convergence.png'), format='png', dpi=1200)
        plot.close()

    def trim_outlier_ranks(self, all_rk_evol, num=100):
        b = []
        for col in all_rk_evol.T:
            col = col[np.argpartition(col, num + 1)[:num]]
            b.append(col)
        rk_evol = np.array(b).T
        self.logger.info(f"Sorted Ranks: {np.sort(rk_evol)[:, ::-1]}")
        # if self.leakage_model == HW:
        #    rk_evol = np.sort(rk_evol)[:, ::-1]
        return rk_evol

    def _store_results(self):
        # Store the final evaluated results to .npy files & .svg file
        results_path = get_results_path(folder=f"{RESULTS}")

        for dir in [self.dataset_type, self.model_name]:
            results_path = os.path.join(results_path, dir)
            create_directory_safely(results_path)
            self.logger.info("Creating Directory {}: ".format(results_path))
        result_file_path = os.path.join(results_path, 'final_results.h5')
        self.logger.info("Creating results at path {}: ".format(result_file_path))

        with h5py.File(result_file_path, 'w') as hdf:
            model_params_group = hdf.create_group('model_parameters')
            model_params_group.create_dataset(TRAINABLE_PARAMS, data=self.trainable_params)
            model_params_group.create_dataset(NON_TRAINABLE_PARAMS, data=self.non_trainable_params)
            model_params_group.create_dataset(TOTAL_PARAMS, data=self.total_params)
            model_params_group.create_dataset(N_CONV_LAYERS, data=self.n_conv_layers)
            model_params_group.create_dataset(N_DENSE_LAYERS, data=self.n_dense_layers)
            model_metrics_group = hdf.create_group('model_metrics_group')
            model_metrics_group.create_dataset(SCORES, data=self.model_scores)
            model_metrics_group.create_dataset(GUESSING_ENTROPY_FINAL, data=self.guessing_entropy_final)
            model_metrics_group.create_dataset(ACCURACY, data=self.model_accuracy)
            model_metrics_group.create_dataset(QTE_NUM_TRACES, data=self.model_qte_traces)
            model_metrics_group.create_dataset(GUESSING_ENTROPIES, data=self.guessing_entropies)
            model_metrics_group.create_dataset(QTE_NUM_TRACES + '3', data=self.model_qte_traces_3)
            hdf.close()

        self._plot_model_attack_results(results_path)


    def _load_attack_model_(self):
        trained_models_path = get_trained_models_path()
        model_file_name = os.path.join(trained_models_path, f'{self.model_name}.tf')
        self.logger.info("Model stored at {}".format(model_file_name))
        attack_model = load_model(model_file_name)
        return attack_model
