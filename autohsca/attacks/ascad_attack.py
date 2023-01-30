import logging

import numpy as np
from sklearn.utils import shuffle

from autohsca.constants import *
from autohsca.core.attack_class import AttackModel


# Attack Code from Ranking Loss Paper Github Repository (https://github.com/gabzai/Ranking-Loss-SCA)
class ASCADAttack(AttackModel):
    def __init__(self, model_name, num_attacks, dataset_type, real_key, byte, plaintext_ciphertext=None, num_classes=256, seed=None, shuffle=True, **kwargs):
        # Logger
        self.logger = logging.getLogger(ASCADAttack.__name__)
        super().__init__(AES_SBOX=ASCAD_AES_Sbox, model_name=model_name, num_attacks=num_attacks,
                         dataset_type=dataset_type, real_key=real_key, byte=byte, plaintext_ciphertext=plaintext_ciphertext,
                         num_classes=num_classes, seed=seed, shuffle=shuffle, **kwargs)
        self.dataset = dataset_type

    def attack(self, X_attack, Y_attack):
        super().attack(X_attack=X_attack, Y_attack=Y_attack)

    def _rank_compute(self, prediction, plaintext, secret_key, byte):
        (nb_trs, nb_hyp) = prediction.shape
        key_log_prob = np.zeros(nb_hyp)
        rank_evol = np.full(nb_trs, 255)
        prediction = np.log(prediction + 1e-40)
        for i in range(nb_trs):
            for k in range(nb_hyp):
                idx = self.AES_SBOX[k ^ plaintext[i, byte]]
                key_log_prob[k] += prediction[i, idx]
            rank_evol[i] = self._rk_key(key_log_prob, secret_key[byte])

        return rank_evol

    def _perform_attacks_(self, predictions, plain_cipher):
        nb_traces = predictions.shape[0]
        nb_attacks = self.num_attacks
        key = self.real_key
        byte = self.byte
        all_rk_evol = np.zeros((nb_attacks, nb_traces))

        for i in range(nb_attacks):
            if self.shuffle:
                predictions_shuffled, plt_shuffled = shuffle(predictions, plain_cipher, random_state=self.random_state)
            else:
                predictions_shuffled, plt_shuffled = predictions, plain_cipher

            all_rk_evol[i] = self._rank_compute(predictions_shuffled, plt_shuffled, key, byte=byte)
        all_rk_evol = self.trim_outlier_ranks(all_rk_evol, num=int(self.num_attacks/2))
        rk_avg = np.mean(all_rk_evol, axis=0)
        self.logger.info(f"All Ranks \n {all_rk_evol}")
        return rk_avg

    def _plot_model_attack_results(self, model_results_dir_path):
        super()._plot_model_attack_results(model_results_dir_path=model_results_dir_path)

    def _store_results(self,):
        super()._store_results()

    def _load_attack_model_(self):
        return super()._load_attack_model_()
