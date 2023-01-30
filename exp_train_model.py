import argparse
import os
import time
from datetime import timedelta

import numpy as np
from keras.saving.save import load_model

from autohsca import *
from autohsca.constants import *
from autohsca.utils import *
from autohsca.utils import print_dictionary


def _load_attack_model(model_file, logger):
    try:
        attack_model = load_model(model_file)
    except Exception as e:
        import traceback
        logger.info(traceback.format_exc())
        logger.info(str(e))
        attack_model = None
    return attack_model

model_dictionary = {ASCAD_CNN_BASELINE: ASCADCNNBaseline, ASCAD_MLP_BASELINE: ASCADMLPBaseline, NAS_MODEL: NASBasic5}


def perform_attack(dataset_reader_obj):
    num_attacks = 100
    real_key = dataset_reader_obj.get_key()
    byte = dataset_reader_obj.attack_byte
    attack_class_params = dict(model_name=model_name, dataset_type=dataset_name, num_classes=num_classes,
                               num_attacks=num_attacks, plaintext_ciphertext=plaintext_ciphertext_attack,
                               real_key=real_key, byte=byte, seed=seed, shuffle=True)
    attack_params = dict(X_attack=X_attack, Y_attack=Y_attack)
    if "ASCAD" in dataset_name:
        attack_obj = ASCADAttack(**attack_class_params)
    attack_obj.attack(**attack_params)
    logger.info(f'The model is already trained')
    logger.info('Best model summary:')
    attack_model.summary(print_fn=logger.info)


if __name__ == "__main__":
    # Make results deterministic
    seed = 1234
    os.environ['PYTHONHASHSEED'] = str(seed)
    parser = argparse.ArgumentParser(description='Model HP Tuner & Model Training')
    parser.add_argument('--dataset', type=str, required=True,
                        help='An ASCAD dataset for the attack. Possible values are ASCAD_desync0, ASCAD_desync50, ASCAD_desync100, ASCAD_desync0_variable, ASCAD_desync50_variable, ASCAD_desync100_variable')
    parser.add_argument('--model_name', type=str, required=True,
                        help='Name of the model to use. Possible values are ascad_mlp_baseline, ascad_cnn_baseline, nas_model')
    parser.add_argument('--max_trials', default=100, type=int,
                        help='Number of trials to use for random or bayesian tuner. Default value is 50.')
    parser.add_argument('--tuner_type', default='random', type=str,
                        help='Type of tuner to use for hyperparameter search. Possible values are greedy, hyperband, random, bayesian. Default value is random.')

    arguments = parser.parse_args()
    print(f"Arguments {arguments}")
    dataset_name = arguments.dataset
    args_model_name = arguments.model_name
    tuner_type = arguments.tuner_type
    max_trials = arguments.max_trials
    epochs = 200
    batch_size = 200

    # Load dataset
    if "ASCAD" in dataset_name:
        dataset_reader_obj = ASCADDatasetReader(dataset_type=dataset_name, load_key=True, load_metadata=True,
                                                load_ciphertext=True)
    (plaintext_ciphertext_profiling, plaintext_ciphertext_attack) = dataset_reader_obj.get_plaintext_ciphertext()
    (X_profiling, Y_profiling), (X_attack, Y_attack) = dataset_reader_obj.get_train_test_dataset()
    input_dim = X_profiling.shape[1]
    num_classes = len(np.unique(Y_profiling))
    metrics = ['accuracy']
    if 'nas' in args_model_name:
        model_name = f'{dataset_name.lower()}_{args_model_name}_{tuner_type}'
    else:
        model_name = f'{dataset_name.lower()}_{args_model_name}'

    log_path = os.path.join(os.getcwd(), 'logs', f'{model_name}.log')
    create_dir_recursively(log_path, is_file_path=True)
    logger = setup_logging(log_path=log_path)
    start_time = time.time()
    # objective = Objective("val_mean_rank", direction="min")
    condition = False
    model_file = os.path.join(get_trained_models_path(), f'{model_name}.tf')
    attack_model = _load_attack_model(model_file, logger)
    if attack_model is None:
        objective = 'val_accuracy'
        learner_params = {'num_classes': num_classes, 'metrics': metrics, 'input_dim': input_dim,
                          'seed': seed, 'max_trials': max_trials, 'tuner': tuner_type,
                          'dataset': dataset_name, 'model_name': model_name, 'objective': objective}
        model_class = model_dictionary[args_model_name]
        model = model_class(**learner_params)
        setup_random_seed(seed=seed)
        logger.info(f'Model name {model_name}')
        config = vars(arguments)
        logger.info(f"Arguments {print_dictionary(config)}")
        logger.info(f"Log File {log_path}")

        verbose = 1
        if tuner_type in [RANDOM_TUNER, BAYESIAN_TUNER]:
            n_e = 20
            model.fit(X=X_profiling, y=Y_profiling, batch_size=batch_size, epochs=n_e, final_model_epochs=epochs - n_e,
                      verbose=verbose)
        elif tuner_type == GREEDY_TUNER:
            n_e = 50
            model.fit(X=X_profiling, y=Y_profiling, batch_size=batch_size, epochs=n_e, final_model_epochs=epochs - n_e,
                      verbose=verbose)
        else:
            model.fit(X=X_profiling, y=Y_profiling, batch_size=batch_size, final_model_epochs=epochs, epochs=epochs,
                      verbose=verbose)

        logger.info('Best model summary:')
        model.summary(print_fn=logger.info)
        logger.info('Search Space summary:')
        model.search_space_summary()
        perform_attack(dataset_reader_obj)
    else:
        perform_attack(dataset_reader_obj)
    end_time = time.time()
    time_taken = timedelta(seconds=(end_time - start_time))
    logger.info(f'The total time elapsed for model {model_name} is {time_taken}')
    # model.evaluate(X_profiling, Y_profiling)
    # model.predict(X_profiling)
    # model.summary(print_fn=logger.info)

