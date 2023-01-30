import os

import numpy as np
from keras.optimizers import RMSprop
from keras.utils import to_categorical
from tensorflow.python.keras.callbacks import ModelCheckpoint
from autohsca.core.sca_base_model import SCABaseModel
from autohsca.utils import get_trained_models_path, check_file_exists

__all__ = ['SCANNModel']

class SCANNModel(SCABaseModel):
    def __init__(self, model_name, num_classes, input_dim, kernel_regularizer=None,
                 kernel_initializer="he_uniform", optimizer=RMSprop(learning_rate=0.00001), metrics=['accuracy'],
                 **kwargs):

        self.num_classes = num_classes
        self.classes_ = np.arange(num_classes)
        self.input_dim = input_dim
        self.loss_function = 'categorical_crossentropy'
        self.kernel_regularizer = kernel_regularizer
        self.kernel_initializer = kernel_initializer
        self.optimizer = optimizer
        self.metrics = metrics
        self.model_name = model_name
        # check the model path, make the default one in the deep-learning-sca, fileformat dataset_type_model_lf
        self.model_file = os.path.join(get_trained_models_path(), f'{self.model_name}.tf')
        self.model, self.scoring_model = self._construct_model_(kernel_regularizer=self.kernel_regularizer,
                                                                kernel_initializer=self.kernel_initializer)

        self.model.compile(loss=self.loss_function, optimizer=self.optimizer, metrics=self.metrics)
        self.scoring_model.compile(loss=self.loss_function, optimizer=self.optimizer, metrics=self.metrics)


    def _construct_model_(self, **kwargs):
        raise NotImplemented

    def reshape_inputs(self, X, y):
        X = X.reshape((X.shape[0], X.shape[1], 1))
        if y is not None:
            y = to_categorical(y, num_classes=self.num_classes)
        return X, y

    def fit(self, X, y, epochs=200, batch_size=100, verbose=1, **kwargs):
        X, y = self.reshape_inputs(X, y)
        check_file_exists(os.path.dirname(self.model_file))
        save_model = ModelCheckpoint(self.model_file)
        callbacks = [save_model]
        self.model.fit(x=X, y=y, batch_size=batch_size, epochs=epochs, callbacks=callbacks, verbose=verbose,
                           **kwargs)
        return self

    def predict(self, X, verbose=0, **kwargs):
        X, _ = self.reshape_inputs(X, None)
        scores = self.predict_scores(X=X, verbose=verbose, **kwargs)
        pred = np.argmax(scores, axis=1)
        return pred

    def predict_scores(self, X, verbose=0, batch_size=200, **kwargs):
        X, _ = self.reshape_inputs(X, None)
        predictions = self.model.predict(x=X, batch_size=batch_size, verbose=verbose, **kwargs)
        return predictions

    def evaluate(self, X, y, verbose=1, batch_size=200, **kwargs):
        X, y = self.reshape_inputs(X, y)
        model_metrics = self.model.evaluate(x=X, y=y, batch_size=batch_size, verbose=verbose, **kwargs)
        return model_metrics

    def summary(self, **kwargs):
        self.model.summary(**kwargs)

