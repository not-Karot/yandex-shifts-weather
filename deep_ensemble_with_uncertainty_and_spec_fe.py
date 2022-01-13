import codecs
import copy
import gc
import json
import logging
import os
import pickle
import random
import time
from typing import List, Tuple, Union

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import median_absolute_error, r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import Binarizer, KBinsDiscretizer
import tensorflow as tf
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras.utils import losses_utils, tf_utils
from tensorflow.python.ops.losses import util as tf_losses_util
import tensorflow_addons as tfa
import tensorflow_probability as tfp
import umap
# %%
from assessment import calc_uncertainty_regection_curve, f_beta_metrics
from uncertainty import ensemble_uncertainties_regression



class LossFunctionWrapper(tf.keras.losses.Loss):
    def __init__(self,
                 fn,
                 reduction=losses_utils.ReductionV2.AUTO,
                 name=None,
                 **kwargs):
        super(LossFunctionWrapper, self).__init__(reduction=reduction, name=name)
        self.fn = fn
        self._fn_kwargs = kwargs

    def call(self, y_true, y_pred):
        if tensor_util.is_tensor(y_pred) and tensor_util.is_tensor(y_true):
            y_pred, y_true = tf_losses_util.squeeze_or_expand_dimensions(y_pred, y_true)
        return self.fn(y_true, y_pred, **self._fn_kwargs)

    def get_config(self):
        config = {}
        for k, v in six.iteritems(self._fn_kwargs):
            config[k] = tf.keras.backend.eval(v) if tf_utils.is_tensor_or_variable(v) \
                else v
        base_config = super(LossFunctionWrapper, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def npairs_loss(labels, feature_vectors):
    feature_vectors_normalized = tf.math.l2_normalize(feature_vectors, axis=1)
    logits = tf.divide(
        tf.matmul(
            feature_vectors_normalized, tf.transpose(feature_vectors_normalized)
        ),
        0.5  # temperature
    )
    return tfa.losses.npairs_loss(tf.squeeze(labels), logits)

class NPairsLoss(LossFunctionWrapper):
    def __init__(self, reduction=losses_utils.ReductionV2.AUTO,
                 name='m_pairs_loss'):
        super(NPairsLoss, self).__init__(npairs_loss, name=name,
                                         reduction=reduction)

def build_preprocessor(X: np.ndarray, colnames: List[str]) -> Pipeline:
    X_ = Pipeline(steps=[
        (
            'imputer', SimpleImputer(
                missing_values=np.nan, strategy='constant',
                fill_value=-1.0
            )
        ),
        (
            'scaler',
            MinMaxScaler()
        )
    ]).fit_transform(X)
    X_ = np.rint(X_ * 100000.0).astype(np.int32)
    binary_features = dict()
    categorical_features = dict()
    removed_features = []
    for col_idx in range(X.shape[1]):
        values = set(X_[:, col_idx].tolist())
        print(f'Column {col_idx} "{colnames[col_idx]}" has ' \
              f'{len(values)} unique values.')
        if len(values) > 1:
            if len(values) < 3:
                binary_features[col_idx] = np.min(X[:, col_idx])
            else:
                categorical_features[col_idx] = len(values)
        else:
            removed_features.append(col_idx)
        del values
    del X_
    all_features = set(range(X.shape[1]))
    useful_features = sorted(list(all_features - set(removed_features)))
    if len(useful_features) == 0:
        raise ValueError('Training inputs are bad. All features are removed.')
    print(f'There are {X.shape[1]} features.')
    if len(removed_features) > 0:
        print(f'These features will be removed: ' \
              f'{[colnames[col_idx] for col_idx in removed_features]}.')
    transformers = []
    if (len(categorical_features) > 0) and (len(binary_features) > 0):
        print(f'There are {len(categorical_features)} categorical ' \
              f'features and {len(binary_features)} binary features.')
    elif len(categorical_features) > 0:
        print(f'There are {len(categorical_features)} categorical features.')
    else:
        print(f'There are {len(binary_features)} binary features.')
    for col_idx in categorical_features:
        n_unique_values = categorical_features[col_idx]
        transformers.append(
            (
                colnames[col_idx],
                KBinsDiscretizer(
                    n_bins=min(max(n_unique_values // 3, 3), 256),
                    encode='ordinal',
                    strategy=('quantile' if n_unique_values > 50 else 'kmeans')
                ),
                (col_idx,)
            )
        )
    for col_idx in binary_features:
        transformers.append(
            (
                colnames[col_idx],
                Binarizer(threshold=0.0),
                (col_idx,)
            )
        )
    preprocessor = Pipeline(steps=[
        (
            'imputer', SimpleImputer(
                missing_values=np.nan, strategy='constant',
                fill_value=-1.0
            )
        ),
        (
            'minmax_scaler',
            MinMaxScaler()
        ),
        (
            'composite_transformer', ColumnTransformer(
                transformers=transformers,
                sparse_threshold=0.0,
                n_jobs=1
            )
        ),
        (
            'selector',
            VarianceThreshold()
        ),
        (
            'standard_scaler',
            StandardScaler(with_mean=True, with_std=True)
        ),
        (
            'pca',
            PCA(random_state=42)
        )
    ])
    return preprocessor.fit(X)


def reduce_dimensions_of_data(features: np.ndarray) -> np.ndarray:
    preprocessed_features = Pipeline(
        steps=[
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=features.shape[1] // 3,
                        random_state=42))
        ]
    ).fit_transform(features)
    print('Features are preprocessed.')
    reduced_features = umap.UMAP(
        low_memory=False,
        n_jobs=-1,
        random_state=42,
        verbose=True
    ).fit_transform(preprocessed_features)
    print('Feature space is reduced.')
    del preprocessed_features
    return reduced_features


# %%
def show_temperature(features: np.ndarray, targets: np.ndarray,
                     title: str = '', figure_id: int = 0):
    if features.shape[0] != targets.shape[0]:
        err_msg = f'Features do not correspond to targets! ' \
                  f'{features.shape[0]} != {targets.shape[0]}'
        raise ValueError(err_msg)
    if len(features.shape) != 2:
        err_msg = f'Features are wrong! Expected 2-D array, got ' \
                  f'{len(features.shape)}-D one.'
        raise ValueError(err_msg)
    if features.shape[1] != 2:
        err_msg = f'Features are wrong! Expected number of ' \
                  f'columns is 2, got {features.shape[1]}.'
        raise ValueError(err_msg)
    if len(targets.shape) != 1:
        err_msg = f'Targets are wrong! Expected 1-D array, got ' \
                  f'{len(targets.shape)}-D one.'
        raise ValueError(err_msg)
    sorted_targets = sorted(targets.tolist())
    n_percentile2 = max(int(round(0.01 * len(sorted_targets))), 1)
    min_target = sorted_targets[n_percentile2]
    max_target = sorted_targets[-n_percentile2]
    del sorted_targets
    clipped_targets = np.empty(targets.shape, dtype=np.float64)
    for sample_idx in range(targets.shape[0]):
        if targets[sample_idx] < min_target:
            clipped_targets[sample_idx] = min_target
        elif targets[sample_idx] > max_target:
            clipped_targets[sample_idx] = max_target
        else:
            clipped_targets[sample_idx] = targets[sample_idx]
    temperature_colors = clipped_targets.tolist()
    temperature_norm = Normalize(vmin=np.min(temperature_colors),
                                 vmax=np.max(temperature_colors))
    fig = plt.figure(figure_id, figsize=(11, 11))
    plt.scatter(x=features[:, 0], y=features[:, 1],
                marker='o', cmap=plt.cm.get_cmap("jet"), c=temperature_colors,
                norm=temperature_norm)
    if len(title) > 0:
        plt.title(f'UMAP projections of weather data {title} (temperature)')
    else:
        plt.title(f'UMAP projections of weather data (temperature)')
    plt.colorbar()
    plt.show()


# %%
def filter_dataset(y: np.ndarray) -> List[int]:
    all_values = sorted(y.tolist())
    n = len(all_values)
    if n <= 10000:
        err_msg = f'y is wrong! Expected length of y is greater than 10000, ' \
                  f'but got {n}.'
        raise ValueError(err_msg)
    y001 = all_values[int(round((n - 1) * 0.001))]
    y999 = all_values[int(round((n - 1) * 0.999))]
    del all_values
    filtered_indices = list(filter(
        lambda idx: (y[idx] > y001) and (y[idx] < y999),
        range(n)
    ))
    return filtered_indices


# %%
def build_neural_network(input_size: int, layer_size: int, n_layers: int,
                         dropout_rate: float, scale_coeff: float,
                         nn_name: str) -> tf.keras.Model:
    feature_vector = tf.keras.layers.Input(
        shape=(input_size,), dtype=tf.float32,
        name=f'{nn_name}_feature_vector'
    )
    outputs = []
    hidden_layer = tf.keras.layers.AlphaDropout(
        rate=dropout_rate,
        seed=random.randint(0, 2147483647),
        name=f'{nn_name}_dropout1'
    )(feature_vector)
    for layer_idx in range(1, (2 * n_layers) // 3 + 1):
        try:
            kernel_initializer = tf.keras.initializers.LecunNormal(
                seed=random.randint(0, 2147483647)
            )
        except:
            kernel_initializer = tf.compat.v1.keras.initializers.lecun_normal(
                seed=random.randint(0, 2147483647)
            )
        hidden_layer = tf.keras.layers.Dense(
            units=layer_size,
            activation='selu',
            kernel_initializer=kernel_initializer,
            bias_initializer='zeros',
            name=f'{nn_name}_dense{layer_idx}'
        )(hidden_layer)
        hidden_layer = tf.keras.layers.AlphaDropout(
            rate=dropout_rate,
            seed=random.randint(0, 2147483647),
            name=f'{nn_name}_dropout{layer_idx + 1}'
        )(hidden_layer)
    try:
        kernel_initializer = tf.keras.initializers.LecunNormal(
            seed=random.randint(0, 2147483647)
        )
    except:
        kernel_initializer = tf.compat.v1.keras.initializers.lecun_normal(
            seed=random.randint(0, 2147483647)
        )
    projection_layer = tf.keras.layers.Dense(
        units=50,
        activation=None,
        use_bias=False,
        kernel_initializer=kernel_initializer,
        name=f'{nn_name}_projection'
    )(hidden_layer)
    for layer_idx in range((2 * n_layers) // 3 + 1, n_layers + 1):
        try:
            kernel_initializer = tf.keras.initializers.LecunNormal(
                seed=random.randint(0, 2147483647)
            )
        except:
            kernel_initializer = tf.compat.v1.keras.initializers.lecun_normal(
                seed=random.randint(0, 2147483647)
            )
        hidden_layer = tf.keras.layers.Dense(
            units=layer_size,
            activation='selu',
            kernel_initializer=kernel_initializer,
            bias_initializer='zeros',
            name=f'{nn_name}_dense{layer_idx}'
        )(hidden_layer)
        hidden_layer = tf.keras.layers.AlphaDropout(
            rate=dropout_rate,
            seed=random.randint(0, 2147483647),
            name=f'{nn_name}_dropout{layer_idx + 1}'
        )(hidden_layer)
    try:
        kernel_initializer = tf.keras.initializers.LecunNormal(
            seed=random.randint(0, 2147483647)
        )
    except:
        kernel_initializer = tf.compat.v1.keras.initializers.lecun_normal(
            seed=random.randint(0, 2147483647)
        )
    output_layer = tf.keras.layers.Dense(
        units=2,
        activation=None,
        use_bias=False,
        kernel_initializer=kernel_initializer,
        name=f'{nn_name}_output'
    )(hidden_layer)
    bayesian_layer = tfp.layers.DistributionLambda(
        lambda t: tfp.distributions.Normal(
            loc=t[..., :1],
            scale=1e-6 + tf.math.softplus((1.0 / scale_coeff) * t[..., 1:])
        ),
        name=f'{nn_name}_distribution'
    )(output_layer)
    neural_network = tf.keras.Model(
        inputs=feature_vector,
        outputs=[bayesian_layer, projection_layer],
        name=nn_name
    )
    negloglik = lambda y, rv_y: -rv_y.log_prob(y)
    radam = tfa.optimizers.RectifiedAdam(learning_rate=3e-4)
    ranger = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)
    losses = {
        f'{nn_name}_distribution': negloglik,
        f'{nn_name}_projection': NPairsLoss()
    }
    loss_weights = {
        f'{nn_name}_distribution': 1.0,
        f'{nn_name}_projection': 0.5
    }
    metrics = {
        f'{nn_name}_distribution': [
            tf.keras.metrics.MeanAbsoluteError()
        ]
    }
    neural_network.compile(
        optimizer=ranger,
        loss=losses,
        loss_weights=loss_weights,
        metrics=metrics
    )
    return neural_network


# %%
def show_training_process(history: tf.keras.callbacks.History, metric_name: str,
                          figure_id: int = 1, comment: str = ''):
    val_metric_name = 'val_' + metric_name
    if metric_name not in history.history:
        err_msg = f'The metric "{metric_name}" is not found! Available metrics are: ' \
                  f'{list(history.history.keys())}.'
        raise ValueError(err_msg)
    plt.figure(figure_id, figsize=(5, 5))
    interesting_metric = history.history[metric_name]
    plt.plot(list(range(len(interesting_metric))), interesting_metric,
             label=f'Training {metric_name}')
    if val_metric_name in history.history:
        interesting_val_metric = history.history[val_metric_name]
        assert len(interesting_metric) == len(interesting_val_metric)
        plt.plot(list(range(len(interesting_val_metric))),
                 interesting_val_metric,
                 label=f'Validation {metric_name}')
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    if len(comment) > 0:
        plt.title(f'Training process of {comment}')
    else:
        plt.title('Training process')
    plt.legend(loc='best')
    plt.show()


# %%
def predict_with_single_nn(input_data: np.ndarray, model_for_prediction: tf.keras.Model,
                           batch_size: int, output_scaler: StandardScaler) \
        -> Tuple[np.ndarray, np.ndarray]:
    if len(input_data.shape) != 2:
        err_msg = f'The `input_data` argument is wrong! Expected 2-D array, ' \
                  f'got {len(input_data.shape)}-D one!'
        raise ValueError(err_msg)
    n_batches = int(np.ceil(input_data.shape[0] / float(batch_size)))
    pred_mean = []
    pred_std = []
    for batch_idx in range(n_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(input_data.shape[0], batch_start + batch_size)
        instant_predictions = model_for_prediction(input_data[batch_start:batch_end])[0]
        if not isinstance(instant_predictions, tfp.distributions.Distribution):
            err_msg = f'Minibatch {batch_idx}: predictions are wrong! ' \
                      f'Expected tfp.distributions.Distribution, ' \
                      f'got {type(instant_predictions)}.'
            raise ValueError(err_msg)
        instant_mean = instant_predictions.mean()
        instant_std = instant_predictions.stddev()
        del instant_predictions
        if not isinstance(instant_mean, np.ndarray):
            instant_mean = instant_mean.numpy()
        if not isinstance(instant_std, np.ndarray):
            instant_std = instant_std.numpy()
        instant_mean = instant_mean.astype(np.float64).flatten()
        instant_std = instant_std.astype(np.float64).flatten()
        pred_mean.append(instant_mean)
        pred_std.append(instant_std)
        del instant_mean, instant_std
    pred_mean = np.concatenate(pred_mean)
    pred_std = np.concatenate(pred_std)
    pred_mean = output_scaler.inverse_transform(
        pred_mean.reshape((input_data.shape[0], 1))
    ).flatten()
    pred_std *= output_scaler.scale_[0]
    return pred_mean, pred_std * pred_std


# %%
def evaluate_single_nn(pred_means: np.ndarray, pred_vars: np.ndarray,
                       true_outputs: np.ndarray) -> float:
    if len(pred_means.shape) != 1:
        err_msg = f'The `pred_means` argument is wrong! Expected 1-D array, ' \
                  f'got {len(pred_means.shape)}-D one.'
        raise ValueError(err_msg)
    if len(pred_vars.shape) != 1:
        err_msg = f'The `pred_vars` argument is wrong! Expected 1-D array, ' \
                  f'got {len(pred_vars.shape)}-D one.'
        raise ValueError(err_msg)
    if len(true_outputs.shape) != 1:
        err_msg = f'The `true_outputs` argument is wrong! Expected 1-D array, ' \
                  f'got {len(true_outputs.shape)}-D one.'
        raise ValueError(err_msg)
    n_test_samples = true_outputs.shape[0]
    if n_test_samples < 5:
        raise ValueError(f'Number of test samples = {n_test_samples} is too small!')
    if n_test_samples != pred_means.shape[0]:
        err_msg = f'The `pred_means` does not correspond to the `true_outputs`! ' \
                  f'{pred_means.shape[0]} != {n_test_samples}'
        raise ValueError(err_msg)
    if n_test_samples != pred_vars.shape[0]:
        err_msg = f'The `pred_vars` does not correspond to the `true_outputs`! ' \
                  f'{pred_vars.shape[0]} != {n_test_samples}'
        raise ValueError(err_msg)

    all_preds_ = np.empty((1, n_test_samples, 2), dtype=np.float32)
    all_preds_[0, :, 0] = pred_means
    all_preds_[0, :, 1] = pred_vars
    all_uncertainty_ = ensemble_uncertainties_regression(all_preds_)
    uncertainties = all_uncertainty_['tvar']
    del all_preds_, all_uncertainty_

    errors = (pred_means - true_outputs) ** 2
    rejection_mse_ = calc_uncertainty_regection_curve(errors, uncertainties)
    return np.mean(rejection_mse_)


# %%
def predict_by_ensemble(input_data: np.ndarray,
                        preprocessing: Pipeline,
                        ensemble: List[tf.keras.Model],
                        postprocessing: List[StandardScaler],
                        minibatch: int) -> np.ndarray:
    num_samples = input_data.shape[0]
    ensemble_size = len(postprocessing)
    if ensemble_size != len(ensemble):
        err_msg = f'Ensemble of preprocessors does not correspond to ' \
                  f'ensemble of models! {ensemble_size} != {len(ensemble)}'
        raise ValueError(err_msg)
    predictions_of_ensemble = np.empty((ensemble_size, num_samples, 2),
                                       dtype=np.float64)
    X = preprocessing.transform(input_data).astype(np.float32)
    for model_idx, (cur_model, post_) in enumerate(zip(ensemble, postprocessing)):
        y_mean, y_var = predict_with_single_nn(
            input_data=X,
            model_for_prediction=cur_model,
            output_scaler=post_,
            batch_size=minibatch
        )
        predictions_of_ensemble[model_idx, :, 0] = y_mean
        predictions_of_ensemble[model_idx, :, 1] = y_var
    return predictions_of_ensemble


# %%
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
# %%
data_dir = os.path.join('data', 'yandex-shifts', 'weather')
#print(f'{data_dir} {os.path.isdir(data_dir)}')
# %%
model_dir = os.path.join('models', 'yandex-shifts', 'weather')
#print(f'{model_dir} {os.path.isdir(model_dir)}')
# %%
traindata_name = os.path.join(data_dir, 'train.csv')
#print(f'{traindata_name} {os.path.isfile(traindata_name)}')
# %%
dev_in_name = os.path.join(data_dir, 'dev_in.csv')
#print(f'{dev_in_name} {os.path.isfile(dev_in_name)}')
# %%
dev_out_name = os.path.join(data_dir, 'dev_out.csv')
#print(f'{dev_out_name} {os.path.isfile(dev_out_name)}')
# %%
eval_name = os.path.join(data_dir, 'eval_in.csv')
#print(f'{eval_name} {os.path.isfile(eval_name)}')
# %%
df_train = pd.read_csv(traindata_name)
#print(f'Row number is {df_train.shape[0]}.')
#print(f'Column number is {df_train.shape[1]}.')
# %%
#df_train.head()
# %%
X_train = df_train.drop(['fact_temperature', 'climate'], axis=1).to_numpy().astype(np.float64)
y_train = df_train['fact_temperature'].to_numpy().astype(np.float64)


# %%
print(f'X_train: dtype = {X_train.dtype}, shape = {X_train.shape}')
print(f'y_train: dtype = {y_train.dtype}, shape = {y_train.shape}')
# %%
common_preprocessor = build_preprocessor(X_train, df_train.drop(['fact_temperature', 'climate'], axis=1).columns)
# %%
print(common_preprocessor)



# %%
X_train = common_preprocessor.transform(X_train)
num_features = X_train.shape[1]
# %%
correct_values = all(np.isfinite(X_train).ravel())
if not correct_values:
    raise ValueError('Some values of input values are not correct (NaNs or infinite)!')
# %%
print(f'Maximal value of input matrix is {np.max(X_train)}.')
print(f'Minimal value of input matrix is {np.min(X_train)}.')
# %%
all_indices = list(range(X_train.shape[0]))
random.shuffle(all_indices)
X_train = X_train[all_indices]
y_train = y_train[all_indices]
del all_indices
gc.collect()
# %%
indices_for_projections = random.sample(
    population=list(range(X_train.shape[0])),
    k=100000
)
X_train_prj = reduce_dimensions_of_data(X_train[indices_for_projections])
# %%
X_train
# %%
show_temperature(X_train_prj, y_train[indices_for_projections],
                 figure_id=0)
# %%
gc.collect()
# %%
all_temperatures = sorted(y_train.tolist())
min_temperature = all_temperatures[0]
max_temperature = all_temperatures[-1]
n_samples_in_trainset = len(all_temperatures)
temperature_001 = all_temperatures[int(round(0.001 * n_samples_in_trainset))]
temperature_999 = all_temperatures[int(round(0.999 * n_samples_in_trainset))]
print(f'Minimal temperature is {min_temperature}.')
print(f'Maximal temperature is {max_temperature}.')
print(f'0.1% of temperature is {temperature_001}.')
print(f'99.9% of temperature is {temperature_999}.')
max_temperature = int(np.ceil(temperature_999))
min_temperature = int(np.floor(temperature_001))
n_classes = max_temperature - min_temperature + 1
dict_of_classes = dict()
for class_idx in range(n_classes):
    dict_of_classes[min_temperature + class_idx] = class_idx
print(f'Number of temperature classes is {n_classes}.')
print('They are:')
for temperature_val in dict_of_classes:
    class_idx = dict_of_classes[temperature_val]
    print('  Class {0:>2}: temperature = {1:4.1f}'.format(class_idx, temperature_val))
y_train_class = np.empty(y_train.shape, dtype=np.int32)
for sample_idx in range(y_train.shape[0]):
    temperature_val = int(round(y_train[sample_idx]))
    if temperature_val in dict_of_classes:
        class_idx = dict_of_classes[temperature_val]
    else:
        if temperature_val < min_temperature:
            class_idx = 0
        else:
            class_idx = n_classes - 1
    y_train_class[sample_idx] = class_idx
# %%
print(f'X_train: dtype = {X_train.dtype}, shape = {X_train.shape}')
print(f'y_train: dtype = {y_train.dtype}, shape = {y_train.shape}')
# %%
skf = StratifiedKFold(n_splits=20, shuffle=True, random_state=42)
splitter = skf.split(X_train, y_train_class)
splits = [(train_index, test_index) for train_index, test_index in splitter]
del splitter, skf
# %%
postprocessing_scalers = []
deep_ensemble = []
BATCH_SIZE = 4096
MAX_EPOCHS = 1
PATIENCE = 15
new_figure_id = 5
BEST_LAYER_SIZE = 512
best_hyperparams = (18, 3e-4)
# %%
gc.collect()
tf.keras.backend.clear_session()
# %%
for train_index, test_index in splits:
    model_name = f'weather_snn_{len(deep_ensemble) + 1}'
    serialization_name = os.path.join(model_dir, model_name + '.h5')
    regression_output_name = f'{model_name}_distribution'
    projection_output_name = f'{model_name}_projection'
    printable_name = f'Self-Normalizing Network ' \
                     f'{len(deep_ensemble) + 1} for weather prediction'
    print('========================================')
    print(' ' + printable_name)
    print('========================================')
    print('')
    new_postprocessing_scaler = StandardScaler().fit(
        y_train[train_index].reshape((len(train_index), 1))
    )
    X_train_ = X_train[train_index].astype(np.float32)
    y_train_ = new_postprocessing_scaler.transform(
        y_train[train_index].reshape((len(train_index), 1))
    ).flatten().astype(np.float32)
    y_train_class_ = y_train_class[train_index]
    X_val_ = X_train[test_index].astype(np.float32)
    y_val_ = y_train[test_index]
    y_val_class_ = y_train_class[test_index]
    y_val_scaled_ = new_postprocessing_scaler.transform(
        y_val_.reshape((len(test_index), 1))
    ).flatten().astype(np.float32)
    steps_per_epoch = X_train_.shape[0] // BATCH_SIZE
    postprocessing_scalers.append(new_postprocessing_scaler)
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (
            X_train_,
            (
                y_train_,
                y_train_class_
            )
        )
    ).repeat().shuffle(1000000).batch(BATCH_SIZE)
    val_dataset = tf.data.Dataset.from_tensor_slices(
        (
            X_val_,
            (
                y_val_scaled_,
                y_val_class_
            )
        )
    ).batch(BATCH_SIZE)
    del X_train_, y_train_, y_val_scaled_, y_train_class_, y_val_class_
    new_model = build_neural_network(
        input_size=num_features,
        layer_size=BEST_LAYER_SIZE,
        n_layers=best_hyperparams[0],
        dropout_rate=best_hyperparams[1],
        scale_coeff=new_postprocessing_scaler.scale_[0],
        nn_name=model_name
    )
    new_model.summary()
    print('')
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor=f'val_{regression_output_name}_mean_absolute_error',
            restore_best_weights=True,
            patience=PATIENCE, verbose=True
        )
    ]
    model_history = new_model.fit(
        train_dataset, epochs=MAX_EPOCHS, steps_per_epoch=steps_per_epoch,
        callbacks=callbacks, validation_data=val_dataset, verbose=1
    )
    new_model.save_weights(serialization_name, overwrite=True, save_format='h5')
    del train_dataset, val_dataset
    print('')
    show_training_process(history=model_history, metric_name='loss',
                          figure_id=new_figure_id, comment=printable_name)
    new_figure_id += 1
    show_training_process(history=model_history,
                          metric_name=f'{regression_output_name}_loss',
                          figure_id=new_figure_id, comment=printable_name)
    new_figure_id += 1
    show_training_process(history=model_history,
                          metric_name=f'{regression_output_name}_mean_absolute_error',
                          figure_id=new_figure_id, comment=printable_name)
    new_figure_id += 1
    del model_history, callbacks
    print('')
    deep_ensemble.append(new_model)
    instant_res = predict_with_single_nn(X_val_, new_model, BATCH_SIZE,
                                         output_scaler=new_postprocessing_scaler)
    y_pred_mean = instant_res[0]
    y_pred_var = instant_res[1]
    rauc_mse_score = evaluate_single_nn(y_pred_mean, y_pred_var, y_val_)
    print('Test quality:')
    print(f'  mean absolute error   = {mean_absolute_error(y_val_, y_pred_mean)}')
    print(f'  mean squared error    = {mean_squared_error(y_val_, y_pred_mean)}')
    print(f'  median absolute error = {median_absolute_error(y_val_, y_pred_mean)}')
    print(f'  r2 score              = {r2_score(y_val_, y_pred_mean)}')
    print(f'  R-AUC MSE             = {rauc_mse_score}')
    del X_val_, y_val_
    del new_model
# %%
config_name = os.path.join(model_dir, 'weather_snn_config.json')
best_hyperparams_for_saving = {
    'input_size': num_features,
    'ensemble_size': len(splits),
    'layer_size': BEST_LAYER_SIZE,
    'n_layers': best_hyperparams[0],
    'alpha_dropout_rate': best_hyperparams[1]
}
with codecs.open(config_name, mode='w', encoding='utf-8') as fp:
    json.dump(best_hyperparams_for_saving, fp, ensure_ascii=False, indent=4)
# %%
preprocessing_name = os.path.join(model_dir, 'preprocessing_pipeline.pkl')
with open(preprocessing_name, 'wb') as fp:
    pickle.dump(common_preprocessor, fp, protocol=pickle.HIGHEST_PROTOCOL)
# %%
postprocessing_name = os.path.join(model_dir, 'postprocessing_scalers.pkl')
with open(postprocessing_name, 'wb') as fp:
    pickle.dump(postprocessing_scalers, fp, protocol=pickle.HIGHEST_PROTOCOL)
# %%
df_in = pd.read_csv(dev_in_name)
print(f'Row number is {df_in.shape[0]}.')
print(f'Column number is {df_in.shape[1]}.')
# %%
df_in.head()
# %%
df_out = pd.read_csv(dev_out_name)
print(f'Row number is {df_out.shape[0]}.')
print(f'Column number is {df_out.shape[1]}.')
# %%
df_out.head()
# %%
inputs = np.vstack([
    df_in.iloc[:, 6:].to_numpy().astype(np.float64),
    df_out.iloc[:, 6:].to_numpy().astype(np.float64)
])
targets = np.concatenate([
    df_in['fact_temperature'].to_numpy().astype(np.float64),
    df_out['fact_temperature'].to_numpy().astype(np.float64)
])
# %%
all_preds = predict_by_ensemble(
    input_data=inputs,
    preprocessing=common_preprocessor,
    postprocessing=postprocessing_scalers,
    ensemble=deep_ensemble,
    minibatch=BATCH_SIZE
)
# %%
all_uncertainty = ensemble_uncertainties_regression(all_preds)
uncertainties = all_uncertainty['tvar']
# %%
all_preds_mean = all_preds[:, :, 0]
avg_preds = np.squeeze(np.mean(all_preds_mean, axis=0))
errors = (avg_preds - targets) ** 2
# %%
rejection_mse = calc_uncertainty_regection_curve(errors, uncertainties)
retention_mse = rejection_mse[::-1]
retention_fractions = np.linspace(0, 1, len(retention_mse))
# %%
print(f'R-AUC MSE = {np.mean(retention_mse)}.')
# %%
plt.figure(new_figure_id, figsize=(7, 7))
new_figure_id += 1
plt.plot(retention_fractions, retention_mse)
plt.ylabel('MSE')
plt.xlabel('Retention Fraction')
plt.show()
plt.clf()
# %%
thresh = 1.0
f_auc, f95, retention_f1 = f_beta_metrics(errors, uncertainties, thresh, beta=1.0)
print(f'F1 score at 95% retention: {f95}')
retention_fractions = np.linspace(0, 1, len(retention_f1))
# %%
plt.figure(new_figure_id, figsize=(7, 7))
new_figure_id += 1
plt.plot(retention_fractions, retention_f1)
plt.ylabel('F1')
plt.xlabel('Retention Fraction')
plt.show()
plt.clf()
# %%
ids = np.arange(1, inputs.shape[0] + 1)
preds = np.mean(all_preds[:, :, 0], axis=0)
df_submission = pd.DataFrame(data={
    'ID': ids,
    'PRED': preds,
    'UNCERTAINTY': uncertainties
})
# %%
df_submission.head()
# %%
out_file = os.path.join(model_dir, 'df_submission_dev.csv')
df_submission.to_csv(out_file, index=False)
# %%
df_eval = pd.read_csv(eval_name)
df_eval.head()
# %%
eval_inputs = df_eval.to_numpy().astype(np.float64)
# %%
all_preds = predict_by_ensemble(
    input_data=eval_inputs,
    preprocessing=common_preprocessor,
    postprocessing=postprocessing_scalers,
    ensemble=deep_ensemble,
    minibatch=BATCH_SIZE
)
# %%
all_uncertainty = ensemble_uncertainties_regression(all_preds)
uncertainties = all_uncertainty['tvar']
# %%
ids = np.arange(1, len(df_eval) + 1)
preds = np.mean(all_preds[:, :, 0], axis=0)
df_submission = pd.DataFrame(data={
    'ID': ids,
    'PRED': preds,
    'UNCERTAINTY': uncertainties
})
df_submission.head()
# %%
out_file = os.path.join(model_dir, 'df_submission.csv')
df_submission.to_csv(out_file, index=False)