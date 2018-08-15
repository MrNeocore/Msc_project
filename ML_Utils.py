import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from datetime import datetime
import warnings
from tqdm import tqdm  
import time

import keras
from keras.models import Sequential
import keras.layers
import keras.models
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback, ModelCheckpoint
from keras import backend as K
from keras.models import load_model

from Utils import *
import subprocess

# TODO : Decapitalize file name

##############################################################################
##############################################################################

#=================
#### TRAINING ####
#================
def train_model(X_train, y_train, arch, optimizer={'optimizer_name':'Adam', 'lr':0.0025}, epochs=25, lossplot=True, impplot=True, tensorboard=False, 
        backend='tensorflow', use_gpu=True, validation_data=None, early_stopping=False, reduce_lr_plateau=False, verbose=1, batch_size=256):
    """ Trains a model using the given parameters """
    
    # NOTE : Only works, supposedly, if using a Python virtual environment
    #os.environ['KERAS_BACKEND'] = backend
    #reload(keras)

    if type(arch) in [keras.models.Sequential, keras.models.Model]: 
        model = arch
    else:
        raise Exception("Invalid model - Keras models only.")

    optimizer_name = optimizer['optimizer_name']
    lr = optimizer['lr']

    SGD_optimizer_settings = {k:v for k,v in optimizer.items() if k not in ['optimizer_name', 'lr']} if optimizer['optimizer_name'] == 'SGD' else {}

    optimizer = get_optimizer(optimizer_name)(lr=lr, **SGD_optimizer_settings)
    model.compile(optimizer, loss='mean_squared_error')

    #### LR Identification 
    #### Could be automatised, but not really worth it
    #lr_finder = LRFinder(model)
    #lr_finder.find(X_train, y_train, start_lr=0.0001, end_lr=1.0, batch_size=512, epochs=50)
    #lr_finder.plot_loss(n_skip_beginning=1, n_skip_end=1)
    #lr_finder.plot_loss_change(sma=3, n_skip_beginning=10, n_skip_end=5, y_lim=(-0.01, 0.01))

    model_fname = None

    time_epochs_cb = TimeEpochsCallback()

    cb = []
    if lossplot: 
        from livelossplot import PlotLossesKeras
        cb.append(PlotLossesKeras())
    if tensorboard:
        tbCallback = keras.callbacks.TensorBoard(log_dir='./logs/{0}'.format(datetime.now().isoformat()), histogram_freq=0, write_graph=True, write_images=True)
        tbCallback.set_model(model)
        cb.append(tbCallback)

        if backend != "tensorflow":
            warnings.warn("Not using tensorflow backend -> Limited tensorboard output")
    
    if validation_data != None:
        # NOTE : The return model, trained with early stopping isn't the best, but the one "patience" steps after the best !
        # Workaround : Save best model then reload after training
        if early_stopping:
            cb.append(EarlyStopping(patience=25, verbose=0))
            model_fname = os.path.join("model_checkpoints", "best_model_{0}.h5".format(int(time.time())%12345))  # Shorten name with modulo
            cb.append(ModelCheckpoint(model_fname, verbose=0, monitor='val_loss', save_best_only=True, mode='auto'))

        if reduce_lr_plateau:
            cb.append(ReduceLROnPlateau(patience=12, factor=0.2, verbose=verbose, min_lr=lr/10))

    if not use_gpu:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        if "CUDA_DEVICE_ORDER" in os.environ and "CUDA_VISIBLE_DEVICES" in os.environ:
            del os.environ["CUDA_DEVICE_ORDER"]
            del os.environ["CUDA_VISIBLE_DEVICES"]

        gpu_usage_cb = GpuUsageCallback()
        cb.append(gpu_usage_cb)

    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=verbose, callbacks=cb+[time_epochs_cb], validation_data=validation_data)

    # Early stopping with backup was used
    if model_fname:
        model = load_model(filepath=model_fname)
        print("Reloading best model")

    if use_gpu:
        gpu_usage = gpu_usage_cb.get_mean_gpu_usage()
    else:
        gpu_usage = -1

    return model, np.array([time_epochs_cb.training_time, time_epochs_cb.epoch_count, gpu_usage])


#====================
## KERAS CALLBACKS ##
#====================

class TimeEpochsCallback(Callback):
    """ Records the total training time 
        Todo:
            * Should use on_train_end callback instead of on_epoch_end
    """
    def __init__(self):
        self.epoch_count = 0
        self.first_epoch = True

    def on_epoch_begin(self, epoch, logs={}):
        if self.first_epoch:
            self.start = time.time()
            self.first_epoch = False

    def on_epoch_end(self, epoch, logs={}):
        self.training_time = time.time()-self.start
        self.epoch_count += 1


class GpuUsageCallback(Callback):
    """ Periodically records the GPU usage %. Only support Nvidia GPUs """
    def __init__(self):
        self.gpu_usage = []
        self.epoch_count = 0

    # on_batch_begin may be more accurate but slows training down more
    def on_epoch_begin(self, batch, logs):
        self.epoch_count +=  1
        # Do not run everytime as the function call is blocking therefore slowing down training
        if self.epoch_count == 5:
            self.epoch_count = 0
            self.gpu_usage.append(int(subprocess.Popen(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv"], stdout=subprocess.PIPE).stdout.read().decode("ascii").split("\n")[1][:-2]))

    def get_mean_gpu_usage(self):
        if not len(self.gpu_usage):
            return np.array([0])
        else:
            return np.mean(np.array(self.gpu_usage)) 


#===================
#### PREDICTION ####
#===================

def predict_evaluate(model, sc_y, X_test, y_test, load_difference_start=None):
    """ Standard predition method used to get model's prediction on a given test set 
        Args :
            sc_y : Normalizer object used to denormalize data
            load_difference_start : Initial time series value - used to reconstruct differenced time series.
    """
    start_time = time.time()
    y_pred = model.predict(X_test)
    inference_time = time.time() - start_time

    y_pred = sc_y.inverse_transform(y_pred.reshape(-1,1))
    index = y_test.index
    results = get_results(y_pred, y_test, index, load_difference_start=load_difference_start)
    
    return results, inference_time

def recurrent_predict_evaluate(model, sc_y, X_test_arr, y_test_arr, load_difference_start=None):
    """ Top level recurrent prediction method - get results for a model evaluated in a recurrent manner (see reccurent_prediction) """
    tmp_results = []

    start_time = time.time()

    for n,df in tqdm(enumerate(X_test_arr), total=len(X_test_arr)):
        y_test = y_test_arr[n]
        tmp = recurrent_prediction(model, df)
        y_pred = sc_y.inverse_transform(np.array(tmp).reshape(-1,1)) 
        tmp_results.append(np.hstack([y_pred,y_test]))
    
    inference_time = time.time() - start_time

    results = np.vstack(tmp_results)
    results = get_results(results[:,0].reshape(-1,1), results[:,1].reshape(-1,1), pd.concat(y_test_arr).index)

    return results, inference_time


def recurrent_prediction(model, X_test):
    """ Recurrent prediction method - identical to the predict_evalute but allows the test_set to contain forecast data """ 
    prev_cols = [x for x in X_test.columns if 'Previous' in x]  # Previous_1, Previous_10..
    prev_steps = [int(x.split('_')[1]) for x in prev_cols]      # 1, 10
    prev_idx = X_test.columns.isin(prev_cols).nonzero()[0]      # idx(Previous_1), idx(Previous_2)
    max_history = max(prev_steps)
    preds = np.zeros(max_history)

    final_preds = []
    X_test = X_test.values

    for n in range(len(X_test)):
        pred = model.predict(np.array([X_test[n],]))[0] 
        
        # Output predictions are in 2D array for Keras models and 1D arrays for sklearn models (at least SVR)
        if type(pred) not in [np.float32, np.float64]:  
            pred = pred[0]

        preds = np.append(pred, preds)[:max_history]
        final_preds.append(pred)

        if len(prev_cols) == 0:
            continue

        if n+1 != len(X_test):
            X_test[n+1] = _build_next_test_sample(X_test[n+1], preds, prev_idx)
        
    return final_preds


def _build_next_test_sample(X_test_sample, preds, prev_idx):
    """ Constructs a sample for the recurrent_prediction mechanics by merging forecast data and already present data """   
    existing = X_test_sample[prev_idx]
    new_tmp = preds[prev_idx]
    new = np.where(existing == 0, new_tmp, existing)
    X_test_sample[prev_idx] = new

    return X_test_sample


##############################################################################
##############################################################################


####################
#### EVALUATION ####
####################

def get_results(y_pred, y_true, index, load_difference_start=None):
    """ Evaluates performance using given 'prediction' and 'true' test set 
        Used to get sample level forecasting error -> Mainly used for plotting
    """
    if load_difference_start is not None:
         y_true = np.r_[load_difference_start, y_true[1:].ravel()].cumsum().reshape(-1,1)
         y_pred = np.r_[load_difference_start, y_pred[1:].ravel()].cumsum().reshape(-1,1)

    abs_err = np.abs(y_pred - y_true)
    abs_perc_err = abs_err / y_true * 100

    results = np.hstack([y_pred, y_true, abs_err, abs_perc_err])
    results = pd.DataFrame(index=index, data=results, columns=['Prediction','True','abs_err','Error (%)'])

    return results

def get_measures(results, period, metric):
    """ Evaluates results build by the 'get_results' methods -> Get MAPE / RMSE, optionnaly using a specific grouping (e.g. monthly) """
    if isinstance(results, pd.DataFrame) and hasattr(results, 'index'):
        metrics = {'MAPE': mean_absolute_percentage_error,
                   'RMSE': root_mean_squared_error,
                   'MAE' : mean_absolute_error}

        metric = metrics.get(metric)

        if period == 'global':
            return pd.Series(metric(results['Prediction'], results['True'])) # Series for type consistency.
        else:
            groupby = results.groupby(get_groupby_time(results, period)[0])
            return groupby.apply(lambda x: metric(x['Prediction'], x['True']))



#===========================
#### EVALUATION METRICS #### 
#===========================

def mean_absolute_percentage_error(y_pred, y_test):
    """ MAPE metric """
    return np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
def root_mean_squared_error(y_pred, y_test):
    """ RMSE metric """
    from math import sqrt
    return sqrt(mean_squared_error(y_test, y_pred))

def mean_absolute_error(y_pred, y_test):
    """ MAE metric """
    return np.mean(np.abs(y_test - y_pred))


#======================
#### DATASET UTILS ####
#======================

def get_test_sets(X_test, y_test, split_on="day"):
        """ Returns a test set splits at a given interval (forecast horizon) and removs "true" data where it's not supposed to be known on true testing data. E.g. H-4 with day ahead forecasting 
            Used when doing recurrent_prediction"
        """
        data = X_test.copy()
        index = data.index.name
        data.reset_index(inplace=True)

        per = {'week': '7 days',
               'day' : '1 day',
               'hour': '1 hour'}

        len_elem = data.loc[data['Period'] == data.iloc[0]['Period'] + pd.Timedelta(per.get(split_on))].index[0]

        prev_cols = [x for x in data.columns if 'Previous' in x]

        for x in prev_cols:
            p = int(x.split("_")[1])
            locations = np.array([range(p+x*len_elem, len_elem * (x+1)) for x in range(len(data)//len_elem+1)]).ravel()
            data.loc[data.index.isin(locations), x] = 0

        data.set_index(index, inplace=True)
    
        return split_dataset_by_date(data, split_on=split_on), split_dataset_by_date(y_test, split_on=split_on)



def split_dataset_by_date(data, split_on="day"):
        """ Splits a dataset at an interval ('hour', 'day', 'month') """
        grp = {'hour' : [data.index.year, data.index.month, data.index.day,data.index.hour],
               'day'  : [data.index.year, data.index.month, data.index.day],
               'week' : [data.index.year, data.index.week]}

        # Remove incomplete blocks (e.g. days). The first block can be incomplete due to some records being removed because they contained nan (due to historical shifting)
        # Note  : Used not to be a problem because year filtering was done after the historical shifting, meaning that blocks would remain complete 
        #         even though some data is actually from the previous (non-requested) year.
        # Note2 : This operation is basically like removing the first data block (i.e. day, month etc) from the first split of the training data if historical data shifting ("Previous_X") within the same block exists (e.g. "Previous_1" with day blocks) 
        #         For more robustness a more general solution has been implemented here even if marginally slower and more complex looking.
        records_per = {}
        records_per['hour'] = len(data.loc[(data.index >= data.index[0]) & (data.index < data.index[0] + pd.Timedelta('1h'))])
        records_per['day'] = records_per['hour'] * 24
        records_per['week'] = records_per['day'] * 7

        block_len = records_per[split_on]

        return [group for _, group in data.groupby(grp.get(split_on)) if len(group) == block_len]


#=======================
#### MODEL BUILDERS ####
#=======================

def build_model(cell_type, **kwargs):
    if cell_type in ['LSTM', 'GRU', 'LSTM_NO_CUDA', 'GRU_NO_CUDA']:
        return build_RNN_model(cell_type, **kwargs)
    elif cell_type == 'Dense':
        return build_Dense_model(**kwargs)
    elif cell_type == 'CNN':
        return build_CNN_model(**kwargs)


def build_Dense_model(layers_config, input_shape, activation='relu'):
    model = Sequential()

    layers_config = [int(float((l[:-1])) * input_shape[2]) if l.endswith('X') else int(l) for l in layers_config]

    model.add(keras.layers.Dense(layers_config[0], activation=activation, input_shape=(input_shape[1],)))

    for l in layers_config[1:]:
        model.add(keras.layers.Dense(l, activation=activation))

    model.add(keras.layers.Dense(1, activation='linear'))

    return model


def build_RNN_model(cell_type, layers_config, input_shape):
    cells_types = {'LSTM':keras.layers.CuDNNLSTM, 'GRU':keras.layers.CuDNNGRU, 'LSTM_NO_CUDA':keras.layers.LSTM, 'GRU_NO_CUDA':keras.layers.GRU}
    cell = cells_types[cell_type]
    layers_config = [int(float((l[:-1])) * input_shape[2]) if l.endswith('X') else int(l) for l in layers_config]

    model = Sequential()

    model.add(cell(layers_config[0], input_shape=input_shape[1:3], return_sequences=True))

    for l in layers_config[1:]:
        model.add(cell(l, return_sequences=True))

    model.add(keras.layers.Dense(1, activation='linear'))

    return model


def build_CNN_model(nb_filters, kernel_size, input_shape, activation='relu', last_layer_type='Dense'):
    model = Sequential()

    model.add(keras.layers.Conv1D(filters=nb_filters[0], kernel_size=kernel_size[0], activation=activation, padding='same', input_shape=input_shape[1:3]))

    for nf, fl in zip(nb_filters[1:], kernel_size[1:]):
        model.add(keras.layers.Conv1D(filters=nf, kernel_size=fl, padding='same', activation=activation))

    if last_layer_type == 'Dense':
        model.add(keras.layers.Dense(1, activation='linear'))
    elif last_layer_type == 'Conv1D':
        model.add(keras.layers.Conv1D(filters=1, kernel_size=kernel_size[-1], padding='same', activation='linear'))

    return model


#=======================
####  MODEL  UTILS  ####
#=======================
# NOTE : Adapted from "ZFTurbo" at "https://stackoverflow.com/questions/43137288/how-to-determine-needed-memory-of-keras-model/45242364"
def model_memory_usage(model, batch_size):
    # Sum of product of layer dimensions
    shapes_mem_count = sum([np.product([x for x in l.output_shape if x]) for l in model.layers])
    trainable_count = model.count_params()
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    total_memory = 4.0 * batch_size * (shapes_mem_count + trainable_count + non_trainable_count)
    mbytes = np.round(total_memory / (1024.0 ** 2), 3)

    return mbytes