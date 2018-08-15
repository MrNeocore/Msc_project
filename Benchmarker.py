""" Benchmarking method file, methods should be rather straight forward. They only are string / result handling to produce charts and test the effect of a given variable """  

import ML_Utils as mlu
import numpy as np
import Load_forecasting as lf
import pandas as pd 
from tqdm import tqdm
#from pandas.plotting import table
from numbers import Number
from itertools import groupby
import json
from IPython.core import display as ICD
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import os
import imgkit


def get_train_shape(dataset, RNN):
    df = lf.Load_Forecaster()
    df.load_data(dataset)
    return df.get_train_data_shape(RNN)

#########################
### GENERIC BENCHMARK ###
#########################

#def tqdm(x, total=None):
#    return x
benchmarking_max_epochs = 200

def _benchmark_base(benchmark_dataset, flag, verbose, early_override):
    df = lf.Load_Forecaster()
    df.load_data(benchmark_dataset)
    df._override_training_settings(epochs=benchmarking_max_epochs, lossplot=False)
    df.db["flag"] = flag + '_benchmark'
    df.verbose = verbose

    if type(early_override) == dict and len(early_override):
        override_method, var_name, var_value = early_override.values()
        getattr(df, override_method)(**{var_name:var_value})

    return df

def _benchmark_loop(df, run_count, model):
    for n in tqdm(range(run_count)):
        df.train_model(model)
        df.predict_load(graph=False, store=True)


# Complex looking but actually simple function returning a list to be looped over for the benchmark
# Complexity comes from checking the input
def _get_iterable(var_range, var_list):
    # Check if continuous -> list of 3 numbers range of number [start, end, step]
    if var_range is not None and len(var_range) == 3 and all([isinstance(x, Number) for x in var_range]):
        start, stop, steps = var_range
        print("Continuous variable benchmark - from {0} to {1} with {2} steps".format(start, stop, steps))
        dtype = type(var_range[0])
        iterable = np.linspace(start, stop, steps, dtype=dtype).tolist()

    # Check if categorical -> homogeneous list
    elif var_list is not None and len(list(groupby(var_list, type))) == 1:
        print("Categorical variable benchmark - {0} variables.".format(len(var_list)))
        iterable = var_list
    else:
        print("Couldn't understand benchmark settings !")
        return []

    return iterable


def benchmark_variable(benchmark_dataset, var_name, override_method, var_range=None, var_list=None, run_count=3, verbose=0, early_override={}, decompose=True, model='LSTM'):
    df = _benchmark_base(benchmark_dataset, var_name, verbose, early_override)

    # Get iterable (continuous /categorical)
    iterable = _get_iterable(var_range, var_list)

    for var in tqdm(iterable):
        if decompose: 
            var = {var_name:var}

        getattr(df, override_method)(**var)
        _benchmark_loop(df, run_count, model)


def plot_benchmark(var_name, mode='barplot', rot=0, title=None, xlabel=None, split_labels_line=True, secondary_y='training_time', secondary_y_label='Seconds', return_data=False, merge_models=True):
    df = lf.Load_Forecaster()
    res = df.load_results(filter_flag=var_name + '_benchmark')

    cols = [var_name, 'testing_MAPE', 'training_MAPE']
    if secondary_y:
        cols.append(secondary_y)

    ret = []

    if merge_models:
        res['model_type'] = res['model_type'].apply(lambda x: x if not x.startswith('CuDNN') else x[5:])

    for loc, d in res.groupby('location'):
        for model, data in d.groupby('model_type'):
            if mode not in ['boxplot', 'detailed_table']:
                data_mean = data[cols].groupby(var_name).mean()

                if mode == 'lineplot':
                    data_mean = data_mean.sort_values(var_name)
                elif mode == 'barplot':
                    data_mean = data_mean.sort_values('testing_MAPE')
                elif mode == 'table':
                    if not return_data:
                        print("Results for location {0} and model type {1}".format(loc, model))
                        ICD.display(data_mean.sort_values('testing_MAPE'))
                        print("------------------------")
                    else:
                        ret.append(((loc, model), data_mean.sort_values('testing_MAPE')))

                if mode in ['lineplot', 'barplot']:
                    ax = data_mean.plot(secondary_y=secondary_y, kind=mode[:-4], fontsize=11, rot=rot)
                    _set_labels(ax, loc, var_name, title, xlabel, rot, split_labels_line, secondary_y, secondary_y_label)

            elif mode == 'boxplot':
                ax = data.boxplot(column='testing_MAPE', by=var_name, rot=rot)
                _set_labels(ax, loc, var_name, title, xlabel, rot, split_labels_line, False)

            elif mode == 'detailed_table':
                if not return_data:
                    print("Results for location {0} and model type {1}".format(loc, model))
                    ICD.display(data[cols])
                    print("------------------------")
                else:
                    ret.append(((loc, model), data[cols].sort_values('testing_MAPE')))

    if return_data:
        return ret

def _set_labels(ax, loc, var_name, title, xlabel, rot, split_labels_line, secondary_y, secondary_y_label=None, font_size=13):
    import matplotlib.pyplot as plt

    plt.suptitle("")

    if title is None:
        ax.set_title("Results for location {0}".format(loc), fontsize=font_size)
    else:
        ax.set_title(title, fontsize=font_size)

    if xlabel is None:
        ax.set_xlabel(str.capitalize(var_name.replace("_", " ")), fontsize=font_size)
    else:
        ax.set_xlabel(xlabel)

    ax.set_ylabel('MAPE', fontsize=font_size)
   
    if secondary_y:
        ax.right_ax.set_ylabel(secondary_y_label, fontsize=font_size)
    
    if split_labels_line:
        _split_labels_line(ax)
        _align_rotated_labels(ax, rot)


def _split_labels_line(ax):
    labels = ax.get_xticklabels()
    fallback = False
                
    # Only apply when xlabels are categorical 
    if labels[0].get_text():
        for n, label in enumerate(labels):
            # Try splitting on dict element instead of all commas (e.g. load propagation plot)
            try:
                label_dict = json.loads(labels[n].get_text().replace("'",'"'))
                labels[n] = "{"+ ',\n'.join(list([str(k)+":"+str(v) for k,v in label_dict.items()])) + "}"
            except: # Not a dictionnary
                break
                fallback = True
        
        # Fallback to simple spliting (on all commas)
        if fallback:
            print('fallback')
            labels = [',\n'.join(x.get_text().split(',')) for x in labels]

        ax.set_xticklabels(labels)

def _align_rotated_labels(ax, rot):
    labels = ax.get_xticklabels()
    if rot not in [0,90]:
        if rot > 0:
            align = 'right'
        else:
            align = 'left'
        ax.set_xticklabels(labels, ha=align)        


###########################
### SPECIFIC BENCHMARKS ###
###########################

def benchmark_RNN_structure(benchmark_data_folder, layer_range, neuron_range, neuron_steps, cell_types, run_count=3, verbose=0):
    df = lf.Load_Forecaster()
    df.load_data(benchmark_data_folder)
    df._override_training_settings(epochs=benchmarking_max_epochs, lossplot=False)
    df.verbose = verbose
    df.db["flag"] = "RNN_structure_benchmark"

    input_shape = df.get_train_data_shape(RNN=True)

    models = []
    layer_start, layer_end = [int(x) for x in layer_range.split('-')]
    layer_range = range(layer_start, layer_end+1)

    neuron_start, neuron_end = [int(float(x[:-1]) * input_shape[2]) if x.endswith('X') else int(x) for x in neuron_range.split('-')]
    neuron_range = [int(x) for x in np.linspace(neuron_start, neuron_end, neuron_steps)]

    for cell_type in cell_types:
        for layer_count in layer_range:
            for neuron_count in neuron_range:
                conf = [str(neuron_count)] * layer_count
                models.append(mlu.build_RNN_model(cell_type, conf, input_shape=input_shape))

    for model in tqdm(models):
        for n in tqdm(range(run_count)):
            df.train_model(model=model, RNN=True)
            df.predict_load(graph=False, store=True)


def benchmark_CNN_structure(benchmark_data_folder, layer_range, filters_range, filters_steps, last_layer_type, kernel_size, run_count=3, verbose=0):
    df = lf.Load_Forecaster()
    df.load_data(benchmark_data_folder)
    df._override_training_settings(epochs=benchmarking_max_epochs, lossplot=False)
    df.verbose = verbose
    df.db["flag"] = "CNN_structure_benchmark"

    input_shape = df.get_train_data_shape(RNN=True)

    models = []
    layer_start, layer_end = [int(x) for x in layer_range.split('-')]
    layer_range = range(layer_start, layer_end+1)

    filters_start, filters_end = [int(float(x[:-1]) * input_shape[2]) if x.endswith('X') else int(x) for x in filters_range.split('-')]
    filters_range = [int(x) for x in np.linspace(filters_start, filters_end, filters_steps)]

    for layer_count in layer_range:
        kernel_size_arr = [kernel_size] * layer_count
        for filter_count in filters_range:
            nb_filters_arr = [filter_count] * layer_count
            models.append(mlu.build_CNN_model(nb_filters_arr, kernel_size_arr, input_shape=input_shape, last_layer_type=last_layer_type))

    for model in tqdm(models):
        for n in tqdm(range(run_count)):
            df.train_model(model=model, RNN=True)
            df.predict_load(graph=False, store=True)



def benchmark_Dense_structure(benchmark_data_folder, layer_range, neuron_range, neuron_steps, run_count=3, verbose=0):
    df = lf.Load_Forecaster()
    df.load_data(benchmark_data_folder)
    df._override_training_settings(epochs=benchmarking_max_epochs, lossplot=False)
    df.verbose = verbose
    df.db["flag"] = "Dense_structure_benchmark"

    input_shape = df.get_train_data_shape(RNN=False)

    models = []
    layer_start, layer_end = [int(x) for x in layer_range.split('-')]
    layer_range = range(layer_start, layer_end+1)

    neuron_start, neuron_end = [int(float(x[:-1]) * input_shape[1]) if x.endswith('X') else int(x) for x in neuron_range.split('-')]
    neuron_range = [int(x) for x in np.linspace(neuron_start, neuron_end, neuron_steps)]

    for layer_count in layer_range:
        for neuron_count in neuron_range:
            conf = [str(neuron_count)] * layer_count
            models.append(mlu.build_Dense_model(conf,input_shape=input_shape))

    for model in tqdm(models):
        for n in tqdm(range(run_count)):
            df.train_model(model=model, RNN=False)
            df.predict_load(graph=False, store=True)


def plot_structure_bench_RNN_Dense(Dense=None, RNN=None, db=None):
    import seaborn as sns
    import matplotlib.pyplot as plt
    df = lf.Load_Forecaster()

    if db is not None:
        df.db["filename"] = db

    if Dense is not None:
        res = df.load_results(filter_flag="Dense_structure_benchmark")
        unit = 'Neurons'
    elif RNN is not None:
        res = df.load_results(filter_flag="RNN_structure_benchmark")
        unit = 'Units'
    else:
        return

    # Assuming all hidden layers has the same number of hidden neurons. 
    res['neuron_per_layer'] = res['layer_config'].apply(lambda x: x[0])
    grp_model_type = res.groupby('model_type')

    for model, grp in grp_model_type:
        # MultiLineplot neurons per layer vs MAPE, one line per layer count
        ax = grp.groupby(['layer_count', 'neuron_per_layer']).mean()[['testing_MAPE']].unstack(level=1).transpose().xs('testing_MAPE').plot(fontsize=13)
        ax.set_title("Model : {0}".format(model), fontsize=13)
        ax.set_xlabel("{0} per hidden layer".format(unit), fontsize=13)
        ax.set_ylabel("Testing MAPE", fontsize=13)

        # Plot on new figure (plt.figure) of heatmap "x:layer_count vs y:neurons per layer vs z:MAPE"
        plt.figure()
        # Annot 'fmt' -> '.Xg' : float type annotations, if more than X digits, uses scientific notation.
        ax2 = sns.heatmap(grp.groupby(['neuron_per_layer', 'layer_count']).mean()[['testing_MAPE']].reset_index().pivot("neuron_per_layer","layer_count","testing_MAPE"), annot=True, fmt='.3g', cbar_kws={'label':'MAPE'})
        ax2.set_title("Testing performance (MAPE)\nModel : {0}".format(model), fontsize=13)
        ax2.set_xlabel("Layer count", fontsize=13)
        ax2.set_ylabel("{0} per hidden layer".format(unit), fontsize=13)

        # Plot on new figure (plt.figure) of heatmap "x:layer_count vs y:neurons per layer vs z:MAPE"
        plt.figure()
        # Annot 'fmt' -> '.Xg' : float type annotations, if more than X digits, uses scientific notation.
        ax2 = sns.heatmap(grp.groupby(['neuron_per_layer', 'layer_count']).mean()[['training_MAPE']].reset_index().pivot("neuron_per_layer","layer_count","training_MAPE"), annot=True, fmt='.3g', cbar_kws={'label':'MAPE'})
        ax2.set_title("Training performance (MAPE)\nModel : {0}".format(model), fontsize=13)
        ax2.set_xlabel("Layer count", fontsize=13)
        ax2.set_ylabel("{0} per hidden layer".format(unit), fontsize=13)

        # Plot on new figure (plt.figure) of heatmap "x:layer_count vs y:neurons per layer vs z:training_time"
        plt.figure()
        # Annot 'fmt' -> '.Xg' : float type annotations, if more than X digits, uses scientific notation.
        ax2 = sns.heatmap(grp.groupby(['neuron_per_layer', 'layer_count']).mean()[['training_time']].reset_index().pivot("neuron_per_layer","layer_count","training_time"), annot=True, fmt='.3g', cbar_kws={'label':'Seconds'})
        ax2.set_title("Training time (seconds)\nModel : {0}".format(model), fontsize=13)
        ax2.set_xlabel("Layer count", fontsize=13)
        ax2.set_ylabel("{0} per hidden layer".format(unit), fontsize=13)


def plot_structure_bench_CNN(database=None):
    import seaborn as sns
    import matplotlib.pyplot as plt
    df = lf.Load_Forecaster()
    if database is not None:
        df._override_database_settings(filename=database)

    res = df.load_results(filter_flag="CNN_structure_benchmark")

    # Assuming all hidden layers have the same number of hidden neurons. 
    res['filters_per_layer'] = res['layer_config'].apply(lambda x: x[0])

    # MultiLineplot filters per layer vs MAPE, one line per layer count
    ax = res.groupby(['layer_count', 'filters_per_layer']).mean()[['testing_MAPE']].unstack(level=1).transpose().xs('testing_MAPE').plot(fontsize=13)
    ax.set_title("Model : CNN", fontsize=13)
    ax.set_xlabel("Filters per hidden layer", fontsize=13)
    ax.set_ylabel("Testing MAPE", fontsize=13)

    # Plot on new figure (plt.figure) of heatmap "x:layer_count vs y:neurons per layer vs z:MAPE"
    plt.figure()
    # Annot 'fmt' -> '.Xg' : float type annotations, if more than X digits, uses scientific notation.
    ax2 = sns.heatmap(res.groupby(['filters_per_layer', 'layer_count']).mean()[['testing_MAPE']].reset_index().pivot("filters_per_layer","layer_count","testing_MAPE"), annot=True, fmt='.3g', cbar_kws={'label':'MAPE'})
    ax2.set_title("Testing performance (MAPE)\nModel : CNN", fontsize=13)
    ax2.set_xlabel("Layer count", fontsize=13)
    ax2.set_ylabel("Filters per hidden layer", fontsize=13)

    # Plot on new figure (plt.figure) of heatmap "x:layer_count vs y:neurons per layer vs z:training_time"
        
    plt.figure()
    # Annot 'fmt' -> '.Xg' : float type annotations, if more than X digits, uses scientific notation.
    ax2 = sns.heatmap(res.groupby(['filters_per_layer', 'layer_count']).mean()[['training_time']].reset_index().pivot("filters_per_layer","layer_count","training_time"), annot=True, fmt='.3g', cbar_kws={'label':'Seconds'})
    ax2.set_title("Training time (seconds)\nModel : CNN", fontsize=13)    
    ax2.set_xlabel("Layer count", fontsize=13)
    ax2.set_ylabel("Filters per hidden layer", fontsize=13)

    # Plot on new figure (plt.figure) of heatmap "x:layer_count vs y:neurons per layer vs z:training_time"
    plt.figure()
    # Annot 'fmt' -> '.Xg' : float type annotations, if more than X digits, uses scientific notation.
    ax2 = sns.heatmap(res.groupby(['filters_per_layer', 'layer_count']).mean()[['training_MAPE']].reset_index().pivot("filters_per_layer","layer_count","training_MAPE"), annot=True, fmt='.3g', cbar_kws={'label':'Seconds'})
    ax2.set_title("Training performance (MAPE)\nModel : CNN", fontsize=13) 
    ax2.set_xlabel("Layer count", fontsize=13)
    ax2.set_ylabel("Filters per hidden layer", fontsize=13)


def benchmark_models(dataset, models, RNN=None, run_count=3, verbose=0):
    df = lf.Load_Forecaster()
    df.db["flag"] = "models_benchmark"
    df.db["save_detailed_results"] = True
    df.load_data(dataset)
    df._override_training_settings(epochs=benchmarking_max_epochs, lossplot=False)
    df.evaluate_training_set = False

    for model in tqdm(models):
        for n in tqdm(range(run_count)):
            df.train_model(model, RNN)
            df.predict_load(graph=False, store=True)


def plot_models_benchs(location):
    import matplotlib.pyplot as plt
    df = lf.Load_Forecaster()
    data = df.load_results('models_benchmark') 
    ax = data[['testing_MAPE', 'training_MAPE', 'location', 'training_time', 'model_summary']].groupby(['location','model_summary']).mean().xs(location).sort_values('testing_MAPE').plot.bar(secondary_y='training_time', rot=0) 
    ax.set_title("Dataset : {0}\nDay ahead forecasting".format(location), fontsize=13)    
    ax.set_xlabel("Model", fontsize=13)
    ax.set_ylabel("MAPE", fontsize=13)
    ax.right_ax.set_ylabel('Seconds', fontsize=13)
    plt.show()
        

def benchmark_datasets(benchmark_datasets, flag, run_count=3, verbose=0):
    for dataset in tqdm(benchmark_datasets):
        if verbose != 0:
            print("Benchmarking dataset {0}".format(dataset))

        df = lf.Load_Forecaster()
        df.db["flag"] = flag + "_benchmark"
        df.verbose = verbose

        df.load_data(dataset)
        df._override_training_settings(epochs=benchmarking_max_epochs, lossplot=False)
        _benchmark_loop(df, run_count)



def plot_NYISO_forecast_error(true_path, pred_path):
    """ Plots and give metrics regarding the NYISO forecasts 
        Args :
            true_path : (file / folder string)  : True NYISO load data file / folder 
            pred_path : (file / folder string)  : NYISO forecasts load data file / folder 
    """
    import Loader as ld
    import Preprocessor as pre
    import matplotlib.pyplot as plt

    true_ld = ld.NY_Loader(true_path)
    nyiso_ld = ld.NY_Loader(pred_path)
    true_pre = pre.NY_Preprocessor(true_ld.data, 'Integrated Load', year_range=list(range(2008, 2018)))
    nyiso_pre = pre.NY_Preprocessor(nyiso_ld.data, 'Integrated Load', year_range=list(range(2008, 2018)), fix_duplicates='keep_last')
    results = mlu.get_results(true_pre.get_data().values, nyiso_pre.get_data(), true_pre.get_data().index)

    import Plotter
    Plotter.plot_results(results, groupby='month')

    print("Global error : {0}".format(mlu.get_measures(results, 'global', 'MAPE')))
    print("Global error : {0}".format(mlu.get_measures(results, 'global', 'RMSE')))
    plt.show()