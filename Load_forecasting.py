"""
This module hosts the high-level API class for doing electricy Load / Demand forecasting.

Usage example:
    # Import the module
    import Load_forecasting as lf

    # Instantiate a Load_Forecaster object
    forecaster = lf.Load_Forecaster()

    # Attach historical load data
    ## <LOAD_DATA> can either be a folder path or a single file path.
    ## Data files are either National Grid's UK demand or NYISO load formatted files. 
    forecaster.attach_load(<LOAD_DATA>) # <LOAD_DATA> can either be a folder or a single file. Each file must

    # Attach historical weather data
    forecaster.attach_weather(filepath_stations=<WEATHER_STATIONS>,
                              filespath_data=<WEATHER_DATA>,
                              variables=<WEATHER_VARIABLES>,
                              drop_stations=<STATIONS_TO_DROP>)

    ## <WEATHER_STATIONS> is NOAA NCDC weather stations list filepath. 
    ## <WEATHER_DATA> is a list of files containing NOAA NCDC weather data collected as per the tutorial.
    ## <WEATHER_VARIABLES> is any of the following : ['W_Spd', 'Air_Temp', 'RHx ', 'Dew_temp']
    ## <STATIONS_TO_DROP> is a list of coordinates [(x,y), ..., (x,y)] of stations from the weather dataset that should be excluded.

    # Process input data using internal default settings.
    forecaster.process_data()

    # Train model
    forecaster.train_model(<MODEL>)
    ## <MODEL> can either be a string refering to one of the pre-included models ['dense', 'dense_small', 'GRU', 'LSTM', 'CNN']
    ## Alternatively, a Keras model can be passed.

    # Evaluate model
    forecaster.predict_load()
"""

import Loader as ld
import Preprocessor as pr
import ML_Utils as mlu
import ML_Preprocessor as mlp
from Utils import *
import pickle
import warnings
import keras
from enum import Enum
from keras.models import model_from_json
from tinydb import TinyDB, Query  # Note : Install Python package "ujson" for faster database operations.
import json
from sklearn.svm import SVR
import sys
import os
from copy import deepcopy
from keras.models import Sequential
from keras.layers import CuDNNLSTM, Dense, Convolution1D, MaxPooling1D, Flatten, LSTM, GRU, CuDNNGRU
from keras import backend as K
import time



class Status(Enum):
    """ Enumeration to track the status of various processes in the Load_Forecaster class"""
    NOT_LOADED = 0
    LOADED = 1
    READY = 2


class Load_Forecaster:
    """ High-level load forecasting class """

    def __init__(self):
        """ Initialize an instance of the class using default settings """
        self.loaders = {"Load":None, "Weather":None}
        self.preprocessors = {"Load":None, "Weather":None}

        self._data = None
        self.models = []
        self.sc_X = None
        self.sc_y = None

        self.settings = {'Weather':{}, 'Load':{}}
        self.settings["Load"]["Preprocessor"] = {'fixes':['zeros', 'extremes_global', 'derivatives']}
        self.settings["Weather"]["Preprocessor"] = {'stations_to_drop':[], 'countries_to_keep':None, 'resample':None, 
                                                    'pre_clustering_data_quality':('1h', 85.0), 'post_clustering_data_quality':98.0, 
                                                    'desired_cluster_count':16, 'min_cluster_size':2, 'algorithm':'meanshift'}
        
        self.settings["ML_Preprocessor"] = {'time_encoding':'cyclical', 'train_test':(6,2), 'load_difference':False, 'stdz':'RobustScaler'}
        self.settings["ML_Preprocessor"]["historical_ts"] = {'points':['1d', '7d'], 'averages':['1d', '7d']}

        self.settings["Data"] = {'location':None, 'forecast_horizon':'day', 'year_range':None, 'data_shape_mode':None, 'weather_data':False, 'weather_variables':{}, 'float_precision':64}
        self.settings["Model"] = {'arch':None, 'optimizer':{'optimizer_name':'Adam', 'lr':0.0005, 'decay':None, 'momentum':None, 'nesterov':None}, 
                                  'batch_size':64, 'use_gpu':True, 'epochs':200, 'lossplot':True, 'tensorboard':False, 'early_stopping':True, 'reduce_lr_plateau':True, 'backend':'tensorflow'} # NOTE : "backend" is read-only
        self.settings["Evaluation"] = {'period':'global', 'metric':'MAPE'}

        self.evaluate_training_set = True

        self.last_training_results = None
        self.last_testing_results = None

        self.last_training_info = None
        self.last_predict_info = None

        self.splits = {'raw':[], 'RNN':[], 'non-RNN':[]}
        self.status = {"Load":Status.NOT_LOADED, "Weather":Status.NOT_LOADED, "ML":Status.NOT_LOADED}

        self.db = {'filename':'results.db', 'save_detailed_metrics':False, 'save_detailed_results':False, 'save_weigths':False, 'flag':None}
        # Base size : 3.4kB, with details : 6.2MB, with details + weights : 13MB

        self.verbose = 1


    def attach_load(self, filename, location):
        """ Registers / attach the load data within the class for later processing 
            Args:
                filename : (list of str / str): Filepath to the NYISO / NG formatted historical load files.
                location : (str): Load data location from : ['UK', 'ENGLAND_WALES', 'NYC', 'MHK', 'NEW_YORK']
            Returns:
                None
        """
        self.settings["Data"]["location"] = location

        # Get an instance of the appropriate loader class object for this specific location.
        loader = self.__get_loader(location)
        error = loader.load(filename)
        self.__attach_loader(loader, "Load", error)


    def attach_weather(self, filepath_stations, filespath_data, variables, drop_stations):
        """ Registers / attach the weather data within the class for later processing 
            Args:
                filepath_stations : (str): Filepath to the NOAA NCDC weather sattions list file
                filespath_data : (list of str): Filepaths to the NOAA NCDC weather data
                variables : (list of str): Weather variables to be used - can only contain elements from ['W_Spd', 'Air_Temp', 'RHx ', 'Dew_temp']
                drop_stations : (list of list of real numbers) : 
            Returns:
                None
        """
        loader = ld.NCDC_Loader()
        self.settings["Weather"]["Preprocessor"]["stations_to_drop"] = drop_stations
        self.settings["Data"]["weather_variables"] = {var:0 for var in variables}
        error = loader.load(filepath_stations=filepath_stations,
                            filepath_data=filespath_data,
                            load_columns=variables)

        self.__attach_loader(loader, "Weather", error)


    def process_data(self):
        """ Process data according to the module settings. 
            Args:
                None
            Returns:
                None
        """
        work_done = {'load':False, 'weather':False, 'ml':False}

        if self.status["Load"] == Status.NOT_LOADED:
            return MissingDataError("Load data not loaded !")

        # Load Preprocessor 
        if self.status["Load"] == Status.LOADED:
            demand_preprocessor, y_var = self.__get_preprocessor(self.settings["Data"]["location"])
            error = demand_preprocessor.load(self.loaders['Load'].get_data(), y_var)
            self.__attach_load_preprocessor(demand_preprocessor, error)
            
            self.preprocessors["Load"].fix_data(max_error_width=3, threshold=(1.5,0.99), **self.settings["Load"]["Preprocessor"])
            self._load = self.preprocessors["Load"].get_data()
            
            work_done['load'] = True
            self.status["Load"] = Status.READY

        # Weather Preprocessor
        if self.status["Weather"] == Status.NOT_LOADED and self.verbose:
            warnings.warn("Weather data not loaded.")
            self.settings["Data"]["weather_data"] = False

        elif self.status["Weather"] == Status.LOADED:
            self.__set_countries_to_keep(y_var)
            self.settings["Weather"]["Preprocessor"]["resample"] = str(self.preprocessors["Load"].settlement_period_minutes) + " min"
            self.preprocessors["Weather"] = pr.NCDC_Preprocessor(self.loaders['Weather'], year_range=self.settings["Data"]["year_range"])
            self.preprocessors["Weather"].fix_data(**self.settings["Weather"]["Preprocessor"])    
            work_done['weather'] = True
            self.settings["Data"]["weather_data"] = True  # Duplicate of status["Weather"] but only self.settings is backed-up.
            self.status["Weather"] = Status.READY

        # ML_Preprocessor
        if work_done['load'] or work_done['weather'] or self.status["ML"] == Status.NOT_LOADED:
            self.preprocessors['ML'] = mlp.ML_Preprocessor(load=self._load, weather_preprocessor=self.preprocessors["Weather"], y_vars=['TS'])
            if self.settings["ML_Preprocessor"]["load_difference"]:
                    self.preprocessors["ML"].transform_load_difference()
                    
            self.preprocessors["ML"].add_historical_ts(**self.settings["ML_Preprocessor"]["historical_ts"])
            self.preprocessors["ML"].encode_time(time_encoding=self.settings["ML_Preprocessor"]["time_encoding"])

            # TODO : Should be done from the ML_Preprocessor 'encode_time' method directly.
            if self.settings["ML_Preprocessor"]["time_encoding"] == 'categorical':
                self.preprocessors['ML'].encode_vars(cat_vars=['Month', 'Day', 'Hour', 'Minute', 'DoW'])
            
            # Data normalization
            self.sc_X, self.sc_y = self.preprocessors['ML'].std_vars(stdz=self.settings["ML_Preprocessor"]["stdz"])
            work_done['ML'] = True
            self.status["ML"] = Status.READY

        # If any change to the data was done : Regenerate the cross-validation splits
        if any(v==True for v in work_done.values()):
            self.splits["raw"] = self.preprocessors['ML'].get_cv_splits(self.settings["ML_Preprocessor"]["train_test"])
            
            # Identify how many clusters exist for each weather variable.
            data_columns = self.splits['raw'][0][0][0].columns.tolist()
            for var in self.settings["Data"]["weather_variables"]:
                self.settings["Data"]["weather_variables"][var] = sum([1 for x in data_columns if x.startswith(var)])


    def train_model(self, model, RNN=None):
        """ Train a machine learning model on the loaded and processed data.
            Args:
                model : (str / Keras Sequential / Keras Model) : Model to be used. 
                RNN : (boolean) : Refers to weather or not the model expect chunked data (e.g. per day), and if direct prediction can be used.
            Returns:
                None
            Todo:
                * Check that the data is ready, else, raise an exception.
                * Validate support for SVR model
                * Load models from file, trained or not.   
                * Get a fresh non trained model using "model.__class__()" instead of calling '__get_std_model' again.              
                * Fix the 'RNN' argument module wise :
                    -> Terrible name that doesn't carry its meaning anymore : Was historical used to make the distinction between Dense (recurrent forecasting*, non-chunked data) and RNN models (direct forecasting, chunked data)
                        Should be split between 'data_shape'=['chunked', 'not_chunked'] and 'forecasting_method'=['direct', 'recurrent']
                        Better yet : Automatic detection of model input and forecating method (i.e. input contains forecast data)
                    (*) : Not using sub forecast load inputs (e.g. H-4 with day ahead forecasting) with Dense model was not considered due to poor performance (no context) 
        """
        if isinstance(model, str):
            model, RNN = self.__get_std_models(model)

        self.settings["Model"]["backend"] = keras.backend.backend()

        start = time.time()

        self.models = []
        self.last_training_info = np.array([]).reshape(0,3)

        data_mode = 'RNN' if RNN else 'non-RNN'
        self.settings["Data"]["data_shape_mode"] = data_mode

        self.__build_data(RNN)

        for ((X_train, y_train), (X_test, y_test)), n in zip(self.splits[data_mode], range(len(self.splits[data_mode]))):

            [X_train, y_train, X_test, y_test], precision = self._update_floating_point_precision([X_train, y_train, X_test, y_test])

            if self.verbose > 0:
                print("Cross-validation, set {0}".format(n))

            if RNN:
                y_test_tmp = np.array([np.array(self.sc_y.transform(x)) for x in y_test], dtype=precision) 
                validation_data = (X_test,y_test_tmp)
            else:
                validation_data = None

            model_instance, action = self.__get_model(model)
            self.settings["Model"]["arch"] = model_instance

            #if action == 'train_SVR':
            #    model_instance.fit(X_train, y_train.values.ravel())
            #    self.models.append(model_instance)
            if action == 'train_Keras':
                if RNN:
                    y_train = np.array([y.values for y in y_train])

                trained_model, training_info = mlu.train_model(X_train, y_train, validation_data=validation_data, verbose=0, **self.settings["Model"])
                self.models.append(trained_model)
                self.last_training_info = np.vstack([self.last_training_info, training_info])

            # TODO : Differentiate between already trained and not loaded models ? 
            # For now, all saved / loaded models from .h5 are considered already trained.
            elif action == 'load_Keras':
                self.models.append(model_instance) 

        print("TOTAL TRAINING TIME : {0}".format(time.time()-start))

    def __get_std_models(self, model_str):
        """ Returns a model instance for a given pre-defined model.
            Args:
                model_str : (str) : Model from ['LSTM', 'GRU', 'dense', 'dense_small', 'CNN']
            Returns:
                Tuple (Keras Sequential model / sklearn model , boolean): [Instanciated model instance, RNN model (i.e. chunked data + direct prediction or non-chunked + recurrent prediction)] 
            Todo:
                * Move to ML_Utils
                * Add support back for SVR model
                * Add support for loading existing model from file (trained or not) 
        """
        if model_str in ['LSTM', 'GRU']:
            RNN = True
            X_train_shape = self.get_train_data_shape(RNN=RNN)
            RNN_layers = [LSTM, CuDNNLSTM, GRU, CuDNNGRU]

            if model_str == 'GRU':
                RNN_layers = RNN_layers[2:4]

            # CuDNNLSTM / CuDNNGRU only supported when using Tensorflow backend
            if self.settings["Model"]["backend"] == "tensorflow":
                RNN_layer = RNN_layers[self.settings["Model"]["use_gpu"]]
            else:
                RNN_layer = RNN_layers[0]

            model = Sequential([RNN_layer(X_train_shape[2]*2, input_shape=X_train_shape[1:3], return_sequences=True),
                                RNN_layer(X_train_shape[2]*2, return_sequences=True),
                                Dense(1, activation='linear')])
        elif model_str == 'dense':
            RNN = False
            X_train_shape = self.get_train_data_shape(RNN=RNN)
            model = Sequential([
                                Dense(X_train_shape[1]*2, activation='relu', input_shape=(X_train_shape[1],)),
                                Dense(X_train_shape[1]*3, activation='relu'),
                                Dense(X_train_shape[1]*3, activation='relu'),
                                Dense(X_train_shape[1]*3, activation='relu'),
                                Dense(X_train_shape[1]*2, activation='relu'),
                                Dense(1, activation='linear')])
        elif model_str == 'dense_small':
            RNN = False
            X_train_shape = self.get_train_data_shape(RNN=RNN)
            model = Sequential([
                                Dense(X_train_shape[1]*2, activation='relu', input_shape=(X_train_shape[1],)),
                                Dense(X_train_shape[1]*2, activation='relu'),
                                Dense(1, activation='linear')])


        elif model_str == 'CNN':
            RNN = True
            X_train_shape = self.get_train_data_shape(RNN=RNN)
            model = Sequential([
                            Convolution1D(776, 3, activation='relu', padding="same", input_shape=X_train_shape[1:3]),
                            Convolution1D(776, 3, activation='relu', padding="same"),
                            Convolution1D(1, 3, activation='linear', padding="same")])
        else:
            raise Exception("Only Keras models pre-existing model strings ('LSTM', 'GRU', 'dense', 'dense_small', 'CNN')")
    
        return model, RNN

    def predict_load(self, graph=True, store=False, plot_groupby=None):
        """ Evaluate the trained models (one per cross validation set).
            Args:
                graph : (boolean) : Weather or not to plot graphs at the end (true vs prediction graph by default)
                store : (boolean) : Weather or not to store results in the database
                plot_groupby : (str) : Change the graph type (graph=True) to plot grouped error. Legal string are those in method 'Utils.get_groupby_time()' 
            Returns:
                None
            Todo:
                TODO
        """
        # Just in case, since data isn't pickled.
        self.__build_data(self.settings["Data"]["data_shape_mode"] == 'RNN')

        cv_train_results = []
        cv_test_results = []
        self.last_predict_info = np.array([]).reshape(0,2)
        train_results, train_inference_time = -1, -1

        # Get back the original load from the differentiated data. Yes could be improved, for example use np.r_[start, X[1:]].cumsum()
        if self.settings["ML_Preprocessor"]["load_difference"]:
            non_diff_data = self.preprocessors['Load'].get_data()
        else:
            load_difference_start = None

        for ((X_train, y_train), (X_test, y_test)), model, ((_, y_train_raw), (_,_)) in zip(self.splits[self.settings["Data"]["data_shape_mode"]], self.models, self.splits['raw']):

            # TODO : Assign predict function to var in if then do stuff without ifs
            if self.settings["Data"]["data_shape_mode"] == 'non-RNN':
                if self.settings["ML_Preprocessor"]["load_difference"]:
                    load_difference_start = non_diff_data.loc[non_diff_data.index == min(y_test[0].index),"TS"][0]

                if self.evaluate_training_set:
                    X_train_tmp = deepcopy(X_train)
                    X_train_tmp = mlu.split_dataset_by_date(X_train_tmp, split_on=self.settings["Data"]["forecast_horizon"])
                    y_train_tmp = deepcopy(y_train)
                    y_train_tmp['TS'] = self.sc_y.inverse_transform(y_train)
                    y_train_tmp = mlu.split_dataset_by_date(y_train_tmp, split_on=self.settings["Data"]["forecast_horizon"])
                    train_results, train_inference_time = mlu.recurrent_predict_evaluate(model, self.sc_y, X_train_tmp, y_train_tmp, load_difference_start=load_difference_start)
                else:
                    train_inference_time = 0

                test_results, test_inference_time  = mlu.recurrent_predict_evaluate(model, self.sc_y, X_test, y_test, load_difference_start=load_difference_start)
                
            else:
                if self.settings["ML_Preprocessor"]["load_difference"]:
                    load_difference_start = non_diff_data.loc[non_diff_data.index == min(index),"TS"][0]
                
                if self.evaluate_training_set:
                    y_train_tmp = deepcopy(y_train)
                    y_train_tmp = pd.concat(y_train)
                    y_train_tmp['TS'] = self.sc_y.inverse_transform(y_train_tmp)
                    train_results, train_inference_time = mlu.predict_evaluate(model, self.sc_y, X_train, y_train_tmp, load_difference_start=load_difference_start)

                test_results, test_inference_time  = mlu.predict_evaluate(model, self.sc_y, X_test, pd.concat(y_test), load_difference_start=load_difference_start)
            
            if self.verbose > 0:
                if self.evaluate_training_set:
                    train_metric_results = mlu.get_measures(train_results, **self.settings["Evaluation"])
                    print("[TRAINING SET] - {period} error : {res}% {metric}".format(res=train_metric_results, **self.settings["Evaluation"]))
                
                test_metric_results = mlu.get_measures(test_results, **self.settings["Evaluation"])
                print("[TESTING  SET] - {period} error : {res}% {metric}".format(res=test_metric_results, **self.settings["Evaluation"]))
            
            cv_train_results.append(train_results)
            cv_test_results.append(test_results)
            
            self.last_predict_info = np.vstack([self.last_predict_info, np.hstack([train_inference_time, test_inference_time])])

        if graph:
            import Plotter
            Plotter.plot_results(pd.concat([pd.concat(cv_train_results), pd.concat(cv_test_results)]), rolling=24, groupby=plot_groupby)
        
        self.last_training_results = cv_train_results
        self.last_testing_results  = cv_test_results

        if store:
            self.__save_results_to_db()

        # Clear memory
        # NOTE : May not be necessary depending on Keras / TF version
        # NOTE 2 : May even <<<SEGFAULT>>>.
        # Necessary to prevent memory use buildup -> Slower training
        #K.get_session().close()
        #cfg = K.tf.ConfigProto()
        #cfg.gpu_options.allow_growth = True
        #K.set_session(K.tf.Session(config=cfg))

        K.clear_session()


    def save_data(self, folder):
        """ Saves processed data from later fast reloading. Used by the generate_datasets notebook """
        if not os.path.isdir(os.path.join(sys.path[0], 'processed_data')):
            os.makedirs(os.path.join(sys.path[0], 'processed_data'))

        if os.path.isdir(os.path.join(sys.path[0], 'processed_data', folder)):
            warnings.warn("Data folder already exists, delete it or change target folder.")
        else:
            os.makedirs(os.path.join(sys.path[0], 'processed_data', folder))
            self.settings["Data"]["source_folder"] = folder
            arch = self.settings["Model"]["arch"]
            self.settings["Model"]["arch"] = None
            pickle.dump(self.splits['raw'], open(os.path.join('processed_data', folder, "data"), 'wb'), protocol=4)
            json.dump(self.settings, open(os.path.join('processed_data', folder, "settings"), 'w'))
            pickle.dump(self.sc_y, open(os.path.join('processed_data', folder, "sc_y"), 'wb'), protocol=4)
            self.settings["Model"]["arch"] = arch

    def load_data(self, folder):
        """ Reload a dataset saves using save_data """
        self.splits['raw'] = pickle.load(open(os.path.join('processed_data', folder, "data"), 'rb'))
        self.settings = json.load(open(os.path.join('processed_data', folder, "settings"), 'r'))
        self.sc_y = pickle.load(open(os.path.join('processed_data', folder, "sc_y"), 'rb'))


    #====================
    ##### OVERRIDES #####
    #====================
    """ Used to alter the module settings """
    def __override_settings(self, setting_class_levels, new_settings):
        setting_class = self.settings
        for level in setting_class_levels:
            setting_class = setting_class[level]

        valid_keys = set(setting_class.keys()).intersection(new_settings.keys())
        invalid_keys = set(new_settings.keys()).difference(setting_class.keys())

        new_settings = {k:v for k,v in new_settings.items() if k in valid_keys}
        setting_class.update(new_settings)

        if len(invalid_keys):
            warnings.warn("Keys {0} are invalid".format(invalid_keys))

    def _override_optimizer(self, **kwargs):
        self.settings["Model"]["optimizer"] = {'optimizer_name':'Adam', 'lr':0.0005, 'decay':None, 'momentum':None, 'nesterov':None}
        self.__override_settings(["Model", "optimizer"], kwargs)


    # TODO : Not homogeneous, decide if more code but checks (this one) or not (**kwargs)
    def _override_model_data_settings(self, year_range=None, train_test=None, forecast_horizon=None, float_precision=None):
        if year_range is not None:
            self.settings["Data"]["year_range"] = year_range
        
        if train_test is not None:
            self.settings["ML_Preprocessor"]["train_test"] = train_test

        if forecast_horizon in ['day', 'week']:
            self.settings["Data"]['forecast_horizon'] = forecast_horizon

        if float_precision in [32,64]:
            self.settings["Data"]["float_precision"] = float_precision

        self.__change_warning(recompute_demand=True) # TODO : Not needed for floating precision...

    def _override_evaluation_metrics(self, **kwargs):
        self.__override_settings(["Evaluation"], kwargs)

    def _override_weather_data_settings(self, **kwargs):
        self.__override_settings(["Weather", "Preprocessor"], kwargs)
        self.__change_warning(recompute_weather=True)

    def _override_historical_load_propagation(self, **kwargs):
        self.__override_settings(["ML_Preprocessor", "historical_ts"], kwargs)
        self.__change_warning(recompute_demand=True)

    def _override_historical_load_preprocessing(self, fixes):
        self.settings["Load"]["Preprocessor"]["fixes"] = fixes
        self.__change_warning(recompute_demand=True)

    def _override_training_settings(self, **kwargs):
        self.__override_settings(["Model"], kwargs)

    def _override_time_encoding(self, mode):
        if mode in ['categorical', 'cyclical']:
            self.settings["ML_Preprocessor"]["time_encoding"] = mode
        else:
            warnings.warn("Time encoding mode '{0}' invalid - ignoring.".format(mode))
        self.__change_warning(recompute_ml_preprocessor=True)

    def _override_standardizer(self, stdz):
        self.settings["ML_Preprocessor"]["stdz"] = stdz
        self.__change_warning(recompute_ml_preprocessor=True)


    def _override_database_settings(self, filename='results.db', save_details=False, save_weights=False):
        self.db["filename"] = filename
        self.db["save_details"] = save_details
        self.db["save_weights"] = save_weights


    def __change_warning(self, recompute_demand=False, recompute_weather=False, recompute_ml_preprocessor=False):
        """ Used to change the ready status of various processes and alert the user in case some recomputation must be done to apply their new settings """
        if recompute_demand:
            if self.status["Load"] == Status.READY:
                self.status["Load"] = Status.LOADED
        if recompute_weather:
            if self.status["Weather"] == Status.READY:
                self.status["Weather"] = Status.LOADED

        if recompute_ml_preprocessor:
            if self.status["ML"] == Status.READY:
                self.status["ML"] = Status.LOADED

        if any([recompute_demand, recompute_weather, recompute_ml_preprocessor]):
            if self.verbose:
                warnings.warn("Re-run process_data to reflect changes")

    def get_train_data_shape(self, RNN):
        """ Used to get the training data shape in order to build models from outside the class which requires a fixed known input shape """
        self.__build_data(RNN=RNN)
        if RNN:
            return self.splits["RNN"][0][0][0].shape
        else:
            return self.splits["raw"][0][0][0].values.shape


    def __attach_loader(self, loader_object, loader_type, error):
        """ Simply checks if the loader object is in a viable stable, if so, "attaches" it : Raw data considered loaded """
        if not error:
            if isinstance(self.loaders[loader_type], ld.Loader):
                warnings.warn("{0} data already loaded, replacing. Pass :filenames as list to load multiple files.".format(loader_type))

            self.loaders[loader_type] = loader_object
            self.status[loader_type] = Status.LOADED
        else:
            warnings.warn("{0} loader failure, error {1} : {2}".format(loader_type, error['code'], error['msg']))

    def __attach_load_preprocessor(self, preprocessor_object, error):
        """ Simply checks if the preprocessor object is in a viable stable, if so, "attaches" it : preprocessed data considered ready"""
        if not error:
            self.preprocessors["Load"] = preprocessor_object
        else:
            warnings.warn("Load preprocessor failure, error {1} : {2}".format(error['code'], error['msg']))


    def __get_loader(self, location):
        """ Return the loader object for the specific location """
        locations = {'ENGLAND_WALES':{'Loader':ld.NG_Loader},
                     'UK':{'Loader':ld.NG_Loader},
                     'NYC':{'Loader':ld.NY_Loader},
                     'MHK':{'Loader':ld.NY_Loader},
                     'NEW_YORK':{'Loader':ld.NY_Loader}}

        return locations[location]['Loader']()


    def __get_preprocessor(self, location):
        """ Return the preprocessor object for the specific location """
        locations = {'ENGLAND_WALES':{'Preprocessor':pr.NG_Preprocessor, 'y_var':'ENGLAND_WALES_DEMAND'},
                     'UK':{'Preprocessor':pr.NG_Preprocessor, 'y_var':'ND'},
                     'NYC':{'Preprocessor':pr.NY_Preprocessor, 'y_var':'Integrated Load'},
                     'MHK':{'Preprocessor':pr.NY_Preprocessor, 'y_var':'Integrated Load'},
                     'NEW_YORK':{'Preprocessor':pr.NY_Preprocessor, 'y_var':'Integrated Load'}}

        return locations[location]['Preprocessor'](year_range=self.settings["Data"]["year_range"]), locations[location]['y_var'] 


    def __set_countries_to_keep(self, y_var):
        """ Sets the list of countries corresponding to the load location (UK only) """
        countries = {'ENGLAND_WALES_DEMAND':['England','Wales'], 
             'SCOTLAND_IRELAND':['Scotland', 'Northern Ireland', 'Ulster'],
             'ND':["Northern Ireland", "England", "Wales", "Scotland", "Ulster"],
             'Integrated Load':None} # Do not filter

        self.settings["Weather"]["Preprocessor"]["countries_to_keep"] = countries[y_var]


    def __build_data(self, RNN):
        """ Processes data : Split according to forecat horizon and de-normalize y_test as it will be compared to the denormalized y_pred """
        if RNN:
            if not len(self.splits['RNN']):
                for (X_train, y_train), (X_test, y_test) in self.splits["raw"]:
                    X_train = mlu.split_dataset_by_date(X_train, split_on=self.settings["Data"]["forecast_horizon"])
                    X_train = np.array([x.values for x in X_train])
                    y_train = mlu.split_dataset_by_date(y_train, split_on=self.settings["Data"]["forecast_horizon"])

                    X_test = mlu.split_dataset_by_date(X_test, split_on=self.settings["Data"]["forecast_horizon"])
                    X_test = np.array([y.values for y in X_test])
                    y_test['TS'] = self.sc_y.inverse_transform(y_test)
                    y_test = mlu.split_dataset_by_date(y_test, split_on=self.settings["Data"]["forecast_horizon"])

                    self.splits["RNN"].append([(X_train, y_train), (X_test, y_test)])
        else:
            if not len(self.splits['non-RNN']):
                for (X_train, y_train), (X_test, y_test) in self.splits["raw"]:
                    y_test['TS'] = self.sc_y.inverse_transform(y_test)
                    X_test, y_test = mlu.get_test_sets(X_test, y_test, split_on=self.settings["Data"]["forecast_horizon"])                    
                    self.splits["non-RNN"].append([(X_train, y_train), (X_test, y_test)])


    def _update_floating_point_precision(self, sets):
        """ Changed training / testing data floating point precision """ 
        precision = self.settings["Data"]["float_precision"]
        
        # np.dtype format expects floating point precision in byte and not in bit
        precision = np.dtype('f{0}'.format(precision//8))

        new_set = []

        for x in sets:
            if isinstance(x, (np.ndarray, pd.DataFrame)): # 'X's or non_chuncked 'y's (Dense models)
                new_set.append(x.astype(precision))
            elif isinstance(x, list) and isinstance(x[0], pd.DataFrame): # Chunked 'y's (non-dense models)
                new_set.append([y.astype(precision) for y in x])
            else:
                new_set.append()
                raise ValueError("The data doesn't seem to be in the right shape ! Should either be a ndarray, list of DataFrame or single DataFrame. Detected first level type : {0}, detected 2sd level type : {1}".format(type(x), type(x[0])))

        return new_set, precision
    
    def __get_model(self, model):
        # SVR
        if type(model) == list:
            return SVR(kernel='rbf', gamma=model[1], C=model[2]), 'train_SVR'

        else:
            # Keras saved model
            if type(model) == str:
                if model.endswith('h5'):
                    return mlu.load_model(version='3'), 'load_Keras'
                else:
                    return model, 'train_Keras'
            else:
                return model_from_json(model.to_json()), 'train_Keras' 
                # Otherwise the same model instance is used for each cross-validation set. No "model.reset" method existing yet.

    #####################################
    ### TODO : Move to separte class ####
    #####################################

    def __save_results_to_db(self):
        """ Save training results / info to NoSQL database for later plotting and analysis """
        db = TinyDB(self.db['filename'])
        set_tbl = db.table('settings')
        res_tbl = db.table('results')

        # Get new run_id, starting at 0
        run_id = max([x['RUN_ID'] for x in res_tbl.all()]+ [-1]) + 1
        settings = self.__settings_to_json(run_id)
        set_tbl.insert(settings)

        if self.evaluate_training_set:
            training_MAPE = pd.concat([mlu.get_measures(x, period='DoY', metric='MAPE') for x in self.last_training_results])
            training_RMSE = pd.concat([mlu.get_measures(x, period='DoY', metric='RMSE') for x in self.last_training_results])
            mean_training_MAPE = training_MAPE.mean()
            mean_training_RMSE = training_RMSE.mean()
        else:
            training_MAPE = -1
            testing_MAPE = -1
            mean_training_MAPE = -1
            mean_training_RMSE = -1

        testing_MAPE = pd.concat([mlu.get_measures(x, period='DoY', metric='MAPE') for x in self.last_testing_results])
        testing_RMSE = pd.concat([mlu.get_measures(x, period='DoY', metric='RMSE') for x in self.last_testing_results])

        mean_training_time = np.mean(self.last_training_info[:,0])
        mean_epoch_count = np.mean(self.last_training_info[:,1])
        mean_gpu_usage = np.mean(self.last_training_info[:,2])

        mean_inference_train_set, mean_inference_test_set = np.hsplit(np.mean(self.last_predict_info, axis=0), 2)

        res_tbl.insert({'RUN_ID'                    :   run_id, 
                        'flag'                      :   self.db['flag'],
                        'training_MAPE'             :   mean_training_MAPE,
                        'training_RMSE'             :   mean_training_RMSE, 
                        'testing_MAPE'              :   testing_MAPE.mean(), 
                        'testing_RMSE'              :   testing_RMSE.mean(), 
                        'training_time'             :   mean_training_time, 
                        'mean_gpu_usage'            :   mean_gpu_usage,
                        'inference_time_train_set'  :   mean_inference_train_set[0],
                        'inference_time_test_set'   :   mean_inference_test_set[0],
                        'ran_epochs'                :   mean_epoch_count
                        })

        if self.db["save_detailed_metrics"]:
           warnings.warn("Storing detailed results significantly increases database size !")
           dtl_met_tbl = db.table('detailed_metrics')
           dtl_met_tbl.insert({'RUN_ID'             : run_id, 
                               'flag'               : self.db['flag'],
                               'training_MAPE'      : training_MAPE,
                               'training_RMSE'      : training_RMSE,
                               'testing_MAPE'       : testing_MAPE,
                               'testing_RMSE'       : testing_RMSE})

        if self.db["save_detailed_results"]:
            #pickle.dump(open("{0}_train.pkl".format(RUN_ID), "wb"),self.last_traning_results)
            #pickle.dump(open("{0}_test.pkl".format(RUN_ID), "wb"),self.last_testing_results)
            if self.evaluate_training_set:
                training_results = [x.to_dict(orient='records') for x in self.last_training_results]
            else:
                training_results = -1


            dtl_res_tbl = db.table('detailed_results')
            dtl_res_tbl.insert({'RUN_ID'                    : run_id, 
                                'flag'                      : self.db['flag'],
                                'detailed_training_results' : training_results,
                                'detailed_testing_results'  : [x.to_dict() for x in self.last_testing_results]
                                #'cv_results'                : [pd.concat([tr, te]).to_dict() for tr,te in zip(self.last_training_results, self.last_testing_results)]
                                })


        if self.db["save_weigths"]:
            warnings.warn("Storing models weights significantly increases database size !")
            weights_tbl = db.table('model_weights')
            weights = self.__models_weights_to_json()
            weights_tbl.insert({'RUN_ID':run_id, 'flag':self.db['flag'], 'weights':weights})


    def __settings_to_json(self, run_id):
        """ Transforms module settings into json-able object and adds a few other pieces of information """
        # Cannot copy Keras models using deepcopy (thread lock objects)
        arch = self.settings["Model"]["arch"]
        self.settings["Model"]["arch"] = None
        settings = deepcopy(self.settings)
        self.settings["Model"]["arch"] = arch
        
        if hasattr(arch, "to_json"): # Keras models
            settings["Model"]["trainable_weights"] = arch.count_params()
            settings["Model"]["memory_usage_MB"] = mlu.model_memory_usage(arch, self.settings["Model"]["batch_size"]) 
            settings["Model"]["arch"] = arch.to_json()
        else:
            settings["Model"]["arch"] = json.dumps({'arch':str(arch)})

        settings['flag'] = self.db['flag']
        settings['RUN_ID'] = run_id

        return settings

    def __models_weights_to_json(self):
        """ Converts model weights to json-able object """
        weights = list(self.models)

        for model in range(len(self.models)):
            layers = self.models[model].get_weights()
            weights[model] = [x.tolist() for x in layers]

        return weights


    def load_results(self, filter_flag=None, detailed=False):
        """ Reloads data from the NoSQL database. Can filter by using a flag (filter_flag) and load detailed results if needed (used when plotting multiple model predictions at once """
        db = TinyDB(self.db['filename'])
        set_tbl = db.table('settings')
        res_tbl = db.table('results')

        if filter_flag:
            query_filter = Query().flag == filter_flag
        else:
            query_filter = Query().flag.exists()

        res_data = res_tbl.search(query_filter)
        set_data = set_tbl.search(query_filter)

        all_results = pd.concat([pd.DataFrame(res_data).set_index('RUN_ID').drop("flag", axis=1), pd.DataFrame(set_data).set_index('RUN_ID')], axis=1)
        all_results = self._decompose_dict(all_results, 'Model')
        all_results = self._decompose_dict(all_results, 'ML_Preprocessor')
        all_results = self._decompose_dict(all_results, 'Data')
        all_results = self._get_arch_details(all_results)
        all_results['fixes'] = all_results['Load'].apply(lambda x: tuple(x['Preprocessor']['fixes']))
        all_results['weather_variables'] = all_results['weather_variables'].apply(json.dumps)
        all_results['historical_ts'] = all_results['historical_ts'].apply(json.dumps)
        all_results['lr'] = all_results['optimizer'].apply(lambda x: x['lr'])
        all_results['name'] = all_results['optimizer'].apply(lambda x: x['optimizer_name'])
        all_results['optimizer'] = all_results['optimizer'].apply(lambda x: json.dumps({k:v for k,v in x.items() if v is not None}))
        all_results['train_years'] = all_results['train_test'].apply(lambda x:int(x[0]))
        all_results['test_years'] = all_results['train_test'].apply(lambda x:int(x[1]))

        # TODO : Could fix by splitting train_test into train_years, test_years.
        all_results['train_test'] = all_results['train_test'].apply(lambda x:int(x[0]))
        
        all_results.drop(["arch", "weather_data", "lossplot", "tensorboard", "Evaluation", "Load", "Weather", "load_difference"], axis=1, inplace=True)
        
        if detailed:
            dtl_res_tbl = db.table('detailed_results')
            dtl_res_data = dtl_res_tbl.search(query_filter)
            all_results = pd.concat([pd.DataFrame(all_results), pd.DataFrame(dtl_res_data).set_index('RUN_ID').drop("flag",axis=1)], axis=1)
            
        
        return all_results


    def _get_arch_details(self, data):
        """ Add derived features from the model's architecture and add other bits of information """
        data['arch'] = data['arch'].apply(lambda x: json.loads(x))
        data['layer_type'] = data['arch'].apply(lambda x: tuple(y['class_name'] for y in x["config"]))
        data['layer_config'] = data['arch'].apply(self._get_units_per_layer)
        data['average_neuron'] = data['layer_config'].apply(np.mean)
        data['layer_count'] = data['layer_config'].apply(len)
        data['neuron_count'] = data['layer_config'].apply(sum)
        data['hidden_neuron'] = data['layer_config'].apply(lambda x: int(np.mean(x[:-1])))
        data['model'] = data.apply(lambda x: tuple("{0}:{1}".format(l,n) for l,n in zip(x['layer_type'], x['layer_config'])), axis=1)   # Huh that's going to be sloooow, but it's not a big deal here (~100 rows max)
        data['model_type'] = data['layer_type'].apply(lambda x: str(x[0]))
        data['model_summary'] = data.apply(lambda x: x['model_type'] + ":" + str(len(x['layer_config'])) + "x" + str(x['layer_config'][0]), axis=1)

        return data

    def _decompose_dict(self, data, column):
        """ Used a level of dictionnary. 
            Example:
                self.settings["Model"]["Preprocessor"]
                -> loading this would results in having the whole Preprocessor data in a single columns
                => Can decompose the Preprocessor dictionnary so that each of its components are in a separate columns """
        return pd.concat([data.drop(column, axis=1), data[column].apply(pd.Series)], axis=1)

    def _get_units_per_layer(self, x):
        """ Returns the number of units per layer. Handles both CNN (units : 'filter') and RNN / Dense ('units') """
        tmp = []
        for y in x["config"]:
            if 'filters' in y['config']:
                tmp.append(y['config']['filters'])
            else:
                tmp.append(y['config']['units'])

        return tuple(tmp)