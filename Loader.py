"""
This module hosts class to load the electricity load / demand data files along with the weather data.
"""
import pandas as pd
import warnings
from abc import ABC
import os
import numpy as np

debug = True


class FileLoadError(Exception):
    """Exception raised for errors in the file loading process."""
    def __init__(self, message):
        self.message = "File loading error : " + message  

# TODO : Add dtypes and use_cols
class Loader(ABC):        
    """ Base loader class. Data is first temporarily stored then concatenated to form the final data. """
    def __init__(self, loc=None):
        self.data = pd.DataFrame()
        self.loaded_files = []
        self.tmp_data = []
        self.tmp_loaded_files = []
        self.dirty = False
        
        if loc:
            self.load(loc)
         
    def _concat_data(self, mode="append"):
        """ Conctatenates all loaded files into the final DataFrame """
        if mode == "append":
            self.tmp_data.append(self.data)
            self.loaded_files.extend(self.tmp_loaded_files)
        else:
            self.loaded_files = self.tmp_loaded_files
        
        self.tmp_loaded_files = []
        
        self.data = pd.concat(self.tmp_data)
        self.tmp_data = []
        self.dirty = False
        

    def _load_folder(self, folder):
        """ Loads all file from a folder """
        for f in os.listdir(folder):
            self._load_file(os.path.join(folder, f)) 
                

    def load(self, path, mode="append"):
        """ Top level file loading method, handles both file and folder path """
        if mode not in ["append", "replace"]:
            raise FileLoadError("Provided load mode ({mode}) isn't valid - only 'append' and 'replace' are allowed.")
            
        if type(path) == str:
            path = [path]

        for f in path:
            if os.path.isdir(f):
                self._load_folder(f)

            elif os.path.isfile(f):
                self._load_file(f)
            else:
                raise FileLoadError("Provided file/folder path ({0}) doesn't exist".format(path))

        if not self.dirty:
            raise FileLoadError("No valid data found in file/folder '{0}'".format(path))
        else:
            self._concat_data(mode)
            
    def _load_file(self, filepath, load_columns):
        """ Try to load a given file - typically called by subclasses """
        try:
            self.tmp_data.append(pd.read_csv(filepath)[load_columns])
        except FileNotFoundError:
            warnings.warn("Skipping file {0} : Not a valid CSV file".format(filepath), Warning)
        else:
            self.tmp_loaded_files.append(filepath)
            self.dirty = True
            
    
    def get_data(self):
        return self.data


class NG_Loader(Loader):
    """ National Grid UK demand data loader - No dataset specific methods """
    def _load_file(self, filepath, load_columns=['SETTLEMENT_DATE', 'SETTLEMENT_PERIOD', 'ND', 'ENGLAND_WALES_DEMAND']):        
        super(NG_Loader, self)._load_file(filepath, load_columns)


class NY_Loader(Loader):
    """ NYISO load data loader """
    def _load_file(self, filepath, load_columns=['Time Stamp', 'Integrated Load']):
        super(NY_Loader, self)._load_file(filepath, load_columns)
    
    def split_zones(self, folder=None):
        """ Splits the single block data (all zones in same dataset) into a single file per zone """ 
        if len(self.data):
            if folder is None:
                folder = "/".join(self.loaded_files[0].split("/")[:-1])
            
            if not os.path.isdir(folder):
                os.makedirs(folder)
                
            grouped = self.data.groupby('Name')
            for name, group in grouped:
                group.to_csv(os.path.join(folder, name+".csv"), index=False)
        else:
            warnings.warn("No data loaded yet, nothing to split !")
            

# TODO : Could improve / inherit
# No support for multi loading (multiple calls to load multiple files separately) 
# Not going to be flexible due to limited time 
# Could inherit, but a little bit too different (multiple type files) to be worth it.
# 6.5% Execution time
class NCDC_Loader():
    """ NCDC weather data loader 
        Todo : 
            * Inherit from base Loader class 
    """
    def __init__(self, filepath_stations=None, filepath_data=None, load_columns=['W_Spd', 'Air_Temp', 'Dew_temp']):
        """ Load data directly from init if file paths given """
        if all([filepath_stations, filepath_data]):
            self.load(filepath_stations, filepath_data, load_columns=load_columns)

    def load(self, filepath_stations, filepath_data, load_columns=['W_Spd', 'Air_Temp', 'Dew_temp']):
        """ Loads both weather station list and weather data """
        # np.int8 for USAF breaks some code as does np.float16 for float variables 
        col_types = {'USAF':str, 'Date':str, 'HrMn':str, 'W_Spd':np.float32, 'Air_Temp':np.float32, 'Dew_temp':np.float32, 'RHx':int, 'LATITUDE':float, 'LONGITUDE':float}

        self.stations = pd.read_csv(filepath_stations, index_col=False, delim_whitespace=True, usecols=['USAF', 'LATITUDE', 'LONGITUDE'], dtype=col_types)

        tmp_data = []
        for f in filepath_data:
            tmp_data.append(pd.read_csv(f, index_col=False, usecols=['USAF', 'Date', 'HrMn']+load_columns, dtype=col_types))

        self.data = pd.concat(tmp_data)

    def get_data(self):
        return self.data