import numpy as np
from Utils import *
import pandas as pd
from datetime import datetime



class ML_Preprocessor():
    """ Machine learning specific data transformations class """
    def __init__(self, load, weather_preprocessor=None, y_vars=["TS"]): 
        """ Args:
                load : Load DataFrame the a Loader derived class 
                weather_preprocessor : NCDC weather preprocessor *object*
        """ 
        if weather_preprocessor is not None:
            self.data = load.join(list(weather_preprocessor.pivot_data.values()))
        else:
            self.data = load

        self.y_vars = y_vars
        self._update_X_vars()


    def _update_X_vars(self):
        self.X_vars = self.data.columns.difference(self.y_vars)

    def encode_vars(self, cat_vars=['Month', 'Day', 'Hour', 'Minute', 'DoW']):
        """ Categorical one hot encoding """
        for var in np.intersect1d(cat_vars, self.data.columns):
            self.data = pd.get_dummies(self.data, columns=[var], drop_first=True)

        # TODO : Make simpler
        self.X_vars = [x for x in self.data.columns if x.split("_")[0] in [x.split("_")[0] for x in self.X_vars]]
        self.y_vars = [y for y in self.data.columns if y.split("_")[0] in [x.split("_")[0] for x in self.y_vars]]


    def std_vars(self, stdz='MinMaxScaler', test_set=True):
        """ Data normalization """
        sc_X = get_scaler(stdz)()
        sc_y = get_scaler(stdz)()
        
        self.data[self.X_vars] = sc_X.fit_transform(self.data[self.X_vars])
        self.data[self.y_vars] = sc_y.fit_transform(self.data[self.y_vars])

        return sc_X, sc_y


    def get_cv_splits(self, train_test=(2,1), array=False):
        """
            Returns cross-validation splits following the 'train_test' settings 
            Todo :
                * Move to ML_Utils ?
        """
        # Available years in dataset
        years = self.data.index.year.unique()

        tr, te = train_test

        splits = []

        X = self.data[self.X_vars]
        y = self.data[self.y_vars]

        for n in range(len(years) - sum(train_test)+1):
            train_years = years[n:n+tr]
            test_years = years[n+tr:n+tr+te]

            X_train = X.loc[X.index.year.isin(train_years)]
            X_test = X.loc[X.index.year.isin(test_years)]

            y_train = y.loc[X.index.year.isin(train_years)]
            y_test = y.loc[X.index.year.isin(test_years)]

            if array:
                X_train, y_train, X_test, y_test = list(map(get_narray, [X_train, y_train, X_test, y_test]))

            splits.append([(X_train, y_train), (X_test, y_test)])

        return splits


    def transform_load_difference(self):
        """ UNUSED -- IGNORE """
        if any(["Prev_" in col for col in self.data.columns]):
            warnings.warn("Call method 'transform_load_difference()' before 'add_historical_ts()' !")

        start = self.data.loc[min(self.data.index), "TS"]

        self.data["TS"] = self.data["TS"].diff()
        self.data.dropna(inplace=True)

        return start


    def encode_time(self, time_encoding='categorical'):
        """ Time encoding : categorical / cyclical """
        self.data['Year'] = self.data.index.year

        if time_encoding == 'categorical':
            self.data['Day'] = self.data.index.day
            self.data['DoW'] = self.data.index.dayofweek
            self.data['DoY'] = self.data.index.dayofyear
            self.data['Month'] = self.data.index.month

            self.data['Hour'] = self.data.index.hour
            self.data['Minute'] = self.data.index.minute

        elif time_encoding == 'cyclical':
            sec_per_day = 60*60*24
            self.data['seconds_into_day'] = self.data.index.hour * 3600 + self.data.index.minute * 60

            # Encoding 00h00->23h59 to cos / sin
            self.data['sin_hour'] = np.sin(2*np.pi*self.data["seconds_into_day"]/sec_per_day)
            self.data['cos_hour'] = np.cos(2*np.pi*self.data["seconds_into_day"]/sec_per_day)
            self.data.drop('seconds_into_day', axis=1, inplace=True)
            
            # Encoding 1->31 to cos / sin
            self.data['sin_day_in_month'] = np.sin(2*np.pi*self.data.index.day/self.data.index.daysinmonth)
            self.data['cos_day_in_month'] = np.cos(2*np.pi*self.data.index.day/self.data.index.daysinmonth)

            # Encoding 0->6 to cos / sin
            self.data['sin_day_in_week'] = np.sin(2*np.pi*self.data.index.dayofweek/7)
            self.data['cos_day_in_week'] = np.cos(2*np.pi*self.data.index.dayofweek/7)

            # Encoding 1->365 to cos/sin (ignoring leap years, shouldn't be a big deal)
            self.data['sin_day_in_year'] = np.sin(2*np.pi*self.data.index.dayofyear/365)
            self.data['cos_day_in_year'] = np.cos(2*np.pi*self.data.index.dayofyear/365)

            # Encoding 1->12 to cos/sin
            self.data['sin_month'] = np.sin(2*np.pi*self.data.index.month/12)
            self.data['cos_month'] = np.cos(2*np.pi*self.data.index.month/12)            

        self._update_X_vars()


    def add_historical_ts(self, points, averages=None, dropna=True):
        """ Forward data propagation """
        shifts = []

        records_per = {}
        records_per['r'] = 1
        records_per['h'] = len(self.data.loc[(self.data.index >= self.data.index[0]) & (self.data.index < self.data.index[0] + pd.Timedelta('1h'))])
        records_per['d'] = records_per['h'] * 24

        # TODO : Add same week day, same week, previous year ?
        for pt in points:
            pt_type = records_per.get(pt[-1])
            if pt_type is None:
                continue

            pt = pt[:-1]
            if '-' not in pt:
                pts = [int(pt)]
            else:
                pt2 = pt.split('-')
                pts = list(range(int(pt2[0]), int(pt2[1])+1))

            shifts.extend([x*pt_type for x in pts])

        for n in shifts:
            self.data["Previous_"+str(n)] = self.data['TS'].shift(n)


        if isinstance(averages, list):
            # Handle average history
            days_avg = self.data['TS'].groupby([self.data.index.year.rename('Year'), 
                                                self.data.index.month.rename('Month'),
                                                self.data.index.day.rename('Day')]).mean()

            # Flatten multi-index...
            days_avg.index = [datetime(x[0], x[1], x[2]) for x in days_avg.index]
            days_avg = days_avg.to_dict()

            for av in averages:
                if av[-1] != 'd':
                    continue

                day = int(av[:-1])
                self.data["Avg_"+str(day)] = self.data.index.to_series().apply(lambda x: days_avg.get(pd.Timestamp(x.year, x.month, x.day) - pd.Timedelta(str(day)+' days')))

        if dropna:
            self.data.dropna(inplace=True)

        self._update_X_vars()


    def get_xy(self):
        return self.data[self.X_vars].values, self.data[self.y_vars].values.ravel()