import pandas as pd
import warnings
from abc import ABC
from multiprocessing import cpu_count, Pool
import numpy as np
from Utils import * 
from Clusterer import Stations_Clusterer
import reverse_geocoder as rg
from datetime import datetime
from tqdm import tqdm
from numba import jit

MANUAL_ANOMALY_VALIDATION = False

class FileLoadError(Exception):
    """Exception raised for errors in the file loading process."""
    def __init__(self, message):
        self.message = "File loading error : " + message

# TODO : Handle multiple columns
class Preprocessor(ABC):
    def __init__(self, data):
        self.data = data.copy()
        self.cores = cpu_count()
        self.par_parts = self.cores
        self.uncorrected_data = None
        self.errors = {'zeros':0, 'extremes_global':0, 'extremes_hourly':0, 'derivatives':0}

    def to_std_fields(self, var, year_range):
        """ Changes field name to a standard name sorts data by time and filter to desired year range """ 
        self.data.rename(columns={var:"TS"}, inplace=True)
        self.data = self.data[['Period', 'TS']]

        # Sort data... some datasets have out of order time...
        self.data = self.data.sort_values('Period')

        if year_range is not None:
            self._filter_years(year_range)

    # TODO : Should be in Utils
    def _filter_years(self, year_range):
        # Available years in dataset
        available_years = self.data["Period"].dt.year.unique()
        years = np.intersect1d(available_years, year_range)
        self.data = self.data.loc[self.data["Period"].dt.year.isin(years)].reset_index(drop=True)  # Filter data and reset index to start at 0

    def apply_parallel(self, function):
        """ Apply a function using multithreading """
        data_split = np.array_split(self.data, self.par_parts)
        pool = Pool(self.cores)
        data = pd.concat(pool.map(function, data_split))
        pool.close()
        pool.join()
        
        return data
     

    ### DETECTORS (see report for explantion)
    def zeros_detector(self):
        to_fix = self.data.loc[self.data["TS"] == 0].index
        self.errors['zeros'] += len(to_fix)

        return chunk(to_fix)

    def extremes_hourly_detector(self, threshold):
        M = threshold[0]
        Q = threshold[1]
        
        hours_data = self.data.groupby([self.data['Period'].dt.hour])
        hours_max_pctl, hours_min_pctl = hours_data.quantile(Q).values, hours_data.quantile(1-Q).values
        
        self.data['max_hourly_pctl'] = self.data['Period'].apply(lambda x: hours_max_pctl[x.hour][0])
        self.data['min_hourly_pctl'] = self.data['Period'].apply(lambda x: hours_min_pctl[x.hour][0])
        
        to_fix = self.data.loc[(self.data["TS"] > self.data['max_hourly_pctl']*M)  | (self.data["TS"] < self.data['min_hourly_pctl']/M)].index
        self.errors['extremes_hourly'] += len(to_fix)
        
        self.data.drop("max_hourly_pctl", axis=1, inplace=True)
        self.data.drop("min_hourly_pctl", axis=1, inplace=True)
        
        print(self.data.iloc[to_fix[0]])

        return chunk(to_fix)
        
    def extremes_detector(self, threshold):
        M = threshold[0]
        Q = threshold[1]
        to_fix = self.data.loc[(self.data["TS"] > self.data.quantile(Q)["TS"]*M) | (self.data["TS"] < self.data.quantile(1-Q)["TS"]/M)].index
        self.errors['extremes_global'] += len(to_fix)

        return chunk(to_fix)
    
    def derivatives_detector(self, threshold):
        d_pos = 'abs_diff_perc_pos'
        d_neg = 'abs_diff_perc_neg'
        Q = threshold[1]
        M = threshold[0]
        self.data[d_pos] = abs(self.data['TS'].diff() / self.data['TS'])
        self.data[d_neg] = abs(self.data['TS'].diff(periods=-1) / self.data['TS'])

        to_fix = self.data.loc[(self.data[d_pos] > self.data.quantile(Q)[d_pos]*M) & (self.data[d_neg] > self.data.quantile(Q)[d_neg]*M)].index
        self.errors['derivatives'] += len(to_fix)
        
        self.data.drop(d_pos, axis=1, inplace=True)
        self.data.drop(d_neg, axis=1, inplace=True)
        
        return chunk(to_fix)

    ### CORRECTOR
    def fix_outliers(self, to_fix, strat="interpolate", max_error_width=3):
        """ Correct errors returned by a detector (to_fix) using the given strategy (strat) """
        refused = []
        to_fix2 = list(filter(lambda x: len(x) <= max_error_width, to_fix))
        refused.extend([x for x in to_fix if x not in to_fix])
        to_fix = to_fix2

        for x in to_fix:
            ok = True
            if strat == "interpolate":
                min_, max_ = (min(x)-1, max(x)+1)
                new_value_vector = np.interp(x, [min_, max_],[self.data.iloc[min_]["TS"], self.data.iloc[max_]["TS"]])
                
                if MANUAL_ANOMALY_VALIDATION: ok = validate_correction(self.data, x, new_value_vector)
                if ok: 
                    self.data.iloc[x, self.data.columns.get_loc("TS")] = new_value_vector
                    if MANUAL_ANOMALY_VALIDATION : print("Correction accepted.")
                else:
                    refused.append(x)

            elif strat == 'nan':
                self.data.iloc[x, self.data.columns.get_loc("TS")] = np.nan
            elif strat == 'month_hourly_average':
                raise NotImplementedError()

            else:
                raise NotImplementedError("Time series correction strategy {0} not available.".format(strat))

        return refused

    def fix_data(self, fixes=["zeros"], max_error_width=3, threshold=(1.5,0.99), strat="interpolate"):
        """ Top-level method to correct data """
        if not self.uncorrected_data:
            self.uncorrected_data = self.data.copy()
            
        not_fixed = []

        if "zeros" in fixes:
            zeros = self.zeros_detector()
            not_fixed.extend(self.fix_outliers(zeros, strat=strat))
        
        if "extremes_hourly" in fixes:
            extremes = self.extremes_hourly_detector(threshold)
            not_fixed.extend(self.fix_outliers(extremes, strat=strat))
            
        if "extremes_global" in fixes:
            extremes = self.extremes_detector(threshold)
            not_fixed.extend(self.fix_outliers(extremes, strat=strat))
            
        if "derivatives" in fixes:
            derivatives  = self.derivatives_detector(threshold)
            not_fixed.extend(self.fix_outliers(derivatives, strat=strat))

        if not all([s in ['zeros', 'extremes_global', 'extremes_hourly', 'derivatives'] for s in fixes]):
            warnings.warn("Unknown detection strategy") 
        
        # Manual correction of unfixed errors
        for x in not_fixed:
            if MANUAL_ANOMALY_VALIDATION : 
                from Plotter import plot_around
                plot_around(self.data.iloc[x[0]]['Period'], self.data, title=str(x[0]))

        print('Detected errors : {0} \n{1}'.format(sum(self.errors.values()), self.errors))

    def get_data(self):
        # Write frequency into dataframe (with DateTimeIndex)  ### NOT USED by other methods yet.
        freqs = self.data["Period"].diff().value_counts()
        data = self.data.set_index("Period")
        data.freq = (freqs == freqs.iloc[0]).idxmax()

        return data


class NG_Preprocessor(Preprocessor):
    settlement_period_minutes = 30

    def __init__(self, data=None, var=None, year_range=None):
        self.year_range = year_range

        if all([x is not None for x in [data, var]]):
            self.load(data, var, year_range)

    def load(self, data, var, year_range=None):
        if not year_range:
            year_range = self.year_range

        super().__init__(data)
        self.to_std_fields(var, year_range)

    # Basic preprocessing (filter var, add Period column etc)
    def to_std_fields(self, var, year_range):
        self._fix_daylight()
        self.add_date_field()
        super().to_std_fields(var, year_range)
        
    def _set_period_to_hour(self, minutes):
        """ Transforms time information (<date>, 30 minutes interval integer (e.g. 32 -> 4pm)) """
        minutes *= self.settlement_period_minutes
        return "{0}:{1}".format(minutes//60,minutes%60)
    
    def _to_date(self, set_date, set_period):
        return datetime.strptime("{0} {1}".format(set_date, self._set_period_to_hour(set_period-1)), "%d-%b-%Y %H:%M")
    
    def _fix_daylight(self):
        """ Correct data containing too few or too many records due to daylight saving times """
        # Clock advances -> Duplicate 46 to 47 and 48. More logically could use 1 and 2 of next days, but date indexing not possible at this point.
        incomplete_days = pd.Series(self.data.groupby(['SETTLEMENT_DATE'], as_index=False).size() == 46)
        incomplete_days = incomplete_days[incomplete_days == True].index

        for day in incomplete_days:
            record_46 = self.data[(self.data['SETTLEMENT_DATE'] == day) & (self.data['SETTLEMENT_PERIOD'] == 46)].to_dict(orient='records')[0]
            record_47 = record_46.copy()
            record_48 = record_46.copy()
            record_47['SETTLEMENT_PERIOD'] = 47
            record_47['SETTLEMENT_PERIOD'] = 48
            self.data = self.data.append(record_47, ignore_index=True)
            self.data = self.data.append(record_48, ignore_index=True)

        # Clock goes back again -> Remove 49 and 50 (could average or something, but nothing is truly true anyway)
        self.data.drop(self.data[self.data['SETTLEMENT_PERIOD'] > 48].index, inplace=True)

    def _add_date_field_parallel(self, data):
        return data.apply(lambda x: self._to_date(x['SETTLEMENT_DATE'], x['SETTLEMENT_PERIOD']), axis=1)
    
    def add_date_field(self):
        self.data['Period'] = self.apply_parallel(self._add_date_field_parallel)
        self.data.drop('SETTLEMENT_DATE', axis=1, inplace=True)
        self.data.drop('SETTLEMENT_PERIOD', axis=1, inplace=True)
        self.data.sort_values('Period', inplace=True)

class NY_Preprocessor(Preprocessor):
    settlement_period_minutes = 60

    def __init__(self, data=None, var=None, year_range=None, fix_duplicates='merge'):
        self.year_range = year_range

        if all([x is not None for x in [data, var]]):
            self.load(data, var, year_range, fix_duplicates)

    def load(self, data, var, year_range=None, fix_duplicates='merge'):
        if not year_range:
            year_range = self.year_range

        super().__init__(data)
        self.to_std_fields(var, year_range, fix_duplicates)

    # Basic preprocessing (filter var, add Period column, remove duplicates, replace nans etc)
    def to_std_fields(self, var, year_range, fix_duplicates):
        self.add_date_field()
        super().to_std_fields(var, year_range)
        self._replace_nan()
        self._remove_duplicates(fix_duplicates)        
        
    def add_date_field(self):
        self.data['Period'] = pd.to_datetime(self.data["Time Stamp"], infer_datetime_format=True)

    def _replace_nan(self):
        self.data['TS'].fillna(0, inplace=True)
        
    def _remove_duplicates(self, fix_duplicates):
        if fix_duplicates == 'merge':
            duplicated = self.data[self.data.duplicated(keep=False, subset="Period")].groupby("Period")

            for time, rec in tqdm(duplicated):
                values = rec['TS'].values
                values = [x for x in values if x != 0]
                
                if len(values):
                    new_val = sum(values)/len(values)
                else:
                    new_val = 0 # Will be fixed later 
                    
                # Replace first record with new value
                self.data.iloc[rec.index[0], self.data.columns.get_loc("TS")] = new_val
            
            # Keeps first duplicated row 
            self.data = self.data.drop_duplicates(subset="Period").reset_index(drop=True)

        # Keep last only (e.g. for NYISO day ahead forecasts)
        elif fix_duplicates == 'keep_last':
            self.data.drop_duplicates(subset='Period', keep='last', inplace=True)
            self.data.reset_index(inplace=True, drop=True)
        else:
            raise ValueError("Invalid duplicates correction method name. Use 'merge' or 'keep_last'")


# To ABC ?
class NCDC_Stations_Preprocessor:
    def __init__(self):
        print("Filter stations by data availability")
        # Keep stations for which we ACTUALLY have data for...
        self.stations = self.stations[self.stations['USAF'].isin(self._data['USAF'].unique())]

        # ... Some GPS coordinates have floating point errors (e.g 1.099999999999964), which appears rounded but still prevents strict comparison. 
        # numpy allclose would work, but does need arrays to have the same shape...
        self.stations = self.stations.round(3)

    # 6% Execution time
    def filter_stations(self, coords_to_drop, countries_to_keep, data_quality=('2h', 98.5)):

        if coords_to_drop:
            self._drop_stations_by_coords(coords_to_drop)

        if countries_to_keep:
            self._drop_stations_by_countries(countries_to_keep)

        if data_quality:
            self._drop_stations_by_data_availability(*data_quality)


    def _drop_stations_by_coords(self, stations_to_drop):
        stations_to_drop = np.array(stations_to_drop)

        # Not using ~ : Not working with empty arrays
        # Stations not found in self.stations
        not_found = stations_to_drop[np.logical_not(np.array([x.all() for x in np.isin(stations_to_drop, self.stations[['LATITUDE', 'LONGITUDE']].values)]))]

        self.stations = self.stations[np.logical_not(np.array([x.all() for x in np.isin(self.stations[['LATITUDE', 'LONGITUDE']].values, stations_to_drop)]))]

        for st in not_found:
            print("Station {0} not found in dataset !".format(st))

    # 4.6% Execution time
    def _drop_stations_by_countries(self, countries_to_keep):
        self.add_stations_country()
        self.stations.drop(self.stations[~self.stations['country'].isin(countries_to_keep)].index, axis=0, inplace=True)

    # 2% Execution time
    def _drop_stations_by_data_availability(self, max_gap='1h', min_quality=98.5):
        stats = {}
        max_gap = pd.Timedelta(max_gap)
        
        # Analyse data from each stations, only keep those with best time consistency (at least 'min_quality' % of time intervals are < 'max_gap')
        for st in self._data.groupby('USAF'):
            a = st[1]['Period'].diff().value_counts()
            stats[st[0]] = a.loc[a.index <= max_gap].sum() / len(st[1]) * 100.0

        # Keep data from stations with more than 98.5% with 2 houror  less gap datapoints
        good_stations_USAF = [k for k,v in stats.items() if v > min_quality]

        self.stations = self.stations[self.stations['USAF'].isin(good_stations_USAF)]
        self._data = self._data[self._data['USAF'].isin(good_stations_USAF)]

    # 4.6% Execution time
    def add_stations_country(self):
        coords = tuple(map(tuple, self.stations[['LATITUDE', 'LONGITUDE']].values))
        self.stations['country'] = [x['admin1'] for x in rg.search(coords)]


    # TODO : Make Stations_Clusterer behave more like a method than a class
    def clusterize_stations(self, desired_stations_count, cluster_min_size=1, algorithm='meanshift', var=None):
        self.stations = Stations_Clusterer().clusterize(self.stations, desired_stations_count, cluster_min_size, algorithm, var)


# TODO : Pass location for mapping
class NCDC_Data_Preprocessor:
    # 17% Execution time
    def __init__(self, year_range=None):
        print("Date format conversion")
        # Convert period information into a suitable format
        # Only necessary for New York data
        self._data['HrMn'] = self._data['HrMn'].str.zfill(4)

        self._data['Period'] = pd.to_datetime(self._data.Date.str.cat(self._data.HrMn, sep=' ')) #14% Execution time by str stuff
        self._data.drop(['HrMn', 'Date'], axis=1, inplace=True)
        
        # Handle duplicates (New York only - freaking daylight saving time)
        # Improvements : Do not just drop, average. Though they are 1 hour away measurements anyway...
        self._data = self._data.drop_duplicates(subset=["USAF","Period"]).reset_index(drop=True)

        # Memory usage (data) before optimization = 875MB (15.25s up to this point - ~1m33 total weather data processing)
        # Memory usage (data) after optimization = 255MB (10.82s up to this point - ~1m20 total weather data processing)

        if year_range is not None:
            self._filter_years(year_range)

    # TODO : Should be in Utils
    def _filter_years(self, year_range):
        # Available years in dataset
        available_years = self._data["Period"].dt.year.unique()
        years = np.intersect1d(available_years, year_range)

        self._data = self._data.loc[self._data["Period"].dt.year.isin(years)]
        
    # 24% Execution time
    def preprocess_data(self, combine_cluster_data=True):
        print("Pre-processing data")

        # Only keep data from stations we actually have data for
        self._data = self._data[self._data['USAF'].isin(self.stations['USAF'])].reset_index(drop=True)

        # Replace large values (1.3 x larger than the 0.99 percentile)
        quantiles = {'high': self._data.quantile(0.99), 'low': self._data.quantile(0.01)}
        for col in self._data.columns.difference(['USAF', 'Period']):
            # 13% Execution time
            self._data.loc[(self._data[col] > quantiles['high'][col]*1.3) | (self._data[col] < quantiles['low'][col]/1.3), col] = np.nan
    
        # 12% Execution time
        self.pivot_data = {v:self._data.pivot(index='Period', columns='USAF', values=v) for v in self._data.columns.difference(['USAF'])}

        if combine_cluster_data:
            self._combine_cluster_data()

    # 11% Execution time
    def _combine_cluster_data(self):
        print("Combining cluster data")

        # Check that clusterization was done
        if 'cluster_num' in self.stations.columns:
            # For each variables (e.g. Air_Temp)
            for var in self.pivot_data.keys():
                # Combine data from each station in the cluster
                for clu in self.stations.groupby('cluster_num'):
                    usaf = clu[1]['USAF'].values
                    self.pivot_data[var][clu[0]] = self.pivot_data[var][usaf].mean(axis=1)

                    ## TODO : SPEED UP by doing one shot rename and drop
                    # Rename columns [0..N] to [var_0..var_N]   (e.g. [0,1..N] to [Air_Temp_0, Air_Temp_1..Air_Temp_N])
                    self.pivot_data[var].rename(columns={clu[0]:var+"_"+str(clu[0])}, inplace=True)

                    # 9.4% Execution time ?
                    self.pivot_data[var] = self.pivot_data[var].drop(usaf, axis=1)#, inplace=True)

        else:
            warning.warns("Attemp to combine non clustered stations !")

    # 46% total execution time
    def postprocess_data(self, min_quality, resample):
        """ Corrects last errors based on the gap size (number of consecutive errors) 
        Note : the time step is very small here (Union of weater stations time step : ~ 1 minute) 
        """
        print("Post-processing data")

        d = self.pivot_data.copy()

        for var in d.keys():

            d[var] = d[var].resample('5min').asfreq()
            # TODO : Replace by gap size limited interpolation -> Interpolates replaces up to 48, but desired behavior is to *NOT* replace anything if more than 48 consecutive errors. 
            # Use Preprocessor class
            # 27% Execution time
            d[var].interpolate(method='time', limit=48, limit_area='inside', inplace=True)
            d[var] = d[var].resample(resample).asfreq()

            # Drop clusters with too many nan
            to_drop = ((d[var].isna().sum() / len(d[var]) * 100.0) > 100.0-min_quality)
            d[var].drop(to_drop[to_drop == True].index, axis=1, inplace=True)   

            # Repair remaining nans

            # Chunk method only works with integer indexes
            d[var].reset_index(inplace=True)

            for c in d[var].columns.difference(['Period']):
                # Gap sizes calculation
                uuu = d[var][c].isna()
                gaps = chunk(uuu[uuu == True].index)
                short_gap_limit = 12

                # Short gaps (<=6H) -> Interpolate
                to_fix = list(filter(lambda x: len(x) <= short_gap_limit, gaps))
        
                for x in to_fix:
                    if 0 in x or len(d[var])-1 in x:
                        #print("Start or end is nan, can't interpolate, replacing by hourly average...")
                        continue

                    min_, max_ = (min(x)-1, max(x)+1)
                    new_value_vector = np.interp(x, [min_, max_],[d[var].iloc[min_][c], d[var].iloc[max_][c]])
                    d[var].loc[x, c] = new_value_vector
                

                # Larger gaps + those not fixed above -> Replace by same month hourly average. Improvement : Use surrounding X days -> Manual = Slow + long to implement.
                uuu = d[var][c].isna()
                to_fix = uuu[uuu == True].index

                if len(to_fix):
                    # Get hourly mean for each month for each year
                    p = d[var]["Period"].dt
                    hours_data = d[var][c].groupby([p.year.rename("Year"), p.month.rename("Month"), p.hour.rename("Hour")]).mean()

                    ###### GENIUS BUG, can I get my 2 hours back ???
                    #### Setting an empty Series to an empty "loc" (for rows), just does random stuff, like replacing the whole column 
                    ### (while none of the rows were actually selected) to a datetime with random microseconds values. FUN.
                    
                    # 1.4% Execution time
                    d[var].loc[to_fix, c] = d[var].loc[to_fix, 'Period'].apply(lambda x: hours_data[x.year, x.month, x.hour])   

            d[var].set_index("Period", inplace=True)

        self.pivot_data = d
        

# Top level class
class NCDC_Preprocessor(NCDC_Stations_Preprocessor, NCDC_Data_Preprocessor):
    def __init__(self, NCDC_Loader=None, year_range=None):
        self.year_range = year_range

        if NCDC_Loader is not None:
            self.load(NCDC_Loader, year_range)

    # 16% Execution time
    def load(self, NCDC_Loader, year_range=None):
        if not year_range:
            year_range = self.year_range

        self.stations, self._data = NCDC_Loader.stations, NCDC_Loader.data
        NCDC_Stations_Preprocessor.__init__(self)
        NCDC_Data_Preprocessor.__init__(self, year_range)

    # 78% Execution time
    def fix_data(self, stations_to_drop, countries_to_keep, pre_clustering_data_quality, 
                desired_cluster_count, min_cluster_size, resample, algorithm=None, var=None, post_clustering_data_quality=0.98):

        # Plot all stations, prior to any processing (UK only)
        #self.plot_stations_map('M')

        self.filter_stations(stations_to_drop, countries_to_keep=countries_to_keep, data_quality=pre_clustering_data_quality)
        
        if algorithm:
            self.clusterize_stations(desired_cluster_count, min_cluster_size, algorithm, var)

        self.preprocess_data(combine_cluster_data=algorithm != None)
        self.postprocess_data(min_quality=post_clustering_data_quality, resample=resample)

        # Plot stations after filtering and clustering  (UK only)
        #self.plot_stations_map('M')

    def plot_stations_map(self, detail_level='M'):
        from Mapper import Mapper
        mp = Mapper()
        mp.show_stations(self.stations, detail_level=detail_level, show_centroids_only=False)