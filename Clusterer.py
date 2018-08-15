import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN, MeanShift
import warnings
from shapely.geometry import MultiPoint
from geopy.distance import great_circle

### TODO : Reduce use of instance variables if not stateful
### TODO : Get rid of lat_lon, use pd.all()
class Stations_Clusterer():
    algorithms = ['meanshift', 'dbscan']

    def clusterize(self, stations, desired_station_count=20, cluster_min_size=2, algorithm='meanshift', var=None):
        stations = stations.copy()

        algo_params = self._get_algorithm(algorithm)
        
        # Convert coordinates into suitable format
        coords = stations[['LATITUDE', 'LONGITUDE']].values
        
        # Algorithm hyperparameter variable not provided, find a suitable one. 
        if not var:
            warnings.warn("Clustering hyperparameter not set, proceeding to search.")
            var = self._scan_var(coords, desired_station_count, cluster_min_size, algo_params)
        
        # Call the actual algorithm
        stations['cluster_num'] = algo_params['method'](coords, var)

        # Drop stations that do not belong to any clusters (e.g. meanshift cluster_all = False)
        stations = stations[stations['cluster_num'] != -1]

        # Get centroids for each cluster
        stations['is_centroid'] = self._get_clusters_centroids(stations)

        # Filter clusters with size at least cluster_min_size
        tmp = stations.groupby('cluster_num')['USAF'].count() >= cluster_min_size
        valid_clusters = tmp[tmp == True].index 
        stations = stations[stations['cluster_num'].isin(valid_clusters)]

        print("Desired stations count : {0} -> Obtained : {1}".format(desired_station_count, len(valid_clusters)))

        # Checks that every cluster (':') only has a single centroid ('True')...
        if not ((stations.groupby('cluster_num')['is_centroid'].apply(pd.Series.value_counts)[:, True] == 1).all()):
            print("CLUSTER ERROR WTF")
            import pdb; pdb.set_trace()

        return stations


    def _get_algorithm(self, algorithm):
        algo_params = {'dbscan':{'name':'dbscan', 'search':True, 'var_range':[1,150,500], 'method':self.__get_stations_clusters_dbscan},
                       'meanshift':{'name':'meanshift', 'search':True, 'var_range':[0.025, 5, 300], 'method':self.__get_stations_clusters_meanshift}}

        if algorithm not in self.algorithms:
            warnings.warn("Provided algorithm name {0} unknown, using default (meanshift) !".format(algorithm))
            algorithm = 'meanshift'

        return algo_params[algorithm]


    # TODO : Change name to something that doesn't make user think this method is like _scan_var but parallelized...
    def __parallel_scan_var(self, algorithm_method, coords, var, cluster_min_size):
        return self._get_cluster_count(algorithm_method(coords, var), min_size=cluster_min_size)


    def _scan_var(self, coords, target_cluster_count, cluster_min_size, algo_params):
        var_search = np.linspace(*algo_params['var_range'])

        clusters_counts = [self._get_cluster_count(algo_params['method'](coords, v), min_size=cluster_min_size) for v in var_search]
        # #joblib.Parallel(n_jobs=-1)(joblib.delayed(self.__parallel_scan_var)(algorithm_method=algo_params['method'], coords=coords, var=v, cluster_min_size=cluster_min_size) for v in var_search)
    
        diff = [abs(target_cluster_count-x) for x in clusters_counts]
        print("[{0} variables as good as each others]".format(len(np.where(diff == np.min(diff))[0])))
        
        best_var_idx =  np.argmin(diff) #np.random.choice(np.where(diff == np.min(diff))[0])  # For consistency purposes (when building different datasets)
        best_var = var_search[best_var_idx]
        print("Clusterer hyperparameter : {0}".format(best_var))
        
        return best_var


    def _get_cluster_count(self, clusters, min_size):
        # Drop non clustered stations (not really belonging to cluster '-1' then)
        clusters = np.delete(clusters, np.argwhere(clusters==-1))

        return np.count_nonzero(np.bincount(clusters) >= min_size)


    def _get_clusters_centroids(self, stations):
        clusters_centroids = list(stations.groupby('cluster_num').apply(lambda x: self._get_station_from_cluster(x[['LATITUDE','LONGITUDE']].values)))

        # Boolean vector (True if both 'LATITUDE' and 'LONGITUDE' are simultaneously in 'clusters_centroids')
        return [x.all() for x in np.isin(stations[['LATITUDE', 'LONGITUDE']].values, clusters_centroids)]


    def _get_station_from_cluster(self, cluster):
        # Get geometrical centroid of cluster
        centroid = (MultiPoint(cluster).centroid.x, MultiPoint(cluster).centroid.y)
        
        # Get closest station from cluster centerpoint
        centermost_point = min(cluster, key=lambda point: great_circle(point, centroid).m)
        #print(f"Station {centermost_point} is \n {cluster}'s centroid\n")
        #if np.allclose(centermost_point, [53.5, 2.2]) or np.allclose(centermost_point, [53.5,-3.0669999999999997]):

        return centermost_point


    def __get_stations_clusters_meanshift(self, coords, var):
        clusterer = MeanShift(cluster_all=False, bandwidth=var, n_jobs=1) # Faster than multi-threaded, still prevents parallel execution unfortunately...
        pred =  clusterer.fit_predict(coords)
        
        return pred

    # TODO : FIX OUTPUT to be save as for meanshift
    # Very fast algorithm, can use haversine distance.
    # Not ideal in this case, ends up removing too many stations close together (see how it works to understand why)
    def __get_stations_clusters_dbscan(self, coords, var):
        kms_per_radian = 6371.0088
        epsilon = var / kms_per_radian
        db = DBSCAN(eps=epsilon, min_samples=1, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))
        cluster_labels = db.labels_
        num_clusters = len(set(cluster_labels))
        clusters = pd.Series([coords[cluster_labels == n] for n in range(num_clusters)])

        return clusters