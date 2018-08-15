import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import warnings
from shapely.geometry import Point

class Mapper():
    def __init__(self, country='UK'):
        self.country = country 
        self.clusters = {}

    def _load_map(self, show_cities, detail_level, color_scheme=None):

        # Limited use 
        lods = {'H':10, 'M':50, 'L':110}

        # Map color scheme
        if not color_scheme:
            color_scheme = {'world':'xkcd:marine blue', 'urban':'xkcd:dark blue', 'cities':'xkcd:medium grey', 'oceans':'xkcd:light blue'}

        if detail_level not in lods:
            warnings.warn("Provided level of detail {0} doesn't exist - default (low) used.".format(detail_level))

        lod = lods.get(detail_level, 'L') # Default lod

        world = gpd.read_file("/media/jonathan/DATA/HW/Project/DATA/METAR_DATA/geopandas/ne_10m_land.shp")
        urban = gpd.read_file("/media/jonathan/DATA/HW/Project/DATA/METAR_DATA/geopandas/urban{0}.shp".format(lod))
        bounds = gpd.read_file("/media/jonathan/DATA/HW/Project/DATA/METAR_DATA/geopandas/bound{0}.shp".format(lod))
        oceans = gpd.read_file("/media/jonathan/DATA/HW/Project/DATA/METAR_DATA/geopandas/ocean50.shp")

        world.plot(ax=self.ax, color=color_scheme['world'], edgecolor='black', linewidth=0.7)
        urban.plot(ax=self.ax, color=color_scheme['urban'])
        bounds.plot(ax=self.ax, color=color_scheme['world'], edgecolor='black', linewidth=0.7)
        oceans.plot(ax=self.ax, color=color_scheme['oceans'])

        if show_cities:
            cities = self._get_cities(large_cities=14)
            cities.plot(ax=self.ax, marker='D', color=color_scheme['cities'], markersize=10)

            default_loc = ('top', 'left')
            cities_text_loc = {'Glasgow':('bottom', 'left'), 
                               'Belfast':('bottom', 'left'), 
                               'Coventry':('bottom','right'),
                               'Bristol':('bottom', 'right'),
                               'Liverpool':('bottom', 'left'),
                               'Birmingham':('bottom', 'left'),
                               'Sheffield':('top', 'right')}

            def reverse(n):
                dirs = {'right':'left', 'left':'right', 'top':'bottom', 'bottom':'top'}
                return dirs[n]

            for x in cities.iterrows():
                name = x[1]['NAME']
                plt.annotate(x[1]['NAME'], (x[1]['geometry'].x, x[1]['geometry'].y), fontstyle='italic', fontweight='bold', 
                color=color_scheme['cities'], xytext=(1.5,-1), textcoords='offset points', 
                horizontalalignment=reverse(cities_text_loc.get(name, default_loc)[1]), 
                verticalalignment=reverse(cities_text_loc.get(name, default_loc)[0]))


    def _get_cities(self, large_cities=14):
        cities = gpd.read_file("/media/jonathan/DATA/HW/Project/DATA/METAR_DATA/pop10.shp")
        # Terminal (xfce4) freeze when printing Vietnamese characters... 
        cities.drop('name_vi', axis=1, inplace=True)
        cities_uk = cities[cities['ADM0_A3'] == 'GBR']

        # Keep capital cities
        main_cities = cities_uk.loc[cities_uk['FEATURECLA'].str.contains('Admin-0')]        

        # Keep largest cities (MIN -> ignore holiday destinations)
        large_cities = cities_uk.sort_values(by="POP_MIN", ascending=False).iloc[0:large_cities]

        cities_uk = pd.concat([main_cities, large_cities]).drop_duplicates('NAME').reset_index(drop=True)

        return cities_uk

    # TODO : FIX, loading map background a second time doesn't work
    def show_stations(self, stations_df, detail_level="M", country='UK', show_centroids_only=True, show_cities=True):

        self.fig, self.ax = plt.subplots()
        self.ax.set_aspect('equal')

        self._load_map(show_cities, detail_level)

        stations_df['geometry'] = stations_df.apply(lambda x: Point(x['LONGITUDE'], x['LATITUDE']), axis=1)

        # Stations out of "Weather_Utils.cluster_station" method
        if pd.Series(['is_centroid', 'cluster_num']).isin(stations_df.columns).all():
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

            if not show_centroids_only:
                non_centroids = stations_df.loc[stations_df['is_centroid'] == False]
                non_centroids = non_centroids.groupby('cluster_num')

                for x in non_centroids:
                    gpd.GeoSeries(x[1]['geometry']).plot(ax=self.ax, marker='o', color=colors[x[1]['cluster_num'].iloc[0] % len(colors)], markersize=5)


            centroids = stations_df.loc[stations_df['is_centroid'] == True]
            centroids_groupby = centroids.groupby('cluster_num')

            # TODO : One shot ?
            for x in centroids_groupby:
                gpd.GeoSeries(x[1]['geometry']).plot(ax=self.ax, marker='x', color=colors[x[1]['cluster_num'].iloc[0] % len(colors)])

            # TODO : Add lat & lon as annotation

        else:
            gpd.GeoSeries(stations_df['geometry']).plot(ax=self.ax, marker='o', color='red', markersize=10)


        map_loc = {'UK':[-8.405, 1.835, 49.929, 61.161]}

        loc = map_loc.get(country, 'auto')

        plt.axis(loc)
        plt.show()