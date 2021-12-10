import pandas as pd
import numpy as np
import logging 
import os
import geojson

import math 
import itertools

import geopandas as gpd
from geopy.geocoders import Nominatim
from shapely.geometry import Point, Polygon, MultiPolygon
import shapely.ops

from bs4 import BeautifulSoup
import requests
from abc import ABC, abstractmethod


logging.basicConfig(level=logging.INFO)

class OpinionNetworkModel(ABC):
    """ Abstract base class for network model """
        
    def __init__(self, state = None, county = None):
        self.state = state
        self.county = county
        self.geo_df = None
        
        self.num_points = None
        self.n_modes = None
        self.probabilities = None
        self.reach_dict = None
        self.power_law_exponent = None

        self.belief_df = None
        self.prob_df = None
        self.adjacency_df = None

        self.mega_influencer_df = None
        self.clustering_coeff = 0
        self.mean_deg = 0
        
    @classmethod
    def initialize(cls, 
                    point_df = None,
                    state = None, 
                    county = None, 
                    num_points = None, 
                    n_modes = 3, 
                    probabilities = [.45,.1,.45],
                    power_law_exponent = 2.5,
                    openmindedness = 1.5,
                    alpha = 1.6, 
                    beta = 3, 
                    include_opinion = True, 
                    include_weight = True,
                    reach_dict = {0:.8, 2:.8},
                    left_openmindedness = 1.5,
                    right_openmindedness = 1.5,
                    threshold = -1
                    ):
        """
            
        Inputs:
            state - (str) Uppercase two-letter state name abbreviation, or None.
            county - (str) Capitalized county name, or None.
            num_points - (int) number of agents to plot.
            n_modes - (int) number of categorical modes.
            probabilities - (list) probabilities of each mode.
            power_law_exponent - (float) exponent of power law
            openmindedness - (float) inter-mode distanct that agents influence.
            alpha - (float) Scaling factor for weight and distance, > 1.
            beta - (float) Scaling factor for distance, > 0.
            include_opinion - (boolean) If True, include distance in opinion space 
                in the probability measure.
            include_weight - (boolean) If True, include influencer weight in the 
                probability measure.
            reach_dict - (dictionary) value is propotional reach of key.
            left_openmindedness - (float) distance in opinion space that left 
                mega-influencers can reach.
            right_openmindedness - (float) distance in opinion space that right 
                mega-influencers can reach.
            threshold - (int) value below which opinions no longer change.

        Outputs: 
            Fully initialized but untrained OpinionNetwork instance.
        """
        model = cls(state = state, county = county)

        # Set network hyperparameters.
        model.num_points = num_points
        model.n_modes = n_modes
        model.probabilities = probabilities
        model.power_law_exponent = power_law_exponent
        model.reach_dict = reach_dict
        model.openmindedness = openmindedness
        model.alpha = alpha
        model.beta = beta
        model.include_opinion = include_opinion 
        model.include_weight = include_weight
        model.reach_dict = reach_dict
        model.left_openmindedness = left_openmindedness
        model.right_openmindedness = right_openmindedness
        model.threshold = threshold

        # Seed network agents and connections.
        if point_df is None:
            model.geo_df = model.get_county_mapping_data()
            model.point_df = model.random_points_on_triangle()
        else:
            model.point_df = point_df
        belief_df = model.assign_weights_and_beliefs()
        model.belief_df = belief_df
        prob_df = model.compute_probability_array(belief_df)
        model.prob_df = prob_df

        adjacency_df = model.compute_adjacency(prob_df)
        model.adjacency_df = adjacency_df
        model.mega_influencer_df = model.connect_mega_influencers()
        
        # Compute network statistics.
        cc, md = model.compute_network_stats(adjacency_df)
        model.clustering_coeff = cc
        model.mean_deg = md
        
        return model

    def get_county_mapping_data(self):
        """ Return geodataframe with location data.

        Returns: 
            Geodataframe with polygon geometry, population and 
            area data for county, state.
        """
        county = self.county
        if county is None:
            county = "Montgomery"
        state = self.state
        if state is None:
            state = "AL"

        county = county.split(" County")[0].split(" county")[0].capitalize()
        state = state.upper()

        geolocator = Nominatim(user_agent="VaccineHesitancy")
        geo = geolocator.geocode("{} County, {}".format(county, state),
                                geometry='geojson')

        full_state_name = geo.raw["display_name"].split(", ")[1]
        polygon = Polygon(geo.raw["geojson"]["coordinates"][0])

        # Convert to GeoDataFrame
        geo_df = gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[polygon])

        #Add spherical coordinate reference system (crs) to lat/long pairs.
        geo_df.crs = "EPSG:4326" 

        #Project onto a flat crs for mapping.
        geo_df = geo_df.to_crs(epsg=3857) 

        # Get areas
        geo_df["area"] = geo_df["geometry"].area
        geo_df["county"] = county
        if self.county == None:
            geo_df["county"] = "no_name_county"
        if self.state == None:
            geo_df["state"] = "no_name_state"

        # Get population.

        url = "https://en.wikipedia.org/wiki/List_of_United_States_counties_and_county_equivalents"

        # Load county population 
        request = requests.get(url) 
        html_content = request.text
        soup = BeautifulSoup(html_content,"lxml")
        county_tables = soup.find_all("table", {"class":"wikitable sortable"})

        df=pd.read_html(str(county_tables))
        df=pd.DataFrame(df[0])
        df["County or equivalent"] = [g.split("[")[0] for g in df["County or equivalent"]]
        df = df[(df["County or equivalent"] == county)&(df["State or equivalent"
            ] == full_state_name)]

        geo_df["population"] = df["Population (2019 estimate)"].iloc[0]

        return geo_df

    def _make_triangulation(self):
        """ Print geojson dictionary with county triangulation.

        Output: 
            Geojson file printed to ../data/<state abbreviation>/<county>
        """
        geo_df = self.geo_df
        county = geo_df.loc[0,"county"].lower()
        state = geo_df.loc[0,"state"].lower()
        path = "../data/{}".format(state)
        if os.path.exists(path) == False:
            os.mkdir(path)
        path = path + "/{}".format(county)
        if os.path.exists(path) == False:
            os.mkdir(path)

        tri = shapely.ops.triangulate(geo_df.loc[0,"geometry"])
        inner_tri = [t for t in tri if t.within(geo_df.loc[0,"geometry"])]

        tri_dict = {"type":"Feature",
                        "geometry": {
                            "type":"MultiPolygon",
                            "coordinates":[list(t.exterior.coords
                                ) for t in inner_tri]
                                    },
                    "properties":county}
        label = county.split(", {}".format(state))[0].replace(" ","_").lower()

        with open('{}/triangulation_dict.geojson'.format(path), 'w') as the_file:
            geojson.dump(tri_dict, the_file)
        return None

    def random_points_on_triangle(self):
        """ Assign N points on a triangle using Poisson point process.

        Returns: 
            An num_points x 2 dataframe of point coordinates.
        """ 
        geo_df = self.geo_df   
        num_points = self.num_points
        county = geo_df.loc[0,"county"]
        state = geo_df.loc[0,"state"]  

        self._make_triangulation()
        
        c = county.lower().split(" county")[0]
        s = state.lower()
        with open('../data/{}/{}/triangulation_dict.geojson'.format(s,c)) as f:
            tri_dict = geojson.load(f)
        
        # Select largest triangle.
        T = np.array([Polygon(t).area for t in tri_dict["geometry"]["coordinates"]])
        triangle_object = Polygon(tri_dict["geometry"]["coordinates"][T.argmax()])

        # Get boundary of triangle.
        bnd = list(triangle_object.boundary.coords)
        
        # If no number of people is given, use population density.
        if num_points is None:
            density = geo_df.loc[0,"population"] // (geo_df.loc[0,"area"] * 10e-7)
            num_points = int(density * (530833043.467798 * 10e-7))
            
        # Get Vertices
        V1 = np.array(bnd[0])
        V2 = np.array(bnd[1])
        V3 = np.array(bnd[2])
        
        # Sample from uniform distribution on [0,1]
        U = np.random.uniform(0,1,num_points)
        V = np.random.uniform(0,1,num_points)
        
        UU = np.where(U + V > 1, 1-U, U)
        VV = np.where(U + V > 1, 1-V, V) 
        
        # Shift triangle into origin and and place points.
        points = (UU.reshape(len(UU),-1) * (V2 - V1).reshape(-1,2)) + (
                                VV.reshape(len(VV),-1) * (V3 - V1).reshape(-1,2))
        
        # Shift points back to original position.
        points = points + V1.reshape(-1,2)
        
        return pd.DataFrame(points, columns = ["x","y"])


    def assign_weights_and_beliefs(self):
        """ Assign weights and beliefs (i.e. modes) accoring to probabilities.
        
        Inputs: 

            
        Returns: 
            Dataframe with xy-coordinates, beliefs, and weights for each point.
        """
        belief_df = self.point_df.copy()
        power_law_exponent = self.power_law_exponent
        k = 1/(power_law_exponent - 1)
        modes = [i for i in range(self.n_modes)]
        
        assert np.sum(np.array(self.probabilities)) == 1, "Probabilities must sum to 1."
        
        belief_df["weight"] = np.random.uniform(0,1,belief_df.shape[0]) ** (-k)
        belief_df["belief"] = np.random.choice(modes, belief_df.shape[0], 
                                p = self.probabilities)
        belief_df["decile"] = pd.qcut(belief_df["weight"], q = 100, labels = [
                                i for i in range(1,101)])
        
        # Move top influencer toward the center.
        top_influencer = belief_df[belief_df["weight"] == belief_df["weight"].max()
                                        ].index[0]
        x_offset = belief_df.loc[top_influencer,"x"]
        y_offset = belief_df.loc[top_influencer,"y"]

        belief_df["x"] = (belief_df["x"] - x_offset) * .001
        belief_df["y"] = (belief_df["y"] - y_offset) * .001
        return belief_df

    def compute_probability_array(self, belief_df):
        """ Return dataframe of probability that row n influences column m.
        
        Inputs: 
            belief_df - (dataframe) xy-coordinats, beliefs and weights of 
                agents.

        Returns: 
            Dataframe with the probability that agent n influces agent m 
            in row n column m.
        """
        openmindedness = self.openmindedness
        alpha = self.alpha
        beta = self.beta
        include_opinion = self.include_opinion
        include_weight = self.include_weight

        prob_df = pd.DataFrame(index = belief_df.index, columns = belief_df.index)

        # Compute squared distance from point i as row. 
        for i in belief_df.index:
            point_i = Point(belief_df.loc[i,"x"],belief_df.loc[i,"y"])

            # Set distance from i to i as np.nan.
            prob_df.loc[i,:] = [point_i.distance(Point(belief_df.loc[j,"x"],
                                                       belief_df.loc[j,"y"])
                                                    ) for j in belief_df.index]
            

            # Only allow connections to people close enough in opinion space.
            if include_opinion:
                current_belief = belief_df.loc[i,"belief"]
                opinion_diff = np.where(np.abs(belief_df["belief"] - current_belief
                                                        ) > openmindedness,0, 1)
                prob_df.loc[i,:] = opinion_diff * prob_df.loc[i,:]

        # Replace 0 with np.nan to avoid divbyzero error.
        prob_df = prob_df.replace(0,np.nan) ** (-beta)    
        
        # Multiply rows by normalized weight factor.
        if include_weight:
            prob_df = prob_df.mul((belief_df["weight"]), axis = 0)
        
        # Raise to alpha.
        prob_df = prob_df ** alpha
        
        # Make sure probabilities don't exceed 1.
        prob_df = prob_df.clip(upper = 1)
        
        prob_df = prob_df.replace(np.nan,0)
        return prob_df

    def compute_adjacency(self, prob_df):
        """ Compute NxN adjacency dataframe.
        
        Inputs: 
            prob_df - (dataframe) num_agents x num_agents dataframe giving  
                probabiltiy of influce row n on column m.
        Outputs: 
            Dataframe where row n and column n is a 1 if n influces m, 
            and a 0 otherwise.
        """
        adjacency_df = pd.DataFrame(0,index = [int(i) for i in prob_df.index], 
            columns = [int(i) for i in prob_df.columns])
        U = np.random.uniform(0,1,(prob_df.shape[0],prob_df.shape[1]))
        
        for i in prob_df.index:
            adj_idx = np.where(U[i] < prob_df.loc[i,:])[0]
            adjacency_df.loc[i,adj_idx] = 1
                  
        return adjacency_df

    def compute_network_stats(self, adjacency_df):
        """ Return clustering coefficient and mean degree. 
        
        Inputs: 
            adjacency_df - (dataframe) num_points x num_points dataframe, where 
                a 1 in row n column m indicates that agent n influences agent m.
        Returns: 
            Tuple of clustering coefficient and mean degree of 
            the network.
        """
        cc = 0
        degrees = []
        for i in adjacency_df.index:
            nbhd = np.where(adjacency_df.loc[i,:] != 0)[0]
            deg = len(nbhd)
            degrees.append(deg)
            cc_i = 0
            if deg > 2:
                C = list(itertools.combinations(nbhd,2))
                cc_i = np.sum([int(adjacency_df.loc[c[0],c[1]] > 0) for c in C]
                                                    ) / math.comb(deg,2)

        
            cc += cc_i
        clustering_coeff = cc / adjacency_df.shape[0]
        
        return clustering_coeff, np.mean(degrees)

    def connect_mega_influencers(self):
        """ Returns mega_influencer reach dataframe

        Outputs: 
            Dataframe where rows are the keys of reach dict and columns
            are the index of belief_df, where a 1 in row n column m indicates
            that influencer n reaches agent m.
        """
        belief_df = self.belief_df
        reach_dict = self.reach_dict
        mega_influencer_df = pd.DataFrame(0, index = [0,2], columns = list(
            belief_df.index))
        
        for i in mega_influencer_df.index:
            ind = np.where(np.abs(belief_df["belief"] - i) != 2)[0]
            U = np.random.uniform(0,1,len(ind))
            mega_influencer_df.loc[i,ind[np.where(U < reach_dict[i])[0]]] = 1
            
        return mega_influencer_df


class NetworkSimulation(ABC):
    """ Abstract base class for learning algorithm.
    """
    def __init__(self):
        self.model = None
        self.iterations = None
        self.results = []
        self.dynamic_belief_df = None

    def run_simulation(self, model, phases):

        results = []
        self.model = model
        new_belief_df = model.belief_df
        new_adjacency_df = model.adjacency_df
        df = pd.DataFrame(columns = [i for i in range(phases + 1)])
        df[0] = new_belief_df["belief"].values

        for i in range(phases):
            phase_dict = {}
            new_belief_df = self.one_dynamics_iteration(
                belief_df = new_belief_df,
                adjacency_df = new_adjacency_df,
                right_openmindedness = model.right_openmindedness, 
                left_openmindedness = model.left_openmindedness,
                mega_influencer_df = model.mega_influencer_df, 
                threshold = model.threshold)

            phase_dict["belief_df"] = new_belief_df
            prob_df = model.compute_probability_array(new_belief_df)
            new_adjacency_df = model.compute_adjacency(prob_df)
            new_adjacency_df.columns = [int(i) for i in new_adjacency_df.columns]
            phase_dict["adjacency_df"] = new_adjacency_df
            clust_coeff, mean_deg = model.compute_network_stats(new_adjacency_df)
            phase_dict["clust_coeff"] = clust_coeff
            phase_dict["mean_deg"] = mean_deg

            results.append(phase_dict)
            df[i+1] = new_belief_df["belief"].values

        self.results = results
        self.dynamic_belief_df = df

        return None

    def one_dynamics_iteration(self, 
                        belief_df, 
                        adjacency_df,
                        right_openmindedness, 
                        left_openmindedness,
                        mega_influencer_df, 
                        threshold):
        """ Returns updated belief_df.
        Inputs: 
            belief_df - (dataframe)
            adjacency_df - (dataframe)
            left_openmindedness - (float) distance in opinion space that left 
                mega-influencers can reach.
            right_openmindedness - (float) distance in opinion space that right 
                mega-influencers can reach.
            threshold - (int) value below which opinions no longer change.

        Returns: 
            Updated belief_df after one round of Hegselmann-Krause.
        """
        df_new = belief_df.copy()
            
        for i in belief_df.index:
            current_belief = belief_df.loc[belief_df.index[i], "belief"]

            if current_belief > threshold:
                
                # Sum over column
                edges = np.where(adjacency_df[i] == 1)[0]
                n_edges = len(edges)
                
                new_belief = np.sum(belief_df.loc[edges,"belief"]) + current_belief
                
                if right_openmindedness > 0:
                    # Am  I connected to them?
                    if mega_influencer_df.loc[2,i] == 1:
                        # Do I listen to them?
                        if np.abs(current_belief - 2) <= right_openmindedness:
                            n_edges = n_edges + 1
                            new_belief = new_belief + 2
                        
                if left_openmindedness > 0:
                    # Am  I connected to them?
                    if mega_influencer_df.loc[0,i] == 1:
                        # Do I listen to them?
                        if np.abs(current_belief  - 0) <= left_openmindedness:
                            n_edges = n_edges + 1
                            new_belief = new_belief + 0
                    
                new_belief = new_belief / (n_edges + 1)

                df_new.loc[i,"belief"] = new_belief

        return df_new


############################
#  Visualization Tools
############################





if __name__ == "__main__":

    # Make relevant directories.
    path = "../data/symmetric_simulations_60periods_1000_2/"
    if os.path.exists(path) == False:
        os.mkdir(path)

    for r in [0,1,2,4,6,8,10]:
        path = "../data/symmetric_simulations_60periods_1000_2/left_reach_{}/".format(r)
        if os.path.exists(path) == False:
            os.mkdir(path)
    

    methods = ["no_influencers","left_and_right_influencers"]

    for method in methods:

        logging.info("\n {} method initialized".format(method))

        if method == "no_influencers":

            path = "../data/symmetric_simulations_60periods_1000_2/left_reach_0/no_influencers/"
            if os.path.exists(path) == False:
                os.mkdir(path)

            run_simulation(
                    folder_name = path,
                    state = "Alabama", 
                    num_points = 1000,
                    populate_community = True, 
                    initialize_network = True,
                    left_reach = 0,
                    right_reach = 0,
                    left_influence = 0,
                    right_influence = 0,
                    hk_phases = 60,
                    runs = 1)

        else:

            for left_reach in [0,1,2,4,6,8,10]:

                path = "../data/symmetric_simulations_60periods_1000_2/left_reach_{}/left_and_right_influencers/".format(left_reach)
                if os.path.exists(path) == False:
                    os.mkdir(path)

                run_simulation(
                        folder_name = path,
                        state = "Alabama", 
                        num_points = 1000,
                        populate_community = False, 
                        initialize_network = False,
                        left_reach = left_reach/10,
                        right_reach = .8,
                        left_influence = 1.5,
                        right_influence = 1.5,
                        hk_phases = 60,
                        runs = 1)

        logging.info("\n {} complete.".format(method))