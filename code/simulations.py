import pandas as pd
import numpy as np
import logging 
import os
import geojson

import math 
import itertools

import geopandas as gpd
from geopy.geocoders import Nominatim
from shapely.geometry import Point, Polygon, MultiPolygon, shape
import shapely.ops
from pyproj import Proj

from bs4 import BeautifulSoup
import requests
from abc import ABC, abstractmethod

from geolocations import *
from visualizations import *

logging.basicConfig(level=logging.INFO)

class OpinionNetworkModel(ABC):
    """ Abstract base class for network model """
        
    def __init__(self, 
                n_modes = 3, 
                probabilities = [.45,.1,.45], 
                power_law_exponent = 2.5,
                openmindedness = 1.5,
                alpha = 1.6, 
                beta = 3,
                include_opinion = True,
                include_weight = True,
                reach_dict = {0:.8,2:.8},
                left_openmindedness = 1.5,
                right_openmindedness = 1.5,
                threshold = -1
                ):
        """Returns initialized OpinionNetworkModel.
            
        Inputs:
            n_modes: (int) number of categorical modes.
            probabilities: (list) probabilities of each mode.
            power_law_exponent: (float) exponent of power law
            openmindedness: (float) inter-mode distanct that agents influence.
            alpha: (float) Scaling factor for weight and distance, > 1.
            beta: (float) Scaling factor for distance, > 0.
            include_opinion: (boolean) If True, include distance in opinion space 
                in the probability measure.
            include_weight: (boolean) If True, include influencer weight in the 
                probability measure.
            reach_dict: (dictionary) value is propotional reach of key.
            left_openmindedness: (float) distance in opinion space that left 
                mega-influencers can reach.
            right_openmindedness: (float) distance in opinion space that right 
                mega-influencers can reach.
            threshold: (int) value below which opinions no longer change.

        Outputs: 
            Fully initialized OpinionNetwork instance.
        """
        self.n_modes = n_modes
        self.probabilities = probabilities
        self.power_law_exponent = power_law_exponent
        self.openmindedness = openmindedness
        self.alpha = alpha
        self.beta = beta
        self.include_opinion = include_opinion
        self.include_weight = include_weight
        self.reach_dict = reach_dict
        self.left_openmindedness = left_openmindedness
        self.right_openmindedness = right_openmindedness
        self.threshold = threshold

        self.agent_df = None
        self.belief_df = None
        self.prob_df = None
        self.adjacency_df = None
        self.mega_influencer_df = None
        
        self.clustering_coefficient = 0
        self.mean_degree = 0

    def populate_model(self, num_agents = 500, geo_df = None, show_plot = False):
        """ Fully initialized but untrained OpinionNetworkModel instance.

        Input:
            num_agents: (int) number of agents to plot.
            geo_df: (dataframe) location geomatic dataframe typically output 
                from get_county_mapping_data(). 
            show_plot: (bool) if true then plot is shown. 
        
        Output: 
            OpinionNetworkModel instance.
        """
        if geo_df is None:
            geo_df = get_county_mapping_data(county = "Montgomery", state = "AL")
        agent_df = self.add_random_agents_to_triangle(geo_df, 
            num_agents = num_agents)
        belief_df = self.assign_weights_and_beliefs(agent_df)
        prob_df = self.compute_probability_array(belief_df)
        adjacency_df = self.compute_adjacency(prob_df)

        # Connect mega-influencers
        mega_influencer_df = self.connect_mega_influencers(belief_df)
        
        # Compute network statistics.
        cc, md = self.compute_network_stats(adjacency_df)

        self.agent_df = agent_df
        self.belief_df = belief_df
        self.prob_df = prob_df
        self.adjacency_df = adjacency_df
        self.mega_influencer_df = mega_influencer_df
        self.clustering_coefficient = cc
        self.mean_degree = md
        
        if show_plot == True:
            self.plot_initial_network()

        return None

    def plot_initial_network(self):
        plot_network(self)
        return None

    def add_random_agents_to_triangle(self, geo_df, num_agents = None, 
        show_plot = False):
        """ Assign N points on a triangle using Poisson point process.
        
        Input:
            geo_df: (dataframe) location geomatic dataframe typically output 
                from get_county_mapping_data(). 
            num_agents: (int) number of agents to add to the triangle.  If None, 
                then agents are added according to density.
            show_plot: (bool) if true then plot is shown. 

        Returns: 
            An num_agents x 2 dataframe of point coordinates.
        """ 
        county = geo_df.loc[0,"county"]
        state = geo_df.loc[0,"state"]  

        # Initialize triangulation.
        make_triangulation(geo_df)
        
        c = county.lower().split(" county")[0]
        s = state.lower()
        with open('../data/{}/{}/triangulation_dict.geojson'.format(s,c)) as f:
            tri_dict = geojson.load(f)

        T = np.array([Polygon(t).area for t in tri_dict["geometry"]["coordinates"]])
        triangle_object = Polygon(tri_dict["geometry"]["coordinates"][T.argmax()])

        # Get boundary of triangle.
        bnd = list(triangle_object.boundary.coords)

        gdf = gpd.GeoDataFrame(geometry = [triangle_object])

        # Establish initial CRS
        gdf.crs = "EPSG:3857"

        # Set CRS to lat/lon
        gdf = gdf.to_crs(epsg=4326) 

        # Extract coordinates
        co = list(gdf.loc[0,"geometry"].exterior.coords)
        lon, lat = zip(*co)
        pa = Proj(
            "+proj=aea +lat_1=37.0 +lat_2=41.0 +lat_0=39.0 +lon_0=-106.55")
        x, y = pa(lon, lat)
        coord_proj = {"type": "Polygon", "coordinates": [zip(x, y)]}
        area = shape(coord_proj).area / (10 ** 6) # area in km^2
        
        # If no number of people is given, use population density.
        if num_agents is None:
            density = geo_df.loc[0,"density"]
            num_agents = int(area * density)

            logging.info("\n Using population density to place {} agents.".format(
                num_agents)) 
            
        # Get Vertices
        V1 = np.array(bnd[0])
        V2 = np.array(bnd[1])
        V3 = np.array(bnd[2])
        
        # Sample from uniform distribution on [0,1]
        U = np.random.uniform(0,1,num_agents)
        V = np.random.uniform(0,1,num_agents)
        
        UU = np.where(U + V > 1, 1-U, U)
        VV = np.where(U + V > 1, 1-V, V) 
        
        # Shift triangle into origin and and place points.
        agents = (UU.reshape(len(UU),-1) * (V2 - V1).reshape(-1,2)) + (
                                VV.reshape(len(VV),-1) * (V3 - V1).reshape(-1,2))
        
        # Shift points back to original position.
        agents = agents + V1.reshape(-1,2)
        agent_df = pd.DataFrame(agents, columns = ["x","y"])
    
        if show_plot == True:
            plot_agents_on_triangle(triangle_object, agent_df)

        return agent_df

    def assign_weights_and_beliefs(self, agent_df, show_plot = False):
        """ Assign weights and beliefs (i.e. modes) accoring to probabilities.
        
        Inputs: 
            agent_df: (dataframe) xy-coordinates for agents.
            show_plot: (bool) if true then plot is shown. 
            
        Returns: 
            Dataframe with xy-coordinates, beliefs, and weights for each point.
        """
        belief_df = agent_df.copy()
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

        if show_plot == True:
            plot_agents_on_triangle_with_belief(belief_df)

        return belief_df

    def compute_probability_array(self, belief_df):
        """ Return dataframe of probability that row n influences column m.
        
        Inputs: 
            belief_df: (dataframe) xy-coordinats, beliefs and weights of 
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
            prob_df: (dataframe) num_agents x num_agents dataframe giving  
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
            adjacency_df: (dataframe) num_agents x num_agents dataframe, where 
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
        clustering_coefficient = cc / adjacency_df.shape[0]

        return clustering_coefficient, np.mean(degrees)

    def connect_mega_influencers(self, belief_df):
        """ Returns mega_influencer reach dataframe

        Inputs: 
            belief_df: (dataframe) xy-coordinats, beliefs and weights of 
                agents.

        Outputs: 
            Dataframe where rows are the keys of reach dict and columns
            are the index of belief_df, where a 1 in row n column m indicates
            that influencer n reaches agent m.
        """
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
        self.phases = None
        self.results = []
        self.dynamic_belief_df = None

    def run_simulation(self, model, phases, show_plot = False):

        results = []
        self.model = model
        new_belief_df = model.belief_df
        new_adjacency_df = model.adjacency_df
        df = pd.DataFrame(columns = [i for i in range(phases + 1)])
        df[0] = new_belief_df["belief"].values

        self.phases = phases
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
            clust_coeff, mean_degree = model.compute_network_stats(new_adjacency_df)
            phase_dict["clust_coeff"] = clust_coeff
            phase_dict["mean_degree"] = mean_degree

            results.append(phase_dict)
            df[i+1] = new_belief_df["belief"].values

        self.results = results
        self.dynamic_belief_df = df

        if show_plot == True:
            self.plot_simulation_results()

        return None

    def plot_simulation_results(self):
        """ Return ridgeplot of simulation.
        """
        df = self.dynamic_belief_df
        if df is None:
            raise ValueError("It looks like the simulation hasn't been run yet.")
        if self.phases < 5:
            plot_phases = [t for t in range(self.phases +1)]
        else:
            t = self.phases // 5
            plot_phases = [0] + [t * (i+1) for i in range(1,5)]
        get_ridge_plot(df, plot_phases, reach_dict = self.model.reach_dict)
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
            belief_df: (dataframe)
            adjacency_df: (dataframe)
            left_openmindedness: (float) distance in opinion space that left 
                mega-influencers can reach.
            right_openmindedness: (float) distance in opinion space that right 
                mega-influencers can reach.
            threshold: (int) value below which opinions no longer change.

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