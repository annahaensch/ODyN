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

from .geolocations import *
from .visualizations import *

logging.basicConfig(level=logging.WARNING)

class OpinionNetworkModel(ABC):
    """ Abstract base class for network model """
        
    def __init__(self, 
                probabilities = [.45,.1,.45], 
                power_law_exponent = 1.5,
                openness_to_neighbors = 1.5,
                openness_to_influencers = 1.5,
                distance_scaling_factor = 1/10,
                importance_of_weight = 1.6, 
                importance_of_distance = 8.5,
                include_opinion = True,
                include_weight = True,
                left_reach = 0.8,
                right_reach = 0.8,
                threshold = -1
                ):
        """Returns initialized OpinionNetworkModel.
            
        Inputs:
            probabilities: (list) probabilities of each mode; these are the 
                values "p_0,p_1,p_2" from [1].
            power_law_exponent: (float) exponent of power law, must be > 0; 
                this is "gamma" from [1].
            openness_to_neighbors: (float) maximum inter-mode distance that agents
                can influence; this is "b" from [1].
            openness_to_influencers: (float) distance in opinion space that 
                mega-influencers can reach; this is "epsilon" from [1].
            distance_scaling_factor: (float) Scale distancy by this amount, must 
                be >0; this is "lambda" from [1].
            importance_of_weight: (float) Raise weights to this power, must be > 0; 
                this is "alpha" from [1].
            importance_of_distance: (float) Raise adjusted distance to this power,
                must be > 0; this is "delta" from [1].
            include_opinion: (boolean) If True, include distance in opinion space 
                in the probability measure.
            include_weight: (boolean) If True, include influencer weight in the 
                probability measure.
            left_reach: (float) this is the proportion of the susceptible population 
                that the left mega-influencers will actually reach, must be between
                0 and 1; this is p_L from [1] 
            right_reach: (float) this is the proportion of the susceptible population 
                that the right mega-influencers will actually reach, must be between
                0 and 1; this is p_R from [1] 

            threshold: (int) value below which opinions no longer change.

        Outputs: 
            Fully initialized OpinionNetwork instance.
        """
        self.probabilities = probabilities
        self.power_law_exponent = power_law_exponent
        self.openness_to_neighbors = openness_to_neighbors
        self.openness_to_influencers = openness_to_influencers
        self.distance_scaling_factor = distance_scaling_factor
        self.importance_of_weight = importance_of_weight
        self.importance_of_distance = importance_of_distance
        self.include_opinion = include_opinion
        self.include_weight = include_weight
        self.left_reach = left_reach
        self.right_reach = right_reach
        self.threshold = threshold

        self.agent_df = None
        self.belief_df = None
        self.prob_df = None
        self.adjacency_df = None
        self.mega_influencer_df = None
        
        self.clustering_coefficient = 0
        self.mean_degree = 0

    def populate_model(self, num_agents = None, density = None, show_plot = False):
        """ Fully initialized but untrained OpinionNetworkModel instance.

        Input:
            num_agents: (int) number of agents to plot.
            show_plot: (bool) if true then plot is shown. 
        
        Output: 
            OpinionNetworkModel instance.
        """
        agent_df = self.add_random_agents_to_triangle(num_agents = num_agents, 
            density = density)
        logging.info("\n {} agents added.".format(agent_df.shape[0]))
        
        belief_df = self.assign_weights_and_beliefs(agent_df)
        logging.info("\n Weights and beliefs assigned.")

        prob_df = self.compute_probability_array(belief_df)
        adjacency_df = self.compute_adjacency(prob_df)
        logging.info("\n Adjacencies computed.")

        # Connect mega-influencers
        mega_influencer_df = self.connect_mega_influencers(belief_df)
        
        # Compute network statistics.
        logging.info("\n Computing network statistics...")
        cc, md = self.compute_network_stats(adjacency_df)
        logging.info("\n Clustering Coefficient: {}".format(cc))
        logging.info("\n Mean Degree: {}".format(md))

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

    def add_random_agents_to_triangle(self, num_agents, density = None, 
        show_plot = False):
        """ Assign N points on a triangle using Poisson point process.
        
        Input:
            num_agents: (int) number of agents to add to the triangle.  If None, 
                then agents are added according to density.
            density: (float) density of agents, if none, then num_agents are plotted
                in triangle with sarea 1 km^2.
            show_plot: (bool) if true then plot is shown. 

        Returns: 
            An num_agents x 2 dataframe of point coordinates.
        """ 
        if density is not None:
            # Plot num_agents on variable sized triangle with correct density.
            b = 1419 * (num_agents/density) ** (1/2)
            triangle_object = Polygon([[0,0],[b,0], [b/2,b],[0,0]])

        else:
            # If no density is given, fill triangle with area 1 km^2.       
            b = 1419
            triangle_object = Polygon([[0,0],[b,0], [b/2,b],[0,0]])

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
        k = -1/(power_law_exponent)
        modes = [i for i in range(len(self.probabilities))]
        
        assert np.sum(np.array(self.probabilities)) == 1, "Probabilities must sum to 1."
        
        belief_df["weight"] = np.random.uniform(0,1,belief_df.shape[0]) ** (k)
        belief_df["belief"] = np.random.choice(modes, belief_df.shape[0], 
                                p = self.probabilities)
        belief_df["decile"] = pd.qcut(belief_df["weight"], q = 100, labels = [
                                i for i in range(1,101)])

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
        n = belief_df.index.shape[0]
        prob_array = np.ones((n,n))
        
        dist_array = np.zeros((n,n)) 
        for i in range(n):
            point_i = Point(belief_df.loc[i,"x"],belief_df.loc[i,"y"])

            # Get distances from i to each other point.
            dist_array[i,:] = [point_i.distance(Point(belief_df.loc[j,"x"],
                                                       belief_df.loc[j,"y"])
                                                    ) for j in belief_df.index]
            
        # Compute the dimensionless distance metric.
        diam = dist_array.max().max()
        lam = (self.distance_scaling_factor * diam)
        delta = -1 * self.importance_of_distance
        
        dist_array = np.where(dist_array == 0,np.nan, dist_array)
        dist_array = (1 + (dist_array/lam)) ** delta
        
        prob_array = prob_array * dist_array 
        
        # Only allow connections to people close in opinion space.
        if self.include_opinion == True:
            op_array = np.zeros((n,n))
            # If row i connects to column j, that means person i is
            # influencing person j.  This can only happen if j is 
            # already sufficiently close to person i in opinion space.
            for i in range(n):
                i_current_belief = belief_df.loc[i,"belief"]
                opinion_diff = np.where(
                    np.abs(belief_df["belief"] - i_current_belief
                        ) > self.openness_to_neighbors,0, 1)
                op_array[i,:] = opinion_diff
                
            prob_array = prob_array * op_array


        # Incentivize connections with heavily weighted people.
        if self.include_weight == True:
            wt_array = belief_df["weight"] ** self.importance_of_weight
            wt_array = wt_array.values.reshape(-1,1)
            
            prob_array = prob_array * wt_array

        prob_df = pd.DataFrame(prob_array)
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
        U = np.random.uniform(0,1,(prob_df.shape[0], prob_df.shape[1]))
        adjacency_df = pd.DataFrame(np.where(U< prob_df.values,1,0))

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
            nbhd = np.where(adjacency_df.loc[:,i] != 0)[0]
            deg = len(nbhd)
            degrees.append(deg)
            cc_i = 0
            if deg > 2:
                C = list(itertools.permutations(nbhd,2))
                cc_i = np.sum([int(adjacency_df.loc[c[0],c[1]] > 0) for c in C]
                                                    ) / math.perm(deg,2)

        
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
        reach_dict = {0:self.left_reach, 2:self.right_reach}
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

    def run_simulation(self, model, phases, show_plot = False, store_results = False):
        """ Carry out simulation.

        Inputs: 
            model: OpinionNetworkModel instance.
            phases: (int) number of phases of simulation to carry out.
            show_plot: (bool) if true, shows ridge plot.
            store_results: (bool) if True, stores results for all phases, if False
                then only updating beliefs are stored.

        Outputs: 
            Complete simulation.
        """

        results = []
        self.model = model
        new_belief_df = model.belief_df.copy()
        new_adjacency_df = model.adjacency_df.copy()
        new_mega_influencer_df = model.mega_influencer_df.copy()

        df = pd.DataFrame(columns = [i for i in range(phases + 1)])
        df[0] = new_belief_df["belief"].values

        self.phases = phases
        for i in range(phases):
            phase_dict = {}
            new_belief_df, new_mega_influencer_df = self.one_dynamics_iteration(
                            belief_df = new_belief_df,
                            adjacency_df = new_adjacency_df,
                            openness_to_influencers = model.openness_to_influencers,
                            mega_influencer_df = new_mega_influencer_df, 
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

        self.dynamic_belief_df = df

        if show_plot == True:
            self.plot_simulation_results()
        if store_results == True:
            self.results = results

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
        get_ridge_plot(df, plot_phases, reach_dict = {0:self.model.left_reach,
                                                    2:self.model.right_reach})
        return None


    def one_dynamics_iteration(self, 
                        belief_df, 
                        adjacency_df,
                        openness_to_influencers,
                        mega_influencer_df, 
                        threshold):
        """ Returns updated belief_df.
        Inputs: 
            belief_df: (dataframe)
            adjacency_df: (dataframe)
            openness_to_influencers: (float) distance in opinion space that 
                mega-influencers can reach; this is "epsilon" from [1].
            threshold: (int) value below which opinions no longer change.

        Returns: 
            Updated belief_df after one round of Hegselmann-Krause.
        """
        df_new = belief_df.copy()
        new_mega_influencer_df = mega_influencer_df.copy()

        for i in belief_df.index:
            current_belief = belief_df.loc[belief_df.index[i], "belief"]

            if current_belief > threshold:
                
                # Sum over column
                edges = np.where(adjacency_df[i] == 1)[0]
                n_edges = len(edges)
                
                new_belief = np.sum(belief_df.loc[edges,"belief"]) + current_belief
                
                if openness_to_influencers > 0:
                    # Am  I connected to the right mega-influencer?
                    if mega_influencer_df.loc[2,i] == 1:
                        # Do I listen to them?
                        if np.abs(current_belief - 2) <= openness_to_influencers:
                            n_edges = n_edges + 1
                            new_belief = new_belief + 2
                        # If not, flip a biased coin to decide if my influencer connection
                        # changes affiliation.
                        else:
                            new_mega_influencer_df.loc[2,i] == 0
                            u = np.random.uniform(0,1)
                            if u < self.model.left_reach:
                                new_mega_influencer_df.loc[0,i] == 1

                    # Am  I connected to the left mega-influencer?
                    if mega_influencer_df.loc[0,i] == 1:
                        # Do I listen to them?
                        if np.abs(current_belief  - 0) <= openness_to_influencers:
                            n_edges = n_edges + 1
                            new_belief = new_belief + 0
                        # If not, flip a biased coin to decide if my influencer connection
                        # changes affiliation.
                        else:
                            new_mega_influencer_df.loc[0,i] == 0
                            u = np.random.uniform(0,1)
                            if u < self.model.right_reach:
                                new_mega_influencer_df.loc[2,i] == 1
                    
                new_belief = new_belief / (n_edges + 1)

                df_new.loc[i,"belief"] = new_belief

        return df_new, new_mega_influencer_df