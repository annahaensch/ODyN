""" Run Symmetric Simulations.
"""
import sys
sys.path.append('..')

import src as odyn
import logging
import pandas as pd
import os

logging.basicConfig(level=logging.INFO)

def main():

    # Make directory if it doesn't exist.
    isdir = os.path.isdir("../data/anti")
    if isdir == False:
        os.mkdir("../data/anti")
    isdir = os.path.isdir("../data/anti/no_name_county")
    if isdir == False:
        os.mkdir("../data/anti/no_name_county")

    logging.info("Loading OR geographic data.")

    # Load geographic data
    geo_df = odyn.get_county_mapping_data(county = "Multnomah", state = "OR")
    hesitancy_dict = odyn.get_hesitancy_dict(geo_df)
    prob = list(hesitancy_dict.values())

    model = odyn.OpinionNetworkModel(
                                probabilities = [prob[2],prob[1],prob[0]],
                                delta = 5,
                                beta = 1.7,
                                include_weight = True,
                                include_opinion = True,
                               )

    model.populate_model(num_agents = 1000, 
                            show_plot = False)
    initial_belief_df = model.belief_df.copy()

    # Save population data.
    model.agent_df.to_parquet("../data/anti/no_name_county/agent_df.pq")
    model.belief_df.to_parquet("../data/anti/no_name_county/belief_df.pq")
    model.prob_df.to_csv("../data/anti/no_name_county/prob_df.csv")
    model.adjacency_df.to_csv("../data/anti/no_name_county/adjacency_df.csv")

    logging.info("\n Model Loaded.")

    # Run simulations for no influencers.
    logging.info("\n Running simulation for no influencers.")
    
    model.reach_dict = {0:0,2:0}
    mega_influencer_df = model.connect_mega_influencers(
    belief_df = initial_belief_df)
    model.mega_influencer_df = mega_influencer_df

    sim = odyn.NetworkSimulation()
    sim.run_simulation(model = model, phases = 60)
    sim.dynamic_belief_df.to_csv(
        "../data/anti/no_name_county/simulation_results_none.csv")

    # Run simulations for variable left reach with right reach .8.
    reaches = [0,2,4,6,8,10]
    for i in range(len(reaches)):
        r = reaches[i]

        logging.info("\n Running simulation for r = {}".format(r))
        
        model.reach_dict = {0:r/10,2:.8}
        mega_influencer_df = model.connect_mega_influencers(
        belief_df = initial_belief_df)
        model.mega_influencer_df = mega_influencer_df

        sim = odyn.NetworkSimulation()
        sim.run_simulation(model = model, phases = 60, store_results = False)
        sim.dynamic_belief_df.to_csv(
            "../data/anti/no_name_county/simulation_results_{}.csv".format(r))

    return None


if __name__ == '__main__':
    main()