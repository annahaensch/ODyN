""" Run Symmetric Simulations.
"""
import sys
sys.path.append('..')

import src as odyn
import logging
import pandas as pd
import os

logging.basicConfig(level=logging.INFO)

def main(run_n):

    my_path = os.popen("git rev-parse --show-toplevel").read().strip("\n")
    my_path = my_path + "/data/anti/no_name_county/run_{}".format(run_n)
    isdir = os.path.isdir(my_path)
    if isdir == False:
        os.mkdir(my_path)
        
    print("Loading SKEPTICAL geographic data.")

    # Load geographic data
    geo_df = odyn.get_county_mapping_data(county = "Multnomah", state = "OR")
    hesitancy_dict = odyn.get_hesitancy_dict(geo_df)
    prob = list(hesitancy_dict.values())

    model = odyn.OpinionNetworkModel(
                                probabilities = [prob[2],prob[1],prob[0]],
	                            distance_scaling_factor = 1/10,
	                            importance_of_weight = 1.6, 
	                            importance_of_distance = 8.5,
                                include_weight = True,
                                include_opinion = True
                               )

    model.populate_model(num_agents = 1000, 
                            show_plot = False)
    initial_belief_df = model.belief_df.copy()

    # Save population data.
    model.agent_df.to_parquet("{}/agent_df.pq".format(my_path))
    model.belief_df.to_parquet("{}/belief_df.pq".format(my_path))
    model.prob_df.columns = [str(i) for i in model.prob_df.columns]
    model.prob_df.to_parquet("{}/prob_df.pq".format(my_path))
    model.prob_df.columns = [int(i) for i in model.prob_df.columns]

    model.adjacency_df.columns = [str(i) for i in model.adjacency_df.columns]
    model.adjacency_df.to_parquet("{}/adjacency_df.pq".format(my_path))
    model.adjacency_df.columns = [int(i) for i in model.adjacency_df.columns]

    print("\n Model Loaded.")

    # Run simulations for no influencers.
    print("\n Running simulation for no influencers.")
    
    model.left_reach = 0
    model.right_reach = 0
    mega_influencer_df = model.connect_mega_influencers(
    belief_df = initial_belief_df)
    model.mega_influencer_df = mega_influencer_df

    sim = odyn.NetworkSimulation()
    sim.run_simulation(model = model)
    sim.dynamic_belief_df.columns = [str(i) for i in sim.dynamic_belief_df.columns]
    sim.dynamic_belief_df.to_parquet("{}/simulation_results_none.pq".format(my_path))

    # Run simulations for variable left reach with right reach .8.
    reaches = [0,2,4,6,8,10]
    for i in range(len(reaches)):
        r = reaches[i]

        print("\n Running simulation for r = {}".format(r))
        
        model.left_reach = r/10
        model.right_reach = .8
        mega_influencer_df = model.connect_mega_influencers(
        belief_df = initial_belief_df)
        model.mega_influencer_df = mega_influencer_df

        sim = odyn.NetworkSimulation()
        sim.run_simulation(model = model, store_results = False)
        sim.dynamic_belief_df.columns = [str(i) for i in sim.dynamic_belief_df.columns]
        sim.dynamic_belief_df.to_parquet(
            "{}/simulation_results_{}.pq".format(my_path,r))

    return None


if __name__ == '__main__':
    run_n = sys.argv[1]
    main(run_n)