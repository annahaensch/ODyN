import unittest

from src.geolocations import *
from src.simulations import *
from src.visualizations import *

class TestSimulation(unittest.TestCase):

    def test_model(self):
        geo_df = get_county_mapping_data(county = "Multnomah", state = "OR")
        hesitancy_dict = get_hesitancy_dict(geo_df)
        prob = list(hesitancy_dict.values())
        self.assertEqual(int(np.sum(prob)), 1)

        # Load Model.
        model = OpinionNetworkModel(
                            probabilities = prob,
                            beta = 2.2,
                            alpha = 1.5
                           )
        self.assertTrue(model.include_opinion)
        self.assertTrue(model.include_weight)
        
        #Populate model.
        model.populate_model(num_agents = 5)
        self.assertEqual(model.belief_df.shape[0],5)

        # Run simulation.
        sim = NetworkSimulation()
        sim.run_simulation(model = model, phases = 2)
        self.assertEqual(int(sim.dynamic_belief_df.shape[1]),3)

if __name__ == '__main__':
    unittest.main()