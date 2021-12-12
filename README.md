# ODyN for Vaccine Hesitancy

## Background

This repository contains Opinion Dynamics Network (ODyN) tools and data to simulate opinion dyamics on networks graphs relevant to Covid-19 vaccine hesitancy.  For technical details related to this project, please see: 

* *Link to come*

## Simulations

#### Load Geographic Data

We begin by Loading geographic and vaccine hesitancy data for a specific county, such as Montgomery, AL. Eventualy counties will be populated by adding "agents" to triangles with a specified density.  We use triangles because this is a convenient way to decompose the polygonal region into manageable pieces that can be filled using a Poisson point process. In case it's helpful, we've included a tool to visualize the triangulated county. 

```
from geolocations import *
geo_df = get_county_mapping_data(county = "Montgomery", state = "AL")
plot_triangulated_county(geo_df)
```
![al_triangulated.png](https://github.com/annahaensch/VaccineHesitancy/blob/main/images/al_triangulated.png?raw=true)

County mapping data will include area, population estimates, mapping coordinates, and vaccine hesitancy estimates for the county and state. Data for Montgomery, AL and Multnomah, OR has been preloaded so it will run quickly, other counties will download data directly from the [CDC website](https://data.cdc.gov/Vaccinations/Vaccine-Hesitancy-for-COVID-19-County-and-local-es/q9mh-h2tw), which might take a few minutes. From here we can read the probabilities of the three relevant modes
* 0 - *not vaccine hesitant*
* 1 - *hesitant or unsure*
* 2 - *strongly hesitant*

More information about the meaning of these modes can be found in the [ASPE Report of June 16th, 2021](https://aspe.hhs.gov/reports/vaccine-hesitancy-covid-19-state-county-local-estimates). Since it will be helpful in what follows, hesitancy rates can be loaded directly from the `geo_df` as a dictionary.

```
hesitancy_dict = get_hesitancy_dict(geo_df)
probability = list(hesitancy_dict.values())
```

#### Initialize Model

A model can be initialized by supplying only the list of probabilities for the desired modes. 
```
from simulations import *
p = list(hesitancy_dict.values())

# Load model object.
model = OpinionNetworkModel(probabilities = p)
```
There are many more default parameters that can be updated; more information about these parameters can be found in `simulations.py`. 

#### Populate Model

There are several options for populating the model with agents.  The simplest option is to add a fixed number of agents without including a density parameter.  This will default to plotting the requested number of agents on a triangle with area equivalent to 1 km^2. 

```
# Populate model.
model.populate_model(num_agents = 500)

# Plot initial network.
model.plot_initial_network()
```
![network_model.png](https://github.com/annahaensch/VaccineHesitancy/blob/main/images/network_model.png?raw=true)

Another option is to plot agents with attention to density.  In this case, the requested number of agents are plotted on a triangle with variable area, satisfying the density requirement. 

```
# Populate model.
model.populate_model(num_agents = 500, 
		density = 200)

# Plot initial network.
model.plot_initial_network()
```
![network_model_with_density.png](https://github.com/annahaensch/VaccineHesitancy/blob/main/images/network_model_with_density.png?raw=true)

Notice that the clusering coefficient and mean degree are both smaller in the example where agents are plotted with density.  This is because 500 agents are plotted with a density of 200 agents/km^2 so they have greated distance between then, and are therefore less likely to be connected under the present model. 

#### Run Simulation

Now we are ready to run a simulation on a model instance.  This is done by loading the simulator object and running the simulation.

```
sim = NetworkSimulation()
sim.run_simulation(model = model, phases = 60)
sim.plot_simulation_results()
```
![ridge_plot.png](https://github.com/annahaensch/VaccineHesitancy/blob/main/images/ridge_plot.png?raw=true)


## Vaccine Trend Visualizations

This repository also contains tools to visualize vaccine rates at the national and county level.  This can be done by loading `visualizations.py` in a Jupyter notebook.

```
from visualizations import * 

vaccine_trends_plot()

relative_vaccine_trends_plot()
```
![us_trends_absolute.png](https://github.com/annahaensch/VaccineHesitancy/blob/main/images/us_trends_absolute.png?raw=true)

![us_trends_relative.png](https://github.com/annahaensch/VaccineHesitancy/blob/main/images/us_trends_relative.png?raw=true)

For county level data, this repository contains data for Montgomery County, AL and Multnomah County, OR, but additional data can also be downloaded directly from the CDC website by settting the optional `download_data` argument to `True` (Note: running the function with this argument will download the 160 MB dataset of all counties and therefore it might take a few minutes to run). 
```
vaccine_trends_plot(county = "Multnomah", 
					state = "OR", 
					show_us_current = True,
					download_data = False)

relative_vaccine_trends_plot(county = "Multnomah",
					state = "OR",
					download_data = False)
```
![or_county_trends_absolute.png](https://github.com/annahaensch/VaccineHesitancy/blob/main/images/or_county_trends_absolute.png?raw=true)

![or_county_trends_relative.png](https://github.com/annahaensch/VaccineHesitancy/blob/main/images/or_county_trends_relative.png?raw=true)

Feel free to download and adjust the code to customize the charts.

# Data Sources

#### National Level Vaccination Trends

Centers for Disease Control and Prevention. *Trends in Number of COVID-19 Vaccinations in the US*. [https://covid.cdc.gov/covid-data-tracker/#vaccination-trends](https://covid.cdc.gov/covid-data-tracker/#vaccination-trends). Last Accessed: December 8, 2021.

#### County Level Vaccination Trends

Centers for Disease Control and Prevention. *COVID-19 Vaccinations in the United States, County*. [https://data.cdc.gov/Vaccinations/COVID-19-Vaccinations-in-the-United-States-County/8xkx-amqh](https://data.cdc.gov/Vaccinations/COVID-19-Vaccinations-in-the-United-States-County/8xkx-amqh). Last Accessed: December 8, 2021.

#### County Level Hesitancy Estimates

Centers for Disease Control and Prevention. *Vaccine Hesitancy for COVID-19: County and local estimates*. [https://data.cdc.gov/Vaccinations/Vaccine-Hesitancy-for-COVID-19-County-and-local-es/q9mh-h2tw](https://data.cdc.gov/Vaccinations/Vaccine-Hesitancy-for-COVID-19-County-and-local-es/q9mh-h2tw). Last Accessed: December 8, 2021.
