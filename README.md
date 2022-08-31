# ODyN for Vaccine Hesitancy

![ODyN.jpeg](https://github.com/annahaensch/VaccineHesitancy/blob/main/images/ODyN.jpeg?raw=true)

## Background

This repository contains Opinion Dynamics Network (ODyN) tools and data to simulate opinion dyamics on networks graphs relevant to Covid-19 vaccine hesitancy.  For technical details related to this project, please see: 

* [Covid-19 Vaccine Hesitancy and Mega-Influencers](https://arxiv.org/pdf/2202.00630.pdf), Anna Haensch, Natasa Dragovic, Christoph Borgers, Bruce Boghosian.

## Setting Up your ODyN Environment

Before you get started, you'll need to create a new environment using `conda` (in case you need it, [installation guide here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)). If you use `conda` you can 
create a new environment (we'll call it `odyn_env`) with

```
conda create --name odyn_env
```

and activate your new environment, with

```
conda activate odyn_env
```
To run the tools in the libarary will need to install the necessary dependencies. First you'll need to conda install 
`pip` and then install the remaining required Python libraries as follows.

```
conda install pip
pip install -U -r requirements.txt
```

Now your environment should be set up to run anything in this library. 

## Running Simulations with ODyN

#### Load Geographic Data

We begin by Loading geographic and vaccine hesitancy data for a specific county, such as Multnomah, OR. Eventualy counties will be populated by adding "agents" to triangles with a specified density.  We use triangles because this is a convenient way to decompose the polygonal region into manageable pieces that can be filled using a Poisson point process. In case it's helpful, we've included a tool to visualize the triangulated county. 

```
import src as odyn

geo_df = odyn.get_county_mapping_data(county = "Multnomah", state = "OR")
odyn.plot_triangulated_county(geo_df)
```
![or_triangulated.png](https://github.com/annahaensch/VaccineHesitancy/blob/main/images/or_triangulated.png?raw=true)

It's also possible to select for small bounded geographic regions using a `bounding_box`.  For example, we could focus on only the upper right corner of the county. 

```
bounding_box = [[-121.965, 45.62], 
                [-121.92, 45.62], 
                [-121.92, 45.65], 
                [-121.965, 45.65], 
                [-121.965, 45.62]]

odyn.plot_triangulated_county(geo_df, 
                              bounding_box = bounding_box,
                              restricted = True)
```
![or_triangulated_inset.png](https://github.com/annahaensch/VaccineHesitancy/blob/main/images/or_triangulated_inset.png?raw=true)

County mapping data will include area, population estimates, mapping coordinates, and vaccine hesitancy estimates for the county and state. Data for Montgomery, AL and Multnomah, OR has been preloaded so it will run quickly, other counties will download data directly from the [CDC website](https://data.cdc.gov/Vaccinations/Vaccine-Hesitancy-for-COVID-19-County-and-local-es/q9mh-h2tw), which might take a few minutes. From here we can read the probabilities of the three relevant modes
* 0 - *not vaccine hesitant*
* 1 - *hesitant or unsure*
* 2 - *strongly hesitant*

More information about the meaning of these modes can be found in the [ASPE Report of June 16th, 2021](https://aspe.hhs.gov/reports/vaccine-hesitancy-covid-19-state-county-local-estimates). Since it will be helpful in what follows, hesitancy rates can be loaded directly from the `geo_df` as a dictionary.

```
hesitancy_dict = odyn.get_hesitancy_dict(geo_df)
probability = list(hesitancy_dict.values())
```

#### Initialize Model

A model can be initialized by supplying only the list of probabilities for the desired modes. 
```
# Load model object.
model = odyn.OpinionNetworkModel(probabilities = prob,
                                include_weight = True,
                                include_opinion = True,
                                importance_of_weight = 0.1,
                                importance_of_distance = 9)
```
There are many more default parameters that can be updated; more information about these parameters can be found in `simulations.py`. 

#### Populate Model

There are several options for populating the model with agents.  One option is to add the agents using the geodata already stored in `geo_df` (this will include the coordinates for relevant triangles as well as recorded population densities).

```
# Populate model.
model.populate_model(geo_df = geo_df, 
                     bounding_box = bounding_box)

# Plot initial network.
model.plot_initial_network()
```
![or_triangulated_with_connections.png](https://github.com/annahaensch/VaccineHesitancy/blob/main/images/or_triangulated_with_connections.png?raw=true)

Another option is to populate a more generic model, by plotting agents on random triangles with a specified number of agents and belief distribution. 

```
# Populate model.
model = odyn.OpinionNetworkModel(probabilities = [.45,.1,.45])
model.populate_model(num_agents = 1000)

# Plot initial network.
model.plot_initial_network()
```
![network_model.png](https://github.com/annahaensch/VaccineHesitancy/blob/main/images/network_model_with_density.png?raw=true)

This workflow is demonstrated in the included notebook: `workflow_demo_Mutnomah_OR.ipynb.`

#### Run Simulation

Now we are ready to run a simulation on a model instance.  This is done by loading the simulator object and running the simulation.

```
sim = odyn.NetworkSimulation()
sim.run_simulation(model = model, phases = 5)
sim.plot_simulation_results()
```
![ridge_plot.png](https://github.com/annahaensch/VaccineHesitancy/blob/main/images/ridge_plot.png?raw=true)

Alternatively, this repository includes several built in scripts that can run simulations from the commmand line and print results to the data folder.  For example, from the top level directory, run the following
```
> cd scripts
> python accepting_simulation.py 1 #the int argument is the number of times to run the simulation
```
to produce simulation results for Multnomah County, Oregon.

## Vaccine Trend Visualizations

This repository also contains tools to visualize vaccine rates at the national and county level.  This can be done by loading `visualizations.py` in a Jupyter notebook.

```
odyn.vaccine_trends_plot()

odyn.relative_vaccine_trends_plot()
```
![us_trends_absolute.png](https://github.com/annahaensch/VaccineHesitancy/blob/main/images/us_trends_absolute.png?raw=true)

![us_trends_relative.png](https://github.com/annahaensch/VaccineHesitancy/blob/main/images/us_trends_relative.png?raw=true)

For county level data, this repository contains data for Montgomery County, AL and Multnomah County, OR, but additional data can also be downloaded directly from the CDC website by settting the optional `download_data` argument to `True` (Note: running the function with this argument will download the 160 MB dataset of all counties and therefore it might take a few minutes to run). 
```
odyn.vaccine_trends_plot(county = "Multnomah", 
					state = "OR", 
					show_us_current = True,
					download_data = False)

odyn.relative_vaccine_trends_plot(county = "Multnomah",
					state = "OR",
					download_data = False)
```
![or_county_trends_absolute.png](https://github.com/annahaensch/VaccineHesitancy/blob/main/images/or_county_trends_absolute.png?raw=true)

![or_county_trends_relative.png](https://github.com/annahaensch/VaccineHesitancy/blob/main/images/or_county_trends_relative.png?raw=true)

Feel free to download and adjust the code to customize the charts.

# Data Sources

All data needed for these simulations are available in the [data folder of this github repository](https://github.com/annahaensch/ODyN/tree/main/data) or in the case of large datasets, they are stored in a [public ODyN folder on Box](https://tufts.box.com/s/zswz021t98dobclsvv4q6a8nuq2ux2ik).  Original data sources are cited below.

#### National Level Vaccination Trends

Centers for Disease Control and Prevention. *Trends in Number of COVID-19 Vaccinations in the US*. [https://covid.cdc.gov/covid-data-tracker/#vaccination-trends](https://covid.cdc.gov/covid-data-tracker/#vaccination-trends). Last Accessed: December 8, 2021.

#### County Level Vaccination Trends

Centers for Disease Control and Prevention. *COVID-19 Vaccinations in the United States, County*. [https://data.cdc.gov/Vaccinations/COVID-19-Vaccinations-in-the-United-States-County/8xkx-amqh](https://data.cdc.gov/Vaccinations/COVID-19-Vaccinations-in-the-United-States-County/8xkx-amqh). Last Accessed: December 8, 2021.

#### County Level Hesitancy Estimates

Centers for Disease Control and Prevention. *Vaccine Hesitancy for COVID-19: County and local estimates*. [https://data.cdc.gov/Vaccinations/Vaccine-Hesitancy-for-COVID-19-County-and-local-es/q9mh-h2tw](https://data.cdc.gov/Vaccinations/Vaccine-Hesitancy-for-COVID-19-County-and-local-es/q9mh-h2tw). Last Accessed: December 8, 2021.

## Unit Tests

To run unit tests run the following from the top level directory.
```
> python -m unittest tests/test_simulations.py
```
