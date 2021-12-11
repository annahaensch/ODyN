# Opinion Dynamics & COVID Vaccine Hesitancy

## Background

This repository contains tools to simulate opinion dyamics related to Covid-19 vaccine hesitancy.  For technical details related to this project, please see: 

* *Link to come*

## Simulations

To load geo and vaccine hesitancy data for a specific county, such as Montgomery, AL, begin with 

```
from geolocations import *
get_county_mapping_data(county = "Montgomery", state = "AL")
```
This will return a geodataframe with area, population estimates, mapping coordinates, and vaccine hesitancy estimates for the county and state. Data for Montgomery, AL and Multnomah, OR has been preloaded so it will run quickly, other counties will download data directly from the [CDC website](https://data.cdc.gov/Vaccinations/Vaccine-Hesitancy-for-COVID-19-County-and-local-es/q9mh-h2tw), which might take a few minutes. From here we can read the probabilities of the three relevant modes
* 0 - *not vaccine hesitant*
* 1 - *hesitant or unsure*
* 2 - *strongly hesitant*

These can be loaded direction from the geo_df.

```
geo_df = get_county_mapping_data(county = "Montgomery", state = "AL")
hesitancy_dict = get_hesitancy_dict(geo_df)
```

A model can be initialized with 
```
from simulations import *
p = list(hesitancy_dict.values())

# Load model object.
model = OpinionNetworkModel(n_modes = 3, 
                            probabilities = p
                           )

# Populate model.
model.populate_model(num_agents = 500, density = geo_df.loc[0,"density"])

# Plot initial network.
model.plot_initial_network()
```
![network_model.png](https://github.com/annahaensch/VaccineHesitancy/blob/main/images/network_model.png?raw=true)

Now we are ready to run a simulation on a model instance.  This is done by loading the simulator object and running the simulation.

```
sim = NetworkSimulation()
sim.run_simulation(model = model, phases = 60)
sim.plot_simulation_results()
```
![ridge_plot.png](https://github.com/annahaensch/VaccineHesitancy/blob/main/images/ridge_plot.png?raw=true)


## Vaccine Trend Visualizations

To create visualizations for the national level data, use the following code from a jupyter notebook

```
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
