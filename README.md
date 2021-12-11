# Simulations

To load geo and vaccine hesitancy data for a specific county, such as Montgomery, AL, begin with 

```
from geolocations import *
get_county_mapping_data(county = "Montgomery", state = "AL")
```
Data for Montgomery, AL and Multnomah, OR has been preloaded to it will run quickly, other counties will download data directly from the cdc, which might take a few minutes. From here you can read the probabilities of modes
* 0 - *not vaccine hesitant*
* 1 - *hesitant or unsure*
* 2 - *strongly hesitant*
A model can be initialized with 
```
from simulations import *

# Read hesitancies from geo_df.
not_hesitant = geo_df.loc[0,"not_hesitant"]
hesitant_or_unsure = geo_df.loc[0,"hesitant_or_unsure"]
strongly_hesitant = geo_df.loc[0,"strongly_hesitant"]

p = [not_hesitant,hesitant_or_unsure,strongly_hesitant]

# Load model object.
model = OpinionNetworkModel(n_modes = 3, 
                            probabilities = p
                           )

# Populate model.
model.populate_model(num_agents = 500, geo_df = geo_df)

# Plot initial network.
model.plot_initial_network()
```
![network_model.png](https://github.com/annahaensch/VaccineHesitancy/blob/main/images/network_model.png?raw=true)


# Visualizations

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


For the network tools there are two types of built in visualizations: the network plot, and the dynamics ridge plot.  There are accessed as follows. 

```
model = OpinionNetworkModel.initialize(num_points = 500)
plot_initial_networks(model)
```


```
sim = NetworkSimulation()
sim.run_simulation(model = model, phases = 60)

df = sim.dynamic_belief_df
get_ridge_plot(df, periods = [0,10,20,40,60])
```

![ridge_plot.png](https://github.com/annahaensch/VaccineHesitancy/blob/main/images/ridge_plot.png?raw=true)

Feel free to download and adjust the code to customize the charts.

# Data Sources

#### National Level Vaccination Trends

Centers for Disease Control and Prevention. *Trends in Number of COVID-19 Vaccinations in the US*. [https://covid.cdc.gov/covid-data-tracker/#vaccination-trends](https://covid.cdc.gov/covid-data-tracker/#vaccination-trends). Last Accessed: December 8, 2021.

#### County Level Vaccination Trends

Centers for Disease Control and Prevention. *COVID-19 Vaccinations in the United States, County*. [https://data.cdc.gov/Vaccinations/COVID-19-Vaccinations-in-the-United-States-County/8xkx-amqh](https://data.cdc.gov/Vaccinations/COVID-19-Vaccinations-in-the-United-States-County/8xkx-amqh). Last Accessed: December 8, 2021.

#### County Level Hesitancy Estimates

Centers for Disease Control and Prevention. *Vaccine Hesitancy for COVID-19: County and local estimates*. [https://data.cdc.gov/Vaccinations/Vaccine-Hesitancy-for-COVID-19-County-and-local-es/q9mh-h2tw](https://data.cdc.gov/Vaccinations/Vaccine-Hesitancy-for-COVID-19-County-and-local-es/q9mh-h2tw). Last Accessed: December 8, 2021.
