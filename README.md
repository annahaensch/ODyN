# ODyN

![network.png](https://github.com/annahaensch/VaccineHesitancy/blob/main/images/network.png?raw=true)

## Background

This repository contains Opinion Dynamics Network (ODyN) tools and data to simulate opinion dyamics on networks graphs with an application to Covid-19 vaccine hesitancy.  For technical details related to this project, please see: 

* [A geospatial bounded confidence model including mega-influencers with an application to Covid-19 vaccine hesitancy](), Anna Haensch, Natasa Dragovic, Christoph Borgers, Bruce Boghosian.
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

## Using ODyN for To Run Simulations

To get started running simulations, check out `notebooks/demo_workbook.ipynb` to get started.  This workbook will show you how to set up belief networks, run simulaltions, and visualize your results. 

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

# Unit Tests

To run unit tests run the following from the top level directory.
```
> python -m unittest tests/test_simulations.py
```
