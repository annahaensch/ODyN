# Simulations

*In Progress*

# Visualizations

## Vaccine Trend Visualizations

To create visualizations for the national level data, use the following code from a jupyter notebook

```
from visualizations import *
national_vaccine_trends_chart()
```
![us_trends_absolute.png](https://github.com/annahaensch/VaccineHesitancy/blob/main/images/us_trends_absolute.png?raw=true)

```
national_vaccine_expectation_chart()
```
![us_trends_relative.png](https://github.com/annahaensch/VaccineHesitancy/blob/main/images/us_trends_relative.png?raw=true)

For county level data, this repository contains data for Montgomery County, AL and Multnomah County, OR, but additional data can also be downloaded directly from the CDC website by settting the optional `download_data` argument to `True`, but be advised that this is a large data set and it might take a few minutes to download. 
```
county_vaccine_trends_chart(county_1 = "Montgomery County, AL", county_2 = "Multnomah County, OR", download_data = False)
```
![county_trends_absolute.png](https://github.com/annahaensch/VaccineHesitancy/blob/main/images/county_trends_absolute.png?raw=true)



Feel free to download and adjust the code to customize the charts to different counties and date ranges.

# Data Sources

#### National Level Vaccination Trends

Centers for Disease Control and Prevention. *Trends in Number of COVID-19 Vaccinations in the US*. [https://covid.cdc.gov/covid-data-tracker/#vaccination-trends](https://covid.cdc.gov/covid-data-tracker/#vaccination-trends). Last Accessed: December 8, 2021.

#### County Level Vaccination Trends

Centers for Disease Control and Prevention. *COVID-19 Vaccinations in the United States, County*. [https://data.cdc.gov/Vaccinations/COVID-19-Vaccinations-in-the-United-States-County/8xkx-amqh](https://data.cdc.gov/Vaccinations/COVID-19-Vaccinations-in-the-United-States-County/8xkx-amqh). Last Accessed: December 8, 2021.

#### County Level Hesitancy Estimates

Centers for Disease Control and Prevention. *Vaccine Hesitancy for COVID-19: County and local estimates*. [https://data.cdc.gov/Vaccinations/Vaccine-Hesitancy-for-COVID-19-County-and-local-es/q9mh-h2tw](https://data.cdc.gov/Vaccinations/Vaccine-Hesitancy-for-COVID-19-County-and-local-es/q9mh-h2tw). Last Accessed: December 8, 2021.
