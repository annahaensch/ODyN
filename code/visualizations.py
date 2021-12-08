import pandas as pd
import numpy as np

from datetime import timedelta
from functools import reduce

import matplotlib.pyplot as plt

COLORS = {"light_orange":"#E69F00",
             "light_blue":"#56B4E9",
             "teal":"#009E73",
             "yellow":"#F0E442",
             "dark_blue":"#0072B2",
             "dark_orange":"#D55E00",
             "pink":"#CC79A7",
             "purple":"#9370DB",
             "black":"#000000",
             "silver":"#DCDCDC"}

START_DATE = pd.to_datetime("2021-04-19")
US_POPULATION = 329500000

def load_national_trend_data():
    df = pd.read_csv("../data/national_level_vaccination_trends.csv")
    df = df[df["Date Type"] == "Admin"]
    df["Date"] = [pd.to_datetime(d) for d in df["Date"]]
    df.set_index("Date", drop = True, inplace = True)
    return df

def load_county_data(county, state, download_data= False):
    c = county.lower()
    s = state.lower()
    
    if download_data == True:
        # Caution: this will take a long time.
        df = pd.read_csv("https://data.cdc.gov/api/views/8xkx-amqh/rows.csv?accessType=DOWNLOAD")
        df = df[(df["Recip_County"] == "{} County".format(county.capitalize())) & (df["Recip_State"] == state.upper())]

    else:
        df = pd.read_csv("../data/{}_county_{}_vaccination_trends.csv".format(c,s), index_col = 0)
    
    df["Date"] = [pd.to_datetime(d) for d in df["Date"]]
    df.sort_values(by = "Date", inplace = True)
    df.set_index("Date", drop = True, inplace = True)
    
    return df

def national_complete_pct(df):
    """ Returns timeseries of percentage completely vaccinated.
    """
    complete = 100 * df['People Fully Vaccinated Cumulative']/US_POPULATION
    return complete.loc[START_DATE:]

def national_one_dose_pct(df):
    """ Returns timeseries of percentage with one dose. 
    """
    one_dose = 100 * df['People Receiving 1 or More Doses Cumulative']/US_POPULATION
    return one_dose.loc[START_DATE:]

def national_expected_complete_pct(df):
    """ Returns timeseries of expected percentage completely vaccinated.
    """
    one_dose = 100 * df['People Receiving 1 or More Doses Cumulative']/US_POPULATION     
    expected = one_dose.loc[START_DATE - timedelta(days = 42):one_dose.index[-1] - timedelta(days = 42)]
    expected = pd.Series(expected.values, index = pd.date_range(START_DATE, one_dose.index[-1]))
    return expected

def county_complete_pct(df):
    """ Returns timeseries of percentage completely vaccinated.
    """
    return df["Series_Complete_Pop_Pct"].loc[START_DATE:,]

def county_one_dose_pct(df):
    """ Returns timeseries of percentage with one dose.
    """
    return df['Administered_Dose1_Pop_Pct'].loc[START_DATE:,]

def county_expected_complete_pct(df):
    """ Returns timeseries of percentage expected completely vaccinated.
    """
    one_dose = df['Administered_Dose1_Pop_Pct']
    expected = one_dose.loc[START_DATE - timedelta(days = 42):one_dose.index[-1] - timedelta(days = 42)]
    expected = pd.Series(expected.values, index = pd.date_range(START_DATE, one_dose.index[-1]))
    
    return expected

def national_vaccine_trends_chart():
    df = load_national_trend_data()
    
    complete = national_complete_pct(df)
    one_dose = national_one_dose_pct(df)
    expected = national_expected_complete_pct(df)
    
    fig, ax = plt.subplots(figsize = (8,5))
    
    # Plot trends.
    ax.plot(one_dose, 
            color = COLORS["dark_blue"], 
            linewidth = 3, 
            label = "One Dose")
    ax.plot(complete, 
            color = COLORS["light_orange"], 
            linewidth = 3, 
            label = "Completely Vaccinated")
    ax.plot(expected, 
            color = "gray", 
            linestyle = "dotted",
            linewidth = 3, 
            zorder = 0,
            label = "Expected Completely Vaccinated")
    
    ax.set_xlabel("Date")
    ax.set_ylabel("Percentage")
    
    ax.set_yticks([0,20,40,60,80])
    ax.set_yticklabels(["{}%".format(20*i) for i in range(5)])
    
    plt.title("National Vaccine Trends")
    plt.legend(loc = "lower right")
    plt.show
    return None


def national_vaccine_expectation_chart():
    df = load_national_trend_data()
    
    complete = national_complete_pct(df)
    expected = national_expected_complete_pct(df)
    
    diff = complete - expected
    diff_weekly = pd.DataFrame(index = pd.date_range(diff.index[0],diff.index[-1], freq = "W"),
                             columns = ["mean"])

    for i in range(diff.shape[0]-1):
        start = diff.index[i]
        end = diff.index[i+1]
        diff_weekly.loc[start,'mean'] = diff.loc[start:end].mean()

    color = [COLORS["teal"] if t >= 0 else COLORS["pink"] for t in diff_weekly["mean"]]

    fig, ax = plt.subplots(figsize = (8,5))
    
    # Plot trends.
    ax.bar(x = diff_weekly.index, width = 1, height = diff_weekly["mean"],color = color)
    
    ax.plot([],[],color = COLORS["teal"], label= "More people than expected are completely vaccinated", linewidth = 3)
    ax.plot([],[],color = COLORS["pink"], label= "Fewer people than expected are completely vaccinated", linewidth = 3)
    
    ax.set_ylabel("Percentage Points")
    ax.set_xlabel("Date")
    ax.set_ylim(-12,12)
    ax.legend(loc = "lower left")
    
    plt.title("Percentage Points Relative to Expected National Rates")
    
    plt.show
    return None


def county_vaccine_trends_chart(county_1 = "Montgomery County, AL", 
                                county_2 = "Multnomah County, OR", 
                                show_us_current = True,
                                download_data = False):
    df = load_national_trend_data()
    
    complete_us = national_complete_pct(df)
    one_dose_us = national_one_dose_pct(df)
    expected_us = national_expected_complete_pct(df)
    
    fig, ax = plt.subplots(1,2, figsize = (15,5), sharey = True)
    
    # Add horizontal line at current completely vaccinated.
    if show_us_current == True:
        for i in [0,1]:
            ax[i].plot((one_dose_us.index[0], one_dose_us.index[-1]),
                    (complete_us.iloc[-1],complete_us.iloc[-1]), 
                    color = "k", 
                    linewidth = 1, 
                    linestyle = "--", 
                    zorder = 0)
            ax[i].annotate("{}%".format(np.around(complete_us.iloc[-1], decimals = 1)), 
                        (one_dose_us.index[0], complete_us.iloc[-1]+1))

    # Plot trends.
    counties = [county_1, county_2]
    for i in range(len(counties)):
        c = counties[i].split(" County, ")[0].lower()
        s = counties[i].split(" County, ")[1].lower()
        df = load_county_data(county = c, state = s, download_data = download_data)
        complete = county_complete_pct(df)
        one_dose = county_one_dose_pct(df)
        expected = county_expected_complete_pct(df)
        
        ax[i].plot(one_dose, 
                color = COLORS["dark_blue"], 
                linewidth = 3, 
                label = "One Dose")
        ax[i].plot(complete, 
                color = COLORS["light_orange"], 
                linewidth = 3, 
                label = "Completely Vaccinated")
        ax[i].plot(expected, 
                color = "gray", 
                linestyle = "dotted",
                linewidth = 3, 
                label = "Expected Completely Vaccinated", 
                zorder = 0)
        
        ax[i].set_title(counties[i])    

    
        ax[i].set_xlabel("Date")
        
        if i == 0:
            ax[i].set_yticks([0,20,40,60,80])
            ax[i].set_yticklabels(["{}%".format(20*i) for i in range(5)])
            ax[i].set_ylabel("")
        
        if i == 1:
            ax[i].legend(loc = "lower right")
    
    plt.suptitle("County Level Vaccine Trends")
    plt.show
    return None
