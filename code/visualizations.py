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
    """ Returns dataframe of national vaccine trends.
    """
    df = pd.read_csv("../data/national_level_vaccination_trends.csv")
    df = df[df["Date Type"] == "Admin"]
    df["Date"] = [pd.to_datetime(d) for d in df["Date"]]
    df.set_index("Date", drop = True, inplace = True)
    return df

def load_county_data(county, state, download_data= False):
    """ Returns dataframe of county level vaccine trends.

    Inputs: 
        county - (str) Capitalized county name "<Name> County"
        state - (str) Upper-case two letter state abbreviation.
        download_data: (bool) Download full dataset if True

    Returns:
        Returns county and state level vaccination data for 
        county, state, if download_data = True then it is 
        downloaded directly from the CDC website: 
        https://data.cdc.gov/Vaccinations/COVID-19-Vaccinations-
        in-the-United-States-County/8xkx-amqh.

    """
    c = county.lower()
    s = state.lower()
    
    if download_data == True:
        # Caution: this will take a long time.
        df = pd.read_csv(
        "https://data.cdc.gov/api/views/8xkx-amqh/rows.csv?accessType=DOWNLOAD")
        df = df[(df["Recip_County"] == "{} County".format(county.capitalize())
            ) & (df["Recip_State"] == state.upper())]

    else:
        df = pd.read_csv("../data/{}_county_{}_vaccination_trends.csv".format(
            c,s), index_col = 0)
    
    df["Date"] = [pd.to_datetime(d) for d in df["Date"]]
    df.sort_values(by = "Date", inplace = True)
    df.set_index("Date", drop = True, inplace = True)
    
    return df

def national_complete_pct(df):
    """ Returns timeseries of percentage completely vaccinated.

    Input: 
        df - (dataframe) national vaccination trends.

    Output: 
        Dataframe of national percentage fully vaccinated by date.

    """
    complete = 100 * df['People Fully Vaccinated Cumulative']/US_POPULATION
    return complete.loc[START_DATE:]

def national_one_dose_pct(df):
    """ Returns timeseries of percentage with one dose. 

    Input: 
        df - (dataframe) national vaccination trends.

    Output:
        Dataframe of national percentage with one dose by date.
    """
    one_dose = 100 * df[
            'People Receiving 1 or More Doses Cumulative']/US_POPULATION
    return one_dose.loc[START_DATE:]

def national_expected_complete_pct(df):
    """ Returns timeseries of expected percentage completely vaccinated.

    Input: 
        df - (dataframe) national vaccination trends.

    Output: 
        Dataframe of national percentage expected complete by date.
    """
    one_dose = 100 * df['People Receiving 1 or More Doses Cumulative'
                            ]/US_POPULATION     
    expected = one_dose.loc[START_DATE - timedelta(days = 42
        ):one_dose.index[-1] - timedelta(days = 42)]
    expected = pd.Series(expected.values, index = pd.date_range(
        START_DATE, one_dose.index[-1]))
    return expected

def county_complete_pct(df):
    """ Returns timeseries of percentage completely vaccinated.

    Input: 
        df - (dataframe) county level vaccination trends.

    Output: 
        Dataframe of county level percentage complete by date.
    """
    return df["Series_Complete_Pop_Pct"].loc[START_DATE:,]

def county_one_dose_pct(df):
    """ Returns timeseries of percentage with one dose.
    
    Input: 
        df - (dataframe) county level vaccination trends.

    Output: 
        Dataframe of county level percentage with one dose by date.
    """
    return df['Administered_Dose1_Pop_Pct'].loc[START_DATE:,]

def county_expected_complete_pct(df):
    """ Returns timeseries of percentage expected completely vaccinated.
    
    Input: 
        df - (dataframe) county level vaccination trends.

    Output: 
        Dataframe of county level percentage expected complete by date.
    """
    one_dose = df['Administered_Dose1_Pop_Pct']
    expected = one_dose.loc[START_DATE - timedelta(days = 42
        ):one_dose.index[-1] - timedelta(days = 42)]
    expected = pd.Series(expected.values, index = pd.date_range(
        START_DATE, one_dose.index[-1]))
    
    return expected

def vaccine_trends_plot(county = None, 
                        state = None,
                        show_us_current = False,
                        download_data = False):
    """ Returns line plot of county vaccine trends.
    
    Inputs: 
        county - (str) Capitalized county name or None for national 
            level data.
        state - (str) Upper-case two letter state abbreviation or None
            for national level data.
        show_us_current - (bool) set to False to hide vertical line
            at current us vaccination rate.
        download_data - (bool) set to True to download data directly 
            from CDC webiste.  Warning: this is slow.

    Returns: 
        Line plot of percentage complete, one-dose, and expected complete
        over time with optional vertical line at national current level.
    """
    df = load_national_trend_data()
    
    complete = national_complete_pct(df)
    one_dose = national_one_dose_pct(df)
    expected = national_expected_complete_pct(df)
    
    fig, ax = plt.subplots(figsize = (8,5))
    
    # Add horizontal line at current completely vaccinated.
    if show_us_current == True:
        ax.plot((one_dose.index[0], one_dose.index[-1]),
                (complete.iloc[-1],complete.iloc[-1]), 
                color = "k", 
                linewidth = 1, 
                linestyle = "--", 
                zorder = 0)
        ax.annotate("{}%".format(np.around(complete.iloc[-1], decimals = 1)), 
                    (one_dose.index[0], complete.iloc[-1]+1))

    ax.set_title("US National Vaccination Rates") 

    # Load county data.
    if county:

        if state:

            c = county.lower().split(" county")[0]
            s = state.upper()
            df = load_county_data(county = c, state = s, 
                download_data = download_data)
        
            complete = county_complete_pct(df)
            one_dose = county_one_dose_pct(df)
            expected = county_expected_complete_pct(df)

            ax.set_title("Vaccination Rates in {} County, {}".format(
                                c.capitalize(), s))   

        else:
            raise ValueError("A two-letter state abbreviation must be given.")
    
    # Plot trends
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
            label = "Expected Completely Vaccinated", 
            zorder = 0)

    ax.set_xlabel("Date")
    ax.set_yticks([0,20,40,60,80])
    ax.set_yticklabels(["{}%".format(20*i) for i in range(5)])
    ax.set_ylabel("Percentage")
    ax.legend(loc = "lower right")
    plt.show()
    return None

def relative_vaccine_trends_plot(county = None,
                                state = None,
                                download_data = False):
    """ Returns bar chart of percentage +/- expected complete.

    Inputs: 
        county - (str) Capitalized county name or None for national 
            level data.
        state - (str) Upper-case two letter state abbreviation or None
            for national level data.
        download_data - (bool) set to True to download data directly 
            from CDC webiste.  Warning: this is slow.

    Returns: 
        Bar chart showing percentage points above of below the 
        expected vaccine rate as a function of time.
    """
    df = load_national_trend_data()
    complete = national_complete_pct(df)
    expected = national_expected_complete_pct(df)

    fig, ax = plt.subplots(figsize = (8,5))
    ax.set_title("Percentage Points Difference Between Expected \n and Actual "
                        "National Vaccine Rates")

    if county:

        if state:
            c = county.lower().split(" county")[0]
            s = state.upper()
            df = load_county_data(county = c, state = s, 
                download_data = download_data)
        
            complete = county_complete_pct(df)
            one_dose = county_one_dose_pct(df)
            expected = county_expected_complete_pct(df)

            ax.set_title("Percentage Points Difference Between Expected \n and "
                "Actual Vaccine Rates in {} County, {}".format(
                                c.capitalize(), s))  

        else:
            raise ValueError("A two-letter state abbreviation must be given.")
    

    # Compute difference between expected and actual.
    diff = complete - expected
    diff_weekly = pd.DataFrame(index = pd.date_range(diff.index[0],
                                                diff.index[-1], freq = "W"),
                             columns = ["mean"])

    for i in range(diff.shape[0]-1):
        start = diff.index[i]
        end = diff.index[i+1]
        diff_weekly.loc[start,'mean'] = diff.loc[start:end].mean()

    color = [COLORS["teal"] if t >= 0 else COLORS["pink"] for t in 
                                diff_weekly["mean"]]
    
    # Plot trends.
    ax.bar(x = diff_weekly.index, width = 1, 
        height = diff_weekly["mean"],
        color = color)
    
    # Add empty plot to generate legend.
    ax.plot([],[],
        color = COLORS["teal"], 
        label= "More people than expected are completely vaccinated", 
        linewidth = 3)
    ax.plot([],[],
        color = COLORS["pink"], 
        label= "Fewer people than expected are completely vaccinated", 
        linewidth = 3)
    
    ax.set_ylabel("Percentage Points")
    ax.set_xlabel("Date")
    ax.set_ylim(-12,12)
    ax.legend(loc = "lower left")
    
    plt.show()
    return None





def plot_initial_networks(folder_name):

    opinions = [False, True, False, True]
    weights = [False, False, True, True]

    # Load point df.
    point_df = pd.read_csv(
        "../data/symmetric_simulations_60periods_1000_2/point_df.csv",
        index_col = 0)

    fig, ax = plt.subplots(2,2,figsize = (15,15))
    
    for i in range(len(opinions)):
        op = opinions[i]
        wt = weights[i]
        adjacency_df = pd.read_csv(
            "../data/symmetric_simulations_60periods_1000_2/initial_adjacency_df_{}_{}.csv".format(
                str(op), str(wt)),index_col = 0)
        adjacency_df.columns = [int(i) for i in adjacency_df.columns]

        stats_df = pd.read_csv("../data/symmetric_simulations_60periods_1000_2/initial_stats_df.csv", index_col = 0)
        
        # Plot people.
        ax[i//2,i%2].scatter(point_df["x"], point_df["y"], 
                   s = [int(w) for w in point_df["weight"].values],
                   c = [COLOR_MAP[b] for b in point_df["belief"].values])

        for j in point_df.index:
            for k in np.where(adjacency_df.loc[j,:] == 1)[0]:
                ax[i//2,i%2].plot((point_df.loc[j,"x"], point_df.loc[k,"x"]),
                                (point_df.loc[j,"y"], point_df.loc[k,"y"]),                            
                                color = "k", lw = .5, zorder = 0)

        # Turn off axes
        ax[i//2,i%2].set_axis_off()
        
        # Add title
        title = "Connections based on physical distance"
        if i == 1:
            title = "Connections based on physical distance and opinion proximity"
        if i == 2:
            title = "Connections based on physical distance and weight"
        if i == 3:
            title = "Connections based on physical distance, opinion proximity and weight"
            
        cc = stats_df.loc[str(op)+"/"+str(wt),"clust_coeff"]
        md = stats_df.loc[str(op)+"/"+str(wt),"mean_deg"]

        title = title + "\n clustering coefficient: " + str(
            np.around(cc, decimals = 3)) + "\n average degree: " + str(
            np.around(md, decimals = 1))
        ax[i//2,i%2].set_title(title)
        
    # Add legend
    ax[1,1].scatter([],[],color = COLOR_MAP[0], label = "Not Hesitant")
    ax[1,1].scatter([],[],color = COLOR_MAP[1], label = "Hesitant or Unsure")
    ax[1,1].scatter([],[],color = COLOR_MAP[2], label = "Strongly Hesitant")     
    plt.legend(loc = "best")
    plt.axis()

    plt.savefig("{}initial_network_plot.png".format(folder_name))
    #plt.show()
