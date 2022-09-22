import pandas as pd
import numpy as np

from datetime import timedelta
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from pySankey.sankey import sankey
from scipy import stats

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid_spec

from .geolocations import *

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

BELIEF_CMAP = LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cm.PuRd.name, a=0.1, b=0.7),
        cm.PuRd(np.linspace(0.1, 0.7, 100)))

SIM_CMAP = LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cm.BuGn.name, a=0.1, b=0.7),
        cm.BuGn(np.linspace(0.1, 0.7, 100)))

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

def load_county_trend_data(county, state, download_data= False):
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
        y = df.index[-1].year
        m = f"{df.index[-1].month:02}"
        d = f"{df.index[-1].day:02}"
        ax.plot((one_dose.index[0], one_dose.index[-1]),
                (complete.iloc[-1],complete.iloc[-1]), 
                color = "k", 
                linewidth = 1, 
                linestyle = "--", 
                zorder = 0,
                label = "US Complete Vaccination Rate {}-{}-{}".format(y,m,d))
        ax.annotate("{}%".format(np.around(complete.iloc[-1], decimals = 1)), 
                    (one_dose.index[0], complete.iloc[-1]+1))

    ax.set_title("US National Vaccination Rates", fontsize = 15) 

    # Load county data.
    if county:

        if state:

            c = county.lower().split(" county")[0]
            s = state.upper()
            df = load_county_trend_data(county = c, state = s, 
                download_data = download_data)
        
            complete = county_complete_pct(df)
            one_dose = county_one_dose_pct(df)
            expected = county_expected_complete_pct(df)

            ax.set_title("Vaccination Rates in {} County, {}".format(
                                c.capitalize(), s), fontsize = 15)   

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


    ax.set_xlabel("Date", fontsize = 12)
    ax.set_yticks([0,20,40,60,80])
    ax.set_ylim(0,90)
    ax.set_yticklabels(["{}%".format(20*i) for i in range(5)])
    ax.set_ylabel("Percentage", fontsize = 12)
    ax.legend(loc = "lower right", prop = {"size":12})
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
    ax.set_title("Relative US National Vaccination Rates", fontsize = 15)

    if county:

        if state:
            c = county.lower().split(" county")[0]
            s = state.upper()
            df = load_county_trend_data(county = c, state = s, 
                download_data = download_data)
        
            complete = county_complete_pct(df)
            one_dose = county_one_dose_pct(df)
            expected = county_expected_complete_pct(df)

            ax.set_title("Relative Vaccination Rates in {} County, {}".format(
                                c.capitalize(), s), fontsize = 15)  

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
    
    ax.set_ylabel("Percentage Points", fontsize = 12)
    ax.set_xlabel("Date", fontsize = 12)
    ax.set_ylim(-12,12)
    ax.legend(loc = "lower left", prop = {"size":12})
    
    plt.show()
    return None

def plot_triangulated_county(geo_df, bounding_box = None, restricted = False, aspect_ratio = 1):
    """ Plots county with triangular regions.
    
    Inputs: 
        geo_df: (dataframe) geographic datatframe including county geometry.
        bounding_box: (list) list of 4 vertices determining a bounding box 
                where agents are to be added.  If no box is given, then the 
                bounding box is taken as the envelope of the county.
        restricted: (bool) if True then region is restrict to bounding box.
        aspect_ratio: (float) aspect ratio of final plot.

    Returns: 
        Boundary of county and triangulation of region.
    """
    tri_dict = make_triangulation(geo_df)
    tri_df = gpd.GeoDataFrame({"geometry":[Polygon(t) for t in tri_dict["geometry"]["coordinates"]]})
    
    # Establish initial CRS
    tri_df.crs = "EPSG:3857"

    # Set CRS to lat/lon
    tri_df = tri_df.to_crs(epsg=4326) 

    fig, ax = plt.subplots(figsize = (10,10))
    linewidth = 1
    # Get bounding box geometry.
    if bounding_box is not None:
        sq_df = gpd.GeoDataFrame({"geometry":[Polygon(bounding_box)]})
        
        # Get bounded triangles.
        if restricted == True:
            inset = [i for i in tri_df.index if tri_df.loc[i,"geometry"].within(sq_df.loc[0,"geometry"])]
            tri_df = tri_df.loc[inset,:].copy()

            # Set plot limits.
            minx = np.array(bounding_box)[:,0].min()
            miny = np.array(bounding_box)[:,1].min()
            maxx = np.array(bounding_box)[:,0].max()
            maxy = np.array(bounding_box)[:,1].max()
            
            ax.set_xlim(minx - .0005,maxx + .0005)
            ax.set_ylim(miny - .0005,maxy + .0005)

            linewidth = 4

    # Plot triangles
    tri_df.boundary.plot(ax = ax, 
                        alpha=1, 
                        linewidth = linewidth,
                        edgecolor = COLORS["light_blue"])

    # Plot county boundary.
    geo_df.crs = "EPSG:3857"
    geo_df = geo_df.to_crs(epsg=4326) 
    geo_df.boundary.plot(ax = ax,edgecolor = "black", linewidth = linewidth)

    # Plot bounding box.
    if bounding_box is not None:
        if restricted == False:
            sq_df.boundary.plot(ax = ax, 
                                alpha = 1, 
                                linestyle = "--", 
                                linewidth = 2, 
                                color = COLORS["dark_orange"])

    ax.set_axis_off()
    ax.set_aspect(aspect_ratio)
    plt.show()

    return None

def plot_agents_on_triangle(triangle_object, agent_df):
    """ Returns triangle filled with agents.

    Inputs: 
        triangle_object : (polygon) shapely triangle object.
        agent_df: (dataframe) x,y coordinates for agents.

    Outputs: 
        Plot of points on triangle.
    """
    fig, ax = plt.subplots(figsize = (8,8))
    df = gpd.GeoDataFrame({"geometry": triangle_object}, index = [0])
    df.boundary.plot(ax = ax, alpha=1, edgecolor = COLORS["light_blue"])
    ax.scatter(agent_df["x"], agent_df["y"], color = COLORS["dark_blue"], 
        zorder = 0)
    ax.set_axis_off()
    plt.show()
    return None

def plot_agents_with_belief_and_weight(belief_df):
    """ Returns triangle filled with agents.

    Inputs: 
        triangle_object : (polygon) shapely triangle object.
        agents: (dataframe) x,y coordinates for agents.

    Outputs: 
        Plot of points on triangle.
    """
    fig, ax = plt.subplots(figsize = (8,8))
    ax.scatter(x = belief_df["x"], 
                y = belief_df["y"],
                s = [int(w) for w in belief_df["weight"].values],
                c = belief_df["belief"],
                cmap = BELIEF_CMAP)

    ax.scatter([],[],color = BELIEF_CMAP(0), label = "Not Hesitant")
    ax.scatter([],[],color = BELIEF_CMAP(128), label = "Hesitant or Unsure")
    ax.scatter([],[],color = BELIEF_CMAP(256), label = "Strongly Hesitant")
    ax.set_axis_off()     
    plt.legend(loc = "best", prop={'size': 15})
    
    plt.show()
    return None


def plot_network(model):
    """ OpinionNetworkModel instance
    """
    # Load point df.
    belief_df = model.belief_df
    
    op = model.include_opinion
    wt = model.include_weight

    adjacency_df = model.adjacency_df
    adjacency_df.columns = [int(i) for i in adjacency_df.columns]

    cc = model.clustering_coefficient
    md = model.mean_degree
    # Plot people.
    fig, ax = plt.subplots(figsize = (8,8))
    ax.scatter(x = belief_df["x"], 
                y = belief_df["y"],
                s = [int(w) for w in belief_df["weight"].values],
                c = belief_df["belief"],
                cmap = BELIEF_CMAP)

    for j in belief_df.index:
        for k in np.where(adjacency_df.loc[j,:] == 1)[0]:
            ax.plot((belief_df.loc[j,"x"], belief_df.loc[k,"x"]),
                            (belief_df.loc[j,"y"], belief_df.loc[k,"y"]),                            
                            color = "k", lw = .5, zorder = 0)

    # Turn off axes
    ax.set_axis_off()
    
    # Add title
    title = "Connections based on physical distance"
    if op == True:
        if wt == True:
            title = "Connections based on physical distance, opinion proximity and weight"
        if wt == False:
            title = "Connections based on physical distance and opinion proximity"
    if op == False:
        if wt == True:
            title = "Connections based on physical distance and weight"
        if wt == False:
            title = "Connections based on physical distance"



    title = title + "\n clustering coefficient: " + str(
        np.around(cc, decimals = 3)) + "\n average in-degree: " + str(
        np.around(md, decimals = 1))
    ax.set_title(title)
        
    # Add legend
    ax.scatter([],[],color = BELIEF_CMAP(0), label = "Not Hesitant")
    ax.scatter([],[],color = BELIEF_CMAP(128), label = "Hesitant or Unsure")
    ax.scatter([],[],color = BELIEF_CMAP(256), label = "Strongly Hesitant")    
    plt.legend(loc = "best")
    plt.axis()

    plt.show()

    None


def get_ridge_plot(dynamic_belief_df, 
                phases = [], 
                reach_dict = None, 
                show_subplot_labels = True,
                show_title = True):
    """ Ridgeplot of updating beliefs.

    Inputs: 
        dynamic_belief_df: (dataframe) updating beliefs across multiple phases
        phases: (list) phases to show in plot.
        reach_dict: (dictionary) value is propotional reach of key.
        show_subplot_labels: (bool) if True show subplot labels.
        show_title: (bool) if True show plot title.

    Ouputs: 
        Ridgeplot of updating belief distributions over phases.
    """
    if phases == []:
        phases = dynamic_belief_df.shape[1] - 1
        if phases < 5:
            phases = [t for t in range(phases +1)]
        else:
            t = phases // 5
            phases = [0] + [t * (i+1) for i in range(1,5)]

    xx = np.linspace(-2,2,1000)
    gs = grid_spec.GridSpec(len(phases),1)
    fig = plt.figure(figsize=(8,4))

    i = 0

    ax_objs = []

    for p in range(len(phases)):
        ax_objs.append(fig.add_subplot(gs[i:i+1, 0:]))
        x = dynamic_belief_df[phases[p]].values
        kde = stats.gaussian_kde(x)
        ax_objs[-1].plot(xx, kde(xx), color = SIM_CMAP(0))
        c = int((p * 256) / (len(phases) -1))
        ax_objs[-1].fill_between(xx,kde(xx), color=SIM_CMAP(c), alpha = 0.8)

        ax_objs[-1].set_yticks([])
        ax_objs[-1].set_yticklabels([])
        ax_objs[-1].set_ylabel('')
        
        #ax_objs[-1].set_axis_off()
        if show_subplot_labels == True:
            ax_objs[-1].text(2.1,0,"{} time steps".format(phases[p]),
                fontweight = "bold",
                fontsize=10,ha="left")

        # make background transparent
        rect = ax_objs[-1].patch
        rect.set_alpha(0)

        ax_objs[-1].set_xlim(-2,2)
        if i == len(phases)-1:
            ax_objs[-1].set_xticks([-1,0,1])
            ax_objs[-1].set_xticklabels([r"$\leftarrow$ Less Hesitant", 
                                        "Unsure",
                                        r"More Hesitant $\rightarrow$"])
            ax_objs[-1].tick_params(axis='both', which='both', length=0)
        else:
            ax_objs[-1].set_xticks([])
            ax_objs[-1].set_xticklabels([])

        spines = ["top","right","left","bottom"]
        for s in spines:
            ax_objs[-1].spines[s].set_visible(False)
            
        i += 1
        
    
    gs.update(hspace=-0.7)
    left = int(reach_dict[-1] * 100)
    right = int(reach_dict[1] *100)
    if show_title == True:
        plt.title("Left Reach: {}%    Right Reach: {}%".format(left, right), 
            y=-.4, fontweight = "bold")

    return None


def get_sankey_plot(dynamic_belief_df, start_hesitant,vaccination_threshold, hesitant_threshold, figure_name):
    """ Create sankey plot of belief evolution
    
    Input:
        dynamic_belief_df: (dataframe) showing changing beliefs over time
        start_hesotamt: (int) percentage of population that starts hesitant.
        vaccination_threshold: (int) value below which individuals are considered 
            vaccinated; this is typically, model.threshold.
        hesitant_threshold: (int) this is the value above which individuals are 
            considered hesitant.
        figure_name: (str) filepath to save figure.
            
    Output: 
        Sankey plot showing vaccinated, willing, hesitant belief over time
    """
    colors = {}
    base_colors = {"willing": "gold",
              "hesitant": "tomato",
              "vaccinated": "seagreen"
                }
    end_df = dynamic_belief_df.iloc[:,-1]
    end_df = end_df.sort_values(ascending = False)
    end_df = pd.DataFrame(end_df.sort_values(ascending = False))
    end_df = end_df.rename(columns = {end_df.columns[-1]:"value"})
    end_df["class"] = "willing"
    end_df.loc[end_df[end_df["value"] <= vaccination_threshold].index,"class"] = "vaccinated"
    end_df.loc[end_df[end_df["value"] >= hesitant_threshold].index,"class"] = "hesitant"
    end_dict = (end_df["class"].value_counts()/end_df.shape[0]).to_dict()
    for k in ["vaccinated","willing","hesitant"]:
        if k in end_dict.keys():
            v = str(int(100 * end_dict[k]))
            end_df["class"] = end_df["class"].replace(k,v)
            colors[v] = base_colors[k]
        
    start_df = dynamic_belief_df.iloc[:,0]
    start_df = pd.DataFrame(start_df.loc[end_df.index])
    start_df = start_df.rename(columns = {0:"value"})
    start_df["class"] = str(100 - start_hesitant)
    start_df.loc[start_df[start_df["value"] > hesitant_threshold].index,"class"] = str(31)
    start_dict = (start_df["class"].value_counts()/start_df.shape[0]).to_dict()
    colors[str(100 - start_hesitant)] = "gold"
    colors[str(start_hesitant)] = "tomato"

    sankey(left = start_df["class"], right = end_df["class"], aspect=20, colorDict=colors, fontsize=20)
    plt.gcf().set_size_inches(6,6)
    plt.savefig(f'{figure_name}.png',bbox_inches='tight',dpi=150)
    plt.close()