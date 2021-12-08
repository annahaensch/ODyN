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


def load_national_trend_data():
      pd.read_csv("../data/load_national_trend_data.csv")

	return 