import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_generator import xz_plot
from fitting_main import get_profiles
from utils import *


file = "data_new/experimental/experimental_test_xyz.csv"
profile_df = get_profiles(file)[0]
profile_df['x'] -= profile_df['x'].min()
profile_df['z'] -= profile_df['z'].min()

# xz_plot(profile_df['x'].values, profile_df['z'].values, title="Experimental profile (pts=2176)")
plot_data_only(profile_df, title="Experimental profile", plot_show=True)
