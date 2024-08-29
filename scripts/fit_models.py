"""
Fit GLMs.

"""

# imports - general
import numpy as np 
import pandas as pd
from sklearn import linear_model
from sklearn import preprocessing

# imports - custom
import sys
sys.path.append("code")
from settings import *

# settings
X_FEATURES = ['burst_count', 'spike_count']
Y_FEATURE = 'exponent'
Y_STRUCTURE = 'VISp'

def main():
    # load results and reformat 
    results = pd.read_csv('results/feature_df.csv')
    df = results.pivot_table(index=['session_id','sweep','trial','bin'], 
                             values=['exponent', 'burst_count','spike_count'], 
                             columns='brain_structure').reset_index()

    # loop over sessions - fit model for each session
    for session_id in SESSIONS:
        # get data for one session
        df_s = df[df['session_id'] == session_id]
        y = df[Y_FEATURE][Y_STRUCTURE].values
        X = df[X_FEATURES].values

        # scale data
        scaler = preprocessing.StandardScaler().fit(X)
        X_scaled = scaler.transform(X)

        """
        REPLACE CODE BELOW WITH DESIRED MODEL
        """
        # fit model
        clf = linear_model.GammaRegressor(max_iter=10000)
        clf.fit(X_scaled, y)

        """
        STORE MODEL RESULTS
        """

    """
    STORE MODEL RESULTS TO FILE
    """


if __name__ == '__main__':
    main()
