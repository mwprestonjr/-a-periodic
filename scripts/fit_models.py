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
X_FEATURE = ['spike_count','burst_count','proportion_bursting','exponent','periodic_pow','mean_velocity']
Y_FEATURE = ['exponent','periodic_pow']
STRUCTURES = [['LGd','VISp'],['VISl','VISp'],['LGd','VISl'],['VISp', 'VISl']]


def main():
    # load results and reformat 
    
    df = pd.read_csv('scratch/feature_df.csv')
    # do we need this?
    num_models = len(Y_FEATURE)*len(STRUCTURES)*len(SESSIONS)

    df_results = pd.DataFrame({
    'session_id'     : [None]*num_models,
    'brain_structure x': [None]*num_models,
    'brain_structure y': [None]*num_models,
    'y_feature' : [None]*num_models,
    'r_squ' : [None]*num_models})
    for ii, feature in enumerate(X_FEATURE):
        df_results[f"coef_{feature}"] = [None]*num_models

    # loop over sessions - fit model for each session
    c = 0
    for session in SESSIONS:
        print(f"fitting session: " + str(session))
        for s in STRUCTURES:
            xs, ys = s
            for yf in Y_FEATURE:
                # data frame session
                df_s = df[df['session_id'] == session]

                # get data for current session, predictor structure, observation structure
                y = df_s[df_s['brain_structure'] == ys][yf].values
                x = df_s[df_s['brain_structure'] == xs][X_FEATURE].values

                x = np.concatenate((np.ones((len(x),1)),x),axis=1)

                # fit model

                # scaling doesn't seem to be necessary
                scaler = preprocessing.StandardScaler().fit(x)
                x_scaled = scaler.transform(x)

                # fit GLM
                reg = linear_model.LinearRegression().fit(x_scaled, y)

                # r2
                r2 = reg.score(x_scaled, y)

                # coefficients
                coefs = reg.coef_

                # store model results
                df_results['session_id'][c] = session
                df_results['brain_structure x'][c] = xs
                df_results['brain_structure y'][c] = ys
                
                df_results['y_feature'][c] = yf
                df_results['r_squ'][c] = r2
                for ii, feature in enumerate(X_FEATURE):
                    df_results[f"coef_{feature}"][c] = coefs[ii+1]

                # increase count
                c += 1

    # save results
    df_results.to_csv('scratch/glm_results.csv', index=False)


if __name__ == '__main__':
    main()
