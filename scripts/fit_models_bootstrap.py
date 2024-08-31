"""
Run permutation+bootstrap analysis of GLMs. Fit each session X on another session Y.
Bootstrap sessions using leave-one-out method. Apply Holm-Bonferroni correction.

"""

# imports - general
import numpy as np 
import pandas as pd
from sklearn import linear_model
from sklearn import preprocessing
from statsmodels.stats.multitest import multipletests

# imports - custom
import sys
sys.path.append("code")
from settings import *

# settings
X_FEATURE = ['spike_count','burst_count','proportion_bursting','mean_velocity']
Y_FEATURE = ['exponent','periodic_pow']
STRUCTURES = [['LGd','VISp'],['VISl','VISp'],['LGd','VISl'],['VISp', 'VISl']]
ALPHA = 0.05 # significance

def main():
    # load results and reformat 
    df = pd.read_csv('scratch/feature_df.csv')
    df.insert(4, 'trial_idx', np.tile(np.repeat(np.arange(60), 30), 9))
    
    # init results
    num_models = len(Y_FEATURE)*len(STRUCTURES)*len(SESSIONS)*60*(len(SESSIONS)-1)

    df_results = pd.DataFrame({
    'session_id'     : [None]*num_models,
    'brain_structure x': [None]*num_models,
    'brain_structure y': [None]*num_models,
    'y_feature' : [None]*num_models,
    'r_squ' : [None]*num_models,
    'pseudo_session' : [None]*num_models,
    'loo_trial_idx' : [None]*num_models})
    
    # loop over sessions - fit model for each session
    c = 0
    for session in SESSIONS:
        # print(f"fitting session: " + str(session))
        for s in STRUCTURES:
            xs, ys = s
            for yf in Y_FEATURE:
                # pivot y_feature
                df_p = df.pivot_table(index=['session_id','sweep','trial','trial_idx','bin'], 
                                         values=X_FEATURE.copy().append(yf), 
                                         columns='brain_structure').reset_index()

                # get pseudo data
                for pseudo_session in SESSIONS:
                    if session == pseudo_session: continue # skip same session
                    for loo_trial_idx in range(60): # index to drop
                        y = df_p.loc[(df_p['session_id']==pseudo_session) & \
                                        (df_p['trial_idx']!=loo_trial_idx), yf][ys].values
                        x = df_p.loc[(df_p['session_id']==session) & \
                                        (df['trial_idx']!=loo_trial_idx), X_FEATURE]
                        x = x.loc[:, (slice(None), xs)] # drop VISp
                        x = np.concatenate((np.ones((len(x),1)),x),axis=1)
                        
                        # scaling doesn't seem to be necessary
                        scaler = preprocessing.StandardScaler().fit(x)
                        x_scaled = scaler.transform(x)

                        # fit GLM
                        reg = linear_model.LinearRegression().fit(x_scaled, y)

                        # r2
                        r2 = reg.score(x_scaled, y)

                        # store model results
                        df_results['session_id'][c] = session
                        df_results['brain_structure x'][c] = xs
                        df_results['brain_structure y'][c] = ys
                        df_results['y_feature'][c] = yf
                        df_results['r_squ'][c] = r2
                        df_results['pseudo_session'][c] = pseudo_session
                        df_results['loo_trial_idx'][c] = loo_trial_idx

                        # increase count
                        c += 1

    # save results
    df_results.to_csv('scratch/glm_bootstrap.csv', index=False)
    
    # compute significance
    df_glm = pd.read_csv("/scratch/glm_results.csv")
    df_bootstrap = pd.read_csv('/scratch/glm_bootstrap.csv')
    p_values = []
    for ii in range(len(df_glm)):
        row = df_glm.iloc[ii]
        df = df_bootstrap.copy()
        for col in ['session_id', 'brain_structure y', 'brain_structure x', 'y_feature']:
            df = df.loc[df[col]==row[col]]
        p_values.append(compute_pvalue(row['r_squ'], df['r_squ']))
    df_glm['p_value'] = p_values
    holm = multipletests(p_values, ALPHA, method='holm')
    df_glm['p_corr'] = holm[1]
    df_glm['significant'] = holm[0]
    df_glm.to_csv('scratch/glm_stats.csv', index=False)


def compute_pvalue(value, distribution):
    return np.sum(distribution>value) / len(distribution)


if __name__ == '__main__':
    main()
