#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
from sklearn.cluster import KMeans
import random
import matplotlib.pyplot as plt
import seaborn as sns

print('PART A.1')

# importing mug data
df_mugs = pd.read_excel('mugs-analysis-full-incl-demographics.xlsx', sheet_name='for-cluster-analysis')
# importing purchase probabilities
df_prob = pd.read_excel('mugs-analysis-full-incl-demographics.xlsx', sheet_name='mugs-full', skiprows=[i for i in range(1,30)], usecols='A,BD:BF')
df_prob.columns = ['Cust', 'P1', 'P2', 'P3']
# merge df_mugs and df_prob
df_mugs_and_probs = df_mugs.merge(df_prob, on='Cust')
# print(df_mugs_and_probs.head())
# print(df_mugs_and_probs.columns)
print('-------------------------------------------------------------------------------------------------------------------')
for col in [' IPr', 'Iin', ' ICp', ' ICl', 'Icn', ' IBr', 'I*pPr30',
       'I*pPr10', 'I*pPr05', 'I*pIn0.5', 'I*pIn1', 'I*pIn3', 'I*pCp12',
       'I*pCp20', 'I*pCp32', 'I*pClD', 'I*pClF', 'I*pClE', 'I*pCnSl',
       'I*pCnSp', 'I*pCnLk', 'I*pBrA', 'I*pBrB', 'I*pBrC', 'income', 'age',
       'sports', 'gradschl']:
    
    col_name = 'P3*' + col
    df_mugs_and_probs[col_name] = df_mugs_and_probs['P3'] * df_mugs_and_probs[col]

cols_p3 = [col for col in df_mugs_and_probs if col.startswith('P3*')]
df_prob_p3 = df_mugs_and_probs[cols_p3]
df_prob_p3.columns = [col[3:] for col in df_prob_p3]

# sum of purchase probabilities for P3
weight_sum_p3 = df_mugs_and_probs['P3'].sum()
# weighted averages
df_wavg_p3 = pd.DataFrame(df_prob_p3.sum(axis=0)).transpose() / weight_sum_p3
df_wavg_p3.index = ['P3 Mean']
cols_p3 = df_wavg_p3.columns
vals_p3 = df_wavg_p3.values[0]

for i in range(len(cols_p3)):
    print(f'Weighted average of {cols_p3[i]} for P3 is {round(vals_p3[i],2)}')

# reinitializing
df_mugs_and_probs = df_mugs.merge(df_prob, on='Cust')

print('-------------------------------------------------------------------------------------------------------------------')

print('PART A.2, BRAND A')
for col in [' IPr', 'Iin', ' ICp', ' ICl', 'Icn', ' IBr', 'I*pPr30',
       'I*pPr10', 'I*pPr05', 'I*pIn0.5', 'I*pIn1', 'I*pIn3', 'I*pCp12',
       'I*pCp20', 'I*pCp32', 'I*pClD', 'I*pClF', 'I*pClE', 'I*pCnSl',
       'I*pCnSp', 'I*pCnLk', 'I*pBrA', 'I*pBrB', 'I*pBrC', 'income', 'age',
       'sports', 'gradschl']:

    col_name = 'P1*' + col
    df_mugs_and_probs[col_name] = df_mugs_and_probs['P1'] * df_mugs_and_probs[col]

cols_p1 = [col for col in df_mugs_and_probs if col.startswith('P1*')]
df_probs_p1 = df_mugs_and_probs[cols_p1]
df_probs_p1.columns = [col[3:] for col in df_probs_p1]
weight_sum_p1 = df_mugs_and_probs['P1'].sum()
df_wavg_p1 = pd.DataFrame(df_probs_p1.sum(axis=0)).transpose() / weight_sum_p1
df_wavg_p1.index = ['P1 Mean']
cols_p1 = df_wavg_p1.columns
vals_p1 = df_wavg_p1.values[0]
for i in range(len(cols_p1)):
    print(f'Weighted average of {cols_p1[i]} for P1 is {round(vals_p1[i],2)}')
# reinitializing
df_mugs_and_probs = df_mugs.merge(df_prob, on='Cust')

print('---------------------------------------------------------------------------------------------------------')

print('PART A.2, BRAND B')
for col in [' IPr', 'Iin', ' ICp', ' ICl', 'Icn', ' IBr', 'I*pPr30',
       'I*pPr10', 'I*pPr05', 'I*pIn0.5', 'I*pIn1', 'I*pIn3', 'I*pCp12',
       'I*pCp20', 'I*pCp32', 'I*pClD', 'I*pClF', 'I*pClE', 'I*pCnSl',
       'I*pCnSp', 'I*pCnLk', 'I*pBrA', 'I*pBrB', 'I*pBrC', 'income', 'age',
       'sports', 'gradschl']:   
    col_name = 'P2*' + col
    df_mugs_and_probs[col_name] = df_mugs_and_probs['P2'] * df_mugs_and_probs[col]

cols_p2 = [col for col in df_mugs_and_probs if col.startswith('P2*')]
df_prob_p2 = df_mugs_and_probs[cols_p2]
df_prob_p2.columns = [col[3:] for col in df_prob_p2]
weight_sum_p2 = df_mugs_and_probs['P2'].sum()
df_wavg_p2 = pd.DataFrame(df_prob_p2.sum(axis=0)).transpose() / weight_sum_p2
df_wavg_p2.index = ['P2 Mean']
cols_p2 = df_wavg_p2.columns
vals_p2 = df_wavg_p2.values[0]
for i in range(len(cols_p2)):
    print(f'Weighted average of {cols_p2[i]} for P2 is {vals_p2[i]:.2f}')
df_mugs_and_probs = df_mugs.merge(df_prob, on='Cust')
# print(df_wavg_p2.head())
print('---------------------------------------------------------------------------------------------------------')

print('PART A.2, LOG LIFTS')
# descriptors
df_desc = mugs_probs_df[mugs_probs_df.columns[1:-3]]
# overall means
df_avg = pd.DataFrame(df_desc.mean(axis=0)).transpose()
df_avg.index = ['Overall Mean']
cols_avg = df_avg.columns
vals_avg = df_avg.values[0]
for i in range(len(cols_avg)):
    print(f'Overall average of {cols_avg[i]} is {round(vals_avg[i],2)}')

print('---------------------------------------------------------------------------------------------------------')
# dataframe with segment means and overall means
df_final = pd.concat([df_wavg_p1, df_wavg_p2, df_wavg_p3, df_avg], axis=0)
# transpose to simplify the log-lift computations
df_final_T = df_final.transpose()
# construct log lift dataframe
for col in ['P1 Mean', 'P2 Mean', 'P3 Mean']:
    col_name = col[:2] + ' LL'
    df_final_T[col_name] = np.log10(df_final_T[col] / df_final_T['Overall Mean'])
df = df_final_T.transpose()
log_lift_df = df.iloc[-3:]
print(log_lift_df)

# display heatmap of log-lifts
plt.figure(figsize = (12,12))
heatmap = sns.heatmap(log_lift_df, annot=True, center=0,cmap = 'BrBG', xticklabels=log_lift_df.columns, annot_kws={'size':6,'rotation': 90})
plt.savefig('affinity.png',dpi = 600) 
plt.tight_layout()
plt.show()
print('------------------------------------------------------------------------------------------------------------------')

print('PART B, K MEANS ANALYSIS')
# importing data
df_mugs = pd.read_excel('mugs-analysis-full-incl-demographics.xlsx', sheet_name='for-cluster-analysis')
X = np.array(df_mugs.drop(columns=['Cust', 'income', 'age', 'sports', 'gradschl']))
k_vals = [0 for i in range(2,11)]
within_cluster_ss_vals = [0 for i in range(2,11)]

# k means
for i, k in enumerate(range(2,11)):
    random.seed(410014)
    k_means_mod = KMeans(n_clusters=k, n_init=50, max_iter=100)
    k_means_mod.fit(X)
    # within-cluster sum of squares
    within_cluster_ss = k_means_mod.inertia_ / X.shape[0]
    # store values
    k_vals[i] = k
    within_cluster_ss_vals[i] = within_cluster_ss
# plotting results
plt.figure(figsize = (8,8))
plt.plot(k_vals, within_cluster_ss_vals, color = 'red')
plt.xlabel('# clusters')
plt.ylabel('Within-cluster SS')
plt.grid()
plt.savefig('kmeans.png',dpi = 600) 
plt.show()

print('-------------------------------------------------------------------------------------------------------------')

print('PART B.1')
random.seed(410014)
# initializing the model with 4 clusters
k_means_mod = KMeans(n_clusters=4, n_init=50, max_iter=100)
X = np.array(df_mugs.drop(columns=['Cust', 'income', 'age', 'sports', 'gradschl']))
k_means_mod.fit(X)
k_means_df = df_mugs.copy()
k_means_df['Cluster ID'] = k_means_mod.labels_
df_agg = k_means_df.groupby('Cluster ID').mean()
print(df_agg.drop(columns=['Cust']))
print('-------------------------------------------------------------------------------------------------')

print('PART B.2')
# transposing
df_agg_T = df_agg.transpose()
# overall mean
df_agg_T['Overall Mean'] = df_mugs.drop(columns=['Cust']).mean(axis=0)
# log lift dataframe
for col in [i for i in range(4)]:
    col_name = 'Seg. ' + str(col)
    df_agg_T[col_name] = np.log10(df_agg_T[col] / df_agg_T['Overall Mean'])
df_log_lift = df_agg_T.drop(columns=[0, 1, 2, 3, 'Overall Mean']).transpose()
df_log_lift.drop(columns=['Cust'], inplace=True)
print(df_log_lift.head())

# heatmap of log-lifts
plt.figure(figsize = (12,12))
heatmap = sns.heatmap(df_log_lift, annot=True, center=0,cmap = 'BrBG', xticklabels=df_log_lift.columns, annot_kws={'size':6,'rotation': 90})
plt.savefig('kmeans clusters.png',dpi = 600) 
plt.tight_layout()
plt.show()


# In[ ]:




