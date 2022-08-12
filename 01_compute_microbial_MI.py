#!/Users/uryckid/Box Sync/Genohydrology/Projects/2021-Mutual_info/ghml_env/bin python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 12:15:24 2018

@author: uryckid
"""

import os
import time
from datetime import datetime
import argparse
import glob
import genohydro as gh
import numpy as np
import pandas as pd
from math import isnan
from csv import DictWriter
from scipy import stats

#os.chdir(os.path.dirname(__file__))
start_time = time.time() 
timestamp = datetime.now()
datestamp = timestamp.strftime('%Y%m%d')

def load_characteristics (charfile):
    stats_df = pd.read_csv(charfile, sep=',', header = 0, index_col = 'Site').dropna(axis = 0, how = 'all')
    stats_df.dropna(axis = 1, how = 'all', inplace = True)
    return stats_df
    
def getXy (inX, iny):
    sites = list(set(inX.index).intersection(iny.index))
    y = iny.loc[sites]  # target
    X = inX.loc[sites]  # features
    return X,y

def clip_samples(inotus, target_df):
    try: 
        ix = target_df.index.levels[0]
        otus_cur = inotus.loc[inotus.index.intersection(ix)]
    except: pass
    #print('Clipping...\nOTU DATA:\n', inotus.index, '\n\nTARGET DATA:', target_df.index)
    print('Clipping...')    
    joint = inotus.index.intersection(list(target_df.index))
    if joint.size>0:
        otus_cur = inotus.loc[inotus.index.intersection(list(target_df.index))]
    else: 
        inotus.rename(index = sample_dict, inplace=True) 
        otus_cur = inotus.loc[inotus.index.intersection(list(target_df.index))]
    return otus_cur

def normalize_by_range(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def shannon_entropy(x, bins):
    c = np.histogramdd(x, bins)[0]
    p = c/np.sum(c)
    p = p[p>0]
    h = -np.sum(p*np.log2(p))
    return h

def get_mi(otu_table, target, xlog = True, ylog=True, nbins = 5): 
    cols = ['Hx', 'Hy', 'Hxy', 'Ixy', 'Ixy/Hy']
    mis = [] 
    
        # Normalize asvs and targets
    if xlog: otu_table = normalize_by_range(np.log(1+otu_table))  
    else: otu_table = normalize_by_range(otu_table)
    if ylog: target = normalize_by_range(np.log(0.001+target.astype(float)))
    else: target = normalize_by_range(target)
    
    bins = np.linspace(0, 1, nbins + 1)
    Hy = shannon_entropy([target], [bins,])
    
    for asv in otu_table.columns: 
        x = otu_table[asv]
        Hx = shannon_entropy([x], [bins])
        Hxy = shannon_entropy([x, target], [bins, bins])
        Ixy = Hx + Hy - Hxy
        Inorm = Ixy/Hy 
        
        mis.append([Hx, Hy, Hxy, Ixy, Inorm])
    MI = pd.DataFrame(mis, index = otu_table.columns, columns = cols)
    return MI
        
def shuff_surr(data, target, no_shuff):
    otus_sh = data.copy()
    cur_target_sh = target.copy()
    sh_frames = []
    for s in range(no_shuff):
       print("Shuffling {}/{}".format(s, no_shuff)) 
       np.random.shuffle(otus_sh.T.values)
       np.random.shuffle(cur_target_sh.values)
       sh_frames.append(get_mi(otus_sh, cur_target_sh)) 
       mi_ss = pd.concat(sh_frames, axis = 1, keys = list(range(no_shuff)), names = ['runs', 'MI'])
    return mi_ss

def sig_mi(true_df, shuff_df, crit_val=2.33):
    ss_means = shuff_df.groupby(by = 'MI', axis = 1).mean()
    ss_stds = shuff_df.groupby(by = 'MI', axis =1).std()
    sig_mi = true_df.loc[true_df['Ixy']>(ss_means['Ixy']+crit_val*ss_stds['Ixy'])]
    return sig_mi
    
def record_results(file_name, dict_of_elem):
    field_names = pd.read_csv(file_name).columns
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        dict_writer = DictWriter(write_obj, fieldnames=field_names)
        # Add dictionary as wor in the csv
        dict_writer.writerow(dict_of_elem)
    print('Record added:', dict_of_elem['date'])
    

def get_sig_mi(data, target, name, datestamp, records):
    mi_frames, mi_ss_frames, mi_sig_frames = [], [], []
    
    for i in target.columns: # iterate over targets (e.g., months)
        print('Calculating MI for %s%s...' %(name, [i]))
        records.update({'name': name, 'target': i})
        otus, cur_target = getXy(data, target[i].dropna())
        records.update({'n(MI)': len(cur_target)})
        records['study_sites'] = cur_target.index.to_list()
        
        mi = get_mi(otus, cur_target)
        mi_frames.append(mi)
        
        no_shuff = records['no_shuffles']
        mi_sh = shuff_surr(otus, cur_target, no_shuff)
        mi_ss_frames.append(mi_sh)
        
        mi_sig = sig_mi(mi, mi_sh)
        mi_sig_frames.append(mi_sig)
        
    if len(target.columns)>1:  
        mi_true = pd.concat(mi_frames, keys = target.columns, axis = 1)
        mi_true.to_csv(data_folder+mi_folder+'{}_mi_raw.csv'.format(name)) 
        mi_ss = pd.concat(mi_ss_frames, axis = 1, keys = target.columns)
        mi_ss.to_csv(data_folder+mi_folder+'{}_mi_ss_{}.csv'.format(name, no_shuff))     
        MI_sig = pd.concat(mi_sig_frames, keys = target.columns, join = 'outer', axis = 1)
        MI_sig.to_csv(data_folder+mi_folder+'{}_mi_sig.csv'.format(name))
        
    else: 
        mi.to_csv(data_folder+mi_folder+'{}_mi_raw.csv'.format(name))
        mi_sh.to_csv(data_folder+mi_folder+'{}_mi_ss_{}.csv'.format(name, no_shuff))
        mi_sig.to_csv(data_folder+mi_folder+'{}_mi_sig.csv'.format(name))
    record_results(records_file, records)    
    return 

# Arguments
only_joint_ASVs = False
if only_joint_ASVs: mi_folder = 'MI/joint/'
else: mi_folder = 'MI/all/'
useTrim = True # Determine whether to use full period of record for stream gage data
yrs = 10 # Number of years of record prior to sampling date to use for mean discharge calculations
doAnnual_Q = True # Calculate MI of annual absolute and specific discharge
doMonthly_Q = True # Calculate MI of mean monthly absolute and specific discharge'
doSeasonal_Q = True # Calculate MI of mean seasonal absolute and specific discharge
doDurations_Q = True # Calcuate MI of absolute and specific flow durations
dlist = [5, 95] # list of exceedence probabilites (integers
doMean_Qt = True # Calculate MI of daily dishcarge for specific time intervals prior to sampling
timesteps = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30] # List of timesteps prior to sample date (e.g., the value '3' is processed as 't-3 days prior to sampling) 
doArea = True # Calculate MI of sub-catchemnt area
no_shuffles = 3 # Number of shuffled surrogates used to develop significance threshold; default = 100'

###### LOAD DATA ######

data_folder = 'Project_data/'  
records_file = data_folder + 'Records.csv'

records = {'date': pd.to_datetime(datestamp), 'useTrim': useTrim, 'POR(max_yrs)': yrs, 
           'no_shuffles': no_shuffles, 'study_sites': []}

# SAMPLES
sample_meta_filename = data_folder  + "MiSeq692/genohydro_metadata_miseq692_clean.csv"
sample_meta_df= gh.load_metadata(sample_meta_filename)
# perms = sample_meta_df.loc[sample_meta_df.Sample_site.isin(['BEA-SHI', 'WAR-KAH', 'SHI-WAR'])]
# sample_meta_df = sample_meta_df.drop(perms.index, axis = 0)
sample_dict = dict(zip(sample_meta_df.index, sample_meta_df.Sample_site))

print('Sample data successfully loaded.\n')

# SITES
site_meta_filename = data_folder + 'Site_metadata_202110.csv'
site_meta_df = gh.load_metadata(site_meta_filename)
station_dict = dict(zip(site_meta_df['USGS number'], site_meta_df.index))
station_dict = {k: station_dict[k] for k in station_dict if not isnan(k)} # drop Nans (Andrews gages)
print('Site metadata (SS) successfully loaded.\n')

# SEQUENCES
otu_filename = data_folder  + 'MiSeq692/ASV-table-with-taxonomy-1450.txt'   
otu_df, tax_dict = gh.load_sequences(otu_filename)
print('OTU/ASV data successfully loaded.\n')

# FLOWS
try:
    daily_Q = pd.read_csv((glob.glob(data_folder + 'Flows/daily_Q*.csv'))[0], index_col = 0, parse_dates=['date'])
    print('Successfully loaded stored discharge data')
except:
    flows_filename = data_folder  + "Flows/discharge_all_20210822.csv" # cfs
    flows_andrews_filename = data_folder + "Flows/Andrews_DailyDischarge_HF00402_v12.csv" # cfs
    daily_Q = gh.load_flows(flows_filename, flows_andrews_filename, station_dict)
    daily_Q.to_csv(data_folder + 'Flows/daily_Q_%s.csv' %datestamp)
print('Flow files successfullly loaded and processed.\n')


###### TRIM DATA #######
# Subset samples #
# Summer samples
print('Selecting summer samples...')
samples_sum = gh.subsample(sample_meta_df)
filters = [(samples_sum.Watershed == 'Willamette'), (samples_sum.Watershed == 'Deschutes'), (samples_sum.Watershed == 'John Day')]
samples_sum_w = samples_sum.loc[filters[0], :]
samples_sum_d = samples_sum.loc[filters[1], :]
samples_sum_j = samples_sum.loc[filters[2], :]
samples_sum_wdj = pd.concat([samples_sum_w, samples_sum_d, samples_sum_j], axis = 0)
labels_sum_wdj = (samples_sum_wdj['sample-id'])
print('...success!\n')

# Select OTUs common to all three basins
print('Selecting OTUS/ASVs common to selected watersheds...')
otus_sum_w = otu_df.loc[otu_df.index.intersection(list(samples_sum_w['sample-id']))]
otus_sum_w = otus_sum_w.loc[:, otus_sum_w.sum()>0] # df of otus of summer labels with data
otus_sum_d = otu_df.loc[otu_df.index.intersection(list(samples_sum_d['sample-id']))]
otus_sum_d = otus_sum_d.loc[:, otus_sum_d.sum()>0]
otus_sum_j = otu_df.loc[otu_df.index.intersection(list(samples_sum_j['sample-id']))]
otus_sum_j = otus_sum_j.loc[:, otus_sum_j.sum()>0]
otus_sum_wdj = pd.concat([otus_sum_w, otus_sum_d, otus_sum_j], join = 'inner')
otus_sum_all = pd.concat([otus_sum_w, otus_sum_d, otus_sum_j], join = 'outer')
otus_sum_all.fillna(0, inplace=True)
otus_sum_all.rename(index = sample_dict, inplace=True)
otus_sum_all.to_csv('Project_data/MiSeq692/ASV_table_WDJ_summer.csv')
joint_otus = {'Willamette': list(otus_sum_w.columns), 'Deschutes': list(otus_sum_d.columns), 'John Day': list(otus_sum_j.columns)}
print('...success!\n')

if only_joint_ASVs: otus = otus_sum_wdj
else: otus = otus_sum_all

# Trim daily flow data to specified number of years
if useTrim: 
    print('\nTrimming Flow data to %s years prior to sampling...' %yrs)
    try: 
        daily_Q_10 = pd.read_csv((glob.glob(data_folder+'Flows/daily_Q_%s_*.csv' %(yrs)))[0], index_col = 0, parse_dates = ['date'])
        print('\nLoading \"%s\"...' %((glob.glob(data_folder+'Flows/daily_Q_%s_*.csv' %(yrs)))[0]))
    except: 
        print('\nCalculating...')
        daily_Q_10 = gh.trim_flow_dates(daily_Q, samples_sum_wdj, sample_dict, yrs)
        daily_Q_10.to_csv(data_folder+'Flows/daily_Q_%s_%s.csv' %(yrs, datestamp))
        
    # Analyze Period of Record
    por = pd.DataFrame(index = daily_Q_10.Site.unique(), columns = ['days', 'years'])
    for s in daily_Q_10.Site.unique():
        cur_data = daily_Q_10.loc[daily_Q_10.Site == s]
        por['days'].loc[s] = cur_data.date.max() - cur_data.date.min()
    por['years'] = (por.days/pd.Timedelta(1, 'D'))/365
    study_sites_q = set(por.index).intersection(samples_sum_wdj.Sample_site.unique())
    short =por.loc[(por.index.isin(study_sites_q)) & (por.years<(yrs*.5))]   
      
    dailyflows = daily_Q_10[~daily_Q_10.Site.isin(list(short.index))].copy()
    stamp = yrs 
else: 
    dailyflows = daily_Q.copy()
    stamp = 'full'
print('Done.')

ts1_frames = {}
ts1_frames['Period of record (years)'] = por.years.round(decimals = 1)

#### Calculate MI ###### 
# Mean Daily Flows at t days before sampling     

name = 'mean_Qt'
try: 
    flows_Qt = pd.read_csv((glob.glob(data_folder + 'Flows/%s_*.csv' %name))[0], index_col=0)
    print('\n\n%s Loaded Successfully' %name)
except:
    print('\n\nCalculating %s...' %(name))
    flows_Qt = gh.sample_day_Q(samples_sum_wdj.reset_index(), sample_dict, timesteps, daily_Q)    
    flows_Qt.to_csv(data_folder + 'Flows/%s_%s.csv' %(name, datestamp))
    print('\n\n%s Calculated Successfully!' %name)        
flows_Qt.columns = ['t-{} d'.format(i) for i in flows_Qt.columns]
flows_Qt_sites = flows_Qt.set_index(flows_Qt.index.map(sample_dict))
ts1_frames[name] = flows_Qt_sites
otus_cur = clip_samples(otus, flows_Qt_sites)
if doMean_Qt: 
    get_sig_mi(otus_cur, flows_Qt_sites, name, datestamp, records)
    
    name = 'meanqt'
    print('Calculating %s...' %name)
    flows_qt_sites = gh.get_q(flows_Qt_sites, site_meta_df)
    print('\n\n%s Calculated Successfully!' %name)    
    otus_cur = clip_samples(otus, flows_qt_sites)
    get_sig_mi(otus_cur, flows_qt_sites, name, datestamp, records)

# Mean monthly absolute and specific discharge

name = 'annual_Q' 
records['name'] = name
try: 
    annual_Q = pd.read_csv((glob.glob(data_folder + 'Flows/%s_%s*.csv' %(name, stamp)))[0], index_col = 0)
    print('Successfully loaded stored %s data' %name)
except: 
    print('Calculating %s...' %name)
    annual_Q = gh.get_mean_flows(dailyflows, 'annual')
    annual_Q.to_csv(data_folder + 'Flows/%s_%s_%s.csv' %(name, stamp, datestamp))
ts1_frames[name] = annual_Q
otus_cur = clip_samples(otus, annual_Q)
print('Calculating MI for: \n', otus_cur)
if doAnnual_Q: get_sig_mi(otus_cur, annual_Q, name, datestamp, records)
    
name = 'annualq' 
try: 
    annual_q = pd.read_csv((glob.glob(data_folder + 'Flows/%s_%s*.csv' %(name, stamp)))[0], index_col = 0)
    print('Successfully loaded stored %s data' %name)
except: 
    print('Calculating %s...' %name)
    annual_q = gh.get_q(annual_Q, site_meta_df)  
    annual_q.to_csv(data_folder + 'Flows/%s_%s_%s.csv' %(name, stamp, datestamp))
otus_cur = clip_samples(otus, annual_q)
print('Calculating MI for: \n', otus_cur)
if doAnnual_Q: get_sig_mi (otus_cur, annual_q, name, datestamp, records)


name = 'monthly_Q' 
try: 
    monthly_Q = pd.read_csv((glob.glob(data_folder + 'Flows/%s_%s*.csv' %(name, stamp)))[0], index_col = 0)
    print('Successfully loaded stored %s data' %name)
except: 
    print('Calculating %s...' %name)
    monthly_Q = gh.get_mean_flows(dailyflows) 
    monthly_Q.to_csv(data_folder + 'Flows/%s_%s_%s.csv' %(name, stamp, datestamp))
ts1_frames[name] = monthly_Q
otus_cur = clip_samples(otus, monthly_Q)
if doMonthly_Q: 
    get_sig_mi(otus_cur, monthly_Q, name, datestamp, records)

    name = 'monthlyq'
    try: 
        monthly_q = pd.read_csv((glob.glob(data_folder+'Flows/%s_%s*.csv' %(name, stamp)))[0], index_col = 0)
        print('Successfully loaded stored %s data' %name)
    except:
        print('Calculating %s...' %name)
        monthly_q = gh.get_q(monthly_Q, site_meta_df)   
        monthly_q.to_csv(data_folder + 'Flows/%s_%s_%s.csv' %(name, stamp, datestamp))
    otus_cur = clip_samples(otus, monthly_q)
    get_sig_mi(otus_cur, monthly_q, name, datestamp, records)    


name = 'seasonal_Q'    
try: 
    seasonal_Q = pd.read_csv((glob.glob(data_folder + 'Flows/%s_%s*.csv' %(name, stamp)))[0], index_col = 0)
    print('Successfully loaded stored %s data' %name)
except:
    print('Calculating %s...' %name)
    seasonal_Q = gh.get_mean_flows(dailyflows, 'seasonal')
    seasonal_Q.to_csv(data_folder + 'Flows/%s_%s_%s.csv' %(name, stamp, datestamp))
ts1_frames[name] = seasonal_Q
otus_cur = clip_samples(otus, seasonal_Q)
if doSeasonal_Q: 
    get_sig_mi(otus_cur, seasonal_Q, name, datestamp, records)
    
    name = 'seasonalq'
    try:
        seasonal_q = pd.read_csv((glob.glob(data_folder + 'Flows/%s_%s*.csv' %(name, stamp)))[0], index_col = 0)
        print('Successfully loaded stored %s data' %name)
    except:
        print('Calculating %s...' %name)
        seasonal_q = gh.get_q(seasonal_Q, site_meta_df)
        seasonal_q.to_csv(data_folder + 'Flows/%s_%s_%s.csv' %(name, stamp, datestamp))
    otus_cur = clip_samples(otus, seasonal_q)
    get_sig_mi(otus_cur, seasonal_q, name, datestamp, records)

# High and low flows 

name = 'durations_Q'
try: 
    durations_Q = pd.read_csv((glob.glob(data_folder + 'Flows/%s_%s*.csv' %(name, stamp)))[0], header = [0, 1], index_col = 0)
    print('Successfully loaded stored %s data: \n' %name, durations_Q.head(), '\n\n')
except:    
    print('Calculating %s...' %name) 
    durations_Q = gh.get_durations(dailyflows, dlist)  # <---------- Add try/except here to load existing calcs
    durations_Q.to_csv(data_folder + 'Flows/%s_%s_%s.csv' %(name, stamp, datestamp))
ts1_frames[name] = durations_Q   
otus_cur = clip_samples(otus, durations_Q) 
if doDurations_Q: 
    get_sig_mi(otus_cur, durations_Q, name, datestamp, records)
    
    name = 'durationsq'
    try:
        durations_q = pd.read_csv((glob.glob(data_folder + 'Flows/%s_%s*.csv' %(name, stamp)))[0], header = [0, 1], index_col = 0)
        print('Successfully loaded stored %s data' %name)
    except:
        print('Calculating %s...' %name)
        durations_q = gh.get_q(durations_Q, site_meta_df)
        durations_q.to_csv(data_folder + 'Flows/%s_%s_%s.csv' %(name, stamp, datestamp))
    otus_cur = clip_samples(otus, durations_q)
    get_sig_mi(otus_cur, durations_q, name, datestamp, records)

name = 'area'
area = pd.DataFrame(site_meta_df.loc[labels_sum_wdj.map(sample_meta_df.Sample_site), 'DRNAREA (square miles)'])
ts1_frames[name] = area
otus_cur = clip_samples(otus, area)  
if doArea: get_sig_mi(otus_cur, area, name, datestamp, records)
 
import pickle
with open(data_folder+"MI_info.pickle", 'wb') as fp:
    pickle.dump([ts1_frames, otus_cur.index.to_list(), joint_otus, otus], fp)

print('------------- Done. Run time = %.2f minutes --------------- ' %((time.time() - start_time)/60))

