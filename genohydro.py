# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 14:38:51 2021

@author: uryckid
"""
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

def load_metadata (metafile):
    meta_df = pd.read_csv(metafile, sep=',', header=0, engine = 'python')
    meta_df.dropna(axis=1, how='all', inplace=True)
    try:
        meta_df.date = pd.to_datetime(meta_df.date)
        meta_df['sample-id'] = [v.rstrip() for i, v in meta_df['sample-id'].items()]
        meta_df.set_index('sample-id', inplace = True)
    except: meta_df = meta_df.set_index('Sample_site').dropna(axis=0, how='all')
    return meta_df

def subsample(inmeta_df):
    subsamples = inmeta_df.copy()
    subsamples['sample-id'] = subsamples.index
    subsamples.set_index('date', inplace=True)
    samples_sum = pd.concat([subsamples.loc['2017-07-01':'2017-08-08', :],
                             subsamples.loc['2018-08-06':'2018-08-08', :]]) # Summer samples, including John Day (one per site)
    return samples_sum

def load_sequences (otufile):
    print('\nProcessing OTU data...')
    seq_df = pd.read_table(otufile, sep='\t', header=0)
    #seq_df.drop(list(['DES-002', 'DES-003', 'DES-004']), axis = 1, inplace = True) # permission   
    tax_dict = dict(zip(seq_df['#OTU ID'], seq_df.taxonomy))
    seq_df = seq_df.loc[seq_df.sum(axis=1) != 0].set_index(['#OTU ID']).T                    
    seq_df = seq_df.drop('taxonomy').apply(pd.to_numeric)                   
    print('\nOTU dataframe (sites, OTUs): ', seq_df.shape )
    return seq_df, tax_dict

def group_taxa(tax_dict, otu_frame): # Not tested
    tax_df = pd.DataFrame.from_dict(tax_dict, orient = 'index')
    tax_df = tax_df[0].str.split(expand=True)
    tax_df.columns = ['domain', 'phylum', 'class', 'order', 'family', 'genus', 'species']
    
    unique_bac = list(tax_df.loc[tax_df.domain.str.contains('Bacteria')].phylum.unique())
    unique_bac.append(list(tax_df.loc[tax_df.domain.str.contains('Bacteria')]['class'].unique()))
    groups = ['Archaea'] + [i.strip('p__').strip(', ') for i in unique_bac]
      
    group_dict=dict()
    for grp in groups:
        if grp == 'Archaea': cur_seqs = tax_df.loc[tax_df.domain.str.contains(grp)].index
        else: cur_seqs = tax_df.loc[(tax_df.domain.str.contains('Bacteria')) & 
                                    (tax_df.phylum.str.contains(grp))].index
        group_frame = otu_frame[otu_frame.columns.intersection(cur_seqs)]
        if group_frame.sum().sum()>0: group_dict.update({grp:group_frame.sum(axis=1)})
    phy_frame = pd.DataFrame.from_dict(group_dict, orient = 'columns')
    phy_frame['total'] = phy_frame.sum(axis=1)
    phy_comp = pd.DataFrame([phy_frame[c]/phy_frame.total for c in phy_frame.columns], index = phy_frame.columns).T
    return phy_comp

def load_flows (flowsfile1, flowsfile2, stations):
    try: 
        flow_data_1 = pd.read_csv(flowsfile1, usecols = ['station', 'date', 'mean_Q'], 
                                 parse_dates = ['date'], low_memory = False)
        flow_data_1.mean_Q = pd.to_numeric(flow_data_1.mean_Q, errors='coerce')
        flow_data_1['Site'] = flow_data_1['station'].map(stations)
        print('USGS gage data loaded succesfully')
        flow_data = flow_data_1
    except: print('No USGS gage data loaded')
    try: 
        flow_data_2 = pd.read_csv(flowsfile2, usecols = ['SITECODE', 'DATE', 'MEAN_Q'], 
                                parse_dates = ['DATE'], dtype = {'MEAN_Q':'float64'})
        sitecodes = flow_data_2.SITECODE.unique()[1:10]
        sites = ['%s-HJA' %x for x in ['W01', 'W02', 'W03', 'W06', 'W07', 'W08', 'W09', 'W10', 'MAC'] ]
        sites_dict = {sitecodes[x]:sites[x] for x in range(len(sitecodes))}
        flow_data_2['Site'] = flow_data_2['SITECODE'].map(sites_dict)
        print('Andrews gage data loaded succesfully')
        flow_data = flow_data_2
    except: print('No Andrews gage data loaded')
    
    # Concatenate flow data 1 and 2
    try: 
        cols = flow_data_2.columns.to_list()
        cols = [cols[i] for i in [1, 2, 0, 3]]
        flow_data_2 = flow_data_2[cols]
        flow_data_2.columns = flow_data_1.columns
        flow_data = flow_data_2.dropna(subset = ['Site']).append(flow_data_1, ignore_index=True)
        print('Flows files successfully concatenated')
    except: pass
    flow_data['doy'] = flow_data.date.dt.strftime('%j').map(int)   
    return flow_data

def trim_flow_dates(flowdata, subsample_meta, sample_dict, no_years):
    subsample_meta.reset_index(inplace=True)
    subsample_meta.set_index('sample-id', inplace=True)
    Q_frames = []
    for sample in subsample_meta.index:
        sample_day = subsample_meta.loc[sample].date
        cur_station_Q = flowdata.loc[flowdata.Site==sample_dict[sample]]
        cur_Q = cur_station_Q.loc[(cur_station_Q.date <= sample_day) & (cur_station_Q.date > (sample_day - relativedelta(years = no_years)))]
        Q_frames.append(cur_Q)
    daily_Q_trim = pd.concat(Q_frames, axis = 0)
    return daily_Q_trim

def get_mean_flows(flowdata, time_frame = 'monthly'):
    flows = flowdata.dropna(subset = ['mean_Q'])
    try: flows = flows.drop('doy', axis = 1)
    except: pass
    if time_frame == 'annual': mean_flows = flows.groupby('Site').mean() 
     
    if time_frame == 'monthly':
        monthly_Q = pd.DataFrame(columns = flows.date.dt.month.unique(), index = flows.Site.unique())
        for i in monthly_Q.columns: 
            cur_flows = flows.loc[flows.date.dt.month == i]
            monthly_Q[i] = cur_flows.groupby('Site').mean()
        mean_flows = monthly_Q
    
    if time_frame == 'seasonal': 
        seasonal_Q = pd.DataFrame(index = flows.Site.unique())
        seasonal_Q['OND'] = flows.loc[flows.date.dt.month.isin(range(10,13))].groupby('Site').mean().mean_Q
        seasonal_Q['JFM'] = flows.loc[flows.date.dt.month.isin(range(1,4))].groupby('Site').mean().mean_Q  
        seasonal_Q['AMJ'] = flows.loc[flows.date.dt.month.isin(range(4,7))].groupby('Site').mean().mean_Q
        seasonal_Q['JAS'] = flows.loc[flows.date.dt.month.isin(range(7,10))].groupby('Site').mean().mean_Q
        mean_flows = seasonal_Q
        
    return mean_flows

def get_q(in_Q, site_meta):
    areas = site_meta['DRNAREA (square miles)'][set.intersection(set(in_Q.index), set(site_meta.index))]
    q = in_Q.divide(areas, axis = 0)
    return q

def get_durations (flowdata, dlist): 
    flowdata = flowdata.dropna()
    flowdata['date'] = pd.to_datetime(flowdata['date'])
    seasons = {'OND': range(10,13), 'JFM': range(1,4), 'AMJ': range(4,7), 'JAS': range(7,10)}
    cols = pd.MultiIndex.from_product([['annual'] + list(seasons.keys()), dlist], names = ['seasons', 'ex_probs'])
    durations = pd.DataFrame(index=flowdata.Site.unique(), columns=cols)

    for site in flowdata.Site.unique():
        cur_site_flows = flowdata.loc[flowdata.Site == site].set_index('date')
        cur_flows = cur_site_flows.copy()
        cur_flows['ranks'] = cur_flows.mean_Q.rank(ascending = False)
        cur_flows['percentile'] = cur_flows.mean_Q.rank(ascending = False, pct=True)
        for d in dlist:
            d_flow = cur_flows.iloc[np.abs((cur_flows['percentile'])-(d/100)).argmin()].mean_Q 
            durations.loc[site, ('annual', d)] = d_flow
        
        for s in seasons.keys():
            cur_seas_flows = cur_site_flows.loc[cur_site_flows.index.month.isin(seasons[s])]
            cur_seas_flows['ranks'] = cur_seas_flows.mean_Q.rank(ascending = False)
            cur_seas_flows['percentile'] = cur_seas_flows.mean_Q.rank(ascending = False, pct=True)
            for d in dlist:
                d_flow = cur_seas_flows.iloc[np.abs((cur_seas_flows['percentile'])-(d/100)).argmin()].mean_Q 
                durations.loc[site, (s, d)] = d_flow
    return durations     

def sample_day_Q(sample_meta, sample_dict, timesteps, flowdata):
    samples = sample_meta.copy()
    #samples['date'] = samples.index
    samples.set_index('sample-id' , inplace = True)
    flows_t = pd.DataFrame(index = samples.index, columns = timesteps)
    for sample in flows_t.index:
        cur_flows = flowdata.loc[flowdata.Site == sample_dict[sample]]
        sample_day = samples.loc[sample].date
        print(sample, sample_day)
        for t in timesteps:
            try:
                flow_day = sample_day - relativedelta(days = t)
                flow = cur_flows.set_index('date').loc[flow_day].mean_Q 
                print(t, flow_day, flow)
                flows_t.loc[sample][t] = flow
            except: 
                print('nope')
                flows_t.loc[sample][t] = np.nan
    return flows_t


    