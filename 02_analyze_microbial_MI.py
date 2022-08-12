# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 14:24:17 2021

@author: uryckid
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib_venn import venn3, venn2
from scipy import stats, optimize
import statistics
import seaborn as sns
import genohydro as gh
from matplotlib import rc
import re
import pickle

os.chdir(os.path.dirname(__file__))
save_figs = False
por = 10
useRaw = False
subsets = ['joint', 'all'] # set of ASVs
suite = subsets[1]

def collapse_levels(indf):
    indf.sort_index(axis=1, level=1, sort_remaining = False, inplace=True)
    # seas = indf.columns.get_level_values(0).unique().to_list()
    # seas = seas[1:]+[seas[0]]
    # newdf = indf.reindex(seas, axis=1, level = 0)
    #indf = newdf.swaplevel(0, 2, axis = 1)
    indf = indf.swaplevel(0, 2, axis = 1)
    cols = []
    frames = {}
    for l in indf.columns.levels[0]: 
        indf[l].sort_index(axis=1, level=0, sort_remaining=False, inplace=True)
        
        c = ([('|'.join(col).strip(), l) for col in indf[l].columns.values]) # collapse multiIndex
        cols.append(c)    
    for i in list(range(5)):
        key = indf.columns.levels[0][i]
        frame = indf[key]
        frame.columns = [l1 for l1, l2 in cols[i]]
        frames[key] = frame   
    outdf = pd.concat(frames, axis = 1)
    outdf = outdf.swaplevel(0, 1, axis = 1)
    return outdf

def holm_p(pframe):
    pcorr = {}
    n=len(pframe)
    psorted = pframe.sort_values(ascending = True)
    for i, v in psorted.iteritems():
        pcorr[i] = v*n
        n-=1
    return pd.DataFrame.from_dict(pcorr, orient = 'index')    

def second_order_fxn(x, a, b, c):
    yfit = a*x**2 + b*x + c
    return pd.Series(yfit, index = x)

# Load Sample Metadata
sample_meta_filename = 'Project_data/MiSeq692/genohydro_metadata_miseq692_clean.csv'
sample_meta_df = gh.load_metadata(sample_meta_filename)
basin_dict = {row.Sample_site:row.Watershed for i, row in sample_meta_df.iterrows()}

# Load study sites
site_meta_filename = 'Project_data/Site_metadata_202110.csv'
site_meta_df = gh.load_metadata(site_meta_filename)
with open ('Project_data/MI_info.pickle', 'rb') as fp:
    ts1_dict, study_sites, joint_otus, sample_otu_df, = pickle.load(fp)
    ts1_dict['durations_Q'].columns = ts1_dict['durations_Q'].columns.swaplevel(0,1)
    ts1_dict['durations_Q'].sort_index(axis = 1, level = 0, sort_remaining=False, inplace=True)
print('\nMEAN PRECIPITATION:\n')
for site in ['WIL-POR', 'DES-BIG', 'JOH-MCD']:
    cur_data = site_meta_df.loc[site]
    print('{}\nAnnual = {}\n Jan = {}\n Jul = {}\n'.format(site, cur_data['PRECIP (inches)'] * 25.4,
                                                     cur_data['JANAVPRE2K (inches)'] *25.4, 
                                                     cur_data['JULAVPRE2K (inches)'] * 25.4))
print('Range:\nArea = {}-{} square kilometers'.format(
    site_meta_df.loc[study_sites, 'DRNAREA (square miles)'].min()* 2.58999, 
    site_meta_df.loc[study_sites, 'DRNAREA (square miles)'].max() * 2.58999 ))    
print('Annual precipitation = {}-{} mm'.format(
    site_meta_df.loc[study_sites, 'PRECIP (inches)'].min()* 25.4, 
    site_meta_df.loc[study_sites, 'PRECIP (inches)'].max()* 25.4 ))    

# Load otus
otu_all = pd.read_csv('Project_data/MiSeq692/ASV-table-with-taxonomy-1450.txt', sep='\t', header=0)   
tax_dict = dict(zip(otu_all['#OTU ID'], otu_all.taxonomy))
otu_all = otu_all.loc[otu_all.sum(axis=1) != 0].set_index(['#OTU ID']).T 
otu_df = otu_all.copy()                   
otu_df = otu_df.drop('taxonomy').apply(pd.to_numeric)                   
print('\nOTU dataframe (sites, OTUs): ', otu_df.shape )
abund = np.sum(sample_otu_df, axis = 0)
var = sample_otu_df.var(axis = 0)
no_sites = sample_otu_df.astype(bool).sum(axis=0)
no_watersheds = sample_otu_df.copy()
no_watersheds.index = no_watersheds.index.map(basin_dict)
no_basins = no_watersheds.groupby(by = no_watersheds.index).sum()
no_basins = no_basins.astype(bool).sum(axis=0)

dna_sequences = pd.read_csv('Figures/256/Table_s2_with_sequences.csv', usecols=['Asv_ID', 'DNA sequence']) # from Byron
dna_sequences.set_index('Asv_ID', inplace=True)

namelist = [ 'mean_Qt', 'meanqt', 'monthly_Q', 'monthlyq', 'durations_Q', 'durationsq', 'area'] 
# 'delta_Q', 'deltaq', 'annual_Q', 'annualq', 'seasonal_Q', 'seasonalq', 'upstr_len', 'drn_dens',  

two_levels = ['mean_Qt', 'meanqt', 'delta_Q', 'deltaq', 'monthly_Q', 'monthlyq', 'seasonal_Q', 'seasonalq']
 

one_level = ['annual_Q', 'annualq', 'drn_dens','area', 'upstr_len']

absolute = [name for name in namelist if 'Q' in name] + ['area','upstr_len', ]
specific = [name for name in namelist if 'q' in name] + ['drn_dens']

MI_frames = {}

for name in namelist:
    print('\n\n%s...' %name)
    if "durations" in name: head = [0, 1, 2]
    elif name in two_levels: head = [0, 1]   
    else: head = [0]
    try:
        if useRaw: MI = pd.read_csv(glob.glob('Project_data/MI/{}/{}_mi_raw.csv'.format(suite, name))[0], 
                     header = head, index_col = 0)
        else: MI = pd.read_csv(glob.glob('Project_data/MI/{}/{}_mi_sig.csv'.format(suite, name))[0], 
                     header = head, index_col = 0)
    except: print('No MI data file found for %s' %name)
    if 'durations' in name: 
        MI = collapse_levels(MI)
        MI = MI[MI.columns.drop(list(MI.filter(regex='50')))] # Drop Q50
    if name in one_level: MI  = pd.concat({name: MI}, axis = 1)
    MI.dropna(axis=0, how = 'all')
    if name != 'area': MI_frames[name] = MI
    MI_all = pd.concat(MI_frames, axis = 1) # MultiIndex with 3 column levels
Inorm = MI_all.loc[:, MI_all.columns.get_level_values(2)=='Ixy/Hy']
sig_otus = Inorm.loc[Inorm.fillna(0).sum(axis=1)!=0].index.to_list()
MI_all = MI_all.loc[sig_otus]
print('...done')

# all_otus = []
# all_otus += [v.index.tolist() for k, v in MI_frames.items() if k != 'area']
# sig_otus = set([item for sublist in all_otus for item in sublist])
sig_otus = set(sig_otus).intersection(set(abund.index))
otu_abund = abund[sig_otus]
otu_abund.sort_values(ascending = False, inplace = True)
ranks = otu_abund.rank(ascending=False, method = 'first').astype(int) # unique ranks, including for duplicate abundance
MI_frames_area = MI_frames.copy()
MI_frames_area['area'] = MI.loc[MI.index.isin(ranks.index)]
MI_sorted = {k: v.reindex(otu_abund.index) for k, v in MI_frames_area.items()}
MI_sorted_df = pd.concat(MI_sorted, axis =1).melt(var_name = ['Name', 'Char', 'MI'], ignore_index = False)

otu_var = var[sig_otus]
otu_var.sort_values(ascending = False, inplace = True)

asv_no_sites = no_sites[sig_otus]
asv_no_sites.sort_values(ascending = False, inplace = True)

##################################
###########  TABLES  #############
##################################

'''TABLE 1. '''
print('Generating Table 1...')
means = MI_all.mean(axis =0)
sds = MI_all.std(axis =0)
medians = MI_all.median(axis =0)
ranges = MI_all.max(axis=0) - MI_all.min(axis=0)

hy = medians[slice(None), slice(None), 'Hy']
ixy = medians[slice(None), slice(None), 'Ixy']
ixy_hy = medians[slice(None), slice(None), 'Ixy/Hy']

#sdhy = ranges[slice(None), slice(None), 'Hy']
sdixy = ranges[slice(None), slice(None), 'Ixy']
sdixy_hy = ranges[slice(None), slice(None), 'Ixy/Hy']

t1 = {}
#hsd = pd.Series(['({})'.format(sdhy[i].round(decimals=3)) for i in sdhy.index], index =sdhy.index, dtype=str)
t1['H(Y) [bits]'] = hy.round(decimals=3)

isd = pd.Series([('({})'.format(np.round(sdixy[i], 3))) for i in ixy.index], index =ixy.index, dtype=str)
t1['I(X; Y) [bits]'] = pd.concat([np.round(ixy, 3), isd], axis = 1, keys = ['Median', '(Range)'])

i_hsd = pd.Series([('({})'.format(np.round(sdixy_hy[i], 3))) for i in ixy_hy.index], index = ixy_hy.index, dtype=str) 
t1['I(X; Y)/H(Y) [bits/bit]'] = pd.concat([ixy_hy.round(decimals=3), i_hsd], axis=1, keys = ['Median', '(Range)'])

table1 = pd.concat(t1, axis = 1)
table1 = table1.reindex(absolute+specific, axis = 0, level = 0).round(decimals=3)
if save_figs: table1.to_csv('Figures/{}/Table1.csv'.formate(suite))
chars = list([ix[1] for ix in table1.index])

MI_area = MI_frames_area['area'].area
print('\n\nArea:\nH(Y) = {}\nI(X;Y)/H(Y) = {}; range = {}\n\n'.format(
    MI_area.Hy[0], MI_area["Ixy/Hy"].median(), MI_area["Ixy/Hy"].max()-MI_area["Ixy/Hy"].min()))

#%%"
'''TABLE S1'''
# Convert hydrology to metric
ts1_met = {}
for k, v in ts1_dict.items(): 
    if k in namelist + ['Period of record (years)']:
        if 'Q' in k: ts1_met[k] = v*0.02832 # Convert cfs to cms
        elif k == 'upstr_len':
            ts1_met[k] = (v * 1.60934).round(decimals=1 )
            ts1_met[k].columns = ['Upstream length (kilometers)']
        else: ts1_met[k] = pd.DataFrame(v, dtype=float)
    else: pass

ts1 = {}
lat = site_meta_df['Latitude']
lon = site_meta_df['Longitude']
coords = pd.concat([lon, lat], axis = 1)
coords.columns = pd.MultiIndex.from_product([['Coordinates'], coords.columns])
basin = pd.DataFrame(site_meta_df.Basin)
basin.columns = pd.MultiIndex.from_product([['Basin'], basin.columns])
area = pd.DataFrame((site_meta_df['DRNAREA (square miles)'] * 2.58999).round(decimals =3)) # Covnert mi2 to km2
area = area.loc[study_sites]
area.columns = pd.MultiIndex.from_product([['Basin drainage area'], ['Drainage area (square kilometers)']])
ts1['Coordinates'] = coords
ts1['Basin drainage area'] = area
#ts1['Basin'] = basin
for k, v in ts1_met.items(): 
    if k not in ['durations_Q']:
        cur_frame = v.copy()
        cur_frame.columns = pd.MultiIndex.from_product([[k], v.columns])
        ts1[k] = cur_frame
    else: ts1[k] = v
table_s1 = pd.concat(ts1, axis = 1, join = 'outer').T
table_s1.dropna(axis = 1, thresh = 5, inplace=True)
table_s1 = table_s1.loc[:, (set(study_sites) & set(table_s1.columns))].copy()
table_s1.sort_values(axis = 1, by=table_s1.index[0], inplace=True)
minima = table_s1[table_s1!=0].min(axis =1).round(decimals=2)
maxima = table_s1.max(axis = 1).round(decimals = 2)
medians = table_s1.median(axis = 1).round(decimals=2)

#pd.concat([table_s1, basin.T], axis=0, join='inner')

survey2 = table_s1.round(decimals=3).copy().T
table_s1['Min'], table_s1['Max'], table_s1['Median'] = minima, maxima, medians 
#table_s1.loc['Basin', :] = table_s1.columns.map(basin)
basin_counts = site_meta_df.loc[table_s1.columns[:-3], 'Basin'].value_counts()

if save_figs: table_s1.round(decimals=3).to_csv('Figures/{}/Table_S1.csv'.format(suite))

print ('\nWillamette R. outlet  seasonal flow durations:', table_s1['WIL-POR']['durations_Q'])
print ('\nDeschutes R. outlet  seasonal flow durations:\n', table_s1['DES-BIG']['durations_Q'])
print ('\nJohn Day R. outlet  seasonal flow durations:\n', table_s1['JOH-MCD']['durations_Q'])


'''Table S2. Abuandance rank, log abudance, log variance, and number of sites at which each 
microbial amplified sequence variant (ASV) was detected in study sites across ORegon, USA.'''

asv_stats = pd.concat([ranks, np.log(otu_abund).round(2), np.log(var).round(2), no_sites], axis=1, join = 'inner',  
                      keys = ['Abundance rank', 'Log abundance', 'Log variance', 'Number of sites'])
asv_stats['Taxonomy'] = asv_stats.index.map(tax_dict)
asv_stats['DNA sequence'] = dna_sequences

table_s2 = asv_stats.copy()
table_s2['Asv_ID'] = table_s2.index
table_s2.set_index('Abundance rank', inplace=True)



if save_figs: table_s2.to_csv('Figures/{}/Table_S2.csv'.format(suite))

asv_stats.reset_index(inplace=True)
asv_stats.set_index('Abundance rank', drop=True, inplace=True)

# Compare with 256 ASVs abundance rank
comp = pd.DataFrame(table_s2['Asv_ID']).reset_index()
comp.set_index('Asv_ID', inplace=True)
abund_256 = pd.read_csv('Figures/256/Table_S2.csv', index_col='Asv_ID')
ranks_256 = abund_256['Abundance rank']
comp['ranks_256'] = comp.index.map(ranks_256)

drops_df = comp.loc[~comp.ranks_256.isnull()]
drop_list = drops_df['Abundance rank'].to_list()

#%%
# RESULTS - HYDROLOGY

## Absolute Discharge
print ('\n\nABSOLUTE DISCHARGE')
print('\nDaily mean discharge (cms) at time lags:')
for t in [2, 10, 30]:
    flows = survey2['mean_Qt']['mean_Qt']['t-{} d'.format(t)]
    print('t-{} d : {} - {}'.format(t, round(flows.min(), 3), round(flows.max(), 3)))

print('\nMonthly mean discharge (cms):')
for m in [1, 7]:
    flows = survey2['monthly_Q']['monthly_Q']['{}'.format(m)]
    print('month {} = : {} - {}'.format(m, round(flows.min(), 3), round(flows.max(), 3)))
    
print('\nFlow Durations (cms):')
flows = survey2['durations_Q']   
for d in ['95', '5']:
    print('Summer Q{}\t{} - {}'.format(d, flows[d]['JAS'].min(), flows[d]['JAS'].max()))
    print('Winter Q{}\t{} - {}'.format(d, flows[d]['JFM'].min(), flows[d]['JFM'].max()))

## Specific discharge
print ('\n\nSPECIFIC DISCHARGE')
#area.hist(bins = 20)
print('\nDrainage area (km2) = {} - {}'.format(area.min(), area.max()))

table_s1q = survey2.iloc[:, 4:].div(area[area.columns[0]], axis = 'index')

print('\nDaily specific discharge (cms/km2) at time lags:')
for t in [2, 10, 30]:
    flows = table_s1q['mean_Qt']['mean_Qt']['t-{} d'.format(t)]
    print('t-{} d : {} - {}'.format(t, round(flows.min(), 3), round(flows.max(), 3)))

print('\nMedian (range) monthly specific discharg9e (cms/km2):')
for m in ['1', '7']:
    flows = table_s1q['monthly_Q']['monthly_Q']['{}'.format(m)]
    print('month {} : {} - {}'.format(m, round(flows.min(), 3), round(flows.max(), 3)))

print('\nFlow Durations (cms):')
flows = table_s1q['durations_Q']   
for d in ['95', '5']:
    print('Summer Q{}\t{} - {}'.format(d, flows[d]['JAS'].min(), flows[d]['JAS'].max()))
    print('Winter Q{}\t{} - {}'.format(d, flows[d]['JFM'].min(), flows[d]['JFM'].max()))    
    
#%%
##################################
###########  FIGURES  ############
##################################

import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import cartopy.feature as cfeature
from matplotlib.transforms import offset_copy
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.ticker as ticker
#matplotlib.rcParams["text.usetex"]=True

# Elsevier: single-column = (90 mm, 3.54 in wide); full-width = (190 mm, 7.48 in) 
# Min font size = 7
twocol = (7.5, 5.63) #full 2-col width, 4:3 aspect
onepluscol = (5.51, 4.13)
onecol = (3.5, 2.63)

labelfont = 6
axisfont = 10
legendfont = 8


'''FIGURE 1'''
# Figure 1. Map of co-located stream gages and summer DNA sample collections in streams across 
# Willamette (2017), Deschutes (2017), and John Day (2018) watersheds in Oregon, USA. 
# Marker colors indicate mean annual precipitation (mm) in sample sub-catchments (StreamStats reference). 
# Inset shows number of unique and shared microbial amplified sequence variants (ASVs) detected in each basin. 

fig=plt.figure(1)
transform = ccrs.PlateCarree()
curCol = 'PRECIP (inches)'
X0 = site_meta_df.loc[site_meta_df.index.isin(study_sites) & (site_meta_df.Basin=='Willamette')].Longitude.values
Y0 = site_meta_df.loc[site_meta_df.index.isin(study_sites) & (site_meta_df.Basin=='Willamette')].Latitude.values
Z0 = site_meta_df.loc[site_meta_df.index.isin(study_sites) & (site_meta_df.Basin=='Willamette')][curCol].values
X1 = site_meta_df.loc[site_meta_df.index.isin(study_sites) & (site_meta_df.Basin=='Deschutes')].Longitude.values
Y1 = site_meta_df.loc[site_meta_df.index.isin(study_sites) & (site_meta_df.Basin=='Deschutes')].Latitude.values
Z1 = site_meta_df.loc[site_meta_df.index.isin(study_sites) & (site_meta_df.Basin=='Deschutes')][curCol].values
X2 = site_meta_df.loc[site_meta_df.index.isin(study_sites) & (site_meta_df.Basin=='John Day')].Longitude.values
Y2 = site_meta_df.loc[site_meta_df.index.isin(study_sites) & (site_meta_df.Basin=='John Day')].Latitude.values
Z2 = site_meta_df.loc[site_meta_df.index.isin(study_sites) & (site_meta_df.Basin=='John Day')][curCol].values

# Convert to metric:
if curCol == 'PRECIP (inches)': 
    Z0 = Z0*25.4
    Z1 = Z1 * 25.4
    Z2 = Z2 * 25.4
    curCol = 'Mean annual precipitation (mm)'


# Create a Stamen terrain background instance.
stamen_terrain = cimgt.Stamen('terrain-background')
ax = fig.add_subplot(1, 1, 1, projection=stamen_terrain.crs)
ax.set_extent([-124.3, -116.75, 41.97, 46.2], crs=transform)

ax.add_feature(cfeature.NaturalEarthFeature
               ('cultural', 'admin_1_states_provinces_lines', '10m'), facecolor = 'None', edgecolor = 'k', linewidth = 1.2)
ax.add_image(stamen_terrain, 8)

ax.add_feature(cfeature.NaturalEarthFeature
                ('physical', 'rivers_north_america', '10m'), facecolor='None', edgecolor='steelblue', alpha = 0.8)
ax.add_feature(cfeature.NaturalEarthFeature
                ('physical', 'rivers_lake_centerlines', '10m'), facecolor='None', edgecolor='steelblue', alpha = 0.8)
ax.add_feature(cfeature.NaturalEarthFeature
                ('cultural', 'urban_areas', '10m'), facecolor='grey', edgecolor='grey', alpha=0.4)


# Determine color and range
cmap = cm.plasma_r
zs = np.concatenate([Z0, Z1, Z2], axis=0)
min_, max_ = zs.min(), zs.max()
edges = 'k'
lw = 0.4

msize = 26 #scale by drainage area?
outline = 'whitesmoke'

# Plot two datasets on one scale
im = plt.scatter(X0, Y0, marker = 's', transform = transform, edgecolors=edges, linewidth = lw, c= Z0, cmap= cmap, 
                  label = 'Willamette', zorder = 20, s = msize, alpha = 0.9) #  
plt.clim(min_, max_)
plt.scatter(X1,Y1,transform=transform, marker = '^', edgecolors=edges, linewidth = lw, c=Z1, cmap = cmap, 
            label = 'Deschutes ', zorder =10, s = msize, alpha = 0.9) #  
plt.clim(min_, max_)
plt.scatter(X2, Y2, transform=transform, marker = 'o', edgecolors=edges, linewidth = lw, c=Z2, cmap = cmap, 
            label = 'John Day', zorder = 10, s = msize, alpha = 0.9)
plt.clim(min_, max_)

platecarree_transform = ccrs.PlateCarree()._as_mpl_transform(ax)
text_transform = offset_copy(platecarree_transform, units='dots', x=+10 )

# Cities
ax.text(-122.6, 45.5051, u'Portland',
            verticalalignment='center', horizontalalignment='left',
            transform=text_transform)
ax.text(-121.25, 44.0582, u'Bend',
            verticalalignment='center', horizontalalignment='left',
            transform=text_transform)
# ax.text(-117.25, 45.5, u'OREGON', fontsize = 16,
#             verticalalignment='center', horizontalalignment='right',
#             transform=text_transform)
# Rivers
fs = 8
ax.text(-123.2, 45.1, u'Willamette R.', va='center', ha='center', fontstyle = 'italic', fontsize = fs,
        rotation = 78, color = 'steelblue', transform=text_transform) #, bbox=dict(facecolor='white', edgecolor = 'white', alpha=0.3, boxstyle='round'))
ax.text(-121.0, 45.1, u'Deschutes R.', va='center', ha='right', fontstyle = 'italic', fontsize = fs,
        rotation = 75, color = 'steelblue', transform=text_transform) #, bbox=dict(facecolor='white', edgecolor = 'white', alpha=0.3, boxstyle='round'))
ax.text(-120.5, 45.12, u'John Day R.', va='center', ha='right', fontstyle = 'italic', fontsize = fs,
        rotation = 94, color = 'steelblue', transform=text_transform)
ax.text(-120.35, 45.55, u'Columbia R.', va='bottom', ha='left', fontstyle = 'italic', fontsize = fs, 
        rotation = 16, color = 'steelblue', transform=text_transform)



ax.set_xticks([-124.0, -122.0, -120.0, -118.0], crs=ccrs.PlateCarree())
ax.set_yticks([42.0, 43.0, 44.0, 45.0, 46.0], crs=ccrs.PlateCarree()) 
ax.set_xticklabels(ax.get_xticklabels(), ha='left')
ax.set_yticklabels(ax.get_yticklabels(), va='bottom')

lon_formatter = LongitudeFormatter(number_format='.1f')
lat_formatter = LatitudeFormatter(number_format='.1f')
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
ax.xaxis.tick_top()

leg = plt.legend(loc='lower left', ncol=1, fontsize = 8)
for marker in leg.legendHandles:
    marker.set_color('k')
        
plt.colorbar(im, pad=0.01, fraction=0.06).set_label('Mean annual precipitation [mm]')
ax.gridlines(crs=ccrs.PlateCarree(), linewidth=1.5, color='gray', alpha=0.5, linestyle='dotted')

# Add Venn diagram inset
inset_size = 0.37
ipx, ipy = 0.535, 0.03
ax2 = fig.add_subplot(facecolor='white', projection=ccrs.PlateCarree(), alpha=1.0)
ax3 = fig.add_subplot(projection=ccrs.PlateCarree(), alpha=1.0)
ip = InsetPosition(ax, [1.06*ipx, 1.05*ipy, inset_size, inset_size])
ip2 = InsetPosition(ax, [ipx, ipy, 1.2*inset_size, 1.2*inset_size])
ax2.set_axes_locator(ip2)
ax3.set_axes_locator(ip)
ax2.set_aspect('auto')
ax3.set_aspect('auto')

vcolors = plt.cm.tab20b(range(20))
ybr = ["fde74c","456990","ac3931"]
rgbs = ['#{}'.format(v) for v in ybr]

v = venn3(subsets = [set(v) for k, v in joint_otus.items()], ax = ax3,
            set_labels = ('Willamette\n($n = {}$)   '.format(basin_counts['Willamette']), 
                          'Deschutes\n  ($n = {}$)'.format(basin_counts['Deschutes']), ''), set_colors = rgbs) #set_labels = [k for k, v in otus.items()])
ax2.text(-150, -70, 'John Day\n ($n = {}$)'.format(basin_counts['John Day']), fontsize =6.5)
for text in v.set_labels:
    text.set_fontsize(6.5)
for text in v.subset_labels:
    text.set_fontsize(6.9)
# a, b, c = [set(v) for k, v in joint_otus.items()]
# flat_list = a.union(b, c)
#flat_list = [item for sublist in joint_otus for item in sublist]

if save_figs: plt.savefig('Figures/{}/FIG01.jpg'.formate(suite), dpi=600)
#%%
'''FIGURE 2'''
phy_df = pd.read_csv('Project_data/MiSeq692/ASV_table_WDJ_summer.csv', index_col=0).fillna(0)
Qt = table_s1.loc[('mean_Qt', 'mean_Qt', 't-0 d'), :][:-3].sort_values(ascending = True)
Qjan = table_s1.loc[('monthly_Q', 'monthly_Q', '1'), :][:-3].sort_values(ascending = True)
Qarea = table_s1.loc[('Basin drainage area', 'Basin drainage area', 'Drainage area (square kilometers)'), :][:-3].sort_values(ascending = True)

#phy_df=phy_df.loc[Qt16.index].T
phy_df=phy_df.loc[Qt.index].T

phy_df = phy_df.divide(phy_df.sum().median())

tax = pd.read_csv('Project_data/MiSeq692/taxonomy_20210924.csv', index_col = 0, delimiter = '\t')
grp_df = pd.concat([phy_df, tax], axis = 1, join = 'inner')

grp_df.loc[grp_df.phylum=='p__Actinobacteriota', 'group'] = 'Actinobacteriota'
grp_df.loc[grp_df.phylum=='p__Bacteroidota', 'group'] = 'Bacteroidota'
grp_df.loc[grp_df.phylum=='p__Cyanobacteria', 'group'] = 'Cyanobacteria'
grp_df.loc[grp_df.phylum=='p__Planctomycetota', 'group'] = 'Planctomycetota'
grp_df.loc[grp_df['class']=='c__Gammaproteobacteria', 'group'] = 'Gammaproteobacteria'
grp_df.loc[grp_df['class']=='c__Alphaproteobacteria', 'group'] = 'Alphaproteobacteria'
grp_df.loc[grp_df.phylum=='p__Verrucomicrobiota', 'group'] = 'Verrucomicrobiota'
grp_df.loc[grp_df.group.isnull(), 'group'] = 'Other'

grp_bar = grp_df.groupby('group').sum()
new_ix = grp_bar.sum(axis=1).sort_values(ascending=False).index
grp_bar = grp_bar.reindex(new_ix)

colors = plt.cm.gist_rainbow(np.linspace(0, 1, 10))[:8]
colors_dict = {"Red":"ff0000","Cyber Yellow":"ffd300","Azure":"147df5",
               "Spring Bud":"a1ff0a","Electric Purple":"be0aff",
               "Electric Blue":"0aefff","Han Purple":"580aff", "Medium Spring Green":"0aff99"}#,"Dark Orange":"ff8700"}#,"Chartreuse Traditional":"deff0a",}
colors_bar = ['#{}'.format(v) for k, v in colors_dict.items()]
colors_bar.reverse()

fig, ax = plt.subplots(figsize = (twocol[0], 4.0))
ax1 = grp_bar.T.plot.bar(stacked='True', color=colors_bar, ax=ax, width=0.95)
plt.xticks(fontsize = 7.5)
#plt.xlabel(r'Sample site (by increasing discharge $Q_{\it(t-16}$ $_{days)}$)', fontsize = axisfont)
plt.xlabel(r'Sample site by increasing discharge on DNA sample date (${Q}_{t}$)', fontsize = axisfont)

plt.ylim(0, 1.0)
plt.ylabel('Community composition', fontsize=axisfont)
plt.legend(ncol=4, loc = 'lower center', fontsize = legendfont, handletextpad = 0.2, 
           bbox_to_anchor=(0.5,  1.0), columnspacing = 0.6, frameon=False)
plt.tight_layout()
if save_figs: plt.savefig('Figures/{}/FIG2_Qt.jpg'.format(suite), dpi = 600)

#%%
'''FIGURE 3'''
# Figure 3. 
optrotation = 0

doDiffs=False
heatcmap = 'gist_rainbow_r'
common_colorbar = False
box_lw = 1.5
yfont = 6
fig, axs = plt.subplots(2, 1, sharex=True, figsize = (7.5, 7.5)) # double_column, 1:1 aspect

data = MI_sorted_df.loc[MI_sorted_df.MI=='Ixy/Hy']
data = data.loc[drops_df.index]
months = data.loc[data.Name.str.contains('monthly')].Char.unique().astype(int)
month_names = ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 
                'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep']
months_dict = {months[i]: month_names[i] for i in range(len(months))}

# Absolute
data_abs = data.loc[data.Name.isin(absolute)].drop('MI', axis =1)
data_abs.index = data_abs.index.map(ranks)
char_order = data_abs.Char.unique()
data_pivot_abs = data_abs.reset_index().pivot(index = "Char", columns = '#OTU ID', values = 'value')
heatdata_abs = data_pivot_abs.loc[char_order]#data_abs['split'] = data_abs.Char.str.split("|")
num_Q = heatdata_abs.drop('area', axis = 0).fillna(0).astype(bool).sum(axis=0)
asv_stats['No. Q stats'] = num_Q
asv_stats['Median MI (Q)'] = heatdata_abs.drop('area', axis = 0).median(axis = 0, skipna=True)
asv_stats['Max MI (Q)'] = heatdata_abs.drop('area', axis = 0).max(axis = 0)

# Specific
data_spec = data.loc[data.Name.isin(specific)].drop('MI', axis =1)
data_spec.index = data_spec.index.map(ranks)
char_order = data_spec.Char.unique()
data_pivot_spec = data_spec.reset_index().pivot(index = "Char", columns = '#OTU ID', values = 'value')
heatdata_spec = data_pivot_spec.loc[char_order]#data_spec['split'] = data_spec.Char.str.split("|")
num_q = heatdata_spec.iloc[:-1, :].fillna(0).astype(bool).sum(axis=0)

asv_stats['No. q stats'] = num_q
asv_stats['Median MI (q)'] = heatdata_spec.median(axis = 0, skipna=True)
asv_stats['Max MI (q)'] = heatdata_spec.max(axis = 0)
if save_figs: asv_stats.to_csv('Figures/{}/Table_S2_with_summary_stats.csv'.format(suite))

mimax = max(heatdata_abs.max().max(), heatdata_spec.max().max())
mimin = min(heatdata_abs.min().min(), heatdata_spec.min().min())
if common_colorbar: vmin, vmax = mimin, mimax
else: vmin, vmax = None, None

g1 = sns.heatmap(heatdata_abs, cmap = heatcmap, center = 0.27, ax = axs[0], 
                 xticklabels=False, yticklabels=1, vmin = vmin, vmax =vmax, cbar_kws={"pad": 0.01})
#axs[0].set_ylabel('Absolute discharge', fontsize='small', labelpad=10)
axs[0].yaxis.label.set_visible(False)
axs[0].xaxis.set_visible(False)

axs[0].text(-10, 8, '$Q_{(t-n}$ $_{days)}$', rotation=90, va='center')
axs[0].text(-10, 22, r'$\bar{Q}_{mon}$', rotation=90, va='center')
axs[0].text(-10, 30.5, '$Q_{5,s}$', rotation=90, va='center')
axs[0].text(-10, 35, '$Q_{95,s}$', rotation=90, va='center')


axs[0].set_xlabel(None)
labs_g1 = list(heatdata_abs.index)
for i in range(len(labs_g1)):
    lab = labs_g1[i]
    try: labs_g1[i] = months_dict[int(lab)]
    except: 
        if i < 28: 
            lab = lab.replace('t-', '')
            labs_g1[i] = lab.replace(' d', '')
        else: 
            lab = lab.replace('annual', 'Ann')
            if lab != 'area': labs_g1[i] = lab[-3:]
    

g1.set_yticklabels(labs_g1, fontsize = yfont)
g1.axhline(y = 0, color='k',linewidth = box_lw)
g1.axhline(y=44, color = 'k', linewidth = box_lw)

miQarray = heatdata_abs.to_numpy().flatten()
Qmed = np.median(miQarray[~np.isnan(miQarray)])

miqarray = heatdata_spec.to_numpy().flatten()
qmed = np.median(miqarray[~np.isnan(miqarray)])


# Statistical tests
# Median Q vs q
miabs = heatdata_abs.iloc[:-1, :].copy()
mispec = heatdata_spec.copy()

asvcount_area = heatdata_abs.iloc[-1, :].fillna(0).astype(bool).sum()
asvcount_abs = heatdata_abs.iloc[:-1, :].fillna(0).astype(bool).sum(axis=1) #exluding area
asvcount_spec = heatdata_spec.fillna(0).astype(bool).sum(axis=1)

asv_mean = (pd.concat([asvcount_abs, asvcount_spec], axis = 0)).mean()
asv_std =  (pd.concat([asvcount_abs, asvcount_spec], axis = 0)).std()
print('Mean number of informative ASVs across all metrics [Q and q] = %.2f (+-%.2f SD)'%(asv_mean, asv_std))

mi_med = (pd.concat([heatdata_abs, heatdata_spec], axis =1)).median(axis=1).median()
mi_max = (pd.concat([heatdata_abs, heatdata_spec], axis =1)).max(axis=1).max()
mi_min = (pd.concat([heatdata_abs, heatdata_spec], axis =1)).min(axis=1).min()
print('Median MI across all metrics = %.3f (range = %.3f - %.3f bits/bit)'%(mi_med, mi_max, mi_min))

res = stats.mannwhitneyu(miabs, mispec,  alternative = 'greater')
print('\nMedian MI Q {} vs. q {}, Mann-Whitney U = {}, p = {}'.format(
    Qmed, qmed, res[0], res[1]))

print('Mean number of informative ASVs\n\tAbsolute: {} (SD ={})\n\tSpecific: {} (SD ={})\n\tArea: {}'.format(
    asvcount_abs.mean(), asvcount_abs.std(), asvcount_spec.mean(), asvcount_spec.std(), asvcount_area))
tstat, p = stats.ttest_rel(asvcount_abs, asvcount_spec, alternative = 'greater')
print('Two-sample t-test for difference in no. ASVs: t = {}, p = {}'.format(tstat, p))


Q = data_abs.loc[data_abs.Name == 'mean_Qt']
Qbar = data_abs.loc[data_abs.Name == 'monthly_Q']
Qdur = data_abs.loc[data_abs.Name == 'durations_Q']

print('\nAbsolute discharge [daily discharge Q]: \nMax:\n {}\n'.format(
    Q.sort_values(by='value', ascending=False).iloc[0][1:]))
print('Min:\n {}\n'.format(
    Q.sort_values(by='value', ascending=True).iloc[0][1:]))  
    
print('\nAbsolute discharge [monthly dishcarge Qbar]: \nMax:\n {}\n'.format(
    Qbar.sort_values(by='value', ascending=False).iloc[0][1:]))
print('Min:\n {}\n'.format(
    Qbar.sort_values(by='value', ascending=True).iloc[0][1:])) 
 
print('\nAbsolute discharge [seasonal flow durations Qdur]: \nMax:\n {}\n'.format(
    Qdur.sort_values(by='value', ascending=False).iloc[0][1:]))
print('Min:\n {}\n'.format(
    Qdur.sort_values(by='value', ascending=True).iloc[0][1:])) 
      

if doDiffs: 
    diffs = heatdata_abs.drop('area', axis = 0).fillna(0) - heatdata_spec.fillna(0)
    diffscmap = 'seismic'
    g2 = sns.heatmap(diffs, cmap = diffscmap, center=0.00, ax = axs[1], 
                      xticklabels=False, yticklabels=1, vmin = vmin, vmax =vmax, cbar_kws={"pad": 0.01})#, xticklabels = data.columns.get_level_values(0))
else: g2 = sns.heatmap(heatdata_spec, cmap = heatcmap, center = 0.185, ax = axs[1], 
                      xticklabels=False, yticklabels=1, vmin = vmin, vmax =vmax, cbar_kws={"pad": 0.01})#, xticklabels = data.columns.get_level_values(0))
topmiX = pd.DataFrame(heatdata_spec.max().sort_values(ascending=False))
#topmiX['rank']=topmiX.index.map(ranks)
topmiY = heatdata_spec.max(axis=1).sort_values(ascending=False).dropna()

print('\nSpecific discharge metrics [all]:\n\t Max: I(ASV {}; {})norm = {} bits/bit\n\t Min: I(ASV {}; {})norm = {} bits/bit\n\t Median: I(X;Y)norm = {} bits/bit'.format(
    topmiX.index[0], topmiY.index[0], topmiX.iloc[0, 0],  
    topmiX.index[-1],  topmiY.index[-1], topmiX.iloc[-1, -1], qmed))

q = data_spec.loc[data_spec.Name == 'meanqt']
qbar = data_spec.loc[data_spec.Name == 'monthlyq']
qdur = data_spec.loc[data_spec.Name == 'durationsq']

print('\nSpecific discharge [daily discharge q]: \nMax:\n {}\n'.format(
    q.sort_values(by='value', ascending=False).iloc[0][1:]))
print('Min:\n {}\n'.format(
    q.sort_values(by='value', ascending=True).iloc[0][1:]))  
    
print('\nSpecific discharge [monthly dishcarge qbar]: \nMax:\n {}\n'.format(
    qbar.sort_values(by='value', ascending=False).iloc[0][1:]))
print('Min:\n {}\n'.format(
    qbar.sort_values(by='value', ascending=True).iloc[0][1:])) 
 
print('\nSpecific discharge [seasonal flow durations qdur]: \nMax:\n {}\n'.format(
    qdur.sort_values(by='value', ascending=False).iloc[0][1:]))
print('Min:\n {}\n'.format(
    qdur.sort_values(by='value', ascending=True).iloc[0][1:]))

axs[1].set_xlabel ('ASV abundance rank', fontsize = axisfont)
g2.set_yticklabels(labs_g1[:-1], fontsize = yfont)
axs[1].yaxis.label.set_visible(False)
axs[1].text(-10, 8, '$q_{(t-n}$ $_{days)}$', rotation=90, va='center')
axs[1].text(-10, 22, r'$\bar{q}_{mon}$', rotation=90, va='center')
axs[1].text(-10, 30, '$q_{5,s}$', rotation=90, va='center')
axs[1].text(-10, 35, '$q_{95,s}$', rotation=90, va='center')


plt.xticks(ticks=[0, 24, 49, 74, 99], labels = [1, 25, 50, 75, 100], va = 'center', rotation = optrotation)
axs[1].tick_params(axis='x', pad = 8)
ax = fig.add_subplot(111, frameon = False)
ax.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
ax.set_ylabel('Hydrologic metrics', fontsize = axisfont, labelpad = 18)
ax.text(0.98, 0.5, 'Normalized mutual information ($I(X;Y)_{norm}$ $[bits/bit]$)', rotation = 'vertical', ha='center', va = 'center')
for g in [g1, g2]:
    g.axvline(x=0, color='k', linewidth = box_lw)
    g.axvline(x=heatdata_abs.shape[1], color='k', linewidth = box_lw)
    g.axhline(y = 0, color='k',linewidth = box_lw)
    g.axhline(y=len(g.get_yticklabels()), color = 'k', linewidth = box_lw)
    g.axhline(y=16, color='k', linewidth = box_lw*0.3) #xmin = -0.11, clip_on = False, 
    g.axhline(y=28, color='k', linewidth = box_lw*0.3)
    g.axhline(y=33, color = 'k', linewidth = box_lw*0.15)
    g.axhline(y=38, color = 'k', linewidth = box_lw*0.15)

    
    if g == g1:g.axhline(y=(heatdata_abs.shape[0]-1), color='k', linewidth = box_lw*0.3)

plt.tight_layout()
plt.subplots_adjust(hspace=0.02, wspace = 0.02)
# cbar1 = fig.colorbar(fig, pad=0.05)
if save_figs: plt.savefig('Figures/{}/FIG03.jpg'.format(suite), dpi = 600)
#%%
'''FIGURE 3 (remaining ASVs)'''
# Figure 3. 

heatcmap = 'gist_rainbow_r'
common_colorbar = False
box_lw = 1.5
yfont = 6
fig, axs = plt.subplots(2, 1, sharex=True, figsize = (7.5, 7.5)) # double_column, 1:1 aspect

data = MI_sorted_df.loc[MI_sorted_df.MI=='Ixy/Hy']
data2=data.copy()
data2.loc[drops_df.index,'value']=np.nan
months = data2.loc[data2.Name.str.contains('monthly')].Char.unique().astype(int)
month_names = ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 
                'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep']
months_dict = {months[i]: month_names[i] for i in range(len(months))}

# Absolute
data2_abs = data2.loc[data2.Name.isin(absolute)].drop('MI', axis =1)
data2_abs.index = data2_abs.index.map(ranks)
char_order = data2_abs.Char.unique()
data2_pivot_abs = data2_abs.reset_index().pivot(index = "Char", columns = '#OTU ID', values = 'value')
heatdata2_abs = data2_pivot_abs.loc[char_order]#data2_abs['split'] = data2_abs.Char.str.split("|")
num_Q = heatdata2_abs.drop('area', axis = 0).fillna(0).astype(bool).sum(axis=0)
asv_stats['No. Q stats'] = num_Q
asv_stats['Median MI (Q)'] = heatdata2_abs.drop('area', axis = 0).median(axis = 0, skipna=True)
asv_stats['Max MI (Q)'] = heatdata2_abs.drop('area', axis = 0).max(axis = 0)

# Specific
data2_spec = data2.loc[data2.Name.isin(specific)].drop('MI', axis =1)
data2_spec.index = data2_spec.index.map(ranks)
char_order = data2_spec.Char.unique()
data2_pivot_spec = data2_spec.reset_index().pivot(index = "Char", columns = '#OTU ID', values = 'value')
heatdata2_spec = data2_pivot_spec.loc[char_order]#data2_spec['split'] = data2_spec.Char.str.split("|")
num_q = heatdata2_spec.iloc[:-1, :].fillna(0).astype(bool).sum(axis=0)

asv_stats['No. q stats'] = num_q
asv_stats['Median MI (q)'] = heatdata2_spec.median(axis = 0, skipna=True)
asv_stats['Max MI (q)'] = heatdata2_spec.max(axis = 0)
if save_figs: asv_stats.to_csv('Figures/{}/Table_S3_with_summary_stats.csv'.format(suite))

mimax = max(heatdata2_abs.max().max(), heatdata2_spec.max().max(), 
            heatdata_abs.max().max(), heatdata_spec.max().max())
mimin = min(heatdata2_abs.min().min(), heatdata2_spec.min().min(),
            heatdata_abs.min().min(), heatdata_spec.min().min())
vmin, vmax = mimin, mimax

g1 = sns.heatmap(heatdata2_abs, cmap = heatcmap, center = 0.27, ax = axs[0], 
                 xticklabels=False, yticklabels=1, 
                 vmin = vmin, vmax = vmax, cbar_kws={"pad": 0.01})
#axs[0].set_ylabel('Absolute discharge', fontsize='small', labelpad=10)
axs[0].yaxis.label.set_visible(False)
axs[0].xaxis.set_visible(False)

axs[0].text(-200, 8, '$Q_{(t-n}$ $_{days)}$', rotation=90, va='center')
axs[0].text(-200, 22, r'$\bar{Q}_{mon}$', rotation=90, va='center')
axs[0].text(-200, 30.5, '$Q_{5,s}$', rotation=90, va='center')
axs[0].text(-200, 35, '$Q_{95,s}$', rotation=90, va='center')


axs[0].set_xlabel(None)
labs_g1 = list(heatdata2_abs.index)
for i in range(len(labs_g1)):
    lab = labs_g1[i]
    try: labs_g1[i] = months_dict[int(lab)]
    except: 
        if i < 28: 
            lab = lab.replace('t-', '')
            labs_g1[i] = lab.replace(' d', '')
        else: 
            lab = lab.replace('annual', 'Ann')
            if lab != 'area': labs_g1[i] = lab[-3:]
    

g1.set_yticklabels(labs_g1, fontsize = yfont)
g1.axhline(y = 0, color='k',linewidth = box_lw)
g1.axhline(y=44, color = 'k', linewidth = box_lw)

miQarray = heatdata2_abs.to_numpy().flatten()
Qmed = np.median(miQarray[~np.isnan(miQarray)])

miqarray = heatdata2_spec.to_numpy().flatten()
qmed = np.median(miqarray[~np.isnan(miqarray)])


# Statistical tests
# Median Q vs q
miabs = heatdata2_abs.iloc[:-1, :].copy()
mispec = heatdata2_spec.copy()

asvcount_area = heatdata2_abs.iloc[-1, :].fillna(0).astype(bool).sum()
asvcount_abs = heatdata2_abs.iloc[:-1, :].fillna(0).astype(bool).sum(axis=1) #exluding area
asvcount_spec = heatdata2_spec.fillna(0).astype(bool).sum(axis=1)

asv_mean = (pd.concat([asvcount_abs, asvcount_spec], axis = 0)).mean()
asv_std =  (pd.concat([asvcount_abs, asvcount_spec], axis = 0)).std()
print('Mean number of informative ASVs across all metrics [Q and q] = %.2f (+-%.2f SD)'%(asv_mean, asv_std))

mi_med = (pd.concat([heatdata2_abs, heatdata2_spec], axis =1)).median(axis=1).median()
mi_max = (pd.concat([heatdata2_abs, heatdata2_spec], axis =1)).max(axis=1).max()
mi_min = (pd.concat([heatdata2_abs, heatdata2_spec], axis =1)).min(axis=1).min()
print('Median MI across all metrics = %.3f (range = %.3f - %.3f bits/bit)'%(mi_med, mi_max, mi_min))

res = stats.mannwhitneyu(miabs, mispec,  alternative = 'greater')
print('\nMedian MI Q {} vs. q {}, Mann-Whitney U = {}, p = {}'.format(
    Qmed, qmed, res[0], res[1]))

print('Mean number of informative ASVs\n\tAbsolute: {} (SD ={})\n\tSpecific: {} (SD ={})\n\tArea: {}'.format(
    asvcount_abs.mean(), asvcount_abs.std(), asvcount_spec.mean(), asvcount_spec.std(), asvcount_area))
tstat, p = stats.ttest_rel(asvcount_abs, asvcount_spec, alternative = 'greater')
print('Two-sample t-test for difference in no. ASVs: t = {}, p = {}'.format(tstat, p))


Q = data2_abs.loc[data2_abs.Name == 'mean_Qt']
Qbar = data2_abs.loc[data2_abs.Name == 'monthly_Q']
Qdur = data2_abs.loc[data2_abs.Name == 'durations_Q']

print('\nAbsolute discharge [daily discharge Q]: \nMax:\n {}\n'.format(
    Q.sort_values(by='value', ascending=False).iloc[0][1:]))
print('Min:\n {}\n'.format(
    Q.sort_values(by='value', ascending=True).iloc[0][1:]))  
    
print('\nAbsolute discharge [monthly dishcarge Qbar]: \nMax:\n {}\n'.format(
    Qbar.sort_values(by='value', ascending=False).iloc[0][1:]))
print('Min:\n {}\n'.format(
    Qbar.sort_values(by='value', ascending=True).iloc[0][1:])) 
 
print('\nAbsolute discharge [seasonal flow durations Qdur]: \nMax:\n {}\n'.format(
    Qdur.sort_values(by='value', ascending=False).iloc[0][1:]))
print('Min:\n {}\n'.format(
    Qdur.sort_values(by='value', ascending=True).iloc[0][1:])) 
      

if doDiffs: 
    diffs = heatdata2_abs.drop('area', axis = 0).fillna(0) - heatdata2_spec.fillna(0)
    diffscmap = 'seismic'
    g2 = sns.heatmap(diffs, cmap = diffscmap, center=0.00, ax = axs[1], 
                      xticklabels=False, yticklabels=1, vmin = vmin, vmax =vmax, cbar_kws={"pad": 0.01})#, xticklabels = data2.columns.get_level_values(0))
else: g2 = sns.heatmap(heatdata2_spec, cmap = heatcmap, center = 0.185, ax = axs[1], 
                      xticklabels=False, yticklabels=1, vmin = vmin, vmax =vmax, cbar_kws={"pad": 0.01})#, xticklabels = data2.columns.get_level_values(0))
topmiX = pd.DataFrame(heatdata2_spec.max().sort_values(ascending=False))
#topmiX['rank']=topmiX.index.map(ranks)
topmiY = heatdata2_spec.max(axis=1).sort_values(ascending=False).dropna()

print('\nSpecific discharge metrics [all]:\n\t Max: I(ASV {}; {})norm = {} bits/bit\n\t Min: I(ASV {}; {})norm = {} bits/bit\n\t Median: I(X;Y)norm = {} bits/bit'.format(
    topmiX.index[0], topmiY.index[0], topmiX.iloc[0, 0],  
    topmiX.index[-1],  topmiY.index[-1], topmiX.iloc[-1, -1], qmed))

q = data2_spec.loc[data2_spec.Name == 'meanqt']
qbar = data2_spec.loc[data2_spec.Name == 'monthlyq']
qdur = data2_spec.loc[data2_spec.Name == 'durationsq']

print('\nSpecific discharge [daily discharge q]: \nMax:\n {}\n'.format(
    q.sort_values(by='value', ascending=False).iloc[0][1:]))
print('Min:\n {}\n'.format(
    q.sort_values(by='value', ascending=True).iloc[0][1:]))  
    
print('\nSpecific discharge [monthly dishcarge qbar]: \nMax:\n {}\n'.format(
    qbar.sort_values(by='value', ascending=False).iloc[0][1:]))
print('Min:\n {}\n'.format(
    qbar.sort_values(by='value', ascending=True).iloc[0][1:])) 
 
print('\nSpecific discharge [seasonal flow durations qdur]: \nMax:\n {}\n'.format(
    qdur.sort_values(by='value', ascending=False).iloc[0][1:]))
print('Min:\n {}\n'.format(
    qdur.sort_values(by='value', ascending=True).iloc[0][1:]))

axs[1].set_xlabel ('ASV abundance rank (overall)', fontsize = axisfont)
g2.set_yticklabels(labs_g1[:-1], fontsize = yfont)
axs[1].yaxis.label.set_visible(False)
axs[1].text(-200, 8, '$q_{(t-n}$ $_{days)}$', rotation=90, va='center')
axs[1].text(-200, 22, r'$\bar{q}_{mon}$', rotation=90, va='center')
axs[1].text(-200, 30, '$q_{5,s}$', rotation=90, va='center')
axs[1].text(-200, 35, '$q_{95,s}$', rotation=90, va='center')


plt.xticks(ticks=[0, 499, 999, 1499, 1999], labels = [1, 500, 1000, 1500, 2000], va = 'center', rotation = optrotation)
axs[1].tick_params(axis='x', pad = 8)
ax = fig.add_subplot(111, frameon = False)
ax.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
ax.set_ylabel('Hydrologic metrics', fontsize = axisfont, labelpad = 18)
ax.text(0.98, 0.5, 'Normalized mutual information ($I(X;Y)_{norm}$ $[bits/bit]$)', rotation = 'vertical', ha='center', va = 'center')
for g in [g1, g2]:
    g.axvline(x=0, color='k', linewidth = box_lw)
    g.axvline(x=heatdata2_abs.shape[1], color='k', linewidth = box_lw)
    g.axhline(y = 0, color='k',linewidth = box_lw)
    g.axhline(y=len(g.get_yticklabels()), color = 'k', linewidth = box_lw)
    g.axhline(y=16, color='k', linewidth = box_lw*0.3) #xmin = -0.11, clip_on = False, 
    g.axhline(y=28, color='k', linewidth = box_lw*0.3)
    g.axhline(y=33, color = 'k', linewidth = box_lw*0.15)
    g.axhline(y=38, color = 'k', linewidth = box_lw*0.15)

    
    if g == g1:g.axhline(y=(heatdata_abs.shape[0]-1), color='k', linewidth = box_lw*0.3)

plt.tight_layout()
plt.subplots_adjust(hspace=0.02, wspace = 0.02)
# cbar1 = fig.colorbar(fig, pad=0.05)
if save_figs: plt.savefig('Figures/{}/FIGS5.jpg'.format(suite), dpi = 600)

#%%

'''########### BOX PLOTS ###########
Figure 4. The number microbial amplified sequence variants (ASVS)
that share mutual information with absolute (color) and specific (color) 
DAILY DISCHARGE at different time lags before sampling.'''

coord_colors = plt.get_cmap(heatcmap)(np.linspace(0, 1, 10))
frames = []
ms = 3
malpha = 0.6
boxcolor = 'lightgrey'

for name in ['meanqt', 'mean_Qt']:
    cur_MI = MI_all.xs((name, 'Ixy/Hy'), level = (0, 2), axis = 1) #Xould use MI_sorted_df
    asvcount = cur_MI.fillna(0).astype(bool).sum(axis=0)
    daily_asvs = asvcount.copy()
    MI_long = pd.melt(cur_MI, var_name=['Lag'], value_name = 'MI' )
    cur_data = MI_long
    lags_all = cur_data.copy()
    ints = [re.findall(r'\d+', i) for i in cur_data.Lag.unique()]
    days = [int(item) for sublist in ints for item in sublist]
    lags_dict = {cur_data.Lag.unique()[i]: days[i] for i in range(len(cur_data.Lag.unique()))}
    cur_data['days'] = cur_data.Lag.map(lags_dict)
    
    fig, axs = plt.subplots(1, 1,figsize = (twocol[0], onecol[0]))
    # sw = sns.swarmplot (x = 'Lag', y = 'MI', data = cur_data, dodge = True, ax=axs,
    #                     s = ms, palette = ['limegreen'], alpha = malpha)
    bx = sns.boxplot(x = 'Lag', y = 'MI', data = cur_data, linewidth= 0.4, ax = axs,
                      fliersize=2, medianprops=dict(lw = 1, zorder=5, color = 'k', alpha = 0.5), whiskerprops = dict(lw = 0.5), capprops = dict(lw=0.5), 
                      boxprops =dict(facecolor=boxcolor, edgecolor=boxcolor))
    
    axs.set_ylabel('Normalized mutual information\n$[bits/bit]$', fontsize = axisfont)
    axs.tick_params(axis='both', which = 'both', labelsize=labelfont)
    axs.set_yscale('log')
    axs.yaxis.set_minor_formatter('{x:.2f}')
    axs.yaxis.set_major_formatter('{x:.2f}')
    axs.set_xticklabels(days, fontsize=labelfont, rotation = optrotation, ha = 'center')
    axs.set_yticks([0.1, .15, .20, .25, .3, .35, .40])
    axs.set_ylim(0.122,0.46)
    
    ax1 = axs.twinx()
    
    lag_medians = cur_data.groupby(by='days').median()
    print('\nMedian MI by lag:\n', lag_medians.sort_values(by='MI', ascending =False))
    days_arr = np.array(days) 
      
    # Median value of MI
    popt, pcov = optimize.curve_fit(second_order_fxn, days_arr, lag_medians.MI.values)
    yfit = second_order_fxn(days_arr, *popt)
    qt_r = stats.pearsonr(lag_medians.MI.values, yfit.values)
    
    if qt_r[1]<0.05: a = 1
    else: a = 0
    axs.plot(days_arr/2, yfit.values, ls='dashed', color = 'r', lw=0.75, alpha = a, 
             label = 'Median $I(X;Y)_{norm}$\n($r = %.2f, p = %.3f$)' %(qt_r[0], qt_r[1]))
    
     # Number of ASVs
    popt, pcov = optimize.curve_fit(second_order_fxn, days_arr, asvcount.values)
    yfit = second_order_fxn(days_arr, *popt)
    qt_r = stats.pearsonr(asvcount.values, yfit.values)
    
    if qt_r[1]<0.05: a = 1
    else: a = 0
    ax1.plot(days_arr/2, yfit, ls='dashed', color = 'forestgreen', lw=0.75, alpha = a, 
             label = 'Number of ASVs\n($r = %.2f, p = %.3f$)' %(qt_r[0], qt_r[1]))
    ax1.scatter(days_arr/2, asvcount.values, marker = '^', s = 36, color = 'forestgreen', alpha = 0.8)
    ax1.set_ylabel('No. of informative microbial taxa', fontsize = axisfont, color='forestgreen')
    ax1.yaxis.set_major_formatter('{x:.0f}')
    plt.yticks(fontsize=labelfont)
    axs.set_xticklabels(days, fontsize=labelfont, rotation = optrotation, ha = 'center')
    ax1.set_ylim(12,33)

    plt.tight_layout()

    lines, labels = axs.get_legend_handles_labels()
    lines2, labels2 = ax1.get_legend_handles_labels()
    

    
    if name == 'mean_Qt': 
        axs.set_xlabel('Daily discharge ($Q_{(t-n}$ $_{days)}$)')
        axs.legend(lines + lines2, labels + labels2, loc='upper left', ncol =2, fontsize = legendfont, handletextpad = 0.1)
        axs.set_ylim(0.122,0.46)
        ax1.set_ylim(12,33)
        if save_figs: plt.savefig('Figures/{}/FIG04.jpg'.format(suite), dpi = 600)
    if name == 'meanqt': 
        axs.set_xlabel('Daily discharge ($q_{(t-n}$ $_{days)}$)')
        axs.legend(lines, labels, loc='upper right', ncol =2, fontsize = legendfont, handletextpad = 0.1)
        axs.set_ylim(0.122,0.36)
        ax1.set_ylim(6,33)
        if save_figs: plt.savefig('Figures/{}/FIGS1.jpg'.format(suite), dpi = 600)   

#%%      
'''Figure 5 (S2). The number microbial amplified sequence variants (ASVS)
that share mutual information with absolute MONTHLY DISCHARGE.'''

for name in ['monthlyq', 'monthly_Q']:
    cur_MI = MI_all.xs((name, 'Ixy/Hy'), level = (0, 2), axis = 1) #Xould use MI_sorted_df
    asvcount = cur_MI.fillna(0).astype(bool).sum(axis=0)
    monthly_asvs = asvcount.copy()
    MI_long = pd.melt(cur_MI, var_name=['Month'], value_name = 'MI' )
    cur_data = MI_long
    months_all = cur_data.copy()
    
    fig, axs = plt.subplots(1, 1,figsize = (twocol[0], onecol[0]))
    # sw = sns.swarmplot (x = 'Month', y = 'MI', data = cur_data, dodge = True, ax=axs,
    #                     s = ms, palette = [colors_bar[1]], alpha = malpha)
    bx = sns.boxplot(x = 'Month', y = 'MI', data = cur_data, linewidth= 0.4, ax = axs,
                      fliersize=2, medianprops=dict(lw = 1, zorder=5, color = 'k', alpha = 0.5), whiskerprops = dict(lw = 0.5), capprops = dict(lw=0.5), 
                      boxprops =dict(facecolor=boxcolor, edgecolor=boxcolor))
    
    axs.set_ylabel('Normalized mutual information\n$[bits/bit]$', fontsize = axisfont)
    axs.tick_params(axis='both', which = 'both', labelsize=labelfont)
    axs.set_yscale('log')
    axs.yaxis.set_minor_formatter('{x:.2f}')
    axs.yaxis.set_major_formatter('{x:.2f}')
           
    # Median MI
    month_medians = cur_data.groupby(by='Month', sort=False).median()
    print('\nMedian MI by month:\n', month_medians.sort_values(by='MI', ascending =False))
    
    popt, pcov = optimize.curve_fit(second_order_fxn, np.arange(1, 13), month_medians.MI)
    yfit = second_order_fxn(np.arange(1, 13), *popt)
    q_r = stats.pearsonr(month_medians.MI, yfit)
    if q_r[1]<0.05: a = 1
    else: a = 0
    axs.plot(np.arange(0, 12), yfit, ls='dashed', alpha = a, color = 'r', lw=0.75, 
             label ='Median $I(X;%s)_{norm}$\n($r = %.2f, p = %.3f$)' %(name[-1],qt_r[0], qt_r[1]))
    axs.set_xticklabels(month_names, fontsize=labelfont, rotation = optrotation)
    axs.set_yticks([0.1, .15, .20, .25, .3, .35, .40, 0.45, 0.5])

    ax1 = axs.twinx()

    # ASV Count    
    popt, pcov = optimize.curve_fit(second_order_fxn, np.arange(0, 12), asvcount.values)
    yfit = second_order_fxn(np.arange(0, 12), *popt)
    qt_r = stats.pearsonr(asvcount.values, yfit.values)
    
    if qt_r[1]<0.05: a = 1
    else: a = 0
    ax1.plot(yfit, ls='dashed', color = colors_bar[1], lw=0.75, alpha = a, 
             label = 'Number of ASVs\n($r = %.2f, p = %.3f$)' %(qt_r[0], qt_r[1]))
    ax1.scatter(np.arange(0, 12), asvcount.values, marker = '^', s = 36, color = colors_bar[1], alpha = 0.8)

    ax1.set_ylabel('No. of informative microbial taxa', fontsize = axisfont, color=colors_bar[1])
    ax1.yaxis.set_major_formatter('{x:.0f}')
    plt.yticks(fontsize=labelfont)
    
    lines, labels = axs.get_legend_handles_labels()
    lines2, labels2 = ax1.get_legend_handles_labels()
    plt.tight_layout()

    if name == 'monthlyq': 
        axs.set_xlabel(r'Mean monthly discharge ($\bar{q}_{mon}$)', fontsize = axisfont)
        ax1.set_ylim(0,28)
        axs.set_ylim(0.128, 0.40)
        axs.legend(lines, labels, loc='upper center', ncol =2, fontsize = legendfont, handletextpad = 0.1)
        if save_figs: plt.savefig('Figures/{}/FIGS2.jpg'.format(suite), dpi = 600)
    if name == 'monthly_Q': 
        axs.set_xlabel(r'Mean monthly discharge ($\bar{Q}_{mon}$)', fontsize = axisfont)
        ax1.set_ylim(8,33)  
        axs.set_ylim(0.128, 0.49)
        axs.legend(lines2, labels2, loc='upper right', ncol =2, fontsize = legendfont, handletextpad = 0.1)
        if save_figs: plt.savefig('Figures/{}/FIG05.jpg'.format(suite), dpi = 600)

#%%

from matplotlib.patches import PathPatch

def adjust_box_widths(g, fac):
    """
    Adjust the withs of a seaborn-generated boxplot.
    """
    xmidlist=[]
    # iterating through Axes instances
    for ax in g.axes:

        # iterating through axes artists:
        for c in ax.get_children():

            # searching for PathPatches
            if isinstance(c, PathPatch):
                # getting current width of box:
                p = c.get_path()
                verts = p.vertices
                verts_sub = verts[:-1]
                xmin = np.min(verts_sub[:, 0])
                xmax = np.max(verts_sub[:, 0])
                xmid = 0.5*(xmin+xmax)
                xhalf = 0.5*(xmax - xmin)
                xmidlist.append(xmid)
                # setting new width of box
                xmin_new = xmid-fac*xhalf
                xmax_new = xmid+fac*xhalf
                verts_sub[verts_sub[:, 0] == xmin, 0] = xmin_new
                verts_sub[verts_sub[:, 0] == xmax, 0] = xmax_new

                # setting new width of median line
                for l in ax.lines:
                    if np.all(l.get_xdata() == [xmin, xmax]):
                        l.set_xdata([xmin_new, xmax_new])
    return(xmidlist)
                        
''' FIGURE 6 (S3). Normalized mutual information scores among microbial amplificed sequence variants (ASVs) for 
MEAN SEASONAL DISCHARGE at high, median, and low flows (, respectivetly).  Boxes show median and interquartile range; 
whiskers show values within 1.5 times the ineterquartile range.''' 

colors = plt.cm.tab10(np.linspace(0, 1, 10))
# colors_dict = {"Red":"ff0000","Cyber Yellow":"ffd300","Azure":"147df5",
#                "Spring Bud":"a1ff0a","Electric Purple":"be0aff",
#                "Electric Blue":"0aefff","Han Purple":"580aff", "Medium Spring Green":"0aff99"}
colors_dict = {'ANN': colors[4], 'OND': colors[1], 'JFM': colors[0], 'AMJ': colors[2], 'JAS': colors[5], }
clrs = [colors[1], colors[0], colors[2], colors[5], colors[4]]
clrs = [coord_colors[2], coord_colors[3], coord_colors[1]]
seasons = list(colors_dict.keys())
ep_list = ['High', 'Low']

for name in ['durationsq','durations_Q']:
    cur_MI = MI_all.xs((name, 'Ixy/Hy'), level = (0, 2), axis = 1)
    cur_MI.columns = cur_MI.columns.str.replace("annual", "Ann")
    cur_MI = cur_MI[cur_MI.columns.drop(list(cur_MI.filter(regex='50')))] # Drop Q50
    mi = tuple(cur_MI.columns.str.split("|", expand = True))
    cur_MI.columns = mi
    asvcount = cur_MI.fillna(0).astype(bool).sum(axis=0)
    durs_asvs = asvcount.copy()
    asvcount.index = asvcount.index.swaplevel(0,1)
    print('Mean no. informative ASVs by flow duration (EP):')
    [print('\t', a, asvcount.loc[(slice(None), a)].mean()) for a in asvcount.index.get_level_values(1).unique()]
    counts = [asvcount.loc[(slice(None), a)].values for a in asvcount.index.get_level_values(1).unique()]
    print(stats.f_oneway(*counts))
    
    MI_long = pd.melt(cur_MI, var_name = ['EP','Season'], value_name = 'MI')
    dur_all = MI_long
    plots = list(MI_long.EP.unique())
    
    dur_medians = MI_long.groupby(['EP', 'Season']).median()
    print(dur_medians.sort_values(by = 'MI', ascending = False))
    
    fig, axs = plt.subplots(figsize = (twocol[0], onecol[0]))
    cur_data = MI_long   
    cur_stats = cur_data.groupby(['Season'])['MI'].describe()


    bx = sns.boxplot(x = 'Season', y = 'MI', hue = 'EP', data = cur_data, linewidth= 0.4, palette = ['grey', 'lightgrey'],
                          fliersize=2, medianprops=dict(lw = 1, zorder=5, alpha = 0.5), whiskerprops = dict(lw = 0.5), capprops = dict(lw=0.5)) 
                          #boxprops =dict(facecolor=boxcolor, edgecolor=boxcolor))    
    xlist = adjust_box_widths(fig, 0.9)
    plt.tick_params(axis='both', which = 'both', labelsize=labelfont)
    plt.yscale('log')
    axs.set_yticks([0.1, .15, .20, .25, .3, .35, .40, 0.45, 0.5])
    axs.yaxis.set_minor_formatter('{x:.2f}')
    axs.yaxis.set_major_formatter('{x:.2f}')
    axs.set_ylim(0.12, 0.45)
    axs.legend().remove()
    plt.xlabel('Seasonal flow durations ($%s_{P,s}$)' %name[-1], fontsize = axisfont, labelpad = 7)
    plt.ylabel('Normalized mutual information\n$[bits/bit]$', labelpad = 7, fontsize = axisfont)
    
    ax1 = axs.twinx()
    ax1.scatter([xlist[::2]], counts[0], marker = '^', color = clrs[0])
    ax1.scatter([xlist[1::2]], counts[1], marker = 'v', color = clrs[1])
    ax1.set_ylabel('No. of informative microbial taxa', fontsize = axisfont, color = clrs[0])

    markers, labels = axs.get_legend_handles_labels()
    full_labs = [r'%s flow (${%s}_{%s}$)'%(ep_list[i], name[-1], labels[i]) for i in [0,1]]
    lgnd = plt.legend(markers, full_labs, fontsize=legendfont, loc = 'upper left')

    plt.tight_layout()
    
    if name=='durations_Q': 
        axs.set_ylim(0.125, 0.54)
        ax1.set_ylim(0, 39.5)
        if save_figs: plt.savefig('Figures/{}/FIG06.jpg'.format(suite), dpi = 600)
    if name=='durationsq': 
        axs.set_ylim(0.125, 0.54)

        ax1.set_ylim(0, 38)

        if save_figs: plt.savefig('Figures/{}/FIGS3.jpg'.format(suite), dpi = 600)

    dur_medians = MI_long.groupby(by='EP').median()
    print('\nMedian MI by durations:\n', dur_medians.sort_values(by='MI', ascending =False))
    
    ### Differences in median MI between seasons and durations ###
    print('\n\nFlow Durations Median MI\n (High (5%) vs. Low (95%)) **Kruskal**')
    by_durs = MI_long[['EP', 'MI']]
    print('All', stats.kruskal(by_durs.loc[by_durs['EP']=='5']['MI'], 
                               #by_durs.loc[by_durs['EP']=='50']['MI'], 
                               by_durs.loc[by_durs['EP']=='95']['MI'], nan_policy='omit'), '\n')

#%%
# Differences between three hydrologic categories
# Medians
plt.figure()
plt.hist(months_all.iloc[:, -1], bins = 20, label = 'months')
plt.hist(dur_all.iloc[:, -1], bins = 20, label = 'durs')
plt.legend()
plt.show()

print('\n\nMedian value of MI for daily discharge over all time lags = {} (range = {}-{})'.format(
    round(lags_all.iloc[:, -1].median(), 4), round(lags_all.iloc[:, -1].min(), 3), round(lags_all.iloc[:, -1].max(), 3)))
print('Median value of MI for monthly discharge over all time months = {} (range = {}-{})'.format(
    round(months_all.iloc[:, -1].median(), 4), round(months_all.iloc[:, -1].min(), 3), round(months_all.iloc[:, -1].max(), 3)))
print('Median value of MI for seasonal flow durations over all time lags = {} (range = {}-{})'.format(
    round(dur_all.iloc[:, -1].median(), 4), round(dur_all.iloc[:, -1].min(), 3), round(dur_all.iloc[:, -1].max(), 3)))

print('Mann-Whitney U results:\nMean momthly vs. Daily discharge:', stats.mannwhitneyu(lags_all.iloc[:, -1], months_all.iloc[:, -1]) )
print('Daily vs Seasonal flow durations:', stats.mannwhitneyu(dur_all.iloc[:, -1], lags_all.iloc[:, -1]))
print('Mean monthly vs Seasonal flow durations:', stats.mannwhitneyu(dur_all.iloc[:, -1], months_all.iloc[:, -1]))


# Number of ASVs
print('Mean number (+-SD) number of informative ASVs:\nMean Monhthly: {} (+-{})\nDaily: {} (+-{})\nDuraions: {} (+-{})'.format(
    daily_asvs.mean(), daily_asvs.std(), monthly_asvs.mean(), monthly_asvs.std(), durs_asvs.mean(), durs_asvs.std()))
print('One-way ANOVA difference in number of ASVs:\n', stats.f_oneway(daily_asvs, monthly_asvs, durs_asvs))

#%%
''' Mutual Information between microbial ASVs and A) total abuncance of the ASV across all sites, 
B) Variance of ASV across all sites, and C) number of sites at which the ASV was detected for 
daily mean absolute and specific discharge at different time lags from day of sampling (t = 0)'''

                
print('\n\nMI vs abundance, number of sites [Linear regression]):\n')
ivs = {'Log Abundance': np.log(otu_abund), 'Number of sites': asv_no_sites}
#ivs = {'Abundance': otu_abund, 'Variance': otu_var}#, 'Number of sites': asv_no_sites}
print('\n\n\n MI vs.', k)
names = {'mean_Qt': '$Q_{(t-n}$ $_{days)}$', 'monthly_Q': r'$\bar{Q}_{mon}$', 'durations_Q': r'${Q}_{P, s}$'}

msize = 4
markers = ['o', 'v', 's']
colors = ['forestgreen', colors_bar[1], clrs[1]]

i = 0
for k, v in ivs.items():
    fig, axs = plt.subplots(figsize=(onecol)) 
    x = v 
    j = 0  
    for k2, v2 in names.items():
        mi = MI_all.xs((k2, 'Ixy/Hy'), level = (0, 2), axis = 1)
        cur_MI = pd.melt(mi, var_name=['Time'], value_name = 'MI', ignore_index = False )
        #cur_MI_256 = cur_MI.loc[drops_df.index]
        cur_MI.dropna(axis = 0, subset =['MI'], inplace=True)
        x_cur = cur_MI.index.map(v)
        print('\n', k2)
        y_cur = cur_MI['MI']
        res = stats.linregress(x_cur, y_cur)
        plt.scatter(x_cur, y_cur, marker = markers[j], facecolors='none', 
                       edgecolors = colors[j], s = msize, alpha = 0.3, linewidth = 0.4)
        plt.scatter(x_cur, y_cur, marker = markers[j], facecolors='none', 
                       edgecolors = colors[j], s = msize, alpha = 0.3, linewidth = 0.4)
        if res[3]>0.001: pstring = '$p={}$'.format(np.round(res[3], decimals = 2))
        else: pstring = '$p<0.001$'
        xs = np.linspace(x_cur.min(), x_cur.max(), 5)
        plt.plot(xs, res.slope*xs+res.intercept, ls = 'dashed', lw = 0.75, color = colors[j], 
                 label = '{} ($r={}$, {})'.format(v2, np.round(res[2], decimals=2), pstring))
        plt.ylabel('Normalized mutual information\n[bits/bit]', fontsize=axisfont)
        plt.legend(fontsize=legendfont, handletextpad = 0.2)
        plt.xticks(fontsize=labelfont)
        plt.yticks(fontsize=labelfont)
        if i == 0: plt.xlabel('Log abundance of microbial taxa', fontsize=axisfont)
        if i==1: plt.xlabel('Number of sites detected', fontsize=axisfont)
        j+=1
    plt.tight_layout()
    if save_figs: 
        if k == 'Log Abundance': plt.savefig('Figures/{}/FIG7.jpg'.format(suite), dpi = 600)  
        else: plt.savefig('Figures/{}/FIGS6.jpg'.format(suite), dpi = 600)
        
    i+=1
    
#%%  
if suite == '4701':
    '''Mutual Information between microbial ASVs and number of watersheds in which the ASV was detected for 
    daily mean absolute and specific discharge at different time lags from day of sampling (t = 0)'''
    
                    
    print('\n\nMI vs number of watersheds:\n')
    names = {'mean_Qt': '$Q_{(t-n}$ $_{days)}$, $q_{(t-n}$ $_{days)}$', 'monthly_Q': r'$\bar{Q}_{mon}$, $\bar{q}_{mon}$', 'durations_Q': r'${Q}_{P, s}$, ${q}_{P, s}$',
             'meanqt': '$Q_{(t-n}$ $_{days)}$, $q_{(t-n}$ $_{days)}$', 'monthlyq': r'$\bar{Q}_{mon}$, $\bar{q}_{mon}$', 'durationsq': r'${Q}_{P, s}$, ${q}_{P, s}$'}
        #'mean_qt': '$Q_{(t-n}$ $_{days)}$', 'monthly_q: r'$\bar{Q}_{mon}$', 'durations_Q': r'${Q}_{P, s}$'}
    colors = ['forestgreen', colors_bar[1], clrs[1]]
    
    fig, axs = plt.subplots(figsize=(onepluscol))
    
    mi = MI_all.xs('Ixy/Hy', level = 2, axis = 1)
    cur_MI = pd.melt(mi, var_name=['Time'], value_name = 'MI', ignore_index = False )
    cur_MI.dropna(axis = 0, subset =['MI'], inplace=True)
    cur_MI['no_basins'] = cur_MI.index.map(no_basins)
    cur_MI['label'] = cur_MI.Time.map(names)
    daily_df = cur_MI.loc[cur_MI.label==names['mean_Qt']]
    monthly_df = cur_MI.loc[cur_MI.label==names['monthly_Q']]
    seasonal_df = cur_MI.loc[cur_MI.label==names['durations_Q']]
    daily = daily_df.no_basins.value_counts().to_dict()
    monthly = monthly_df.no_basins.value_counts().to_dict()
    seasonal = seasonal_df.no_basins.value_counts().to_dict()
    
    sns.boxplot(x = cur_MI.no_basins, y= 'MI', hue='label', data=cur_MI, linewidth= 0.4, palette=colors,
                      fliersize=2, medianprops=dict(lw = 0.5, zorder=5, color = 'k'), whiskerprops = dict(lw = 0.5), capprops = dict(lw=0.5), boxprops =dict(alpha=0.7), width=0.97)
    
    plt.ylabel('Normalized mutual information [bits/bit]', fontsize=axisfont)
    leg = plt.legend(fontsize=legendfont, handletextpad = 0.2)
    for lh in leg.legendHandles: 
        lh.set_alpha(0.7)
    plt.xticks(fontsize=legendfont)
    plt.yticks(fontsize=legendfont)
    plt.xlabel('Number of watersheds', fontsize=axisfont)
    
    for k, v in monthly.items():
        if k == 1: 
            hd = monthly_df.loc[monthly_df.no_basins==k].MI.max()+0.0275
            col = 'dimgrey'
        else: 
            hd = monthly_df.loc[monthly_df.no_basins==k].MI.median()
            col = 'white'
        plt.text(k-1, hd+0.003, 'n= %s'%v, ha = 'center', fontsize = legendfont, fontstyle='italic', color=col)
    for k, v in daily.items():
        if k == 1: 
            hm = daily_df.loc[daily_df.no_basins==k].MI.max()+0.01
            col = 'dimgrey'
        else: 
            hm = daily_df.loc[daily_df.no_basins==k].MI.median()
            col = 'white'
        plt.text(k-1.32, hm+0.003, 'n= %s'%v, ha = 'center', fontsize = legendfont, fontstyle='italic', color=col)
    for k, v in seasonal.items():
        if k == 1: 
            hs = seasonal_df.loc[seasonal_df.no_basins==k].MI.max()+0.01
            col = 'dimgrey'
        else: 
            hs = seasonal_df.loc[seasonal_df.no_basins==k].MI.median()
            col = 'white'
        plt.text(k-0.68, hs+0.003, 'n= %s'%v, ha = 'center', fontsize = legendfont, fontstyle='italic', color=col)
    #plt.ylim(-0.04)
    plt.tight_layout()
    if save_figs: 
        plt.savefig('Figures/{}/FIGS6.jpg'.format(suite), dpi = 600)  
    
