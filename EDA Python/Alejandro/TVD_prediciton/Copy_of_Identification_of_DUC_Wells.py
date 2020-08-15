#!/usr/bin/env python
# coding: utf-8

# Three (3) datasets (Wellheader, Production and PerfTreatment) were provided for wells in Alberta, British Columbia and saskatchewan. The purpose of this Untapped Energy Datathon exercise is to identify the DUC wells and provide insights.
# 
# Key Assumptions made to identify DUC wells are as follows: 
# 
# > A well with a Current Status of 'Flowing', 'Gas Lift', 'Pumping' is not a DUC well.
# 
# > A well with production volumes recorded  is not a DUC well.
# 
# > A well with Completion Activities such as Perforations, Well Stimulations is not a DUC well.
# 
# As part of the insights, we are also assuming that all wells where DUC wells at some point. We will also determine how long the non-DUC wells had DUC status by assuming that as the difference between the First Production Date or Last Activity Date whichever is greater. 
# 
# For exploratory Data Analysis please review separate Jupyter notebook 'EDA for Identification of DUC Wells' and corresponding html files. 
# 

# ## Import Libraries

# In[3]:


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np


# ## Load datasets

# In[4]:



## list date columns to parse dates for wellheader dataset
datecolumns = ['LicenceDate', 'ConfidentialReleaseDate','AbandonDate', 'SurfAbandonDate', 'SpudDate', 'FinalDrillDate', 'RigReleaseDate','StatusDate','CompletionDate']  

# load datasets
wellheader = pd.read_csv('../../../data/WellHeader_Datathon.csv',parse_dates=datecolumns, low_memory=False, na_values='Not Applicable')
perftreatment = pd.read_csv('../../../data/PerfTreatments.csv',parse_dates=['ActivityDate'])
wellproduction = pd.read_csv('../../../data/WellProduction.csv', parse_dates=['ProdPeriod'])


# In[5]:


wellheader = wellheader.iloc[:,0:85]
wellproduction=wellproduction.iloc[:,0:4]
perftreatment=perftreatment.iloc[:,0:8]


# ## Convert datatypes

# In[12]:


## Select objects to convert to category type for wellheader data
def clean_pipeline2(df):
    ## Select objects to convert to category type 
    cat =list(df.dtypes[df.dtypes == 'object'].index)
    df[cat] = df[cat].astype('category')
    surface_columns=['Surf_LSD','Surf_Section','Surf_Township','Surf_Range','BH_LSD','BH_Section','BH_Township','BH_Range']
    df[surface_columns]=df[surface_columns].astype(str)
    #df_s = df[column_selection]
    #df_s['TVD']= df['TVD']
    return df

def duc_welss(wellheader,perftreatment,wellprodcution)
    wellheader = clean_pipeline2(wellheader)

    # change ProdType and ActivityType columns from object to category datatype
    wellproduction['ProdType'] = wellproduction['ProdType'].astype('category')
    perftreatment['ActivityType'] = perftreatment['ActivityType'].astype('category')

    ##create list of columns to be dropped
    dropcols_wellheader= ['CompletionDate','UnitID', 'UnitFlag', 'UnitName', 'Confidential', 'RegulatoryAgency', 'ConfidentialReleaseDate',         'AbandonDate', 'SurfAbandonDate', 'OSArea', 'OSDeposit', 'Municipality']

    wellheader = wellheader.drop(columns=dropcols_wellheader, axis=1, inplace=False)

    ## create perf activity type dictionary to classify as Completion Activity (Yes or No) based on our assumptions and intrepretation for the different perftreatment['ActivityType'].
    perf_activity_dict = {'Perforation': 'Yes',
     'Fracture': 'Yes',
     'Hydraulic Fracture': 'Yes',
     'Sand Fracture': 'Yes',
     'Open Hole': 'No',
     'Chemical Fracture': 'Yes',
     'Other' : 'No',
     'Acid Squeeze' : 'Yes',
     'Bridge Plug Set' : 'No',
     'Acid Wash' : 'Yes',
     'Acidize' : 'Yes',
     'Remedial' : 'Yes',
     'Cement Squeeze' : 'Yes',
     'Hydra Jet Perforation': 'Yes',
     'Slotted Liner' : 'Yes',
     'Open Hole/Barefoot Completion' : 'No',
     'Remedial Casing Cementing' : 'No',
     'Cement Plug' : 'Yes',
     'Multi-Stage Fracture - Port Closed': 'Yes',
     'Bridge Plug - No Cement' : 'No',
     'Packing Device Capped w/Cement' : 'Yes',
     'Chemical Squeeze': 'Yes',
     'Casing Patch' : 'No',
     'Acid Treatment' : 'Yes',
     'Multi-Stage Fracture': 'Yes'}

    perftreatment['CompletionActivity'] = perftreatment['ActivityType'].map(perf_activity_dict)
    perftreatment['CompletionActivity'] = perftreatment['CompletionActivity'].astype('category')


    # ## Identify DUC Wells. 
    # 
    # 1. Assume any Well that doesn't have status as ['Pumping', 'Flowing', 'Gas Lift'] could potentially be a DUC well. Create a subset of         data of wells without this status as subset_wellheader. 
    # 2. Filter out the wells from the subset_wellheader data that are not in the perftreatment or production datasets as DUCS.
    # 3. Filter out any wells in the perftreatment data, that have no recorded production data as DUC_perf.
    # 4. Identify if there are any potential DUCs based on perftreatment data only. 
    # 5. If there are cross check with DUCS_perf (perftreatment wells with no production data). The assumption here is that some wells may have     been completed but had no production volumes. 
    # 6. Filter out wells that have identified to have no completion activities from DUCS_perf as DUCS_perf_final. 


    # filter out wells with Current Status that are not in ['Pumping', 'Flowing', 'Gas Lift']
    subset_wellheader = wellheader[~wellheader['CurrentStatus'].isin(['Pumping', 'Flowing', 'Gas Lift'])]

    # check only required variables for Current Status remain
    subset_wellheader['CurrentStatus'].unique()

    # check number of wells remaining 
    print('number of wells with Current Status not in ["Pumping", "Flowing", "Gas Lift"] ', subset_wellheader['EPAssetsId'].nunique())
    # filtering wells not in perf_treatment nor in well_production from subset_wellheader
    DUCS = subset_wellheader[~subset_wellheader['EPAssetsId'].isin(perftreatment['EPAssetsId'])]
    DUCS = DUCS[~DUCS['EPAssetsId'].isin(wellproduction['EPAssetsId'])]
    print ('number of wells not in perf_treatment data nor in well_production data ', DUCS.shape[0])
    # filtering wells in perftreatment that have no production data
    DUCS_perf = perftreatment[~perftreatment['EPAssetsId'].isin(wellproduction['EPAssetsId'])]
    print('number of wells from perftreatment datatable that have no production data ', DUCS_perf['EPAssetsId'].nunique())
    ## Create subsets of Perf Treatment to obtain wells which have Completion Activity as Yes, No or NaN
    perftreatment_subset_columns = perftreatment[['EPAssetsId', 'CompletionActivity', 'ActivityDate']]
    perftreatment_subset_null_values = perftreatment_subset_columns[pd.isnull(perftreatment_subset_columns['CompletionActivity'])]
    perftreatment_subset2 = perftreatment_subset_columns[perftreatment_subset_columns['CompletionActivity'] !='Yes']
    perftreatment_subset1 = perftreatment_subset_columns[perftreatment_subset_columns['CompletionActivity'] =='Yes']
    print('number of wells in perftreatment data ', perftreatment_subset_columns['EPAssetsId'].nunique())
    print('number of wells in perftreatment data with Completion Activity as "Yes" ', perftreatment_subset1['EPAssetsId'].nunique())
    print('number of wells in perftreatment data with Completion Activity as "No" ', perftreatment_subset2['EPAssetsId'].nunique())
    print('number of wells in perftreatment data with Completion Activity as "NaN" ', perftreatment_subset_null_values['EPAssetsId'].nunique())
    # calculate number of wells with Completion Activity with only No or NaN
    if perftreatment_subset_columns['EPAssetsId'].nunique() > perftreatment_subset1['EPAssetsId'].nunique():
        Potential_DUC_wells = perftreatment_subset_columns['EPAssetsId'].nunique() - perftreatment_subset1['EPAssetsId'].nunique()
        print('Potential number of DUC wells using only the CompletionActivity criteria from perftreatment table = ', Potential_DUC_wells)
    else: 
        print('No wells identified as potential DUC wells using only CompletionActivity criteria from perftreatment table')
    # check to see if there are any wells with no production that have also not been completed
    DUCS_perf_final= DUCS_perf[~DUCS_perf['EPAssetsId'].isin(perftreatment_subset1['EPAssetsId'])]

    # ## Append DUC well lists to get complete final list
    Final_DUC_List = DUCS['EPAssetsId'].append(DUCS_perf_final['EPAssetsId'].drop_duplicates(), ignore_index=True)
    # In[24]:
    DUCS_dataframe = wellheader[wellheader['EPAssetsId'].isin(Final_DUC_List)]
    return DUCS_dataframe

