import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import altair as alt
from altair.utils import sanitize_dataframe
plt.style.use('ggplot')


def clean_pipeline2(df):
    ## Select objects to convert to category type 
    cat =list(df.dtypes[df.dtypes == 'object'].index)
    if len(cat)>0:
        df[cat] = df[cat].astype('category')
        surface_columns=['Surf_LSD','Surf_Section','Surf_Township','Surf_Range','BH_LSD','BH_Section','BH_Township','BH_Range']
        df[surface_columns]=df[surface_columns].astype(str)
    #df_s = df[column_selection]
    #df_s['TVD']= df['TVD']
    return df

def duc_wells(wellheader,wellproduction,perftreatment,):
    wellheader = clean_pipeline2(wellheader)
    #return wellheader.columns
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


    # check number of wells remaining 

    m1 = subset_wellheader['EPAssetsId'].nunique()
    # filtering wells not in perf_treatment nor in well_production from subset_wellheader
    ducs = subset_wellheader[~subset_wellheader['EPAssetsId'].isin(perftreatment['EPAssetsId'])]
    ducs = ducs[~ducs['EPAssetsId'].isin(wellproduction['EPAssetsId'])]
    m2=ducs.shape[0]
    #print ('number of wells not in perf_treatment data nor in well_production data ', )
    # filtering wells in perftreatment that have no production data
    ducs_perf = perftreatment[~perftreatment['EPAssetsId'].isin(wellproduction['EPAssetsId'])]
    #print('number of wells from perftreatment datatable that have no production data ', ducs_perf['EPAssetsId'].nunique())
    ## Create subsets of Perf Treatment to obtain wells which have Completion Activity as Yes, No or NaN
    perftreatment_subset_columns = perftreatment[['EPAssetsId', 'CompletionActivity', 'ActivityDate']]
    perftreatment_subset_null_values = perftreatment_subset_columns[pd.isnull(perftreatment_subset_columns['CompletionActivity'])]
    perftreatment_subset2 = perftreatment_subset_columns[perftreatment_subset_columns['CompletionActivity'] !='Yes']
    perftreatment_subset1 = perftreatment_subset_columns[perftreatment_subset_columns['CompletionActivity'] =='Yes']
    #print('number of wells in perftreatment data ', perftreatment_subset_columns['EPAssetsId'].nunique())
    #print('number of wells in perftreatment data with Completion Activity as "Yes" ', perftreatment_subset1['EPAssetsId'].nunique())
    #print('number of wells in perftreatment data with Completion Activity as "No" ', perftreatment_subset2['EPAssetsId'].nunique())
    #print('number of wells in perftreatment data with Completion Activity as "NaN" ', perftreatment_subset_null_values['EPAssetsId'].nunique())
    # calculate number of wells with Completion Activity with only No or NaN
    if perftreatment_subset_columns['EPAssetsId'].nunique() > perftreatment_subset1['EPAssetsId'].nunique():
        m3= perftreatment_subset_columns['EPAssetsId'].nunique() - perftreatment_subset1['EPAssetsId'].nunique()
        #print('Potential number of DUC wells using only the CompletionActivity criteria from perftreatment table = ', Potential_DUC_wells)
    else: 
        m3 =0
        #print('No wells identified as potential DUC wells using only CompletionActivity criteria from perftreatment table')
    # check to see if there are any wells with no production that have also not been completed
    ducs_perf_final= ducs_perf[~ducs_perf['EPAssetsId'].isin(perftreatment_subset1['EPAssetsId'])]
    m4 = ducs_perf_final.shape[0]
    # ## Append DUC well lists to get complete final list
    ducs = ducs['EPAssetsId'].append(ducs_perf_final['EPAssetsId'].drop_duplicates(), ignore_index=True)
    #
    return ducs,m1,m2,m3,m4,wellheader,wellproduction,perftreatment

def duc_time(ducs,wellheader,wellproduction):
    Latest_production_period = wellproduction['ProdPeriod'].max()
    ducs_df = wellheader[wellheader['EPAssetsId'].isin(ducs)]     
    #print ('The latest production period is ', Latest_production_period)
    #Merge Final DUC list with wellheader data to get full wellheader information
    ducs_df = wellheader[wellheader['EPAssetsId'].isin(ducs)]
    # create new column for how long well has been a DUC
    ducs_final_df = ducs_df.copy()
    ducs_final_df.loc[:,'Days of Uncompleted Status'] = Latest_production_period - ducs_final_df['RigReleaseDate']
    # transform days to numeric datatype
    ducs_final_df.loc[:,'Days of Uncompleted Status'] = ducs_final_df['Days of Uncompleted Status']/np.timedelta64(1, 'D')
    #check statistics of 'Days of Uncompleted Status'
    #ducs_final_df ['Days of Uncompleted Status'].describe()
    #create histogram of 'Days of Uncompleted Status'
    return ducs_final_df

def hist_days_uncomplete(ducs_final_df,step):
    ducs_final_df=sanitize_dataframe(ducs_final_df)
    chart = alt.Chart(ducs_final_df).mark_bar().encode(
        alt.X('Days of Uncompleted Status',bin=alt.Bin(extent=[0, 500], step=step)),
        alt.Y('count()',title='Number Wells'),
        color=alt.value('green'),
        opacity=alt.value(0.7)
    ).properties(width=700).interactive()
    return st.altair_chart(chart)
    # ducs_final_df['Days of Uncompleted Status'].hist()
    # plt.xlabel('Days of Uncompleted Status')
    # plt.title('Number of DUC wells')
    # st.pyplot()

def hist_days_uncomplete2(ducs_final_df,step):
     plot = sns.distplot(ducs_final_df['Days of Uncompleted Status'], kde=False, rug=False,bins=step);
     plot.set(xlabel="Days", ylabel = "Number of Wells")
     st.pyplot()
    

    # create bins for range of days of uncompleted status
def ducs_binned(ducs_final_df):
    bins = [0, 60, 90, 365, 730, 1095]
    labels = ['<60days', '60-90days', '90-365days', '365-730days', '730-1095days']
    ducs_final_df['Binned'] = pd.cut(ducs_final_df['Days of Uncompleted Status'], bins=bins, labels=labels)
    ducs_final_df.head(5)
    # groupby number of wells by bin label
    ducs_df_binned = ducs_final_df.groupby(by='Binned', as_index=False).agg({'EPAssetsId':'size'})
    # create bar graph
    ducs_df_binned=sanitize_dataframe(ducs_df_binned)
    #ducs_df_binned.plot.barh(x='Binned', y='EPAssetsId')
    chart = alt.Chart(ducs_df_binned).mark_bar().encode(
        alt.Y('Binned',sort='-x'),
        alt.X('EPAssetsId',title='Number Wells')
    ).properties(width=700).interactive()
    #plt.xlabel('Number of DUC Wells')
    #plt.ylabel('Days of Uncompleted Status - Range')
    #st.pyplot()
    return st.altair_chart(chart)
   # """## DUC status duration for non-DUC wells"""

    ## non DUC wells dataframe
def non_duc_wells_duration(wellheader,wellproduction,perftreatment,ducs):
    non_ducs_df = wellheader[~wellheader['EPAssetsId'].isin(ducs)]
    #non_ducs_df.info()
    ## groupby production data to remove duplicates, and calculate first production period per well
    first_production_by_well = wellproduction.groupby(by='EPAssetsId', as_index=False, observed=True).agg({'ProdPeriod': 'min'})
    #first_production_by_well.head(5)
    # groupby perforation data by EPAssetsId and Last Activity Date
    last_Activity_Date_by_well = perftreatment.groupby(by='EPAssetsId', as_index=False).agg({'ActivityDate': 'max'})
    ## merge non DUC wells dataframe['EPAssetsId', 'RigReleaseDate'] with production data and perftreatment data
    non_ducs_df = non_ducs_df.merge(first_production_by_well, how='left', on='EPAssetsId')
    non_ducs_df = non_ducs_df.merge(last_Activity_Date_by_well, how='left', on='EPAssetsId')
    #non_ducs_df.info()
    non_ducs_df.loc[:,'DUCStatusEndDate'] = non_ducs_df['ProdPeriod'].fillna(value=non_ducs_df['ActivityDate'])
    #non_ducs_df['DUCStatusEndDate'].isnull().sum()
    ## drop off rows that don't have Perftreatment or Production data
    non_ducs_df.dropna(subset=['DUCStatusEndDate'], inplace = True)
    #non_ducs_df.shape
    #non_ducs_df.head(2)
    #dataframe_check = non_ducs_df[non_ducs_df['DUCStatusEndDate'] <= non_ducs_df['RigReleaseDate']]
    #print(dataframe_check)
    # create new column for DUC status duration
    non_ducs_final_df = non_ducs_df.copy()
    non_ducs_final_df.loc[:,'Days of Uncompleted Status'] = non_ducs_df['DUCStatusEndDate'] - non_ducs_final_df['RigReleaseDate']
    # transform days to numeric datatype
    non_ducs_final_df.loc[:,'Days of Uncompleted Status'] = non_ducs_final_df['Days of Uncompleted Status']/np.timedelta64(1, 'D')
    # create histogram for DUC status duration for non_DUC_wells
    return non_ducs_final_df

def info_single_well(well_uwi,wh):
    df= wh[wh['UWI']==well_uwi]
    df = clean_pipeline(df)
    df=df
    return df

def clean_pipeline(df):
    ## Select objects to convert to category type 
    cat =list(df.dtypes[df.dtypes == 'object'].index)
    if len(cat)>0:
        df[cat] = df[cat].astype('category')
        surface_columns=['Surf_LSD','Surf_Section','Surf_Township','Surf_Range','BH_LSD','BH_Section','BH_Township','BH_Range']
        df[surface_columns]=df[surface_columns].astype(str)
        df[surface_columns]=df[surface_columns].astype('category')
    feature_selection = ['EPAssetsId', 'Province','UWI', 'CurrentOperator',
       'CurrentStatus', 'WellType',
        'Formation', 'Field', 'Pool',
        'Surf_Location', 'Surf_Longitude',
       'Surf_Latitude','GroundElevation', 'KBElevation', 'TotalDepth',
       'SurfaceOwner', 'DrillingContractor', 'FinalDrillDate',
       'RigReleaseDate', 'DaysDrilling', 'DrillMetresPerDay',
       'WellProfile', 'PSACAreaCode', 'PSACAreaName',
       'ProjectedDepth']
    df = df[feature_selection]
    return df
def non_ducs_per_formation(df,step,facet):
    
    df=sanitize_dataframe(df)
    chart = alt.Chart(df).mark_bar().encode(
        alt.Y('Days of Uncompleted Status',bin=alt.Bin(extent=[0, 500], step=step)),
        alt.X('count()',title='Number Wells'),
        facet=alt.Facet(facet, columns=3),
        color=alt.value('#e68805'),
        opacity=alt.value(0.7)
    ).properties(width=200).interactive()
    return st.altair_chart(chart)

def non_ducs_bins(non_ducs_final_df):
    # create bins for range of days of uncompleted status
    bins = [0, 60, 90, 365, 730, 1095]
    labels = ['<60days', '60-90days', '90-365days', '365-730days', '730-1095days']
    non_ducs_final_df['Binned'] = pd.cut(non_ducs_final_df['Days of Uncompleted Status'], bins=bins, labels=labels)
    # group by number of wells by bin label
    non_ducs_df_binned = non_ducs_final_df.groupby(pd.cut(non_ducs_final_df['Days of Uncompleted Status'], bins=bins)).size()
    # groupby number of wells by bin label
    non_ducs_df_binned = non_ducs_final_df.groupby(by='Binned', as_index=False).agg({'EPAssetsId':'size'})
    # create bar graph
    # non_ducs_df_binned.plot.barh(x='Binned', y='EPAssetsId')
    # plt.xlabel('Number of Wells')
    # plt.ylabel('Days of Uncompleted Status - Range')
    # plt.show()
    # # groupby number of wells by formation and bin label
    # groupby number of wells by formation and bin labels
    non_ducs_df_binned_formation = non_ducs_final_df.groupby(['Binned', 'Formation'], as_index=False, observed=True).agg({'EPAssetsId': 'size'})
    # Number of wells binned by Days of Uncompleted Status duration by Formation 
    g=sns.catplot(x='EPAssetsId', y='Binned', data=non_ducs_df_binned_formation, col='Formation',col_wrap=2, kind='bar', height=6, aspect = 0.8)
    g.set(xlabel='Number of Wells', ylabel= 'Days of Uncompleted Status - Range')
    st.pyplot()

#"""## Additional DUC wells Insights"""
def test():
    # Days of Uncompleted Status hist by Formation 
    g=sns.FacetGrid(ducs_final_df, col='Formation', height=5, aspect = 1)
    g.map(plt.hist, 'Days of Uncompleted Status', alpha=0.7)
    plt.show()

    # Average Days of Uncompleted Status and Number of DUC wells by Field

    DUCS_by_field = ducs_final_df.groupby(by=['Field', 'Province'], as_index=False, observed=True).agg({'Days of Uncompleted Status' : 'mean', 'EPAssetsId': 'size'})

    DUCS_by_field.head()

    # Average Days of Uncompleted Status by field
    DUCS_by_field.plot.barh(x='Field', y='Days of Uncompleted Status')
    plt.xlabel('Average Days of Uncompleted Status')
    plt.show()

    # Number of DUC wells by Field
    DUCS_by_field.plot.barh(x='Field', y='EPAssetsId')
    plt.xlabel('Number of DUC wells')
    plt.show()

    # Average Days of Uncompleted Status and Number of DUC wells by CurrentOperator

    DUCS_by_operator = ducs_final_df.groupby(by='CurrentOperator', as_index=False, observed=True).agg({'Days of Uncompleted Status' : 'mean', 'EPAssetsId': 'size'})

    DUCS_by_operator.head(8)

    # Average days of uncompleted status by Operator plot
    DUCS_by_operator.plot.barh(x='CurrentOperator', y='Days of Uncompleted Status')
    plt.xlabel('Average Days of Uncompleted Status')
    plt.show()

    # Number of wells by Operator plot
    DUCS_by_operator.plot.barh(x='CurrentOperator', y='EPAssetsId')
    plt.xlabel('Number of Wells')
    plt.show()

    # DUC wells by PSACAreaName
    DUCS_by_PSACAreaName = ducs_final_df.groupby(by='PSACAreaName', as_index=False, observed=True).agg({'Days of Uncompleted Status' : 'mean', 'EPAssetsId': 'size'})

    DUCS_by_PSACAreaName.plot.barh(x='PSACAreaName', y='Days of Uncompleted Status')
    plt.xlabel('Average Days of Uncompleted Status')
    plt.show()
    pass