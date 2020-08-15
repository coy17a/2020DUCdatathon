import pandas as pd

def clean_pipeline2(df):
    ## Select objects to convert to category type 
    cat =list(df.dtypes[df.dtypes == 'object'].index)
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
    Final_DUC_List = ducs['EPAssetsId'].append(ducs_perf_final['EPAssetsId'].drop_duplicates(), ignore_index=True)
    #ducs_dataframe = wellheader[wellheader['EPAssetsId'].isin(Final_DUC_List)]
    return Final_DUC_List,m1,m2,m3,m4

def duc_time(ducs):
    pass