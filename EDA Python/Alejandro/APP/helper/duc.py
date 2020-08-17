import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import altair as alt
from PIL import Image
import folium 
from altair.utils import sanitize_dataframe
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
plt.style.use('ggplot')


def eda_plot(df,x,y,color):
    sns.scatterplot(x=x,y=y,hue=color,data=df)
    st.pyplot()
    return ""
def eda_plot2(df,x,y,color):
    df=sanitize_dataframe(df)
    plot= alt.Chart(df).mark_bar(size=5).encode(
        x=x,
        y= y,
        color=color
    )
    return plot

@st.cache(persist=True)
def load_production():
     wp = pd.read_csv("data/WellProduction.csv",parse_dates=['ProdPeriod'],index_col=False)
     #wp = pd.read_csv("data/WellProduction.csv",parse_dates=['ProdPeriod'],index_col=False)
     #wp = wp.drop(columns=['WellHeader.Match'])
     return wp
     
@st.cache(persist=True)
def load_header():
    datecolumns = ['LicenceDate', 'ConfidentialReleaseDate','AbandonDate', 'SurfAbandonDate', 'SpudDate', 'FinalDrillDate', 'RigReleaseDate','StatusDate','CompletionDate']  
    wh = pd.read_csv("data/WellHeader_Datathon.csv",index_col=False,parse_dates=datecolumns)
    #wh = clean_pipeline(wh)
    return wh
@st.cache(allow_output_mutation=True,persist=True)
def load_treatment():
     wt = pd.read_csv("data/PerfTreatments.csv",index_col=False,parse_dates=['ActivityDate'])
     #clenaing of wt dataset
     return wt

@st.cache(persist=True)
def load_production_eda():
     wp = pd.read_csv("data/productioneda.csv",index_col=False)
     #wp = pd.read_csv("data/WellProduction.csv",parse_dates=['ProdPeriod'],index_col=False)
     #wp = wp.drop(columns=['WellHeader.Match'])
     return wp

def display_eda(wp,wh,wt,wp_eda):
    
    st.header('Exploration Data Analysis')
        
    options =['Well Header','Production','Perforation Treatments']
    eda_tab=st.sidebar.radio('Select Dataset',options)
    
    if eda_tab == 'Well Header':
        desc = wh.describe()
        st.subheader('Well Header Data Numerical Stats')
        wh1=clean_pipeline(wh)
        st.write(desc)
        st.subheader('Geographical Location')    
        sample=st.slider('Number of well in the map',min_value=10,max_value=10000,value=100,step=1)
        wh_sample=wh.sample(sample)
        map_well = map_wells(wh_sample)
        folium_static(map_well)   
        st.subheader('General Information')
        cols = list(wh1.columns)
        x = st.selectbox('X Variable',cols,index=14)
        y = st.selectbox('Y Variable',cols,index=13)
        color = st.selectbox('Color Variable',cols,index=6)
        st.write(eda_plot(wh1,x,y,color))
       

    if eda_tab == 'Production':
        
        st.subheader('Total Production')
        options=['Formation','Province','WellType','Pool','WellProfile','PSACAreaName']
        x_value=st.selectbox('X value',options,index=0,key='TotalP')
        sns.barplot(x=x_value,y='Total Production',data=wp_eda, hue='ProdTypeT')
        st.pyplot()
        st.subheader('Average Production')
        x_value2=st.selectbox('X value',options,index=0,key='AverageP')
        sns.barplot(x=x_value2,y='Average Production',data=wp_eda,hue='ProdTypeA')
        st.pyplot()
        st.subheader('Single Well production by product type')
        well_list=st.checkbox('Show well list',False)
        if well_list:
            options = wp['EPAssetsId'].head(5).unique()
            well_id = st.selectbox('Please Select wells',options)
            well_plot = single_Well_plot1(well_id,wp)
            st.write(well_plot)
        well_id=st.text_input('Well ID','-')
        if well_id != '-':
            well_id=int(well_id)
            well_plot = single_Well_plot1(well_id,wp)
            st.write(well_plot)
    if eda_tab == 'Perforation Treatments':
         activities,hist1,perf_plot=perf_analysys(wt)
         st.subheader('Unique type of Activities')
         st.write(activities.astype('object'))
         st.subheader('Number of activities per Well')
         step = st.slider('Bins',min_value=10,max_value=100,value=30)
         hist_perf(hist1,step)
         st.subheader('Most Common Activities')
         total=perf_plot['Number of Acitvities'].sum()
         st.write(f'Total Activities: {total}')
         st.write(perf_plot.astype('object'))
         st.subheader('Most Common Activities Plot')
         activities=st.slider('Number Of top Activities',min_value=1,max_value=25,value=10)
         top_activiteis(perf_plot,activities)

def predictions(wp_eda):
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.subheader(' Predict DUC wells from new datasets')
    uploaded_file = st.file_uploader("Upload Well Header CSV file", type="csv")
    if uploaded_file is not None:
        datecolumns = ['LicenceDate', 'ConfidentialReleaseDate','AbandonDate', 'SurfAbandonDate', 'SpudDate', 'FinalDrillDate', 'RigReleaseDate','StatusDate','CompletionDate'] 
        wh = pd.read_csv(uploaded_file,parse_dates=datecolumns,index_col=False)
        #st.write(data.head())
    
    uploaded_file2 = st.file_uploader("Upload Production CSV file", type="csv")
    if uploaded_file2 is not None:
        wp = pd.read_csv(uploaded_file2,parse_dates=['ProdPeriod'],index_col=False)
        #st.write(production.head())

    uploaded_file3= st.file_uploader("Upload PerfTreatment CSV file", type="csv")
    if uploaded_file3 is not None:
        wt= pd.read_csv(uploaded_file3,index_col=False,parse_dates=['ActivityDate'])
        #st.write(perf.head())
    st.subheader('Results')
    if (uploaded_file != None) & (uploaded_file2 != None) & (uploaded_file3 != None):
        selection=st.radio('Analysys',['EDA','DUCs'],index=1)
        if selection=='EDA':
            display_eda(wp,wh,wt,wp_eda)
        else:
            duc_analysis(wh,wp,wt)
        
def home():
    st.title('DUC or Not!! ')
    st.markdown('This web application was designed as part of the 2020 DUC Datathon submission of the team Data Conquerors and shows the work performed in exploring and analyzing a dataset conformed of 10438 wells to determine possible drilled but uncompleted wells. With this information oil services companies can identify potential clients and/or forecast market size; For that, a prediction option is available in the main menu whereby simply dropping any data sets alike, real-time segregation can be done on the fly.')
    image = Image.open('Img/duc.png')
    
    st.image(image,use_column_width=True)
    st.subheader('Untapped Energy DUC Datathon 2020: Data Conquerors')
    home = Image.open('Img/home.png')
    st.image(home,use_column_width=True)
    st.header('www.ducornot.herokuapp.com')
        

def hist_perf(hist,step):
     plot = sns.distplot(hist['PerfShots'], kde=False, rug=False,bins=step);
     plot.set(xlabel="Number Treatments", ylabel = "Number of Wells")
     st.pyplot()

def single_Well_plot1(well_id,df1):
    df_well = df1[(df1['EPAssetsId']==well_id) & (df1['ProdType'] != 'Production Hours')]
    fig, ax = plt.subplots(figsize = (12,6))    
    fig = sns.barplot(x='ProdPeriod',y='Volume',hue='ProdType',data=df_well, ax=ax)
    x_dates = df_well['ProdPeriod'].dt.strftime('%Y-%m-%d')
    ax.set_xticklabels(labels=x_dates, rotation=45, ha='right')
    st.pyplot() 
    return ""  

def map_wells(wh):
    latitude = 54.73
    longitude = -117.52704
    wells_map = folium.Map(location=[latitude, longitude], zoom_start=5)
    mc = MarkerCluster()
    # add markers to map
    for row in wh.itertuples():
        mc.add_child(folium.CircleMarker(
        #label = 'UWI: {}<br> Formation: {}<br> Current Operator: {}'.format(row.UWI,row.Formation,row.CurrentOperator)
        #label = folium.Popup(label,parse_html=False)
        location=[row.Surf_Latitude, row.Surf_Longitude],
        radius=3, 
        popup ='UWI: {uwi}<br>Formation: {formation}<br>Current Operatonr:{currentop}'.format(uwi=row.UWI,formation=row.Formation,currentop=row.CurrentOperator)
        #popup=label, 
        #color='blue',
        #fill=True,
        #parse_html=False,
        
        #fill_opacity=0.7,
         ))
    wells_map.add_child(mc)
    #wells_map.save('wells.html’)
    return(wells_map) 


def what_is_duc():
    st.markdown('''
## Introduction

The DUC (Drilled but Uncompleted) wells play an important role in balancing the economy of oil and gas production by providing smoothing effect on the fluctuating difference between supply and demand in the market. 

The subject of 2020 Datathon competition is identifying DUC wells.

## DUC Well characteristics

DUC wells are drilled but not completed! More specifically, these are wells that have reached the target depth and casings are installed to protect the well and the formation. The wells however are not completed which means the production casing is not installed and no other completion activities are performed to bring the well in production. This might suggest the criteria for identifying the wells, as follows:

-  Drilling is complete and well has reached the target total depth
-  Well is not completed as in:
    - Production casing is not installed
    - No perforation or fracking has been performed
    - No stimulation / acidization activity has been performed
- The well has no production. Any production record associated with a well disqualifies it as a DUC Well.     
    

## Roadmap and strategy for DUC well modeling

- DUCs will be identified using the criteria in above. The criteria will be coded to create a binary identifier to tag well that are DUC with 1 and the rest as 0. The process has two step filter application to the "wellHeader" subeset as the master list. 
    - First, all the wells from the "wellPerf" which meet the criteria for being through completion activities are removed from the master list resulting in an interim list.
    - Second, the wells from "wellProduction" which are showing production records (any product type) will be removed from the interim list, which results in a table listing Wells that are potentially qualified as DUC.


## EDA

Well header file will be the master list providing all the wells. before filtering for DUC wells, the subsets are manipulated for more suitable format and structure for exploratory data analysis (EDA) and subsequent modeling. 

1. Duplicate records exist in "wellProduction" table. Group by for ['EPAssetsId', 'ProdPeriod', 'ProductType'] with 'Volume' aggregrated performed to get rid of duplicate values
2. Table is transformed into wide format using production types
3. Table is summarised to create new columns for: 
    a. min and max of production periods, 
    b. count of production period.
4.  Table also aggregrated to create total production for each ['EPAssetsId', 'ProductType']
    
## DUC Wells Identifications Strategy
''' )
    wf = Image.open('Img/workflow.png')
    st.image(wf)

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
    # filtering wells not in wt nor in well_production from subset_wellheader
    ducs = subset_wellheader[~subset_wellheader['EPAssetsId'].isin(perftreatment['EPAssetsId'])]
    ducs = ducs[~ducs['EPAssetsId'].isin(wellproduction['EPAssetsId'])]
    m2=ducs.shape[0]
    #print ('number of wells not in wt data nor in well_production data ', )
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
        alt.Y('Binned',sort='x'),
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
def about_us():
    st.header('The Team')
    st.markdown(''' 
### Ijeoma Odoko:''') 
    ij = Image.open('Img/ijeoma.jpeg')
    st.image(ij,width=200)
    st.markdown('''
    A Business Intelligence Analyst with approximately 8 years of experience as a Project Engineer within the Canadian Oil and Gas Industry delivering pipeline and facility projects across Alberta and British Columbia. I enjoy performing data cleaning, exploratory data analysis, and preparing data visualizations to provide actionable insights.    
    **Contributions:** *Exploratory Data Analysis, DUC well Identification Code, DUC Identification Strategy and Visualizations*  
    [Linkedin](https://www.linkedin.com/in/ijeoma-odoko-peng-meng-3b7b2430/)
### Alejandro Coy:''')
    image = Image.open('Img/me.jpeg')
    st.image(image,width=200)
    st.markdown('''
    I'am Chemical Engineer, Data Geek and Amateur Triathlete. I’ve been working on research, design, and implementation of new technologies for the heavy oil industry for the past 10 years.
    Since I worked on my bachelor’s thesis (where I analyzed thousands of flow measurements for gas natural mixtures), I’ve been passionate about data and the powerful insights obtained from experiments where the information is collected and process correctly.  
    **Contributions:** *Streamlit App Code, Exploratory Data Analysis, Visualizations and TVD Kaggle Competition Strategy and Code*  
    [Linkedin](https://www.linkedin.com/in/luisalejandrocoy/)  
    [AlejandroCoy.ca](https://www.alejandrocoy.ca)
### Gabriel Garcia:''')
    gb = Image.open('Img/Gabriel.png')
    st.image(gb,width=200)
    st.markdown('''
    A geoscience engineer with 11 years’ experience in the oil and gas industry as a data analyst, performing processing and interpretation of well logs reservoir characterization for basins in Canada, USA and Mexico; I can also hold my breath for almost 2 minutes. 
    **Contributions:** *DUC Identification and TVD Competition Strategy,Documentation*  
    [Linkedin](https://www.linkedin.com/in/gabriel-garcia-rosas/)
### Mohammed Alaudding:''')
    mh = Image.open('Img/Muhammad.png')
    st.image(mh,width=200)
    st.markdown('''
    Md Alauddin is a PhD student at " the Centre for Risk, Integrity and Safety Engineering (C-RISE)" in the Department of Process Engineering, Memorial University of Newfoundland, Canada. His research interest includes abnormal situation management, fault detection and diagnosis, evolutionary computation, and data mining application in oil and gas systems. He is currently working on prediction and control of COVID-19 using stochastic modeling.  
    **Contributions:** *DUC Identification and TVD competition Strategy,Documentation, Visualizations*  
    [Linkedin](https://www.linkedin.com/in/mohammad-alauddin-002b6512/)
### Korang Modaressi:''')
    km = Image.open('Img/korang.png')
    st.image(km,width=200)
    st.markdown('''
Korang has been working in industrial projects for most of his career. He is a Professional
Engineer (P.Eng.) with master’s degree in Mechanical Engineering. Through his career he
has been working with disciplines and stakeholders in projects to gather meaningful project
data that he used to provide information needed for making decision about the projects.
With years of experience in projects he believes in the power of data efficacy and the
important role the analytical skills play in information-based decision making. He is a perpetual learner
and enjoys analyzing data and presenting the results with the power of visualization.  
 **Contributions:** *DUC Identification and TVD competition Strategy,Documentation*  
 [Linkedin](https://www.linkedin.com/in/korang-modaressi/)''')

def duc_analysis(wh,wp,wt):
        st.header('Datathon DUC Well Identification')
        st.sidebar.subheader('DUC Identification Menu')
        ducs,m1,m2,m3,m4,wellheader,wellproduction,perftreatment = duc_wells(wh,wp,wt)
        options = ['General Information','Time Analysis','DUC Duration','More About DUCs']
        nav = st.sidebar.radio('Nav',options)
        ducs_final = duc_time(ducs,wellheader, wellproduction)
        
        if nav == 'General Information':
            ddf= wh[wh['EPAssetsId'].isin(ducs)]
            ddf= clean_pipeline(ddf)
            st.write(f'Number of wells with current Status not in ["Pumping", "Flowing", "Gas Lift"] : {m1}')
            st.write(f'Number of wells not in perf_data nor in well_production data: {m2}') 
            st.write(f'Potential number of DUC wells using only the CompletionActivity criteria from perftreatment table: {m3}')
            st.write(f'Wells with no production that have also not been completed: {m4}')
            st.success(f'Total DUC wells in dataset:{len(ducs)}')
            desc = ddf.describe().T
            #st.write(ddf.astype('object'))
            st.write(desc)
            st.subheader('Geographical Location')
            folium_static(map_wells(ddf))
            st.subheader('DUCs Exploration')
            cols = list(ddf.columns)
            x = st.selectbox('X Variable',cols,index=2)
            y = st.selectbox('Y Variable',cols,index=3)
            color = st.selectbox('Color Variable',cols,index=7)
            ddf=sanitize_dataframe(ddf)
            st.altair_chart(alt.Chart(ddf).mark_bar(size=8).encode(
            x=x,
            y= y,
            color=color).properties(width=700,height=300))
            st.subheader('Single Well Information')
            uwi_list=st.checkbox('Show UWI lis',False)
            if uwi_list:
                st.write(ddf['UWI'])
            well_uwi=st.text_input('Well UWI','-')
            if well_uwi != '-':
                single_well= info_single_well(well_uwi,wh)
                st.write(single_well.astype('object').iloc[0])
        if nav == 'Time Analysis':
            ducs_final = duc_time(ducs,wellheader, wellproduction)
            desc = ducs_final.describe()
            st.subheader('Days of Uncompleted Status')
            step= st.slider('Bins',min_value=5,max_value=100,value=30)
            hist_days_uncomplete(ducs_final,step=step)
            st.subheader('Days of Uncompleted Status grouped by Period')
            ducs_binned(ducs_final)
        if nav =='DUC Duration':
            non_ducs_df=non_duc_wells_duration(wellheader,wellproduction,perftreatment,ducs)
            st.subheader('Time of Uncompleted Status For all Non-DUC wells')
            step= st.slider('Bins',min_value=5,max_value=100,value=30)
            hist_days_uncomplete2(non_ducs_df,step)
            st.subheader('Time of Uncompleted Status in the data set by Formation ')
            non_ducs_bins(non_ducs_df)
            facet = st.selectbox('Facet By:',['Formation','PSACAreaName','Field'],index=0)
            non_ducs_per_formation(non_ducs_df,25,facet)
        if nav == 'More About DUCs':
            #st.write(ducs_final.columns)
            facet = st.selectbox('Facet By:',['Formation','PSACAreaName','Field','CurrentOperator'],index=3)
            non_ducs_per_formation(ducs_final,25,facet)
            
def perf_analysys(wt):
    wt['ActivityType'] = wt['ActivityType'].astype('category')
    wt=wt.iloc[:,0:8]
    activities=wt['ActivityType'].unique()
    activities=pd.DataFrame(activities)
    hist1=wt.groupby('EPAssetsId').count().sort_values(by='PerfShots',ascending=False)
    wt1=wt.loc[ :,['EPAssetsId', 'ActivityDate', 'ActivityType', 'PerfShots'] ]
    perf_plot=wt1.groupby(['ActivityType']).count().sort_values('ActivityDate',ascending=False).reset_index()[['ActivityType','EPAssetsId']]
    perf_plot.columns=['ActivityType','Number of Acitvities']
   
    return activities,hist1,perf_plot

def top_activiteis(perf_plot,activities):
    top=perf_plot.iloc[0:activities]
    sns.barplot(x='Number of Acitvities',y='ActivityType', data=top, order=top['ActivityType'])
    sns.despine(left=True, bottom=True)
    st.pyplot()
