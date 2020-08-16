from collections import defaultdict
from altair.vegalite.v4.api import value
from google.protobuf.symbol_database import Default
import pandas as pd
import streamlit as st
import altair as alt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium 
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
from PIL import Image
from altair.utils import sanitize_dataframe
from helper.duc import *


def main():
    st.sidebar.title('DUC or not?' )
    tab_options = ['Home','What is a Duc?','EDA','Datathon Identification','Prediction','About Us']
    tab =st.sidebar.radio('Pages',tab_options)
    if tab == "What is a Duc?":
        what_is_duc()
    elif tab == 'EDA':        
        dispaly_eda()

    elif tab == 'Datathon Identification':
        wp = load_production()
        wh = load_header()
        wt = load_treatment()
        duc_analysis(wh,wp,wt)
        

    elif tab == 'Prediction':
        st.set_option('deprecation.showfileUploaderEncoding', False)
        st.subheader(' Predict DUC wells from new datasets')
        uploaded_file = st.file_uploader("Upload Well Header CSV file", type="csv")
        if uploaded_file is not None:
            wh = pd.read_csv(uploaded_file)
            #st.write(data.head())
      
        uploaded_file2 = st.file_uploader("Upload Production CSV file", type="csv")
        if uploaded_file2 is not None:
            wp = pd.read_csv(uploaded_file2)
            #st.write(production.head())

        uploaded_file3= st.file_uploader("Upload PerfTreatment CSV file", type="csv")
        if uploaded_file3 is not None:
            wt= pd.read_csv(uploaded_file3)
            #st.write(perf.head())
        st.subheader('Results')
        if (uploaded_file != None) & (uploaded_file2 != None) & (uploaded_file3 != None):
            duc_analysis(wh,wp,wt)

    elif tab == 'About Us':
        about_us()
        
    else: 
        st.title('DUC or Not!! ')
        st.write('Introduciton to the App ...................')
        image = Image.open('Img/duc.png')
       
        st.image(image,use_column_width=True)
        st.subheader('Untapped Energy DUC Datathon 2020: Data Conquerors')
        home = Image.open('Img/home.png')
        st.image(home,use_column_width=True)
        

def dispaly_eda():
    with st.spinner('Loading..'):
        wp = load_production()
        wh = load_header()
        wt = load_treatment()
    options =['Well Header','Production','Perforation Treatments','Geo Info']
    eda_tab=st.radio('Select Dataset',options)
    
    if eda_tab == 'Well Header':
        desc = wh.describe()
        st.subheader('Well Header Data Numerical Stats')
        wh1=clean_pipeline(wh)
        st.write(desc)
        st.subheader('Geographical Location')       
       
        cols = list(wh1.columns)
        x = st.selectbox('X Variable',cols,index=14)
        y = st.selectbox('Y Variable',cols,index=13)
        color = st.selectbox('Color Variable',cols,index=6)
        #wh=sanitize_dataframe(wh)
        # st.altair_chart(alt.Chart(wh).mark_point().encode(
        #     x=x,
        #     y= y,
        #     color=color).properties(width=700))#width=700,height=300))
        st.write(eda_plot(wh1,x,y,color))
       
        
    if eda_tab == 'Production':
        st.subheader('Example: Well production by product type')
        well_id = wp['EPAssetsId'].head(20).unique()
        default_id = list(well_id[0:5])
        wells = st.multiselect('Please Select wells',well_id)
        well_plot = single_Well_plot1(wells,wp)
        st.write(well_plot)
    if eda_tab == 'Geo Info':
        sample=st.slider('Number of well in the map',min_value=10,max_value=10000,value=100,step=1)
        wh_sample=wh.sample(sample)
        map_well = map_wells(wh_sample)
        folium_static(map_well)


def duc_analysis(wh,wp,wt):
        st.header('Datathon DUC Well Identification')
        st.sidebar.subheader('DUC Identification Menu')
        ducs,m1,m2,m3,m4,wellheader,wellproduction,perftreatment = duc_wells(wh,wp,wt)
        options = ['General Information','Time Analysis','DUC Duration','More']
        nav = st.sidebar.radio('Nav',options)
        if nav == 'General Information':
            
            st.write(f'Number of wells with current Status not in ["Pumping", "Flowing", "Gas Lift"] : {m1}')
            st.write(f'Number of wells not in perf_treatment data nor in well_production data: {m2}') 
            st.write(f'Potential number of DUC wells using only the CompletionActivity criteria from perftreatment table: {m3}')
            st.write(f'Wells with no production that have also not been completed: {m4}')
            st.success(f'Total DUC wells in dataset:{len(ducs)}')
            ddf= wh[wh['EPAssetsId'].isin(ducs)]
            ddf= clean_pipeline(ddf)
            desc = ddf.describe().T
            #st.write(ddf.astype('object'))
            st.write(desc)
            st.subheader('Geographical Location')
            folium_static(map_wells(ddf))
            st.subheader('DUCs Exploration')
            cols = list(ddf.columns)
            x = st.selectbox('X Variable',cols,index=13)
            y = st.selectbox('Y Variable',cols,index=14)
            color = st.selectbox('Color Variable',cols,index=7)
            ddf=sanitize_dataframe(ddf)
            st.altair_chart(alt.Chart(ddf).mark_bar(size=8).encode(
            x=x,
            y= y,
            color=color).properties(width=700,height=300))
        if nav == 'Time Analysis':
            ducs_final = duc_time(ducs,wellheader, wellproduction)
            desc = ducs_final.describe()
            st.subheader('Days of Uncompleted Status')
            hist_days_uncomplete(ducs_final,step=50)
            st.subheader('Days of Uncompleted Status grouped by Period')
            ducs_binned(ducs_final)
        if nav =='DUC Duration':
            non_ducs_df=non_duc_wells_duration(wellheader,wellproduction,perftreatment,ducs)
            st.subheader('Time of Uncompleted Status in the data set')
            hist_days_uncomplete2(non_ducs_df)
            st.subheader('Time of Uncompleted Status in the data set by: ')
            facet = st.selectbox('Facet By:',['Formation','PSACAreaName','Field'],index=0)
            non_ducs_bins(non_ducs_df)
            non_ducs_per_formation(non_ducs_df,25,facet)
            



def eda_plot(df,x,y,color):
    sns.scatterplot(x=x,y=y,hue=color,data=df)
    st.pyplot()
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


def single_Well_plot1(well_id,df1):
    df_well = df1[(df1['EPAssetsId'].isin(well_id)) & (df1['ProdType'] != 'Production Hours')]
    chart= alt.Chart(df_well).mark_bar(size=5).encode(
        x=alt.X('ProdPeriod',timeUnit='yearmonthdate'),
        y='Volume',
        color='ProdType',
        facet=alt.Facet('EPAssetsId', columns=3)
        ).properties(title='Wells',
    width=180,
    height=150
    ).interactive()
    return chart

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

def about_us():
    st.header('The Team')
    st.markdown(''' 
### Ijeoma Odoko:''') 
    ij = Image.open('Img/ijeoma.jpeg')
    st.image(ij,width=200)
    st.markdown('''
    I am a Business Intelligence Analyst with approximately 8 years of experience as a Project Engineer within the Canadian Oil and Gas Industry delivering pipeline and facility projects across Alberta and British Columbia. I enjoy performing data cleaning, exploratory data analysis, and preparing data visualizations to provide actionable insights. 
### Gabriel:
### Korang
### Mohammed:
### Alejandro Coy:''')
    image = Image.open('Img/me.jpeg')
    st.image(image,width=200)
    st.markdown('''
I’ve been working on research, design, and implementation of new technologies for the heavy oil industry for the past 10 years.
Since I worked on my bachelor’s thesis (where I analyzed thousands of flow measurements for gas natural mixtures), I’ve been passionate about data and the powerful insights obtained from experiments where the information is collected and process correctly.
        ''')

def what_is_duc():
    st.markdown('''
# Introduction

The DUC (Drilled but Uncompleted) wells play an important role in balancing the economy of oil and gas production by providing smoothing effect on the fluctuating difference between supply and demand in the market. 

The subject of 2020 Datathon competition is identifying DUC wells.

# DUC Well characteristics

DUC wells are drilled but not completed! More specifically, these are wells that have reached the target depth and casings are installed to protect the well and the formation. The wells however are not completed which means the production casing is not installed and no other completion activities are performed to bring the well in production. This might suggest the criteria for identifying the wells, as follows:

-  Drilling is complete and well has reached the target total depth
-  Well is not completed as in:
    - Production casing is not installed
    - No perforation or fracking has been performed
    - No stimulation / acidization activity has been performed
- The well has no production. Any production record associated with a well disqualifies it as a DUC Well.     
    

# Roadmap and strategy for DUC well modeling

- DUCs will be identified using the criteria in above. The criteria will be coded to create a binary identifier to tag well that are DUC with 1 and the rest as 0. The process has two step filter application to the "wellHeader" subeset as the master list. 
    - First, all the wells from the "wellPerf" which meet the criteria for being through completion activities are removed from the master list resulting in an interim list.
    - Second, the wells from "wellProduction" which are showing production records (any product type) will be removed from the interim list, which results in a table listing Wells that are potentially qualified as DUC.


# EDA

Well header file will be the master list providing all the wells. before filtering for DUC wells, the subsets are manipulated for more suitable format and structure for exploratory data analysis (EDA) and subsequent modeling. 

    1. Duplicate records exist in "wellProduction" table. Group by for ['EPAssetsId', 'ProdPeriod', 'ProductType'] with 'Volume' aggregrated performed to get rid of duplicate values
    2. Table is transformed into wide format using production types
    3. Table is summarised to create new columns for: 
        a. min and max of production periods, 
        b. count of production period.
    4.  Table also aggregrated to create total production for each ['EPAssetsId', 'ProductType']
        
# DUC Wells Identifications
## Workflow file
  

''' )



if __name__ == "__main__":
    main()


