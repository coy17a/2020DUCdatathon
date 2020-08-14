from collections import defaultdict
from altair.vegalite.v4.api import value
from google.protobuf.symbol_database import Default
import pandas as pd
import streamlit as st
import altair as alt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

def main():
    st.sidebar.title('DUC or not?' )
    tab_options = ['Home','What is a Duc?','EDA','Datathon Identification','Prediction','About Us']
    tab =st.sidebar.radio('Pages',tab_options)
    if tab == "What is a Duc?":
        st.write('EIA Paper, maybe markdown from github....')
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
 
- Classification methods will be used under unsupervised learning to find patterns which may be associated with the DUC wells. 
 
- the supervised learning might be performed using the DUC identifier from earlier steps.

Methods and algorithms need to be selected to associate the data features with the DUC well types. Number of method might be considered before the appropriate model is fitted. those techniques are discussed as follows.

# EDA

Well header file will be the master list providing all the wells. before filtering for DUC wells, the subsets are manipulated for more suitable format and structure for exploratory data analysis (EDA) and subsequent modeling. 

    1. Duplicate records exist in "wellProduction" table. Those are removed first
    2. Table is transformed into wide format using production types
    3. Table is summarised to create new columns for: 
        a. min and max of production periods, 
        b. count of production period,  
        c. sum of production volume for each production type
        
Grouping on Asset Ids and production periods would be applied to the table summary. 

The resulting production table would be used for analyzing data series with relation to DUC wells. Should the time be ample for further analysis, the economy of DUC Wells and their impact on the overall production profile for the region might be researched.  

## Unsupervised Learning

**_Under Development_**

## Supervised Learning

### Decision Tree

Decision-Tree based classification methods might be considered to classify the data into DUC and non-DUC wells using the criteria defined. 

**_Under Development_**
''' )
    elif tab == 'EDA':
         with st.spinner('Loading..'):
            wp = load_produciton()
            wh = load_header()
            wt = load_treatment()
         options =['Well Header','Production','Perforation Treatments']
         eda_tab=st.radio('Select Dataset',options)
         if eda_tab == 'Well Header':
            desc = wh.describe()
            st.subheader('Well Header Data Numerical Stats')
            st.write(desc)
            cols = list(wh.columns)
            x = st.selectbox('X Variable',cols,index=0)
            y = st.selectbox('Y Variable',cols,index=9)
            color = st.selectbox('Color Variable',cols,index=5)
            plot1 = eda_plot(wh,x,y,color)
            st.write(plot1)
         elif eda_tab == 'Production':
            st.subheader('Example: Well produciton by product type')
            well_id = wp['EPAssetsId'].head(20).unique()
            default_id = list(well_id[0:5])
            wells = st.multiselect('Please Select wells',well_id)
            well_plot = single_Well_plot1(wells,wp)
            st.write(well_plot)
      
       
    elif tab == 'Datathon Identification':
        st.write('Result from Datathon Analysis')

    elif tab == 'Prediction':
        st.subheader(' Predict DUC wells from new datasets')
        uploaded_file = st.file_uploader("Upload Well Header CSV file", type="csv")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            st.write(data.head())
      
        uploaded_file2 = st.file_uploader("Upload Production CSV file", type="csv")
        if uploaded_file2 is not None:
            production = pd.read_csv(uploaded_file2)
            st.write(production.head())

        uploaded_file3= st.file_uploader("Upload PerfTreatment CSV file", type="csv")
        if uploaded_file3 is not None:
            perf = pd.read_csv(uploaded_file3)
            st.write(perf.head())
        st.subheader('Results, same analysis from previous Tab')


    elif tab == 'About Us':
        st.header('The Team')
        st.markdown(''' 
### Ijeoma:''') 
        ij = Image.open('Img/ijeoma.jpeg')
        st.image(ij,width=200)
        st.markdown('''
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
    else: 
        st.title('DUC or Not!! ')
        st.write('Introduciton to the App ...................')
        image = Image.open('Img/duc.png')
       
        st.image(image,use_column_width=True)
        st.subheader('Untapped Energy DUC Datathon 2020: Data Conquerors')
        home = Image.open('Img/home.png')
        st.image(home,use_column_width=True)
        

def eda_plot(df,x,y,color):
    sns.scatterplot(x=x,y=y,hue=color,data=df)
    return st.pyplot()
    
@st.cache     
def load_produciton():
     wp = pd.read_csv("data/WellProduction.csv",parse_dates=['ProdPeriod'],index_col=False)
     wp = pd.read_csv("data/WellProduction.csv",parse_dates=['ProdPeriod'],index_col=False)
     wp = wp.drop(columns=['WellHeader.Match'])
     return wp
@st.cache 
def load_header():
     wh = pd.read_csv("data/WellHeader_Datathon.csv",index_col=False)
     wh = clean_pipeline(wh,)
     return wh
@st.cache 
def load_treatment():
     wt = pd.read_csv("data/PerfTreatments.csv",index_col=False)
     #clenaing of wt dataset
     return wt

def clean_pipeline(df):
    ## Select objects to convert to category type 
    cat =list(df.dtypes[df.dtypes == 'object'].index)
    df[cat] = df[cat].astype('category')
    surface_columns=['Surf_LSD','Surf_Section','Surf_Township','Surf_Range','BH_LSD','BH_Section','BH_Township','BH_Range']
    df[surface_columns]=df[surface_columns].astype(str)
    df[surface_columns]=df[surface_columns].astype('category')
    feature_selection = ['TVD','EPAssetsId','GroundElevation',
                    'KBElevation','Formation','WellType','WellProfile','ProjectedDepth','DaysDrilling','TotalDepth']
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

if __name__ == "__main__":
    main()


