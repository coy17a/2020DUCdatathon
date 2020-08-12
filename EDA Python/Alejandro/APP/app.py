from altair.vegalite.v4.api import value
from google.protobuf.symbol_database import Default
import pandas as pd
from pandas import wide_to_long
import streamlit as st
import altair as alt
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def main():
    st.sidebar.title('DUC or not?' )
    tab_options = ['Home','What is a Duc?','EDA','Datathon Identification','Prediction','About Us']
    tab =st.sidebar.radio('Pages',tab_options)
    if tab == "What is a Duc?":
        st.write('EIA Paper, maybe markdown from github....')
        st.markdown('''
## Introduction

The DUC (Drilled but Uncompleted) wells play an important role in balancing the economy of oil and gas production by providing smoothing effect on the fluctuating difference between supply and demand in the market. 

The subject of 2020 Datathon competition is identifying DUC wells.

## DUC Well characteristics

DUC wells are drilled but not completed! More specifically, these are wells that have reached the target depth and casings are installed to protect the well and the formation. The wells however are not completed which means the production casing is not installed and no other completion activities are performed to bring the well in production. This might suggest the criteria for identifying the wells, as follows:

1. Drilling is complete and well has reached the target total depth
2. Well is not completed as in:
    1. Production casing is not installed
    2. No perforation or fracking has been performed
    3. No stimulation / acidization activity has been performed
3. The well has no production beyond a defined threshold.     
    
There are wells that produce in spite of not being completed. these wells will be excluded from the DUC category if the cumulative production volume or time is beyond a certain threshold, which needs to be defined as part of the criteria.

## Roadmap and strategy for DUC well modeling

Methods and algorithems need to be selected to associate the data features with the DUC well types. Number of method might be considered before the appropriate model is fitted. those techniques are discussed as follows.

### Supervised Learning

#### Decision Tree

Decision-Tree based classification methods might be able to classify the data into DUC and non-DUC wells using the criteria defined.

**_Under Development_**''' )
    elif tab == 'EDA':
         with st.spinner('Loading..'):
            #wh = pd.read_csv('../../../data/WellHeader_Datathon.csv')
            wp = pd.read_csv("data/WellProduction.csv",parse_dates=['ProdPeriod'],index_col=False)
            wp = wp.drop(columns=['WellHeader.Match'])
            st.subheader('Exmple: Well produciton by product type')
            well_id = wp['EPAssetsId'].head(20).unique()
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
### Ijeoma: 
### Gabriel:
### Korang
### Mohammed:
### Alejandro Coy:''')
        image = Image.open('me.jpeg')
        st.image(image,width=200)
        st.markdown('''
I’ve been working on research, design, and implementation of new technologies for the heavy oil industry for the past 10 years.

Since I worked on my bachelor’s thesis (where I analyzed thousands of flow measurements for gas natural mixtures), I’ve been passionate about data and the powerful insights obtained from experiments where the information is collected and process correctly.
        ''')
    else: 
        st.title('DUC or Not')
        st.write('Introduciton to the App ...................')
        image = Image.open('duc.png')
       
        st.image(image,use_column_width=True)
        st.subheader('Untapped Energy DUC Datathon 2020: Data Conquerors')
        home = Image.open('home.png')
        st.image(home,use_column_width=True)
        
    

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


