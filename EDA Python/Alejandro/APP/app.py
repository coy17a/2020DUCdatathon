
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
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
            duc_analysis(wh,wp,wt)

    elif tab == 'About Us':
        about_us()
        
    else: 
        st.title('DUC or Not!! ')
        st.markdown('This web application was designed as part of the 202 DUC Datathon submission of the team Data Conquerors and shows the work performed in exploring and analyzing a dataset conformed of 10438 wells to determine possible drilled but uncompleted wells. With this information oil services companies can identify potential clients and/or forecast market size; For that, a prediction option is available in the main menu whereby simply dropping any data sets alike, real-time segregation can be done on the fly.')
        image = Image.open('Img/duc.png')
       
        st.image(image,use_column_width=True)
        st.subheader('Untapped Energy DUC Datathon 2020: Data Conquerors')
        home = Image.open('Img/home.png')
        st.image(home,use_column_width=True)
        


    

if __name__ == "__main__":
    main()


