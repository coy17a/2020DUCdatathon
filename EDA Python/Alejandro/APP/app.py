
import streamlit as st
from helper.duc import *


def main():
    @st.cache(allow_output_mutation=True)
    def load_app():
        #with st.spinner('...'):
        wp = load_production()
        wh = load_header()
        wt = load_treatment()
        wp_eda = load_production_eda()
        return wp,wh,wt,wp_eda
    wp,wh,wt,wp_eda = load_app()
    st.sidebar.title('DUC or not?' )
    tab_options = ['Home','What is a Duc?','EDA','Datathon Identification','Prediction','About Us']
    tab =st.sidebar.radio('Pages',tab_options)
    if tab == "What is a Duc?":
        what_is_duc()
    elif tab == 'EDA':
        display_eda(wp,wh,wt,wp_eda)

    elif tab == 'Datathon Identification':
        duc_analysis(wh,wp,wt)


    elif tab == 'Prediction':
        predictions(wp_eda)
        
    elif tab == 'About Us':
        about_us()
        
    else: 
        home()
        
if __name__ == "__main__":
    main()


