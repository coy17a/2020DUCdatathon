
import streamlit as st
from helper.duc import *


def main():
    st.sidebar.title('DUC or not?' )
    tab_options = ['Home','What is a Duc?','EDA','Datathon Identification','Prediction','About Us']
    tab =st.sidebar.radio('Pages',tab_options)
    if tab == "What is a Duc?":
        what_is_duc()
    elif tab == 'EDA':
        wp = load_production()
        wh = load_header()
        wt = load_treatment()

        display_eda(wp,wh,wt)

    elif tab == 'Datathon Identification':
        wp = load_production()
        wh = load_header()
        wt = load_treatment()
        duc_analysis(wh,wp,wt)
        

    elif tab == 'Prediction':
        predictions()
        
    elif tab == 'About Us':
        about_us()
        
    else: 
        home()
        
        


if __name__ == "__main__":
    main()


