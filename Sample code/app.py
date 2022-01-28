# This file and its imported files are based on code from https://github.com/baligoyem/dataqtor.git

import streamlit as st
from streamlit import caching
import SessionState
import pandas as pd
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta
from string_grouper import match_strings
import re
from downloader import get_table_download_link
from utils import download_button

# Page configuation
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(page_title="Machine Learning for Beginners", layout="wide")
st.title("Machine Learning for Beginners")
st.image('Machine Learning for Beginners-logos.jpeg', width=300)

hide_streamlit_style = """
            <style>
            footer {
	        visibility: hidden;
	            }
            footer:after {
	            content:'developed by Porshea E and Amanda A'; 
	            visibility: visible;
	            display: block;
	            position: relative;
	            #background-color: red;
	            padding: 5px;
	            top: 2px;
                    }
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


@st.cache(allow_output_mutation=True, persist = True)
def beforeSTable():
    global before
    before = pd.DataFrame(columns=["Column", "Null Records", "Out of Format Records", "Proper Format Records", "Column DQ Score(%)"])
    return before


@st.cache(allow_output_mutation=True, persist = True)
def afterSTable():
    global after
    after = pd.DataFrame(columns=["Column", "Null Records", "Out of Format Records", "Proper Format Records", "Column DQ Score(%)"])
    return after


@st.cache(allow_output_mutation=True, persist=True)
def reading_dataset():
    global dataset
    try:
        dataset = pd.read_excel(uploaded_file)
    except ValueError:
        dataset = pd.read_csv(uploaded_file)
    return dataset

# Sidebar configuration for file upload area
with st.sidebar.subheader('Upload your file'):
    uploaded_file = st.sidebar.file_uploader("Please upload a spreadsheet file (xlsx or csv)", type=["xlsx", "csv"])
st.sidebar.subheader("")
st.sidebar.subheader("")
if st.sidebar.button("Clear Cache"):
	caching.clear_memo_cache()
	st.sidebar.success("Cache is cleared!")

# Data profiling and cleaning utilities
if uploaded_file is not None:
    before = beforeSTable()
    after = afterSTable()
    dataset = reading_dataset()
    task = st.selectbox("Menu", ["Data Profiler", "Data Cleaner", "Build Machine Learning Models"]) # Drop down with options
    st.session_state.beforeSS = before
    st.session_state.afterSS = after
    
    # Data Profiler configuration
    if task == "Data Profiler":
        pr = ProfileReport(dataset, explorative=True, orange_mode=True)
        st_profile_report(pr)
    
    # Data Cleaner configuration
    elif task == "Data Cleaner":
        # Obtain column and row counts and information about duplicated rows
        st.header("Dataset Counts")
        numerical = dataset.select_dtypes(include=['number', 'bool', 'datetime64[ns]', 'timedelta64'])
        st.table(pd.DataFrame([[dataset.shape[0], dataset.shape[1],
                                (dataset.shape[1] - numerical.shape[1]), numerical.shape[1]]],
                              columns=["Row Count", "Column Count", "Nominal Column Count",
                                       "Numeric Column Count"], index=[""]))
        columns = dataset.columns.to_numpy().tolist()
        useless = dataset[dataset.isnull().sum(axis=1) > (dataset.shape[1] / 2)]
        uselessRows_count = useless.shape[0]
        if uselessRows_count > 0:
            st.write(str(uselessRows_count), "rows may be useless:", useless)
            st.write("")
        
        # Edit the dataset
        st.header("Edit Your Dataset")

        # Drop duplicate rows
        st.subheader("Review and Drop Duplicate Rows")
        duplicated = dataset[dataset.duplicated()]
        idx_dup = dataset[dataset.duplicated()].index
        duplicatedRows_count = duplicated.shape[0]
        if duplicatedRows_count == 0:
            st.success("There are no duplicated rows in the dataset.")
        else:
            st.write("There are", str(duplicatedRows_count), "duplicated rows in the dataset:",
                    dataset[dataset.duplicated()])
            if st.button("Drop Duplicated Rows"):
                dataset.drop(index=idx_dup, inplace=True)
                st.success("Duplicated rows were deleted.")
            
        # Drop null values
        st.subheader("Drop Null Values")
        with st.form(key="drop_nulls"):
            if st.form_submit_button(label="Drop null values"):
                try:
                    dataset = dataset.dropna()
                    st.success("Null values were deleted.")
                except KeyError:
                    st.error("Null values were not deleted.")
            
        # Convert values in columns
        st.subheader("Convert Values in Columns")
        convert_expander = st.expander("Convert Values in a Column by Using String Methods",
                                            expanded=False)
        with convert_expander:
            obj_cols = dataset.select_dtypes(include=['object', 'category']).columns
            betacolumns1 = st.columns((0.5, 6, 3.5))
            with betacolumns1[1]:
                convert_toTitle = st.checkbox(
                    label="Convert the first character of each word to upper case, 'Aaa Aaaa'")
            if convert_toTitle:
                selected_col = st.selectbox("Select the column with the values you want to convert", obj_cols,
                                            key="forTitle")
                if st.button("Convert", key="button_for_convertingTitle"):
                    dataset[selected_col] = dataset[selected_col].str.title()
                    st.success("Values were converted.")

            betacolumns2 = st.columns((0.5, 6, 3.5))
            with betacolumns2[1]:
                convert_toLower = st.checkbox(label="Convert values into lower case, 'aaaaaaaa'")
            if convert_toLower:
                selected_col = st.selectbox("Select the column with the values you want to convert", obj_cols,
                                            key="forLower")
                if st.button("Convert", key="button_for_convertingLower"):
                    dataset[selected_col] = dataset[selected_col].str.lower()
                    st.success("Values were converted.")

            betacolumns3 = st.columns((0.5, 6, 3.5))
            with betacolumns3[1]:
                convert_toUpper = st.checkbox(label="Convert values into upper case, 'AAAAA AAAAAA'")
            if convert_toUpper:
                selected_col = st.selectbox("Select the column with the values you want to convert", obj_cols,
                                            key="forUpper")
                if st.button("Convert", key="button_for_convertingUpper"):
                    dataset[selected_col] = dataset[selected_col].str.upper()
                    st.success("Values were converted.")

            betacolumns4 = st.columns((0.5, 6, 3.5))
            with betacolumns4[1]:
                convert_to1space = st.checkbox(label="Remove multiple spaces")
            if convert_to1space:
                selected_col = st.selectbox("Select the column with the values you want to convert", obj_cols,
                                            key="forSpace")
                if st.button("Remove", key="button_for_removingSpace"):
                    dataset[selected_col] = dataset[selected_col].apply(
                        lambda x: re.sub(' +', ' ', str(x))).replace('nan', np.NaN)
                    st.success("Multispaces were removed.")

            betacolumns5 = st.columns((0.5, 6, 3.5))
            with betacolumns5[1]:
                strip = st.checkbox(label="Strip")
            if strip:
                selected_col = st.selectbox("Select the column with the values you want to convert", obj_cols,
                                            key="forStrip")
                strp = st.text_input("Value")
                if st.button("Strip", key="button_for_Strip"):
                    dataset[selected_col] = dataset[selected_col].str.strip(strp)
                    st.success("Changes were applied.")

            betacolumns6 = st.columns((0.5, 6, 3.5))
            with betacolumns6[1]:
                replace = st.checkbox(label="Replace")
            if replace:
                st.info("If you want to remove the value instead of replacing it with another value, type 'none'.")
                selected_col = st.selectbox("Select the column with the values you want to change", obj_cols,
                                            key="forReplace")
                cols = st.columns((3, 2, 3))
                with cols[0]:
                    val1 = st.text_input("Find what:")
                    if val1 == "(":
                        val1 = '\('
                    if val1 == "?":
                        val1 = '\?'
                    if val1 == "|":
                        val1 = '\|'
                    if val1 == '[':
                        val1 = '\['
                    if val1 == '+':
                        val1 = '\+'
                    if val1 == ')':
                        val1 = '\)'
                    if val1 == '*':
                        val1 = '\*'
                    if val1 == '^':
                        val1 = '\^'
                    if val1 == '$':
                        val1 = '\$'
                with cols[2]:
                    valrep = st.text_input("Replace with:" )
                    if valrep == "none":
                        valrep = ""
                    if valrep == "(":
                        valrep = '\('
                    if valrep == "?":
                        valrep = '\?'
                    if valrep == "|":
                        valrep = '\|'
                    if valrep == '[':
                        valrep = '\['
                    if valrep == '+':
                        valrep = '\+'
                    if valrep == ')':
                        valrep = '\)'
                    if valrep == '*':
                        valrep = '\*'
                    if valrep == '^':
                        valrep = '\^'
                    if valrep == '$':
                        valrep = '\$'

                if st.button("Replace", key="button_for_replace"):
                    countrep = dataset[selected_col].str.count(val1).sum()
                    dataset[selected_col] = dataset[selected_col].str.replace(val1, valrep)
                    success_text = str(int(countrep)) + " values were changed."
                    st.success(success_text)

            betacolumns5 = st.columns((0.5, 9.5))
            with betacolumns5[1]:
                strip = st.checkbox(label="Format Corrector for 'Telefon Numarası' (to reduce the character length of examples like '0XXXXXXXXXX' to 10)")
            if strip:
                selected_col = st.selectbox("Select column with 'Telefon Numarası' values",
                                            obj_cols,
                                            key="forReducing")
                first_char = st.text_input("Character")
                if st.button("Remove", key="button_for_Reducing"):
                    dataset[selected_col] = dataset[selected_col].astype('str').apply(lambda x : x[1:] if x.startswith(first_char) else x).replace('nan', np.NaN)
                    st.success("Changes were applied.")

        # Show edited dataset
        st.subheader("Show Edited Dataset")
        if st.button("Show my Dataset", key="display"):
            st.write(dataset)

    bst = st.session_state.beforeSS
    ast = st.session_state.afterSS

# If nothing if selected from the drop down, then a message indicating that a file should be uploaded will display    
else:
    st.write("")
    st.info('Awaiting for file to be uploaded.')
    st.write("")
    """
    **Use our models to see how machine learning works!**

    1. Upload your Excel file in the sidebar.
    2. Get insight into your data using the data profiler.
    3. Use the data cleaner to delete duplicate values and replace special characters.
    4. Select a machine learning model to make predicitons using your cleaned data.
    ---
    """
    