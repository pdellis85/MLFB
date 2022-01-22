# This file is based on code from https://github.com/baligoyem/dataqtor.git

import streamlit as st
from streamlit import caching
import SessionState
import pandas as pd
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from gaugeChart import gauge
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta
from string_grouper import match_strings
import re
from TRnoChecker import isValidTCID, taxnum_checker
import TR_name_gender
from downloader import get_table_download_link
from utils import download_button

# Page configuation
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(page_title="Poject 3 Dashboard Title", layout="wide")
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


# insert logo --> st.image('.png', width=250)

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
    task = st.selectbox("Menu", ["Data Profiler", "Data Quality Detector",
                                 "Data Cleaner", "Review Summary Report and Download Adjusted Data"])
    st.session_state.beforeSS = before
    st.session_state.afterSS = after
    
    # Data Profiler configuration
    if task == "Data Profiler":
        pr = ProfileReport(dataset, explorative=True, orange_mode=True)
        st_profile_report(pr)
    
    # Data Quality Detector configuration
    elif task == "Data Quality Detector":
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
        st.write("---")

        if st.checkbox("Run Column Detector", key="run_col_detector"):
            cols = st.columns(1)
            with cols[0]:
                selected_column = st.selectbox("Column", columns, key="col_select")
            cols = st.columns((7, 1.4, 1.6))
            with cols[0]:
                st.write("Type:", dataset[selected_column].dtype)
            if (dataset[selected_column].dtype == np.int16 or dataset[selected_column].dtype == np.int32 or
                        dataset[selected_column].dtype == np.int64 or dataset[selected_column].dtype == np.float16 or dataset[
                    selected_column].dtype == np.float32 or dataset[selected_column].dtype == np.float64):
                with cols[1]:
                    if st.checkbox("Show Total"):
                        total = dataset[selected_column].sum()
                        with cols[2]:
                            st.write(": ", '{:,.2f}'.format(total))
            elif dataset[selected_column].dtype == 'object':
                    strMinLength = dataset[selected_column].str.len().min()
                    strMaxLength = dataset[selected_column].str.len().max()
                    valueStrMin = dataset.reset_index(drop=True).loc[dataset[selected_column].str.len().argmin(), selected_column]
                    valueStrMax = dataset.reset_index(drop=True).loc[dataset[selected_column].str.len().argmax(), selected_column]
                    valueMin = dataset[selected_column].dropna().astype(str).sort_values().min()
                    valueMax = dataset[selected_column].dropna().astype(str).sort_values().max()
                    strCA = pd.DataFrame(
                        [[strMinLength, valueStrMin, strMaxLength, valueStrMax, valueMin, valueMax]],
                        columns=["Min Length", "Value (minLen)", "Max Length", "Value (maxLen)",
                                 "Min (Alphabetic)", "Max (Alphabetic)"], index=["info"])
                    st.write(strCA)

            filledCount = dataset[selected_column].count()
            nanCount = int(dataset[selected_column].isna().sum())
            nonNullValues_per = (dataset[selected_column].count() / len(dataset) * 100).round(1)
            nullValues_per = (dataset[selected_column].isna().sum() / len(dataset) * 100).round(1)
            nanDF = pd.DataFrame([[filledCount, nonNullValues_per],
                                  [nanCount, nullValues_per]], columns=["count", "percentage(%)"],
                                 index=["non-NaN", "isNaN"])

            def color_survived(val):
                if val == nanDF["percentage(%)"]["isNaN"] and val >= 50:
                    color = '#FA8072'
                    return f'background-color: {color}'
                else:
                    color = 'white'
                    return f'background-color: {color}'

            cols = st.columns((3, 7))
            with cols[0]:
                st.write(
                    nanDF.style.format({'count': '{:,}', 'percentage(%)': '{:.1f}'}).applymap(color_survived, subset=['percentage(%)']))
            with cols[1]:
                describe = dataset[selected_column].describe().to_frame().T
                st.write(describe.style.format(
                    {"count": '{:,.0f}', "mean": '{:,.1f}', "std": '{:,.1f}', "min": '{:,.2f}',
                     "25%": '{:,.2f}', "50%": '{:,.2f}', "75%": '{:,.2f}', "max": '{:,.2f}'}))

            freqCount = dataset[selected_column].value_counts().to_frame(name="count")
            per = (dataset[selected_column].value_counts(normalize=True).round(3) * 100).to_frame(
                name="percentage(%)")
            general = pd.concat([freqCount, per], axis=1)
            if general.shape[0] > 10:
                cols = st.columns(3)
                with cols[0]:
                    st.write("5 most frequent values in this column:",
                             general.iloc[np.r_[0:5]].style.format(
                                 {"count": '{:,}', 'percentage(%)': '{:.1f}'}))
                with cols[1]:
                    st.write("5 least frequent values in this column:",
                             general.iloc[np.r_[-5:0]].style.format(
                                 {'count': '{:,}', 'percentage(%)': '{:.1f}'}))
                with cols[2]:
                    if st.checkbox("Frequency Table", key="freq_table"):
                        st.write(general.style.format({'count': '{:,}', 'percentage(%)': '{:.1f}'}))

            else:
                st.write("Frequency Table:", general.style.format({'count': '{:,}', 'percentage(%)': '{:.1f}'}))

            if dataset[selected_column].value_counts().count() > 10:
                unexpected = general[general["percentage(%)"] < 0.1]
                if not unexpected.empty:
                    unexpected['value'] = unexpected.index.astype('str')
                    if (unexpected["value"].value_counts().count() <= 25):
                        fig = plt.figure(figsize=(20, 6))
                        ax = fig.add_axes([0, 0, 1, 1])
                        plt.title("Unexpected Value Graph\n ", size=20, loc="left")
                        ax.bar(unexpected.value, unexpected["count"], color="#262730")
                        ax = plt.gca()
                        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
                        ax.tick_params(axis='both', which='major', labelsize=14)
                        plt.xticks(rotation=45)
                        plt.xlabel("value", fontsize=15)
                        plt.ylabel("count", fontsize=15)
                        plt.show()
                        st.pyplot(fig)

            if (dataset[selected_column].dtype == np.int16 or dataset[selected_column].dtype == np.int32 or
                        dataset[selected_column].dtype == np.int64 or dataset[selected_column].dtype == np.float16 or dataset[
                    selected_column].dtype == np.float32 or dataset[selected_column].dtype == np.float64):
                cols = st.columns(2)
                with cols[0]:
                    if st.checkbox("Show p-0-n"):
                        countp = dataset[dataset[selected_column] > 0][selected_column].count()
                        count0 = dataset[dataset[selected_column] == 0][selected_column].count()
                        countn = dataset[dataset[selected_column] < 0][selected_column].count()
                        countp_per = (countp / len(dataset) * 100).round(1)
                        count0_per = (count0 / len(dataset) * 100).round(1)
                        countn_per = (countn / len(dataset) * 100).round(1)
                        dfp0n = pd.DataFrame([[countp, countp_per],
                                              [count0, count0_per],
                                              [countn, countn_per]], columns=["count", "percentage(%)"],
                                             index=["positive value", "0-value", "negative value"])
                        st.write(dfp0n.style.format({'count': '{:,}', 'percentage(%)': '{:.1f}'}))

            NOFR = 0
            PFR = (dataset[selected_column].shape[0]) - (nanCount + NOFR)
            dqst = pd.DataFrame([["Null Records", nanCount],
                                 ["Out of Format Records", NOFR],
                                 ["Proper Format Records", PFR]],
                                    columns=["Records Type", "Number of Records"])
            st.write("----")
            dq_score = round((PFR / dataset[selected_column].shape[0] * 100), 2)
            table = st.selectbox("Add to", ["'Before' Summary Table", "'After' Summary Table"], key = "add_st")
            insert = st.button("Insert", key="insert")
            if table == "'Before' Summary Table":
                if insert:
                    before.loc[len(before)] = [selected_column, nanCount, NOFR, PFR, dq_score]
                    st.session_state.beforeSS = before
                    st.success("Values have been added to 'Before' Summary Table.")
            elif table == "'After' Summary Table":
                if insert:
                    after.loc[len(after)] = [selected_column, nanCount, NOFR, PFR, dq_score]
                    st.session_state.afterSS = after
                    st.success("Values have been added to 'After' Summary Table.")

            st.subheader("Data Quality Measurement Results for {}".format(selected_column))
            st.write("")
            cols = st.columns((2, 1))
            with cols[0]:
                rtypes = list(dqst["Records Type"])
                noR = list(dqst["Number of Records"])
                fig = plt.figure(figsize=(10, 6))
                # creating the bar plot
                plt.bar(rtypes, noR, color='#023047',
                        width=0.4)

                def addlabels(x, y):
                    for i in range(len(x)):
                        plt.text(i, y[i] + 2, y[i], ha='center', size='large')

                addlabels(rtypes, noR)
                ax = plt.gca()
                ax.tick_params(axis='both', which='major', labelsize=13)
                plt.ylabel("No. of Records")
                plt.title("Summary Graph")
                plt.show()
                st.pyplot(fig)

            if dq_score <= 25:
                a = 1
            elif dq_score > 25 and dq_score <= 50:
                a = 2
            elif dq_score > 50 and dq_score <= 75:
                a = 3
            else:
                a = 4

            dq_score_str = str(dq_score) + "%"
            with cols[1]:


                gauge(labels=['VERY LOW', 'LOW', 'MEDIUM', 'HIGH'], \
                      colors=["#1b0203", "#ED1C24", '#FFCC00', '#007A00'], arrow=a, title=dq_score_str)
                plt.title("Column DQ Score", fontsize=20)
                st.pyplot()

    # Data Cleaner configuration
    elif task == "Data Cleaner":
        st.write("")
        st.write("---")
        if st.checkbox(label="Run Edit Engine"):
            st.subheader("Edit Your Dataset")
            with st.form(key="drop_nulls"):
                if st.form_submit_button(label="Drop null values"):
                    try:
                        dataset = dataset.dropna()
                        st.success("Null values were deleted.")
                    except KeyError:
                        st.error("Null values were not deleted."
            with st.form(key="edit_form_col"):
                col_for_editing = dataset.columns
                colu = st.multiselect("Edit by Column", col_for_editing)
                if st.form_submit_button(label="Drop the Column"):
                    try:
                        dataset.drop(columns=[colu], axis=1, inplace=True)
                        st.success("The Column was deleted.")
                    except KeyError:
                        st.error("The Column was not found.")

            with st.form(key="edit_form_idx"):
                idx = st.number_input("Edit by Index", format="%i", value=0,
                                      max_value=dataset.index.max(), step=1)
                if st.form_submit_button(label="Drop the Row"):
                    try:
                        dataset.drop(index=idx, axis=0, inplace=True)
                        st.success("The Record was deleted.")
                    except KeyError:
                        st.error("The Index was not found.")

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

            sorting_expander = st.expander("Sorting Transformation",
                                                expanded=False)
            with sorting_expander:
                col_for_sorting = dataset.columns
                colSort = st.multiselect("Select the columns whose values you want to sort", col_for_sorting,
                                         key="multiselect_forSorting22")
                asc = canContain = st.text_input(
                    "Please separate the ascending argument values with commas and use only True/False, for example 'True,False' (Do not put a space after comma)")
                asc = asc.split(',')
                if st.button("Sort", key="Sorting2"):
                    st.success("Changes were applied.")
                    res = list(map(lambda ele: ele == "True", asc))
                    dataset.sort_values(by=colSort, ascending=res, ignore_index=False, inplace=True)

            if st.button("Show my Dataset", key="display"):
                st.write(dataset)

    bst = st.session_state.beforeSS
    ast = st.session_state.afterSS
    if task == "Review Summary Report and Download Adjusted Data":
        st.write("'Before' Summary Table")
        st.dataframe(bst.style.format({"Column DQ Score(%)": '{:,.2f}'}))

        st.write("'After' Summary Table")
        st.dataframe(ast.style.format({"Column DQ Score(%)": '{:,.2f}'}))

        st.write("")
        bodq_score = round(bst["Column DQ Score(%)"].mean(), 2)
        aodq_score = round(ast["Column DQ Score(%)"].mean(), 2)

        if bodq_score <= 25:
            before_arrow = 1
        elif bodq_score > 25 and bodq_score <= 50:
            before_arrow = 2
        elif bodq_score > 50 and bodq_score <= 75:
            before_arrow = 3
        else:
            before_arrow = 4

        if aodq_score <= 25:
            after_arrow = 1
        elif aodq_score > 25 and aodq_score <= 50:
            after_arrow = 2
        elif aodq_score > 50 and aodq_score <= 75:
            after_arrow = 3
        else:
            after_arrow = 4

        odq_graph = st.columns(2)
        with odq_graph[0]:
            gauge(labels=['VERY LOW', 'LOW', 'MEDIUM', 'HIGH'], \
                  colors=["#1b0203", "#ED1C24", '#FFCC00', '#007A00'], arrow=before_arrow, title=str(bodq_score) + '%')
            plt.title("'Before' Overall DQ Score", fontsize=16)
            st.pyplot()
        with odq_graph[1]:
            gauge(labels=['VERY LOW', 'LOW', 'MEDIUM', 'HIGH'], \
                  colors=["#1b0203", "#ED1C24", '#FFCC00', '#007A00'], arrow=after_arrow, title=str(aodq_score) + '%')
            plt.title("'After' Overall DQ Score", fontsize=16)
            st.pyplot()

        prepare_expander = st.expander("Prepare Dataset for Download", expanded=False)
        with prepare_expander:
            session_state = SessionState.get(df=dataset)
            if st.checkbox("Reorder and Eliminate Columns", key = "reorder_eliminate"):
                col_for_order = dataset.columns
                colOrder = st.multiselect("Select the columns in the order you want them to be", col_for_order,
                                              key="multiselect_forOrder22")
                if st.button("Set", key="set_order2"):
                    session_state.df = dataset[colOrder]
                    st.success("The adjustments were applied.")
                    st.write("Sample", session_state.df.head())
                    download_button(get_table_download_link(session_state.df), "AdjustedData.xlsx", "Download (.xlsx)")

            else:
                download_button(get_table_download_link(dataset), "AdjustedData.xlsx", "Download (.xlsx)")

else:
    st.write("")
    st.info('Awaiting for file to be uploaded.')
    st.write("")
    """
    **Use our models to see how machine learning works!**

    1. Upload your Excel file in the sidebar.
    2. Get insight into your data using the data analysis tools.
    3. Check the quality of your data and clean your data.
    4. Select a machine learning model to make predicitons using your cleaned data.
    ---
    """
    