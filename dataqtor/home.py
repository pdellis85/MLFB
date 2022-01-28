from cgi import test
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.utils import shuffle
import streamlit as st
from streamlit import caching
from streamlit_pandas_profiling import st_profile_report
import sklearn
from sklearn import tree
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import pandas as pd
import plotly.express as px
from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta
from string_grouper import match_strings
import re
from TRnoChecker import isValidTCID, taxnum_checker
import TR_name_gender
from downloader import get_table_download_link
from utils import download_button


st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(page_title="Machine Learning For Beginners",
                   page_icon="🔍", layout="wide")
hide_streamlit_style = """
            <style>
            footer {
	        visibility: hidden;
	            }
            footer:after {
	            content:'Clean and Train your data'; 
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


@st.cache(allow_output_mutation=True, persist=True)
def reading_dataset():
    global dataset
    try:
        dataset = pd.read_excel(uploaded_file)
    except ValueError:
        dataset = pd.read_csv(uploaded_file)
    return dataset


st.image('DataQtor.png', width=250)
with st.sidebar.subheader('Upload your file'):
    uploaded_file = st.sidebar.file_uploader(
        "Please upload a file of type: xlsx, csv", type=["xlsx", "csv"])

st.sidebar.subheader("")

st.sidebar.subheader("")


if uploaded_file is not None:

    dataset = reading_dataset()
    task = st.selectbox("Menu", ["Data Profiler", "Data Quality Detector",
                                 "Data Corrector", "Model"])

    if task == "Data Profiler":
        pr = ProfileReport(
            dataset,
            explorative=True,
            missing_diagrams=None,
            # duplicates=None,
            interactions=None)
        st_profile_report(pr)

    elif task == "Data Quality Detector":
        numerical = dataset.select_dtypes(
            include=['number', 'bool', 'datetime64[ns]', 'timedelta64'])
        st.table(pd.DataFrame([[dataset.shape[0], dataset.shape[1],
                                (dataset.shape[1] - numerical.shape[1]), numerical.shape[1]]],
                              columns=["Row Count", "Column Count", "Nominal Column Count",
                                       "Numeric Column Count"], index=[""]))
        columns = dataset.columns.to_numpy().tolist()
        useless = dataset[dataset.isnull().sum(
            axis=1) > (dataset.shape[1] / 2)]
        uselessRows_count = useless.shape[0]
        if uselessRows_count > 0:
            st.write(str(uselessRows_count), "rows may be useless:", useless)
            st.write("")

        duplicated = dataset[dataset.duplicated()]
        idx_dup = dataset[dataset.duplicated()].index
        duplicatedRows_count = duplicated.shape[0]
        if duplicatedRows_count == 0:
            st.success("There is no duplicated rows in the dataset.")
        else:
            st.write("There are", str(duplicatedRows_count), "duplicated rows in the dataset:",
                     dataset[dataset.duplicated()])
            if st.button("Drop Duplicated Rows"):
                dataset.drop(index=idx_dup, inplace=True)
                st.success("Duplicated rows were deleted.")
        st.write("---")

    elif task == "Model":
        st.table(pd.DataFrame(dataset.head(10)))
        columnes = dataset.columns.to_numpy().tolist()
        target = st.selectbox("Target", columnes)
        df = pd.DataFrame(dataset)
        y = df.loc[:, target].ravel()
        X = df
        X.drop(y)
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
            X, y, random_state=78)
        scaler = StandardScaler()
        X_scaler = scaler.fit(X_train)
        X_train_scaled = X_scaler.transform(X_train)
        X_test_scaled = X_scaler.transform(X_test)

        def impPlot(imp, name):
            figure = px.bar(imp,
                            x=imp.values,
                            y=imp.keys(), labels={'x': 'Importance Value', 'index': 'Columns'},
                            text=np.round(imp.values, 2),
                            title=name + ' Feature Selection Plot',
                            width=1000, height=600)
            figure.update_layout({
                'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                'paper_bgcolor': 'rgba(0, 0, 0, 0)',
            })
            st.plotly_chart(figure)

        def randomForest():
            rf_model = RandomForestClassifier(
                n_estimators=500, random_state=78)
            rf_model = rf_model.fit(X_train_scaled, y_train)
            predictions = rf_model.predict(X_test_scaled)
            acc_score = accuracy_score(y_test, predictions)
            # st.info(acc_score)
            importances = rf_model.feature_importances_
            feat_importances = pd.Series(
                importances, index=X.columns).sort_values(ascending=True)
            if st.checkbox("Show Vis", False, 1):
                st.subheader('Random Forest Classifier:')
                impPlot(feat_importances, 'Random Forest Classifier')
                st.write('\n')

        def decisionTree():
            model = tree.DecisionTreeClassifier()
            model.fit(X_train_scaled, y_train)
            predictions = model.predict(X_test_scaled)
            acc = accuracy_score(y_test, predictions)
            # st.info(acc)
            if st.checkbox("Show Vis", False, 2):
                st.subheader('Decision Tree Classifier:')
                # max_depth is maximum number of levels in the tree
                clf = DecisionTreeClassifier(max_depth=3)
                clf.fit(X, y)
                b = plt.figure(figsize=(20, 10))
                a = plot_tree(clf,
                              filled=True,
                              rounded=True,
                              fontsize=14)
                st.pyplot(b)

        def xgbClassTree():
            x_model = GradientBoostingClassifier()
            x_model.fit(X_train_scaled, y_train)
            predictions = x_model.predict(X_test_scaled)
            accs = accuracy_score(y_test, predictions)
            # st.info(accs)
            if st.checkbox("Show Classification Report", False, 3):
                st.subheader('Gradient Boosting Classifier:')
                st.write(classification_report(y_test, predictions))
                st.write('\n')

        task2 = st.selectbox("Chose Model", ["Random Forest", "Decision Tree",
                                             "Gradient Boost Tree"])

        if task2 == "Random Forest":
            randomForest()

        elif task2 == "Decision Tree":
            decisionTree()

        else:
            xgbClassTree()


else:
    st.write("")
    st.info('Awaiting for file to be uploaded.')
    st.write("")

    st.image('Machine Learning for Beginners-logos.jpeg', width=450)
    st.write("")
    """
    **Get your data ready for use before you start working with it:**

    1. Upload your Excel/CSV file 📁
    2. Gain insight into your data 💡
    3. Measure the quality of your data 📊
    4. Repair your data in light of analyzes 🛠
    5. Observe improvement in data quality 📈
    6. Download the dataset you repaired 📥
    ---
    
    """
