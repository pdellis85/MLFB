![Machine Learning for Beginners Logo](https://github.com/pdellis85/MLFB/blob/main/Machine%20Learning%20for%20Beginners-logos.jpeg)

What do self-driving cars, Amazon's Alexa, Facebook ads, and digital insurance companies have in common?  The answer is machine learning.  IBM defines machine learning as "a branch of artificial intelligence (AI) and computer science which focuses on the use of data and algorithms to imitate the way that humans learn, gradually improving its accuracy" (IBM Cloud Education).  SAS elaborates, staing that marchine learning "is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns, and make decisions with minimal human intervention" (SAS).  Machine learning models can process large datasets relatively quickly, using statistics to make predictions.  The more data the model "sees," the more it learns and the better it gets.

Machine learning models are typically built by data scientists using computer code.  Some tools, such as those offered by Amazon Web Services, exist to help the non-coder build machine learning models, but even these require some knowledge of the technology behind machine learning.  We wanted to create a tool that would enable a person new to machine learning to build multiple models and learn how the model building process works.  To do this, we built an interactive dashboard that allows a user to import data, do some basic data cleaning, run the cleaned data through three machine learning models, and view visuals of each model's output as well as generate scores of each model's perfromance.  We added in-context notes explainng what each step does.

For those who want to dive right into building machine learning models, click this link to launch the app:

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/pdellis85/mlfb/main/Code/Sample_code/app.py)

<details>
  
<summary>About the Models</summary>

The Machine Learning for Beginners dashboard has three models available for selection.  

The **Decision Tree** model is a supervised learning method that predicts target values by using the features of a dataset to make decisions.  Decision Trees can be used for both classification and regression.  Decision Trees are fairly simple to understand, and you can create a visualization of the model after its creation.  The Decsion Tree model forms the basis of the other two models we use in our dashboard.
  
The **Random Forest Classifier** is a meta estimator that creates several Decision Trees from sub-sets of data and averages the results of each to make predictions.  Since the model uses multiple Decision Trees (classifier types), Random Forest is an ensemble learning method.  Each Decision Tree in the ensemble makes its own predictions and the results are compiled, with the most common result being identified.  The Decision Trees in a Random Forest model are not correlated with each other.  
  
The **Gradient Boosting Classifier** is an additive model that combines other models together to create one model that performs better than its parts.  Gradient Boosting uses Decision Trees as weak learners, models whose performance is only slightly better than chance.  Additional Trees are added to the weak learners, and their output is added together.  Adding additional Trees to the weak learners "boosts" their performance.

</details>

<details>
  
<summary>How to Set up and Run the Machine Learning Dashboard</summary>
  
If you'd like to run a local version of our dashboard on your computer, here are the steps you'll need to take.
  
1. Create a new conda environment on your computer by running the following command in the Anaconda Powershell:
```
conda create -n mlfb python=3.7.9
```
2. Once the new environment has been created, activate the environment by running the following command in the Anaconda Powershell:
```
conda activate mlfb
```
3.  Run the following command to download the requirements.txt file:  
```
wget https://raw.githubusercontent.com/pdellis85/MLFB/main/Code/Sample_code/requirements.txt
```
4.  Enter the following command to install all of the libraries and dependencies you'll need to run the dashboard:
```
pip install -r requirements.txt
```
5.  Download the contents of this repository from https://github.com/pdellis85/MLFB/archive/refs/heads/main.zip.  Unzip the file and move the MLFB file somewhere on your computer (like your desktop).  You can also clone this repository using the command prompt or [Git Hub Dekstop](https://desktop.github.com/).
  
6.  Use the cd command to change directories.  You'll need to navigate to the directory where you put the MLFB file in step 5.
  
7.  Launch the app with the following command:
```
streamlit run app.py  
```  
</details>

# References and Resources

Bailgoyem.  *DataQtor*.  https://github.com/baligoyem/dataqtor.git.  Retrieve 22 January 2022.

Brownlee, Jason. *A Gentle Introduction to the Gradient Boosting Algorithm for Machine Learning.*. https://machinelearningmastery.com/gentle-introduction-gradient-boosting-algorithm-machine-learning/. Retrieved 3 December 2021.

Brownlee, Jason. *Bagging and Random Forest for Imbalanced Classification*. https://machinelearningmastery.com/bagging-and-random-forest-for-imbalanced-classification/. Retrieved 2 December 2021.

Dataprofessor.  *EDA App*.  https://github.com/dataprofessor/eda-app.git.  Retrieved 22 January 2022.

IBM.  *Random Forest*.  https://www.ibm.com/cloud/learn/random-forest.  Retrieved 27 January 2022.

IBM Cloud Education. *Machine Learning*. https://www.ibm.com/cloud/learn/machine-learning.  Retrieved 27 January 2022.

SAS. *Machine Learning: What It Is and Why It Matters*.  https://www.sas.com/en_us/insights/analytics/machine-learning.html.  Retrieved 27 January 2022.

Sci-kit Learn.  *Decision Trees.*  https://scikit-learn.org/stable/modules/tree.html.  Retrieved 2 February 2022.
