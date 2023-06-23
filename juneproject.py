import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import warnings
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from PIL import Image
import io
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection  import train_test_split
import certifi

warnings.filterwarnings(action='ignore', message='All-NaN slice encountered')

def get_downloadable_data(df):
    return df.to_csv().encode('utf-8')

st.title('**Home Loan Approval Prediction**')

description=''' This dataset contains 13 attributes about 613 home loan requests.'''


st.write(description)

loan_df =pd.read_csv('https://raw.githubusercontent.com/nicoleolivetto/Streamlit_presentation/main/loan_requests.csv', header=0,index_col=0)

st.write(loan_df)

st.download_button('download raw data', get_downloadable_data(loan_df), file_name='loan_requests.csv')

st.subheader('About the dataset:')

show_info=st.checkbox('Show info()')
if show_info:
    st.subheader('loan_df.info()')

    buffer = io.StringIO()
    loan_df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

show_info=st.checkbox('Show describe()')
if show_info:
    st.subheader('loan_df.describe()')
    st.write(loan_df.describe())

show_info=st.checkbox('Show Head and Tail()')
if show_info:
    st.subheader('loan_df.head()')
    st.write(loan_df.head())

    st.subheader('loan_df.tail()')
    st.write(loan_df.tail())


st.subheader('Removing null values')
st.write('Number of null values in each column:')
st.write(pd.isnull(loan_df).sum())


code1= ''' loan_df=loan_df.dropna()'''
show_code1=st.checkbox('Show code1')
if show_code1:
    st.code(code1, language='python')

loan_df=loan_df.dropna()
st.write('Number of null values in each column after removing them:')
st.write(pd.isnull(loan_df).sum())

st.subheader('Creation of column Total_Income')
loan_df=loan_df.assign(Total_Income=loan_df.ApplicantIncome + loan_df.CoapplicantIncome)
st.write(loan_df)

loan_df['CoapplicantIncome'] = loan_df['CoapplicantIncome'].astype('int64')
loan_df['LoanAmount'] = loan_df['LoanAmount'].astype('int64')
loan_df['Loan_Amount_Term'] = loan_df['Loan_Amount_Term'].astype('int64')
loan_df['Credit_History'] = loan_df['Credit_History'].astype('int64')
loan_df['Total_Income'] = loan_df['Total_Income'].astype('int64')

code2=''' loan_df['CoapplicantIncome'] = loan_df['CoapplicantIncome'].astype('int64')
loan_df['LoanAmount'] = loan_df['LoanAmount'].astype('int64')
loan_df['Loan_Amount_Term'] = loan_df['Loan_Amount_Term'].astype('int64')
loan_df['Credit_History'] = loan_df['Credit_History'].astype('int64') '''

show_code2=st.checkbox('Conversion on float values to int:')
if show_code2:
    st.code(code2, language='python')


st.subheader('Info after removing null values and adding a new column:')
buffer = io.StringIO()
loan_df.info(buf=buffer)
s = buffer.getvalue()
st.text(s)

st.subheader('Numerical Data Representation')

option = st.selectbox('Select a numerical attribute',('ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Total_Income'))

fig1, axes = plt.subplots(1,2, figsize=(10, 5))
st.write('Distribution of', option)
sns.set_style('whitegrid')
sns.boxplot(ax=axes[ 0], data=loan_df, x=option, color='pink')
sns.histplot(ax=axes[ 1], data=loan_df, x=option, color='yellow')
st.pyplot(fig1)

st.subheader('Categorical Data Representation')


options = st.selectbox('Select a categorical attribute',('Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Credit_History', 'Property_Area', 'Loan_Status'))

st.write('Data representation with pie plot')
fig2=plt.figure(figsize=(10,5))
st.write('Distribution of', options)

pie_chart = loan_df[options].value_counts()
pie_chart.plot.pie(shadow=True,startangle=90,autopct='%1.1f%%')
plt.legend(loan_df[options].value_counts().index.unique())
plt.show()

st.pyplot(fig2)

st.write('Data representation with barplot plot')

fign=plt.figure(figsize=(5,3))
ax = loan_df[options].value_counts().plot(kind='bar').set(title='Data Representation with Barplot')
st.pyplot(fign)

#fig5=plt.figure(figsize=(10,5))
#st.write('Distribution of', options)
#st.write(options)
#sns.catplot(data=loan_df, x=options, kind="count", palette="Paired").set(title='Data Representation with Catplot')
##plt.show()

#st.pyplot(fig5)



st.subheader('Pairplot')
description2=''' A pairplot is a data visualization that plots pair-wise relationships between all the variables of a dataset. 
The diagonal line shows the distribution of values in that variable using a histogram. Each other cell shows the relationship as a scatterplot between the two variables at their intersection.'''

st.write(description2)

fig6=sns.set(style="ticks", color_codes=True)    
sns.pairplot(loan_df, hue='Loan_Status')
st.pyplot(fig6)


st.subheader('How different attributes impact Loan_Status')

def impact(column):
   
    df = loan_df.pivot_table(index=column, columns='Loan_Status', aggfunc='size')
    sns.set_style('whitegrid')
    st.write(column)
    ax=df.plot(kind='bar', stacked=False, color=['pink', 'purple'])
    for container in ax.containers:
        ax.bar_label(container)
    

fig3=plt.figure(figsize=(10,5))

df = loan_df.pivot_table(index=options, columns='Loan_Status', aggfunc='size')
sns.set_style('whitegrid')
st.write(df)
ax=df.plot(kind='bar', stacked=False, color=['pink', 'purple'])
for container in ax.containers:
    ax.bar_label(container)

st.pyplot(fig3)

st.subheader('Observations:')
st.write('Gender, Education, Dependents and Self_Employed dont have significant impact on the loan status, But loan requests have a higher chance at being approved when Credit_History takes value 1')

st.subheader('Approved and rejected loan amounts across property areas')

figpa=plt.figure(figsize=(10,5))
sns.boxplot( x='Property_Area', y='LoanAmount', hue='Loan_Status', data=loan_df)
st.pyplot(figpa)

fig10, axes = plt.subplots(1,2, figsize=(10, 5))
st.write('Loan_Amount and Total_Income depending on the Property area')
sns.set_style('whitegrid')
sns.boxplot(ax=axes[ 0], data=loan_df, x='Property_Area', y='LoanAmount', palette='Paired')
sns.boxplot(ax=axes[ 1], data=loan_df, x='Property_Area', y='Total_Income', palette='Paired')
st.pyplot(fig10)

st.subheader('Observations:')
st.write('Total_Income is similar in all property areas. The highest number of loans were applied in Semi-urban areas, but the property area doesnt have much impact on Loan_approval')

st.subheader('Correlation matrix')

description3='''A correlation matrix is a table showing correlation coefficients among your variables. 
Each cell in the table shows the correlation between two variables.
The correlation coefficient is a standardized metric that ranges from -1 and +1. Positive values indicate a positive correlation,
negative values indicate a negative correlation. 0 indicates no correlation.'''

st.write(description3)



loan_df['Loan_Status'] = loan_df['Loan_Status'].map({'Y': 1, 'N': 0})
#loan_df['Loan_Status']

#MODEL

loan_num=loan_df.select_dtypes(include=[np.number])

st.header('**Implementation of Gaussian Naive Bayes**')

model = GaussianNB()
y=loan_df.Loan_Status
choices = st.multiselect('Select features', ['ApplicantIncome',	'CoapplicantIncome','LoanAmount',	'Loan_Amount_Term',	'Credit_History', 'Total_Income'])
test_size = st.slider('Test size: ', min_value=0.1, max_value=0.9, step =0.1)
if len(choices) > 0 and st.button('RUN MODEL'):
    with st.spinner('Training...'):
        x = loan_num[choices]
        #y=y.astype('int')
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=2)

        x_train = x_train.to_numpy().reshape(-1, len(choices))
        model.fit(x_train, y_train)

        x_test = x_test.to_numpy().reshape(-1, len(choices))
        y_pred = model.predict(x_test)

        accuracy = accuracy_score(y_test, y_pred)

        st.write(f'Accuracy = {accuracy:.2f}')



fig11=plt.figure(figsize=(5,4))
sns.heatmap(loan_df.corr(), annot=True, cmap='coolwarm')
st.pyplot(fig11)







