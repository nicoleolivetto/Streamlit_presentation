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

st.title('**Top 1000 Most Subscribed Youtube Channels**')

description=''' This data contains 7 attributes about the top YouTube channels as per number of subscribers.
 These attributes with their proper description are as follows: \n

**Rank**: Rank of the channel as per number of subscribers they have\n
**Youtuber**: Channel Official Name\n
**Subscribers**: Number of subscribers channel have\n
**Video views**: Number for which all videos have been watched collectively\n
**Video count**: Number of videos channel has uploaded so far\n
**Category**: Category (genre) of the channel\n
**Started**: Year when the channel was started'''

st.write(description)

topsubscribed_df =pd.read_csv('https://raw.githubusercontent.com/nicoleolivetto/Streamlit_presentation/main/Top_subscribed.csv', header=0,index_col=0)



st.write(topsubscribed_df)

st.download_button('download raw data', get_downloadable_data(topsubscribed_df), file_name='Top_subscribed.csv')

show_info=st.checkbox('Show info()')
if show_info:
    st.subheader('Topsubscribed_df.info()')


    buffer = io.StringIO()
    topsubscribed_df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

topsubscribed_df.rename({'Video Views': 'VideoViews'}, axis=1, inplace=True)

topsubscribed_df['Subscribers']=topsubscribed_df['Subscribers'].str.replace(',','')
topsubscribed_df['VideoViews']=topsubscribed_df['VideoViews'].str.replace(',','')
topsubscribed_df['Video Count']=topsubscribed_df['Video Count'].str.replace(',','')

topsubscribed_df['Subscribers'] = topsubscribed_df['Subscribers'].astype('int64')
topsubscribed_df['VideoViews'] = topsubscribed_df['VideoViews'].astype('int64')
topsubscribed_df['Video Count'] = topsubscribed_df['Video Count'].astype('int64')

topsubscribed_df=topsubscribed_df[(topsubscribed_df != 0).all(1)]


topsubscribed_df = topsubscribed_df.drop(topsubscribed_df[topsubscribed_df['Started'] < 2005].index)

topsubscribed_df = topsubscribed_df.replace('https://us.youtubers.me/global/all/top-1000-most_subscribed-youtube-channels','Unknown')

topsubscribed_df.reset_index(drop=True, inplace=True)

st.subheader('Dataset after changing column types, removing rows with zeros, changing the name of a column, removing raw with wrong value')

st.write(topsubscribed_df)

show_infos=st.checkbox('Show info() of new df')
if show_infos:
    st.subheader('Topsubscribed_df.info()')


    buff = io.StringIO()
    topsubscribed_df.info(buf=buff)
    n = buff.getvalue()
    st.text(n)


show_info=st.checkbox('Show describe()')
if show_info:
    st.subheader('Topsubscribed_df.describe()')
    st.write(topsubscribed_df.describe())

st.subheader('I created a new column named "range_subs" and divided all entries in 6 categories')

topsubscribed_df=topsubscribed_df.assign(range_subs=0)
topsubscribed_df.range_subs = topsubscribed_df.range_subs.astype(str)

#add code
code1 = '''for i in range ((len(topsubscribed_df["Subscribers"]))):
    if (topsubscribed_df.at[(i),'Subscribers'] > 10000000) and (topsubscribed_df.at[(i),'Subscribers'] <= 15000000):
        topsubscribed_df.at[(i),'range_subs']= '10M - 15M'
    elif (topsubscribed_df.at[(i),'Subscribers'] > 15000000) and (topsubscribed_df.at[(i),'Subscribers'] <= 20000000):
        topsubscribed_df.at[(i),'range_subs']= '15M - 20M'
    elif (topsubscribed_df.at[(i),'Subscribers'] > 20000000) and (topsubscribed_df.at[(i),'Subscribers'] <= 50000000):
        topsubscribed_df.at[(i),'range_subs']= '20M - 50M'
    elif (topsubscribed_df.at[(i),'Subscribers'] > 50000000) and (topsubscribed_df.at[(i),'Subscribers'] <= 100000000):
        topsubscribed_df.at[(i),'range_subs']= '50M - 100M'
    elif (topsubscribed_df.at[(i),'Subscribers'] > 100000000) and (topsubscribed_df.at[(i),'Subscribers'] <= 200000000):
        topsubscribed_df.at[(i),'range_subs']= '100M - 200M'
    elif (topsubscribed_df.at[(i),'Subscribers'] > 20000000):
        topsubscribed_df.at[(i),'range_subs']= '200M+'
'''

show_code1=st.sidebar.checkbox('Show code1')
if show_code1:
    st.code(code1, language='python')


for i in range ((len(topsubscribed_df["Subscribers"]))):
    if (topsubscribed_df.at[(i),'Subscribers'] > 10000000) and (topsubscribed_df.at[(i),'Subscribers'] <= 15000000):
        topsubscribed_df.at[(i),'range_subs']= '10M - 15M'
    elif (topsubscribed_df.at[(i),'Subscribers'] > 15000000) and (topsubscribed_df.at[(i),'Subscribers'] <= 20000000):
        topsubscribed_df.at[(i),'range_subs']= '15M - 20M'
    elif (topsubscribed_df.at[(i),'Subscribers'] > 20000000) and (topsubscribed_df.at[(i),'Subscribers'] <= 50000000):
        topsubscribed_df.at[(i),'range_subs']= '20M - 50M'
    elif (topsubscribed_df.at[(i),'Subscribers'] > 50000000) and (topsubscribed_df.at[(i),'Subscribers'] <= 100000000):
        topsubscribed_df.at[(i),'range_subs']= '50M - 100M'
    elif (topsubscribed_df.at[(i),'Subscribers'] > 100000000) and (topsubscribed_df.at[(i),'Subscribers'] <= 200000000):
        topsubscribed_df.at[(i),'range_subs']= '100M - 200M'
    elif (topsubscribed_df.at[(i),'Subscribers'] > 20000000):
        topsubscribed_df.at[(i),'range_subs']= '200M+'

st.write(topsubscribed_df)

st.subheader('The following graphs show how many channels there are in each subscriber range')
st.write('Most channels have 10-50M subscribers')

fig1=plt.figure(figsize=(15,10))
plt.title('Subscriber range')
sns.set_style('whitegrid')
m = sns.barplot(x=topsubscribed_df['range_subs'].value_counts().index, y=topsubscribed_df['range_subs'].value_counts(), palette ='Paired')
m.set_xticklabels(topsubscribed_df['range_subs'].value_counts().index, rotation = 45)
m.set(xlabel ='range_subs', ylabel = 'Values')
plt.show()
   
st.pyplot(fig1)

fig15=plt.figure(figsize=(15,10))    
st.set_option('deprecation.showPyplotGlobalUse', False)
pie_chart = topsubscribed_df['range_subs'].value_counts()
pie_chart.plot.pie(shadow=True,startangle=90,autopct='%1.1f%%')
plt.legend(topsubscribed_df['range_subs'].value_counts().index.unique())
plt.show()

st.pyplot(fig15)
    

st.subheader('How many of these youtubers joined YT each year?')
 

col_1, col_2 = st.columns(2)
with col_1: 
    annual_join = topsubscribed_df.groupby(['Started']).size().reset_index().rename(columns = {0:'Counts'})
    st.write(annual_join)

with col_2:
    fig2=plt.figure(figsize=(15,10))
    plt.title('join per year')
    sns.set_style('whitegrid')
    a=sns.barplot(x=topsubscribed_df['Started'].value_counts().sort_index().index, y=topsubscribed_df['Started'].value_counts())
    a.set_xticklabels(topsubscribed_df['Started'].value_counts().index, rotation = 45)
    a.set(xlabel ='Started', ylabel = 'Values')
    plt.show()

    st.pyplot(fig2)


st.subheader('How many Youtube channels in each category?')

col_1, col_2 = st.columns(2)
with col_1: 
    a=topsubscribed_df.groupby(['Category']).size().reset_index().rename(columns = {0:'Counts'})
    st.write(a)

with col_2:
    fig3=plt.figure(figsize=(15,10))
    plt.title('Categories')
    sns.set_style('whitegrid')
    n = sns.barplot(x=topsubscribed_df['Category'].value_counts().sort_index().index, y=topsubscribed_df['Category'].value_counts())
    n.set_xticklabels(topsubscribed_df['Category'].value_counts().index, rotation = 90)
    n.set(xlabel ='Started', ylabel = 'Values')
    #plt.show()

    st.pyplot(fig3)


col_1, col_2 = st.columns(2)

with col_1:
    st.subheader('Total view per category')
    st.write(topsubscribed_df.groupby("Category")["VideoViews"].sum().sort_values(ascending=False))
with col_2:
    st.subheader('Average views per category')
    st.write(topsubscribed_df.groupby("Category")["VideoViews"].mean().sort_values(ascending=False))


st.subheader('Average video count per sub range')


st.write( topsubscribed_df.groupby(topsubscribed_df['range_subs'])['Video Count'].transform('mean').unique())
fig4, ax = plt.subplots()

ax.plot(topsubscribed_df['range_subs'].unique(), topsubscribed_df.groupby(topsubscribed_df['range_subs'])['Video Count'].transform('mean').unique())
ax.set_xlabel("range")
ax.set_ylabel("media video count")

st.pyplot(fig4)

st.subheader('Average video views per sub range')


st.write( topsubscribed_df.groupby(topsubscribed_df['range_subs'])['VideoViews'].transform('mean').unique())
fig5, ax = plt.subplots()

ax.plot(topsubscribed_df['range_subs'].unique(), topsubscribed_df.groupby(topsubscribed_df['range_subs'])['VideoViews'].transform('mean').unique())
ax.set_xlabel("range")
ax.set_ylabel("media video views")
st.pyplot(fig5)

st.subheader('How do attributes influence each other?')

description2=''' A pair plot is a data visualization that plots pair-wise relationships between all the variables of a dataset. 
The diagonal line shows the distribution of values in that variable using a histogram. Each other cell shows the relationship as a scatterplot between the two variables at their intersection.'''

st.write(description2)

fig6=sns.set(style="ticks", color_codes=True)    
sns.pairplot(topsubscribed_df)
st.pyplot(fig6)

st.subheader('Subscribers depending on start year')
fig17=plt.figure(figsize=(15,10))
plt.scatter(topsubscribed_df['Started'], topsubscribed_df['Subscribers'], s=100, color='green', alpha=0.3)
plt.xlabel('Started')
plt.ylabel('Subscribers')
plt.grid()
st.pyplot(fig17)

st.subheader('Video views depending on subscribers')
fig18=plt.figure(figsize=(15,10))
plt.scatter(topsubscribed_df['Subscribers'], topsubscribed_df['VideoViews'],s=100, color='lime', alpha=0.1)
plt.xlabel('Subscribers')
plt.ylabel('VideoViews')
plt.grid()
st.pyplot(fig18)

st.subheader('Video count depending on start year')
fig19=plt.figure(figsize=(15,10))
plt.scatter(topsubscribed_df['Started'], topsubscribed_df['Video Count'],s=100, color='purple', alpha=0.1)
plt.xlabel('Started')
plt.ylabel('Video count')
plt.grid()
st.pyplot(fig19)

st.subheader('Correlation matrix')

description3='''A correlation matrix is a table showing correlation coefficients among your variables. 
Each cell in the table shows the correlation between two variables.
The correlation coefficient is a standardized metric that ranges from -1 and +1. Positive values indicate a positive correlation,
negative values indicate a negative correlation. 0 indicates no correlation.'''

st.write(description3)

fig7=plt.figure(figsize=(8,6))
sns.heatmap(topsubscribed_df.corr(), annot=True)
st.pyplot(fig7)

st.subheader('How video views change depending on the category')

fig8=plt.figure(figsize=(25,15))
sns.lineplot(x = 'Category', y = 'VideoViews', data=topsubscribed_df,linewidth=2, marker='*',markersize=25, color='blue')
plt.title('How video views change depending on the category', fontsize = 25)
plt.grid()
st.pyplot(fig8)

fig10=plt.figure(figsize=(30,10))
plt.subplot(121)

st.subheader('How does the sub range affect video views and video count?')
plt.title('Video views depending on range subs',fontsize = 20)
plt.ylabel("videoviews")
topsubscribed_df.groupby('range_subs')['VideoViews'].mean().sort_index().plot.bar(color = 'pink')
plt.grid()
plt.subplot(122)
plt.title('Video count depending on subscriber range',fontsize = 20)
plt.ylabel("video count")
topsubscribed_df.groupby('range_subs')['Video Count'].mean().sort_index().plot.bar(color = 'yellow')
plt.grid()
st.pyplot(fig10)

st.subheader('Distribution of video views and start year for each sub range')

description4='''A boxplot is a simple way of representing statistical data on a plot in which a rectangle is drawn to represent the second and third quartiles, usually with a vertical line inside to indicate the median value. 
The lower and upper quartiles are shown as horizontal lines either side of the rectangle.  
Outliers that differ significantly from the rest of the dataset may be plotted as individual points beyond the whiskers on the box-plot.'''

st.write(description4)

fig12, axes = plt.subplots(1,2, figsize=(18, 15))

sns.boxplot(ax=axes[ 0], data=topsubscribed_df, x='range_subs', y='VideoViews', palette='pastel')
sns.boxplot(ax=axes[ 1], data=topsubscribed_df, x='range_subs', y='Started', palette='pastel')
st.pyplot(fig12)

#create two new columns
topsubscribed_df = topsubscribed_df.assign(Years_on_YT = 2023 - topsubscribed_df['Started'])
topsubscribed_df = topsubscribed_df.assign(ThirtyM="0")

#get rid of outliers and create df_filtered
q = topsubscribed_df["Video Count"].quantile(0.75)
q_low = topsubscribed_df["Video Count"].quantile(0.25)
q_hi  = topsubscribed_df["Video Count"].quantile(0.75)

df_filtered = topsubscribed_df[(topsubscribed_df["Video Count"] < q_hi) & (topsubscribed_df["Video Count"] > q_low)]
df_filtered.reset_index(drop=True, inplace=True)

st.subheader('Df after adding two new columns and removing the outliers')

code2='''q = topsubscribed_df["Video Count"].quantile(0.75)
q_low = topsubscribed_df["Video Count"].quantile(0.25)
q_hi  = topsubscribed_df["Video Count"].quantile(0.75)

df_filtered = topsubscribed_df[(topsubscribed_df["Video Count"] < q_hi) & (topsubscribed_df["Video Count"] > q_low)]
df_filtered.reset_index(drop=True, inplace=True)'''

show_code2=st.sidebar.checkbox('Show code2')
if show_code2:
    st.subheader('Outlier removal')
    st.code(code2, language='python')


st.write(df_filtered)

show_infos=st.checkbox('Show info() of df after removing outliers')
if show_infos:
    st.subheader('df_filtered.info()')


    buff = io.StringIO()
    df_filtered.info(buf=buff)
    n = buff.getvalue()
    st.text(n)

st.subheader('Boxplot before vs after removing outliers')
fig16, axes = plt.subplots(1,2, figsize=(20, 15))
sns.boxplot(ax=axes[ 0], data=topsubscribed_df, x='range_subs', y='Video Count', palette='Paired')
sns.boxplot(ax=axes[ 1], data=df_filtered, x='range_subs', y='Video Count', palette='Paired')
st.pyplot(fig16)

#drop three columns and create df2
df_filtered = df_filtered.drop(['Category','Youtube Channel', 'range_subs'], axis=1)

st.subheader('Df after dropping Category, Youtube Channel and range_subs')
st.write(df_filtered)

show_infos=st.checkbox('Show info() of df after dropping columns')
if show_infos:
    st.subheader('df_filtered.info()')


    buff = io.StringIO()
    df_filtered.info(buf=buff)
    n = buff.getvalue()
    st.text(n)

st.subheader('Correlation matrix of the new df')
fig13=plt.figure(figsize=(8,6))
sns.heatmap(df_filtered.corr(), annot=True)
st.pyplot(fig13)


for i in range (len(df_filtered['ThirtyM'])):
    if((df_filtered.at[(i), 'Subscribers']) > 30000000):
        df_filtered.at[(i), 'ThirtyM'] = 1
    else:
        df_filtered.at[(i), 'ThirtyM'] = 0

si = df_filtered[df_filtered.ThirtyM == 1] # >30M subs
no = df_filtered[df_filtered.ThirtyM == 0] # <30M subs  

st.subheader('Replace values of ThirtyM with 1 if the number of subscribers is >=30M or with 0 if it is <30M')

for i in range (len(df_filtered['ThirtyM'])):
    if((df_filtered.at[(i), 'Subscribers']) >= 30000000):
        df_filtered.at[(i), 'ThirtyM'] = 1
    else:
        df_filtered.at[(i), 'ThirtyM'] = 0

code3='''for i in range (len(df_filtered['ThirtyM'])):
    if((df_filtered.at[(i), 'Subscribers']) >= 30000000):
        df_filtered.at[(i), 'ThirtyM'] = 1
    else:
        df_filtered.at[(i), 'ThirtyM'] = 0'''

show_code3=st.sidebar.checkbox('Show code3')
if show_code3:
    st.code(code3, language='python')

st.write(df_filtered)
#MODEL

st.header('**Implementation of Gaussian Naive Bayes**')

model = GaussianNB()
y=df_filtered.ThirtyM
choices = st.multiselect('Select features', ['VideoViews','Rank','Started'])
test_size = st.slider('Test size: ', min_value=0.1, max_value=0.9, step =0.1)
if len(choices) > 0 and st.button('RUN MODEL'):
    with st.spinner('Training...'):
        x = df_filtered[choices]
        y=y.astype('int')
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=2)

        x_train = x_train.to_numpy().reshape(-1, len(choices))
        model.fit(x_train, y_train)

        x_test = x_test.to_numpy().reshape(-1, len(choices))
        y_pred = model.predict(x_test)

        accuracy = accuracy_score(y_test, y_pred)

        st.write(f'Accuracy = {accuracy:.2f}')

