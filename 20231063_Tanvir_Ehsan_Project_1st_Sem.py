#!/usr/bin/env python
# coding: utf-8

# # Roll: 20231091
# Mohammad Taslim Mazumder Sohel
# <br>Batch: 10th

# **Project Plan:**
# <br> I want to predict house rent for Dhaka city base on following labels,
# 
# *   Location
# *   Area
# *   No. of Beds
# *   No. of Baths

# In[105]:


# Import related python library
import pandas as pd
import numpy as np
from decimal import Decimal
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
matplotlib.rcParams["figure.figsize"] = (10,6)
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# # Load data from source CSV file.

# In[106]:


# Load data from source CSV file.
df = pd.read_csv("houserent.csv")

# Print firs 5 rows
df.head()


# # Understanding the Detail of Dataset

# In[107]:


df.shape

print("\nDisplay info from Dataset\n",df.info())

print("\nData set type: ", type(df))

print("\nData types: \n", df.dtypes)

print("\n List of column name: ",df.columns)

print("\nCount empty house rent fields: ",display(df['Price'].notnull().sum()))
print("\nTotal Number of Locations: ",len(df['Location'].unique()))


# # Scrub Errelevant Data

# In[108]:


#removing the unnamed column
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]


# In[109]:


df.head()


# # Checking and Handle Missing Data

# In[110]:


df.isnull().sum()


# # Fix structral data

# In[111]:


df['Area']=df['Area'].str.replace(',','')


# In[26]:


df.head()


# In[27]:


#removing the sqft from the area and making them int values
df['Area'] = df['Area'].apply(lambda x: int(x.split(' ')[0]))


# In[28]:


df.head()


# # Scrub for Irrelevant Data
# 

# In[29]:


df.Bed.unique()


# In[30]:


df.Bath.unique()


# In[31]:


df[df.Bed>5]


# In[32]:


df[df.Bath>6] #8 baths!! noice but why?????


# In[33]:


df1 = df.copy()


# # Standardize

# In[34]:


#I wanted to make the price Thousand/Lakh values to a float values
def price_float(x):
    y = x.split(' ')[1]
    if y == "Thousand":
        return float(x.split(' ')[0]) * 1000
    else:
        return float(x.split(' ')[0]) * 100000


# In[35]:


df1.Price = df1.Price.apply(price_float)


# In[36]:


# Check a single value as, 304 no record
df1.loc[304]


# In[37]:


df1['price_per_sqft'] = df1['Price']/df1['Area']


# In[38]:


df1.head()


# In[39]:


len(df1['Location'].unique())


# In[40]:


#removing the comma in the Area column
df1['Location']=df1['Location'].str.replace(',','')


# In[41]:


df1.head()


# In[42]:


#Finding the locations with most number of houses
location_count = df1['Location'].value_counts(ascending=False)
location_count.head(30)


# In[43]:


#Finding locations with less than 10 houses
len(location_count[location_count <= 10])


# In[44]:


len(df1['Location'].unique())


# In[45]:


#Keeping the locations with less houses together
location_count_under_10 = location_count[location_count <= 10]


# In[46]:


#now leveling them as 'other'
df1.Location = df1.Location.apply(lambda x: 'other' if x in location_count_under_10 else x)


# In[47]:


#now we have less unique values which will incress our accuracy
len(df1['Location'].unique())


# In[48]:


#now founding the unrealistic Area to Bed ratio according to my Civil engineer friend
df1[df1.Area/df1.Bed<300].head(10)


# In[49]:


df1.shape


# In[50]:


#removing them from our df
df2 = df1[~(df1.Area/df1.Bed<300)]
df2.shape


# # Measures of central tendency

# In[51]:


#Lets see the description of the 'price_per_sqft'
df2.price_per_sqft.describe()


# # Outlier Detection Using IQR, and Extraction from Dataset

# In[66]:


def drop_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('Location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        # We know that, Lower Outlier < (Mean-Standard Deviation)
        # and Higher Outlier > (Mean+Standard Deviation)
        # So, we only keep data which gretar than Lower Outlier and samaller than Higher Outlier.
        reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out
df3 = drop_outliers(df2)
df3.shape


# # Data Visualization for Better Understanding the Datasets

# In[67]:


#Building a scatter chart
def plot_scatter_chart(df,Location):
    bed2 = df[(df.Location==Location) & (df.Bed==2)]
    bed3 = df[(df.Location==Location) & (df.Bed==3)]
    matplotlib.rcParams['figure.figsize'] = (15,10)
    plt.scatter(bed2.Area,bed2.Price,color='blue',label='2 Bed', s=50)
    plt.scatter(bed3.Area,bed3.Price,marker='+', color='red',label='3 Bed', s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price (BDT)")
    plt.title(Location)
    plt.legend()

plot_scatter_chart(df3,"Shantinagar Dhaka")


# In[68]:


plot_scatter_chart(df3,"Mirpur Dhaka")


# In[69]:


matplotlib.rcParams["figure.figsize"] = (10,10)
plt.hist(df3.price_per_sqft,rwidth=0.5)
plt.xlabel("Price Per Square Feet")
plt.ylabel("Count")


# In[70]:


df3[df3.Bath>df3.Bed+1] # this number of bed to bath ratio seems normal so not droping them


# # Bulding categorical data

# In[71]:


dummies = pd.get_dummies(df3.Location)
dummies.head()


# In[72]:


df4 = pd.concat([df3,dummies.drop('other',axis='columns')],axis='columns')
df4.head()


# In[73]:


df5 = df4.drop('Location',axis='columns')
df5.head()


# ## Model Building

# In[79]:


X = df5.drop(['Price','price_per_sqft'],axis='columns') #Had to drop 'price_per_sqft' because it was confusing the model and giving random negative predictions
X.head()


# In[75]:


y = df5.Price
y.head()


# # Split Dataset into Training and Test data.

# In[93]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=15)


# # We Prepare 4 Machine Learning Models Using Training Data

# # 1) Linear Regression

# In[98]:



# Create and fit the model
lr_clf = LinearRegression()
lr_clf.fit(X_train, y_train)

# Predict values
y_pred = lr_clf.predict(X_test)

p = lr_clf.score(X_test,y_test)
print("%.2f" % (p*100))


# In[78]:


cv = ShuffleSplit(n_splits=5, test_size=0.25, random_state=10)

cross_val_score(LinearRegression(), X, y, cv=cv)


# # 2) Ridge Regression

# In[80]:


from sklearn.linear_model import Ridge
rdg_clf = Ridge(alpha=1.0)
rdg_clf.fit(X_train,y_train)
p = rdg_clf.score(X_test,y_test)
print("%.2f" % (p*100))


# In[81]:


cv = ShuffleSplit(n_splits=5, test_size=0.25, random_state=10)

cross_val_score(Ridge(), X, y, cv=cv)


# # 3) Bayesian Regression

# In[82]:


from sklearn import linear_model
br_clf = linear_model.BayesianRidge()
br_clf.fit(X_train,y_train)
p = br_clf.score(X_test,y_test)
print("%.2f" % (p*100))


# In[83]:


cv = ShuffleSplit(n_splits=5, test_size=0.25, random_state=10)

cross_val_score(linear_model.BayesianRidge(), X, y, cv=cv)


# # 4) Lasso Regression

# In[84]:


from sklearn.linear_model import Lasso
las_clf = Lasso(alpha=0.1)
las_clf.fit(X_train,y_train)
p = las_clf.score(X_test,y_test)
print("%.2f" % (p*100))


# In[85]:


cv = ShuffleSplit(n_splits=5, test_size=0.25, random_state=10)

cross_val_score(Lasso(), X, y, cv=cv)


# # Lets Test How Well the Model Works

# In[86]:


def predict_price(Location,Area,Bed,Bath):
    index = np.where(X.columns == Location)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = Area
    x[1] = Bed
    x[2] = Bath
    if index >= 0:
        x[index] = 1

    return rdg_clf.predict([x])[0] #Ridge Regression got the best accuracy


# In[88]:


#My family used to live in this area in a 1100 sqft appertment with 3 beds and 3 baths. And the rent was 16500 Taka
predict_price('Matikata Cantonment Dhaka',1250, 3, 3)


# In[ ]:


predict_price('Mirpur Dhaka',1250, 3, 3)


# In[ ]:


predict_price('Gulshan 2 Gulshan Dhaka',2000, 3, 3)


# In[ ]:


predict_price('Badda Dhaka',1000, 2, 2)

