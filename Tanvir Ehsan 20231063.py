#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


df = sns.load_dataset('tips')
df


# In[15]:


df.isnull().sum()


# In[8]:


df.describe


# In[6]:


import seaborn as sns


# In[12]:


df.corr()


# In[18]:


import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns


# In[20]:


sns.histplot(df['total_bill'], kde=True )


# In[19]:


sns.heatmap(correlation_matrix,cmap="cool")


# In[21]:


sns.scatterplot(df['total_bill'], df['sex'])


# In[22]:


correlation_matrix = df.corr()
sns.heatmap(correlation_matrix)


# In[23]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_scor


# In[24]:


tips = sns.load_dataset('tips')


# In[25]:


day_table_count = tips.groupby('day')['size'].count()
hardest_day_to_work = day_table_count.idxmax()
print("The hardest day to work is:", hardest_day_to_work)


# In[29]:


# Sum of tips per day
day_tips_sum = tips.groupby('day')['tip'].sum()

# Tip percent per day
day_tips_percent = tips.groupby('day').apply(lambda x: (x['tip'].sum() / x['total_bill'].sum()) * 100)

best_day_by_tips_sum = day_tips_sum.idxmax()
best_day_by_tips_percent = day_tips_percent.idxmax()
print("The best day to work based on maximum total tips is:", best_day_by_tips_sum)
print("The best day to work based on maximum tip percentage is:", best_day_by_tips_percent)
get_ipython().set_next_input('Who eats more (and tips more)? Smokers or non-smokers');get_ipython().run_line_magic('pinfo', 'smokers')
smoker_vs_nonsmoker_stats = tips.groupby('smoker').agg({'total_bill': 'mean', 'tip': 'mean'})
print(smoker_vs_nonsmoker_stats)

average_tip_by_table_size = tips.groupby('size')['tip'].mean()


print("We can examine the average tip amount for different table sizes to see if there is any trend.")
print(average_tip_by_table_size)


# In[32]:


import seaborn as sns


# In[33]:


plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(tips['total_bill'], bins=20, kde=True)
plt.xlabel('Total Bill')
plt.subplot(1, 2, 2)
sns.histplot(tips['tip'], bins=20, kde=True)
plt.xlabel('Tip Amount')
plt.show()


# In[34]:


sns.scatterplot(data=tips, x='total_bill', y='tip', hue='smoker', style='time')
plt.title("Scatter plot between 'total_bill' and 'tip'")
plt.show()


# In[35]:


correlation_matrix = tips[['total_bill', 'tip', 'size']].corr()

# Create a heatmap of the correlation matrix
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap")
plt.show()


# In[36]:


df.isnull().sum()


# In[37]:


tips = sns.load_dataset('tips')

# Check for missing values and drop instances with missing values
print("Number of missing values in each column:")
print(tips.isnull().sum())

tips = tips.dropna()  # Drop instances with missing values

# Reduce redundant information by removing unnecessary columns
# In this example, let's remove the 'size' column as it may not be relevant for our analysis
tips = tips.drop('size', axis=1)


# In[43]:


# Define the features (X) and the target variable (y)
X = tips.drop('tip', axis=1)
y = tips['tip']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=26) # Using test_size=0.3 for 70%-30% split

# Display the shapes of the resulting datasets
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# In[45]:


X_train


# In[47]:


y_train


# In[49]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)


# In[50]:


predictions=model.predict(X_test)


# In[51]:


new = np.array([0,1,3,1,0,0]).reshape(1,-1)


# In[ ]:





# In[ ]:




