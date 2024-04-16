#!/usr/bin/env python
# coding: utf-8

# In[4]:


#Q1


# In[7]:


import pandas as pd

users = pd.read_csv("https://raw.githubusercontent.com/justmarkham/DAT8/master/data/u.user", sep='|')
 
mean_age_per_occupation = users.groupby('occupation')['age'].mean()

print('mean_age_per_occupation:\n', mean_age_per_occupation)

users['is_male'] = users['gender'].apply(lambda x: 1 if x == 'M' else 0)
male_ratio = users.groupby('occupation')['is_male'].mean().sort_values(ascending=False)

print('male_ratio:\n', male_ratio)

min_max_age = users.groupby('occupation')['age'].agg([min, max])

print('min_max_age:\n', min_max_age)

mean_age_per_occupation_sex = users.groupby(['occupation', 'gender'])['age'].mean()

print('mean_age_per_occupation_sex:\n', mean_age_per_occupation_sex)

gender_counts = users.groupby(['occupation', 'gender']).size().unstack().fillna(0)
total_counts = users.groupby('occupation')['gender'].count()
gender_percentage = (gender_counts.div(total_counts, axis=0) * 100).rename(columns={'F': 'Female%', 'M': 'Male%'})

print('gender_percentage:\n', gender_percentage)


# In[ ]:


#Q2


# In[8]:


import pandas as pd

euro12 = pd.read_csv('https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/02_Filtering_%26_Sorting/Euro12/Euro_2012_stats_TEAM.csv')

goals = euro12['Goals']

number_of_teams = len(euro12['Team'].unique())

print('number_of_teams:', number_of_teams)

number_of_columns = euro12.shape[1]

print('number_of_columns:', number_of_columns)

discipline = euro12[['Team', 'Yellow Cards', 'Red Cards']]

print('discipline:\n', discipline)

discipline_sorted = discipline.sort_values(by=['Red Cards', 'Yellow Cards'], ascending=[False, False])

print('discipline_sorted:\n', discipline_sorted)

average_yellow_cards = discipline['Yellow Cards'].mean()

print('average_yellow_cards:\n', average_yellow_cards)

teams_more_than_6_goals = euro12[euro12['Goals'] > 6]

print('teams_more_than_6_goals:\n', teams_more_than_6_goals)

teams_start_with_g = euro12[euro12['Team'].str.startswith('G')]

print('teams_start_with_G:\n', teams_start_with_g)

first_7_columns = euro12.iloc[:, :7]

print('first_7_columns:\n', first_7_columns)

columns_except_last_3 = euro12.iloc[:, :-3]

print('columns_except_last_3:\n', columns_except_last_3)

shooting_accuracy_specific = euro12.loc[euro12['Team'].isin(['England', 'Italy', 'Russia']), ['Team', 'Shooting Accuracy']]

print('shooting_accuracy_specific:\n', shooting_accuracy_specific)


# In[ ]:


#Q3


# In[9]:


import pandas as pd
import numpy as np

np.random.seed(0)
s1 = pd.Series(np.random.randint(1, 5, 100))
s2 = pd.Series(np.random.randint(1, 4, 100))
s3 = pd.Series(np.random.randint(10000, 30001, 100))

df = pd.DataFrame({'bedrs': s1, 'bathrs': s2, 'price_sqr_meter': s3})

bigcolumn = pd.concat([s1, s2, s3], ignore_index=True).to_frame('bigcolumn')

print(df)
print(bigcolumn)


# In[ ]:


#Q4


# In[12]:


import pandas as pd
from datetime import datetime

data = pd.read_csv('./wind.txt', delim_whitespace=True)

date_series = data['Yr'].astype(str) + '-' + data['Mo'].astype(str) + '-' + data['Dy'].astype(str)

data['Yr_Mo_Dy'] = pd.to_datetime(date_series, format='%y-%m-%d', errors='coerce')

data['Yr_Mo_Dy'] = data['Yr_Mo_Dy'].apply(lambda x: x if x.year < 2000 else datetime(x.year - 100, x.month, x.day))

# Set the datetime as the index
data.set_index('Yr_Mo_Dy', inplace=True)

missing_values = data.isnull().sum()
print('missing_values:\n', missing_values)

non_missing_values = data.notnull().sum().sum()
print('non_missing_values:\n', non_missing_values)

mean_windspeed = data.mean().mean()
print('mean_windspeed:\n', mean_windspeed)

loc_stats = data.describe().T[['min', 'max', 'mean', 'std']]
print('loc_stats:\n', loc_stats)

day_stats = data.agg(['min', 'max', 'mean', 'std'], axis=1)
print('day_stats:\n', day_stats)

january_winds = data[data.index.month == 1].mean()
print('january_winds:\n', january_winds)

yearly_stats = data.resample('A').mean()
print('yearly_stats:\n', yearly_stats)

monthly_stats = data.resample('M').mean()
print('monthly_stats:\n', monthly_stats)

weekly_stats = data.resample('W').mean()
print('weekly_stats:\n', weekly_stats)

first_52_weeks_stats = data[:'1962'].resample('W').agg(['min', 'max', 'mean', 'std'])
print('first_52_weeks_stats:\n', first_52_weeks_stats)


# In[ ]:


#Q5


# In[13]:


import pandas as pd


chipo_data = pd.read_csv('https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv', sep='\t')


print("First 10 entries:\n", chipo_data.head(10))


observations = chipo_data.shape[0]
columns = chipo_data.shape[1]


print("Number of observations:", observations)
print("Number of columns:", columns)


print("Column names:", chipo_data.columns.tolist())


print("Index details:", chipo_data.index)


item_order_freq = chipo_data['item_name'].value_counts()
most_ordered_item_name = item_order_freq.idxmax()
most_ordered_item_qty = item_order_freq.max()
print("Most ordered item:", most_ordered_item_name)
print("Ordered", most_ordered_item_qty, "times")


if 'choice_description' in chipo_data.columns:
    choice_freq = chipo_data['choice_description'].value_counts()
    top_choice_description = choice_freq.idxmax()
    top_choice_qty = choice_freq.max()
    print("Most ordered choice:", top_choice_description)
    print("Ordered", top_choice_qty, "times")


chipo_data['item_price'] = chipo_data['item_price'].str.replace('$', '').astype(float)
print("Data type of item_price:", chipo_data['item_price'].dtype)


chipo_data['total_price'] = chipo_data['item_price'] * chipo_data['quantity']
total_revenue = chipo_data['total_price'].sum()
print("Total revenue: ${:.2f}".format(total_revenue))


total_orders = chipo_data['order_id'].nunique()
print("Total number of orders:", total_orders)


average_revenue = total_revenue / total_orders
print("Average revenue per order: ${:.2f}".format(average_revenue))


unique_items_sold = chipo_data['item_name'].nunique()
print("Number of different items sold:", unique_items_sold)


# In[ ]:


#Q6


# In[15]:


import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('./us-marriages-divorces-1867-2014.csv')

plt.figure(figsize=(10, 5))
plt.plot('Year', 'Marriages_per_1000', data=data, label='Marriages', marker='o', color='blue')
plt.plot('Year', 'Divorces_per_1000', data=data, label='Divorces', marker='x', color='red')

plt.title('Marriages and Divorces per 1000 in the U.S. (1867 - 2014)')
plt.xlabel('Year')
plt.ylabel('Counts per Capita')
plt.legend()
plt.grid(True)
plt.show()



# In[ ]:


#Q7


# In[16]:


import pandas as pd
import matplotlib.pyplot as plt

data_file = './us-marriages-divorces-1867-2014.csv'
data_frame = pd.read_csv(data_file)

target_years = [1900, 1950, 2000]
selected_data = data_frame[data_frame['Year'].isin(target_years)]

index_positions = list(range(len(target_years)))
bar_width = 0.4

fig, ax = plt.subplots(figsize=(10, 6))

marriage_bars = ax.bar([i - bar_width/2 for i in index_positions], selected_data['Marriages_per_1000'], 
                       bar_width, alpha=0.7, color='darkblue', label='Marriages')
divorce_bars = ax.bar([i + bar_width/2 for i in index_positions], selected_data['Divorces_per_1000'], 
                      bar_width, alpha=0.7, color='maroon', label='Divorces')

ax.set_xlabel('Selected Years')
ax.set_ylabel('Counts per 1000 Individuals')
ax.set_title('Marriages vs Divorces per 1000 People in the U.S. Across Selected Years')
ax.set_xticks(index_positions)
ax.set_xticklabels(target_years)

ax.legend()
ax.grid(True)

plt.show()


# In[ ]:


#Q8


# In[17]:


import pandas as pd
import matplotlib.pyplot as plt


actor_data = pd.read_csv('./actor_kill_counts.csv')

sorted_data = actor_data.sort_values('Count')


plt.figure(figsize=(10, 8))
plt.barh(sorted_data['Actor'], sorted_data['Count'], color='darkgreen')


plt.xlabel('Kill Count')
plt.ylabel('Actor')
plt.title('Deadliest Actors in Hollywood by Kill Count')
plt.grid(True)

plt.show()


# #Q9

# In[18]:


import pandas as pd
import matplotlib.pyplot as plt


emperor_data = pd.read_csv('./roman-emperor-reigns.csv')


cause_counts = emperor_data['Cause_of_Death'].value_counts()


colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'purple', 'orange', 'pink']


plt.figure(figsize=(10, 8))
plt.pie(cause_counts, labels=cause_counts.index, autopct='%1.1f%%', startangle=90, counterclock=False, colors=colors)
plt.title('Causes of Death of Roman Emperors')
plt.axis('equal')
plt.show()


# In[ ]:


#Q10


# In[20]:


import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('./arcade-revenue-vs-cs-doctorates.csv')


plt.figure(figsize=(10, 6))


cmap = plt.get_cmap('cool')

scatter_plot = plt.scatter('Total Arcade Revenue (billions)', 'Computer Science Doctorates Awarded (US)', 
                           c='Year', cmap=cmap, s=100, alpha=0.6, edgecolors='w', linewidth=1, data=data)


plt.xlabel('Total Arcade Revenue (billions)')
plt.ylabel('Computer Science Doctorates Awarded (US)')
plt.title('Arcade Revenue vs. CS PhDs (2000-2009)')


color_bar = plt.colorbar(scatter_plot)
color_bar.set_label('Year')


plt.grid(True)


plt.show()


# In[ ]:




