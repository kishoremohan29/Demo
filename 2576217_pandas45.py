#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#4


# In[ ]:


get_ipython().run_line_magic('pinfo', 'years')


# In[2]:


import pandas as pd

# Replace 'cast.csv' with the actual path to your CSV file
file_path = 'cast.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Filter the DataFrame to include only rows with "Superman" or "Batman" characters
superman_batman_roles = df[df['character'].isin(['Superman', 'Batman'])]

# Group the data by 'year' and 'character' and count the number of each character in each year
character_counts = superman_batman_roles.groupby(['year', 'character']).size().unstack(fill_value=0)

# Determine the years where the count of "Superman" characters is greater than "Batman" characters
superman_years = character_counts[character_counts['Superman'] > character_counts['Batman']]

# Count the number of "Superman years"
num_superman_years = len(superman_years)

# Print the result
print("Number of 'Superman years' in film history:", num_superman_years)


# In[ ]:


get_ipython().run_line_magic('pinfo', 'characters')


# In[3]:


import pandas as pd

# Replace 'cast.csv' with the actual path to your CSV file
file_path = 'cast.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Filter the DataFrame to include only rows with "Superman" or "Batman" characters
superman_batman_roles = df[df['character'].isin(['Superman', 'Batman'])]

# Group the data by 'year' and 'character' and count the number of each character in each year
character_counts = superman_batman_roles.groupby(['year', 'character']).size().unstack(fill_value=0)

# Determine the years where the count of "Batman" characters is greater than "Superman" characters
batman_years = character_counts[character_counts['Batman'] > character_counts['Superman']]

# Count the number of "Batman years"
num_batman_years = len(batman_years)

# Print the result
print("Number of 'Batman years' in film history with more Batman characters than Superman characters:", num_batman_years)


# In[ ]:


3.Plot the number of actor roles each year and the number of actress roles each year over the history of film


# In[5]:


import pandas as pd
import matplotlib.pyplot as plt

# Replace 'cast.csv' with the actual path to your CSV file
file_path = 'cast.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Filter the DataFrame to include only rows with 'type' as "actor" or "actress"
actor_actress_roles = df[df['type'].isin(['actor', 'actress'])]

# Group the data by 'year' and 'type' and count the number of roles of each type in each year
roles_by_year_and_type = actor_actress_roles.groupby(['year', 'type']).size().unstack(fill_value=0)

# Create separate DataFrames for actor and actress roles
actor_roles = roles_by_year_and_type['actor']
actress_roles = roles_by_year_and_type['actress']

# Plot the number of actor and actress roles each year
plt.figure(figsize=(12, 6))
plt.plot(actor_roles.index, actor_roles.values, label='Actor Roles', color='blue')
plt.plot(actress_roles.index, actress_roles.values, label='Actress Roles', color='pink')
plt.xlabel('Year')
plt.ylabel('Number of Roles')
plt.title('Number of Actor and Actress Roles Each Year')
plt.legend()
plt.grid()
plt.show()


# In[ ]:


4.Plot the number of actor roles each year and the number of actress roles each year, but this time as a kind='area' plot.


# In[6]:


import pandas as pd
import matplotlib.pyplot as plt

# Replace 'cast.csv' with the actual path to your CSV file
file_path = 'cast.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Filter the DataFrame to include only rows with 'type' as "actor" or "actress"
actor_actress_roles = df[df['type'].isin(['actor', 'actress'])]

# Group the data by 'year' and 'type' and count the number of roles of each type in each year
roles_by_year_and_type = actor_actress_roles.groupby(['year', 'type']).size().unstack(fill_value=0)

# Create separate DataFrames for actor and actress roles
actor_roles = roles_by_year_and_type['actor']
actress_roles = roles_by_year_and_type['actress']

# Plot the number of actor and actress roles each year as an area plot
plt.figure(figsize=(12, 6))
plt.fill_between(actor_roles.index, actor_roles.values, label='Actor Roles', color='blue', alpha=0.5)
plt.fill_between(actress_roles.index, actress_roles.values, label='Actress Roles', color='pink', alpha=0.5)
plt.xlabel('Year')
plt.ylabel('Number of Roles')
plt.title('Number of Actor and Actress Roles Each Year (Area Plot)')
plt.legend()
plt.grid()
plt.show()


# In[ ]:


5.Plot the difference between the number of actor roles each year and the number of actress roles each year over the history of film.


# In[7]:


import pandas as pd
import matplotlib.pyplot as plt

# Replace 'cast.csv' with the actual path to your CSV file
file_path = 'cast.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Filter the DataFrame to include only rows with 'type' as "actor" or "actress"
actor_actress_roles = df[df['type'].isin(['actor', 'actress'])]

# Group the data by 'year' and 'type' and count the number of roles of each type in each year
roles_by_year_and_type = actor_actress_roles.groupby(['year', 'type']).size().unstack(fill_value=0)

# Calculate the difference between the number of actor roles and actress roles each year
difference_roles = roles_by_year_and_type['actor'] - roles_by_year_and_type['actress']

# Plot the difference between actor and actress roles each year
plt.figure(figsize=(12, 6))
plt.plot(difference_roles.index, difference_roles.values, label='Difference (Actor - Actress)', color='green')
plt.xlabel('Year')
plt.ylabel('Difference in Number of Roles')
plt.title('Difference Between Actor and Actress Roles Each Year')
plt.legend()
plt.grid()
plt.show()


# In[ ]:


6.Plot the fraction of roles that have been 'actor' roles each year in the history of film.


# In[8]:


import pandas as pd
import matplotlib.pyplot as plt

# Replace 'cast.csv' with the actual path to your CSV file
file_path = 'cast.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Filter the DataFrame to include only rows with 'type' as "actor" or "actress"
actor_actress_roles = df[df['type'].isin(['actor', 'actress'])]

# Group the data by 'year' and 'type' and count the number of roles of each type in each year
roles_by_year_and_type = actor_actress_roles.groupby(['year', 'type']).size().unstack(fill_value=0)

# Calculate the fraction of 'actor' roles each year
total_roles_each_year = roles_by_year_and_type['actor'] + roles_by_year_and_type['actress']
fraction_actor_roles = roles_by_year_and_type['actor'] / total_roles_each_year

# Plot the fraction of 'actor' roles each year
plt.figure(figsize=(12, 6))
plt.plot(fraction_actor_roles.index, fraction_actor_roles.values, label='Fraction of Actor Roles', color='blue')
plt.xlabel('Year')
plt.ylabel('Fraction of Actor Roles')
plt.title('Fraction of Actor Roles Each Year')
plt.legend()
plt.grid()
plt.show()


# In[ ]:


7.Plot the fraction of supporting (n=2) roles that have been 'actor' roles each year in the history of film.


# In[9]:


import pandas as pd
import matplotlib.pyplot as plt

# Replace 'cast.csv' with the actual path to your CSV file
file_path = 'cast.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Filter the DataFrame to include only rows with 'type' as "actor," 'n' as 2 (supporting roles), and valid 'year'
supporting_actor_roles = df[(df['type'] == 'actor') & (df['n'] == 2) & ~df['year'].isna()]

# Group the data by 'year' and 'type' and count the number of supporting roles of each type in each year
supporting_roles_by_year_and_type = supporting_actor_roles.groupby(['year', 'type']).size().unstack(fill_value=0)

# Calculate the fraction of 'actor' supporting roles each year
total_supporting_roles_each_year = (
    supporting_roles_by_year_and_type['actor'] + supporting_roles_by_year_and_type['actor']
)
fraction_actor_supporting_roles = supporting_roles_by_year_and_type['actor'] / total_supporting_roles_each_year

# Plot the fraction of 'actor' supporting roles each year
plt.figure(figsize=(12, 6))
plt.plot(fraction_actor_supporting_roles.index, fraction_actor_supporting_roles.values, label='Fraction of Actor Supporting Roles', color='blue')
plt.xlabel('Year')
plt.ylabel('Fraction of Actor Supporting Roles')
plt.title('Fraction of Actor Supporting Roles Each Year')
plt.legend()
plt.grid()
plt.show()


# In[ ]:


8.Build a plot with a line for each rank n=1 through n=3, where the line shows what fraction of that rank's roles were 'actor' roles for each year in the history of film.


# In[10]:


import numpy as np
import matplotlib.pyplot as plt

# Load the data
data = np.loadtxt("film_history.csv", delimiter=",")

# Create a dictionary to store the fraction of actor roles for each rank and year
rank_actor_fraction = {}
for rank in range(1, 4):
    rank_actor_fraction[rank] = {}
    for year in data[:, 0]:
        rank_actor_fraction[rank][year] = np.sum(data[data[:, 1] == rank, 2] == "actor") / np.sum(data[data[:, 1] == rank, 2] != "")

# Create a plot
fig, ax = plt.subplots()

# Plot the lines for each rank
for rank in range(1, 4):
    years = np.array(list(rank_actor_fraction[rank].keys()))
    fractions = np.array(list(rank_actor_fraction[rank].values()))
    ax.plot(years, fractions, label=f"Rank {rank}")

# Set the plot title and labels
ax.set_title("Fraction of actor roles for each rank in film history")
ax.set_xlabel("Year")
ax.set_ylabel("Fraction of actor roles")

# Add a legend
ax.legend()

# Show the plot
plt.show()


# In[ ]:





# In[ ]:


1.Make a bar plot of the months in which movies with "Christmas" in their title tend to be released in the USA.


# In[12]:


import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
df = pd.read_csv('release_dates.csv')

# Filter rows where the title contains "Christmas" and the country is USA
christmas_movies_usa = df[(df['title'].str.contains('Christmas', case=False, na=False)) & (df['country'] == 'USA')]

# Extract the release month from the 'year' column and create a new column 'release_month'
christmas_movies_usa['release_month'] = pd.to_datetime(christmas_movies_usa['year'], format='%Y').dt.month

# Count the number of movies released in each month
monthly_counts = christmas_movies_usa['release_month'].value_counts().sort_index()

# Create a bar plot of the months
plt.figure(figsize=(10, 6))
monthly_counts.plot(kind='bar', color='skyblue')
plt.title('Movies with "Christmas" in Title Released in the USA by Month')
plt.xlabel('Month')
plt.ylabel('Number of Movies')
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.show()


# In[ ]:


2.Make a bar plot of the months in which movies whose titles start with "The Hobbit" are realeased in the USA.


# In[13]:


import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
df = pd.read_csv('release_dates.csv')

# Filter rows where the title starts with "The Hobbit" and the country is USA
hobbit_movies_usa = df[(df['title'].str.startswith('The Hobbit', na=False)) & (df['country'] == 'USA')]

# Extract the release month from the 'year' column and create a new column 'release_month'
hobbit_movies_usa['release_month'] = pd.to_datetime(hobbit_movies_usa['year'], format='%Y').dt.month

# Count the number of movies released in each month
monthly_counts = hobbit_movies_usa['release_month'].value_counts().sort_index()

# Create a bar plot of the months
plt.figure(figsize=(10, 6))
monthly_counts.plot(kind='bar', color='skyblue')
plt.title('Movies Starting with "The Hobbit" Released in the USA by Month')
plt.xlabel('Month')
plt.ylabel('Number of Movies')
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.show()


# In[ ]:


3.Make a bar plot of the day of the week which movies whose titles start with "Romance" are realeased in the USA


# In[14]:


import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
df = pd.read_csv('release_dates.csv')

# Filter rows where the title starts with "Romance" and the country is USA
romance_movies_usa = df[(df['title'].str.startswith('Romance', na=False)) & (df['country'] == 'USA')]

# Convert the 'date' column to datetime to extract the day of the week
romance_movies_usa['date'] = pd.to_datetime(romance_movies_usa['date'])

# Extract the day of the week and create a new column 'day_of_week'
romance_movies_usa['day_of_week'] = romance_movies_usa['date'].dt.day_name()

# Count the number of movies released on each day of the week
day_of_week_counts = romance_movies_usa['day_of_week'].value_counts().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

# Create a bar plot of the day of the week
plt.figure(figsize=(10, 6))
day_of_week_counts.plot(kind='bar', color='skyblue')
plt.title('Movies Starting with "Romance" Released in the USA by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Number of Movies')
plt.xticks(rotation=45)
plt.show()


# In[ ]:


4.Make a bar plot of the day of the week which movies with "Action" in their title tend to be realeased in the USA.


# In[15]:


import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
df = pd.read_csv('release_dates.csv')

# Filter rows where the title contains "Action" (case-insensitive) and the country is USA
action_movies_usa = df[(df['title'].str.contains('Action', case=False, na=False)) & (df['country'] == 'USA')]

# Convert the 'date' column to datetime to extract the day of the week
action_movies_usa['date'] = pd.to_datetime(action_movies_usa['date'])

# Extract the day of the week and create a new column 'day_of_week'
action_movies_usa['day_of_week'] = action_movies_usa['date'].dt.day_name()

# Count the number of movies released on each day of the week
day_of_week_counts = action_movies_usa['day_of_week'].value_counts().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

# Create a bar plot of the day of the week
plt.figure(figsize=(10, 6))
day_of_week_counts.plot(kind='bar', color='skyblue')
plt.title('Movies with "Action" in Title Released in the USA by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Number of Movies')
plt.xticks(rotation=45)
plt.show()


# In[ ]:


get_ipython().run_line_magic('pinfo', 'USA')


# In[16]:


import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('release_dates.csv')

# Filter rows where the title contains "Judi Dench" (case-insensitive) and the year is in the 1990s
judi_dench_movies_1990s = df[(df['title'].str.contains('Judi Dench', case=False, na=False)) & (df['year'] >= 1990) & (df['year'] <= 1999)]

# Filter further to select only movies released in the USA
judi_dench_movies_usa = judi_dench_movies_1990s[judi_dench_movies_1990s['country'] == 'USA']

# Display the release date of each movie
print(judi_dench_movies_usa[['title', 'date']])


# In[ ]:


get_ipython().run_line_magic('pinfo', 'USA')


# In[17]:


import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('release_dates.csv')

# Filter rows where the title contains "Judi Dench" (case-insensitive) and the country is USA
judi_dench_movies_usa = df[(df['title'].str.contains('Judi Dench', case=False, na=False)) & (df['country'] == 'USA')]

# Convert the 'date' column to datetime to extract the release month
judi_dench_movies_usa['date'] = pd.to_datetime(judi_dench_movies_usa['date'])

# Extract the release month and create a new column 'release_month'
judi_dench_movies_usa['release_month'] = judi_dench_movies_usa['date'].dt.month

# Count the number of movies released in each month
monthly_counts = judi_dench_movies_usa['release_month'].value_counts().sort_index()

# Create a bar plot of the release months
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
monthly_counts.plot(kind='bar', color='skyblue')
plt.title('Release Months of Films with Judi Dench in the USA')
plt.xlabel('Month')
plt.ylabel('Number of Films')
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.show()


# In[ ]:


get_ipython().run_line_magic('pinfo', 'USA')


# In[18]:


import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
df = pd.read_csv('release_dates.csv')

# Filter rows where the title contains "Tom Cruise" (case-insensitive) and the country is USA
tom_cruise_movies_usa = df[(df['title'].str.contains('Tom Cruise', case=False, na=False)) & (df['country'] == 'USA')]

# Convert the 'date' column to datetime to extract the release month
tom_cruise_movies_usa['date'] = pd.to_datetime(tom_cruise_movies_usa['date'])

# Extract the release month and create a new column 'release_month'
tom_cruise_movies_usa['release_month'] = tom_cruise_movies_usa['date'].dt.month

# Count the number of movies released in each month
monthly_counts = tom_cruise_movies_usa['release_month'].value_counts().sort_index()

# Create a bar plot of the release months
plt.figure(figsize=(10, 6))
monthly_counts.plot(kind='bar', color='skyblue')
plt.title('Release Months of Films with Tom Cruise in the USA')
plt.xlabel('Month')
plt.ylabel('Number of Films')
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.show()

