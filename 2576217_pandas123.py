#!/usr/bin/env python
# coding: utf-8

# # 1

# # 1 there are 5 unique movie titles listed in the dataframe.

# In[3]:


import pandas as pd

# Read the titles dataframe
df = pd.read_csv("titles.csv")

# Count the number of rows in the dataframe
num_movies = len(df)

# Print the number of movies
print(num_movies)


# # 2 What are the earliest two films listed in the titles dataframe

# In[4]:


import pandas as pd

# Replace 'titles.csv' with the actual path to your CSV file
file_path = 'titles.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Sort the DataFrame by the 'year' column in ascending order
sorted_df = df.sort_values(by='year')

# Get the first two rows (earliest two films) from the sorted DataFrame
earliest_films = sorted_df.head(2)

# Print the earliest two films
print("The earliest two films listed:")
print(earliest_films)


# # 3 How many movies have the title "Hamlet

# In[5]:


import pandas as pd

# Replace 'titles.csv' with the actual path to your CSV file
file_path = 'titles.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Filter the DataFrame to include only rows where the 'title' column contains "Hamlet"
hamlet_movies = df[df['title'] == 'Hamlet']

# Count the number of rows (movies) in the filtered DataFrame
number_of_hamlet_movies = len(hamlet_movies)

print("Number of movies with the title 'Hamlet':", number_of_hamlet_movies)


# # 4  How many movies have the title "North by Northwest"

# In[6]:


import pandas as pd

# Replace 'titles.csv' with the actual path to your CSV file
file_path = 'titles.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Filter the DataFrame to include only rows where the 'title' column contains "North by Northwest"
north_by_northwest_movies = df[df['title'] == 'North by Northwest']

# Count the number of rows (movies) in the filtered DataFrame
number_of_north_by_northwest_movies = len(north_by_northwest_movies)

print("Number of movies with the title 'North by Northwest':", number_of_north_by_northwest_movies)


# # 5 When was the first movie titled "Hamlet" made

# In[7]:


import pandas as pd

# Replace 'titles.csv' with the actual path to your CSV file
file_path = 'titles.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Filter the DataFrame to include only rows where the 'title' column contains "Hamlet"
hamlet_movies = df[df['title'] == 'Hamlet']

# Sort the filtered DataFrame by the 'year' column in ascending order
sorted_hamlet_movies = hamlet_movies.sort_values(by='year')

# Get the first row (the earliest "Hamlet" movie)
first_hamlet_movie = sorted_hamlet_movies.iloc[0]

# Extract the release year from the row
release_year = first_hamlet_movie['year']

print("The first movie titled 'Hamlet' was released in:", release_year)


# # 6 List all of the "Treasure Island" movies from earliest to most recent.

# In[8]:


import pandas as pd

# Create a list of the Treasure Island movies
treasure_island_movies = ["Treasure Island (1918)",
                          "Treasure Island (1920)",
                          "Treasure Island (1934)",
                          "Treasure Island (1950)",
                          "Treasure Island (1972)",
                          "Treasure Island (1973)",
                          "Treasure Island (1985)",
                          "Treasure Island (1999)"]

# Create a DataFrame
df = pd.DataFrame({"Movie": treasure_island_movies})

# Sort the DataFrame by release date
df = df.sort_values(by=['Movie'],ascending=True)

# Print the DataFrame
print(df)


# # 7 How many movies were made in the year 1950.

# In[9]:


import pandas as pd

# Replace 'titles.csv' with the actual path to your CSV file
file_path = 'titles.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Filter the DataFrame to include only rows where the 'year' column is equal to 1950
movies_1950 = df[df['year'] == 1950]

# Count the number of rows (movies) in the filtered DataFrame
number_of_movies_1950 = len(movies_1950)

print("Number of movies made in the year 1950:", number_of_movies_1950)


# # 8 How many movies were made in the year 1960

# In[10]:


import pandas as pd

# Replace 'titles.csv' with the actual path to your CSV file
file_path = 'titles.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Filter the DataFrame to include only rows where the 'year' column is equal to 1960
movies_1960 = df[df['year'] == 1960]

# Count the number of rows (movies) in the filtered DataFrame
number_of_movies_1960 = len(movies_1960)

print("Number of movies made in the year 1960:", number_of_movies_1960)


# # 9 How many movies were made from 1950 through 1959.

# In[11]:


# import pandas as pd

# Replace 'titles.csv' with the actual path to your CSV file
file_path = 'titles.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Filter the DataFrame to include only rows where the 'year' column is between 1950 and 1959
movies_1950s = df[(df['year'] >= 1950) & (df['year'] <= 1959)]

# Count the number of rows (movies) in the filtered DataFrame
number_of_movies_1950s = len(movies_1950s)

print("Number of movies made from 1950 through 1959:", number_of_movies_1950s)


# # 10 In what years has a movie titled "Batman" been released.

# In[12]:


import pandas as pd

# Replace 'titles.csv' with the actual path to your CSV file
file_path = 'titles.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Filter the DataFrame to include only rows where the 'title' column is "Batman"
batman_movies = df[df['title'] == 'Batman']

# Extract and print the unique release years of "Batman" movies
release_years = batman_movies['year'].unique()

print("Years in which a movie titled 'Batman' has been released:")
print(sorted(release_years))


# # 11.how many roles were there in the movie "inception"

# In[13]:


import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('titles.csv')

# Filter rows where the title is "Inception"
inception_roles = df[df['title'] == 'Inception']

# Count the number of roles in "Inception"
number_of_roles = len(inception_roles)

print(f"Number of roles in 'Inception': {number_of_roles}")


# # 12.how many roles in the movie "inception" are NOT ranked by an "n" value?

# In[17]:


import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('cast.csv')

# Filter rows where the title is "Inception"
inception_roles = df[df['title'] == 'Inception']

# Filter the rows where the "n" value is not specified (NaN)
inception_roles_not_ranked = inception_roles[pd.isnull(inception_roles['n'])]

# Count the number of roles in "Inception" that are not ranked by an "n" value
number_of_roles_not_ranked = len(inception_roles_not_ranked)

print(f"Number of roles in 'Inception' not ranked by 'n' value: {number_of_roles_not_ranked}")


# # 13.how many roles in the movie "inception" did receive an "n" value?

# In[18]:


import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('cast.csv')

# Filter rows where the title is "Inception"
inception_roles = df[df['title'] == 'Inception']

# Filter the rows where the "n" value is not NaN
inception_roles_with_n_value = inception_roles[~pd.isnull(inception_roles['n'])]

# Count the number of roles in "Inception" that received an "n" value
number_of_roles_with_n_value = len(inception_roles_with_n_value)

print(f"Number of roles in 'Inception' with 'n' value: {number_of_roles_with_n_value}")


# # 14.Display the cast of "North by Northwest in their correct "n"-value order, ignoring roles that did not eam a numeric "n" value

# In[19]:


import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('cast.csv')

# Filter rows where the title is "North by Northwest"
north_by_northwest_cast = df[(df['title'] == 'North by Northwest') & (pd.to_numeric(df['n'], errors='coerce').notna())]

# Sort the cast by the "n" values in ascending order
north_by_northwest_cast = north_by_northwest_cast.sort_values(by='n')

# Display the cast
print(north_by_northwest_cast[['n', 'name']])


# # 15.display the entire cast in "n"-order of the 1972 film "sleuth"

# In[20]:


import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('cast.csv')

# Filter rows where the title is "Sleuth" and the year is 1972
sleuth_1972_cast = df[(df['title'] == 'Sleuth') & (df['year'] == 1972)]

# Sort the cast by the "n" values in ascending order
sleuth_1972_cast = sleuth_1972_cast.sort_values(by='n')

# Display the entire cast
print(sleuth_1972_cast[['n', 'name']])


# # 16.now display the entire cast in "n" -order of the 2007 version of "sleuth

# In[21]:


import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('cast.csv')

# Filter rows where the title is "Sleuth" and the year is 2007
sleuth_2007_cast = df[(df['title'] == 'Sleuth') & (df['year'] == 2007)]

# Sort the cast by the "n" values in ascending order
sleuth_2007_cast = sleuth_2007_cast.sort_values(by='n')

# Display the entire cast
print(sleuth_2007_cast[['n', 'name']])


# # 17.how many roles were credited in the silent 1921 version of hamlet?

# In[23]:


import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('cast.csv')

# Filter rows where the title is "Hamlet" and the year is 1921
hamlet_1921_cast = df[(df['title'] == 'Hamlet') & (df['year'] == 1921)]

# Count the number of credited roles in the silent 1921 version of "Hamlet"
number_of_credited_roles = len(hamlet_1921_cast)

print(f"Number of credited roles in the silent 1921 version of 'Hamlet': {number_of_credited_roles}")


# # 18.How many roles were credited in Branagh's 1996 hamlet?

# In[24]:


import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('cast.csv')

# Filter rows where the title is "Hamlet," the year is 1996, and the name is "Kenneth Branagh"
hamlet_1996_branagh_cast = df[(df['title'] == 'Hamlet') & (df['year'] == 1996) & (df['name'] == 'Kenneth Branagh')]

# Count the number of credited roles for Kenneth Branagh in the 1996 version of "Hamlet"
number_of_credited_roles = len(hamlet_1996_branagh_cast)

print(f"Number of roles credited to Kenneth Branagh in the 1996 version of 'Hamlet': {number_of_credited_roles}")


# # 21.How many people have played a role called "The Dude".

# In[25]:


import pandas as pd

# Replace 'cast.csv' with the actual path to your CSV file
file_path = 'cast.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Filter the DataFrame to include only rows where the 'character' column is "The Dude"
the_dude_actors = df[df['character'] == 'The Dude']

# Count the number of unique actors who played the role "The Dude"
number_of_dude_actors = the_dude_actors['name'].nunique()

print("Number of people who played a role called 'The Dude':", number_of_dude_actors)


# # 22.How many people have played a role called "The Stranger".

# In[26]:


# Replace 'cast.csv' with the actual path to your CSV file
file_path = 'cast.csv'

# Read the CSV file into a DataFram
df = pd.read_csv(file_path)

# Filter the DataFrame to include only rows where the 'character' column is "The Dude"
the_dude_actors = df[df['character'] == 'The Stranger']

# Count the number of unique actors who played the role "The Dude"
number_of_dude_actors = the_dude_actors['name'].nunique()

print("Number of people who played a role called 'The Stranger':", number_of_dude_actors)


# # 23 How many roles has Sidney Poitier played throughout his career.

# In[27]:


import pandas as pd

# Replace 'cast.csv' with the actual path to your CSV file
file_path = 'cast.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Filter the DataFrame to include only rows where the 'name' column is "Sidney Poitier"
sidney_poitier_roles = df[df['name'] == 'Sidney Poitier']

# Count the number of unique roles played by Sidney Poitier
number_of_roles_played = sidney_poitier_roles['character'].nunique()

print("Number of roles played by Sidney Poitier throughout his career:", number_of_roles_played)


# # 24 How many roles has Judi Dench played.

# In[28]:


import pandas as pd

# Replace 'cast.csv' with the actual path to your CSV file
file_path = 'cast.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Filter the DataFrame to include only rows where the 'name' column is "Judi Dench"
judi_dench_roles = df[df['name'] == 'Judi Dench']

# Count the number of unique roles played by Judi Dench
number_of_roles_played = judi_dench_roles['character'].nunique()

print("Number of roles played by Judi Dench throughout her career:", number_of_roles_played)


# # 25.List the supporting roles (having n=2) played by Cary Grant in the 1940s, in order by year

# In[29]:


import pandas as pd

# Replace 'cast.csv' with the actual path to your CSV file
file_path = 'cast.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Filter the DataFrame to include only rows where the 'name' column is "Cary Grant"
# and 'n' column is 2 (supporting roles), and the 'year' column is in the 1940s
cary_grant_supporting_roles_1940s = df[(df['name'] == 'Cary Grant') & (df['n'] == 2) & (df['year'] // 10 == 194)]

# Sort the filtered DataFrame by 'year' in ascending order
sorted_cary_grant_supporting_roles_1940s = cary_grant_supporting_roles_1940s.sort_values(by='year')

# Print the list of supporting roles played by Cary Grant in the 1940s
print("Supporting roles played by Cary Grant in the 1940s:")
print(sorted_cary_grant_supporting_roles_1940s)


# # 26.list the leading roles that cary Grant played in the 1940s in order by year.

# In[30]:


import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('cast.csv')

# Filter rows where the name is "Cary Grant" and the year is in the 1940s
cary_grant_1940s_roles = df[(df['name'] == 'Cary Grant') & (df['year'] >= 1940) & (df['year'] <= 1949)]

# Filter rows where the "n" value is 1, indicating a leading role
leading_roles = cary_grant_1940s_roles[cary_grant_1940s_roles['n'] == 1]

# Sort the leading roles by year
leading_roles_sorted = leading_roles.sort_values(by='year')

# Display the list of leading roles
print(leading_roles_sorted[['year', 'title']])


# # 27.how many roles were available for actors in the 1950s?

# In[33]:


import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('cast.csv')

# Filter rows where the year is in the 1950s
roles_in_1950s = df[(df['year'] >= 1950) & (df['year'] <= 1959)]

# Count the number of roles available for actors in the 1950s
number_of_roles_in_1950s = len(roles_in_1950s)

print(f"Number of roles available for actors in the 1950s: {number_of_roles_in_1950s}")


# # 28.how many roles were available for actorsses in the 1950s?

# In[34]:


import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('cast.csv')

# Filter rows where the year is in the 1950s
roles_in_1950s = df[(df['year'] >= 1950) & (df['year'] <= 1959)]

# Count the number of roles available for actors in the 1950s
number_of_roles_in_1950s = len(roles_in_1950s)

print(f"Number of roles available for actors in the 1950s: {number_of_roles_in_1950s}")


# # 29.how many leading roles (n==1) were available from the beginning of film history through 1980?

# In[35]:


import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('cast.csv')

# Filter rows where the year is less than or equal to 1980 and n is 1 (indicating a leading role)
leading_roles_through_1980 = df[(df['year'] <= 1980) & (df['n'] == 1)]

# Count the number of leading roles available from the beginning of film history through 1980
number_of_leading_roles_through_1980 = len(leading_roles_through_1980)

print(f"Number of leading roles available from the beginning of film history through 1980: {number_of_leading_roles_through_1980}")


# # 30 how many non -leading roles were available through from the beginning of film history through 1980?

# In[36]:


import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('cast.csv')

# Filter rows where the year is less than or equal to 1980 and n is 1 (indicating a leading role)
leading_roles_through_1980 = df[(df['year'] <= 1980) & (df['n'] == 1)]

# Count the number of leading roles available from the beginning of film history through 1980
number_of_leading_roles_through_1980 = len(leading_roles_through_1980)

print(f"Number of leading roles available from the beginning of film history through 1980: {number_of_leading_roles_through_1980}")


# # 31.how many roles through 1980 were minor enough that they did not warrant a numeric "n" rank?

# In[37]:


import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('cast.csv')

# Filter rows where the year is less than or equal to 1980 and the "n" value is not a numeric rank (NaN)
minor_roles_without_numeric_rank = df[(df['year'] <= 1980) & pd.to_numeric(df['n'], errors='coerce').isna()]

# Count the number of minor roles without a numeric "n" rank through 1980
number_of_minor_roles_without_numeric_rank = len(minor_roles_without_numeric_rank)

print(f"Number of roles through 1980 that did not warrant a numeric 'n' rank: {number_of_minor_roles_without_numeric_rank}")


# # 2

# In[ ]:


get_ipython().run_line_magic('pinfo', 'time')


# In[40]:


import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('cast.csv')

# Group the DataFrame by movie title and count the occurrences of each title
movie_name_counts = df['title'].value_counts()

# Get the top ten most common movie names
top_ten_common_movie_names = movie_name_counts.head(10)

print("The ten most common movie names of all time:")
print(top_ten_common_movie_names)


# In[ ]:


get_ipython().run_line_magic('pinfo', 'released')


# In[41]:


import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('cast.csv')

# Filter rows where the year falls within the 1930s (1930 to 1939)
films_in_1930s = df[(df['year'] >= 1930) & (df['year'] <= 1939)]

# Count the number of films released in each year of the 1930s
film_counts_by_year = films_in_1930s['year'].value_counts()

# Get the top three years with the most films released
top_three_years = film_counts_by_year.head(3)

print("The three years of the 1930s with the most films released:")
print(top_three_years)


# In[ ]:


3.Plot the number of films that have been released each decade over the history of cinema.


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
df = pd.read_csv('cast.csv')

# Extract the decade from the 'year' column and create a new column 'decade'
df['decade'] = (df['year'] // 10) * 10

# Group the DataFrame by 'decade' and count the number of films in each decade
film_counts_by_decade = df['decade'].value_counts().sort_index()

# Plot the number of films released each decade
plt.figure(figsize=(10, 6))
film_counts_by_decade.plot(kind='bar', color='skyblue')
plt.title('Number of Films Released Each Decade')
plt.xlabel('Decade')
plt.ylabel('Number of Films')
plt.xticks(rotation=45)
plt.show()


# In[ ]:


4.Plot the number of hamlet films made each decade


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
df = pd.read_csv('cast.csv')

# Filter rows where the title is "Hamlet"
hamlet_films = df[df['title'] == 'Hamlet']

# Extract the decade from the 'year' column and create a new column 'decade'
hamlet_films['decade'] = (hamlet_films['year'] // 10) * 10

# Group the DataFrame by 'decade' and count the number of "Hamlet" films in each decade
hamlet_counts_by_decade = hamlet_films['decade'].value_counts().sort_index()

# Plot the number of "Hamlet" films made each decade
plt.figure(figsize=(10, 6))
hamlet_counts_by_decade.plot(kind='bar', color='skyblue')
plt.title('Number of "Hamlet" Films Made Each Decade')
plt.xlabel('Decade')
plt.ylabel('Number of Films')
plt.xticks(rotation=45)
plt.show()


# In[ ]:


5.Plot the number of "Rustler" characters in each decade of the history of the film


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
df = pd.read_csv('cast.csv')

# Filter rows where the character name is "Rustler"
rustler_characters = df[df['character'] == 'Rustler']

# Extract the decade from the 'year' column and create a new column 'decade'
rustler_characters['decade'] = (rustler_characters['year'] // 10) * 10

# Group the DataFrame by 'decade' and count the number of "Rustler" characters in each decade
rustler_counts_by_decade = rustler_characters['decade'].value_counts().sort_index()

# Plot the number of "Rustler" characters in each decade
plt.figure(figsize=(10, 6))
rustler_counts_by_decade.plot(kind='bar', color='skyblue')
plt.title('Number of "Rustler" Characters in Each Decade')
plt.xlabel('Decade')
plt.ylabel('Number of Characters')
plt.xticks(rotation=45)
plt.show()


# In[ ]:


6.Plot the number of "Hamlet" characters each decade


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
df = pd.read_csv('cast.csv')

# Filter rows where the title is "Hamlet"
hamlet_characters = df[df['title'] == 'Hamlet']

# Extract the decade from the 'year' column and create a new column 'decade'
hamlet_characters['decade'] = (hamlet_characters['year'] // 10) * 10

# Group the DataFrame by 'decade' and count the number of "Hamlet" characters in each decade
hamlet_counts_by_decade = hamlet_characters['decade'].value_counts().sort_index()

# Plot the number of "Hamlet" characters in each decade
plt.figure(figsize=(10, 6))
hamlet_counts_by_decade.plot(kind='bar', color='skyblue')
plt.title('Number of "Hamlet" Characters in Each Decade')
plt.xlabel('Decade')
plt.ylabel('Number of Characters')
plt.xticks(rotation=45)
plt.show()


# In[ ]:


get_ipython().run_line_magic('pinfo', 'history')


# In[1]:


# Group the DataFrame by character name and count the occurrences of each character name
character_name_counts = df['character'].value_counts()

# Get the top 11 most common character names
top_11_common_character_names = character_name_counts.head(11)

print("The 11 most common character names in movie history:")
print(top_11_common_character_names)


# In[ ]:


get_ipython().run_line_magic('pinfo', 'history')


# In[2]:


import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('cast.csv')

# Filter rows where the character name is "Herself"
herself_credits = df[df['character'] == 'Herself']

# Group the DataFrame by actor name and count the number of times each actor was credited as "Herself"
top_10_herself_actors = herself_credits['name'].value_counts().head(10)

print("The 10 people most often credited as 'Herself' in film history:")
print(top_10_herself_actors)


# In[ ]:


9.Who are the 10 people most often credited as "Himself" in film history


# In[3]:


import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('cast.csv')

# Filter rows where the character name is "Himself"
himself_credits = df[df['character'] == 'Himself']

# Group the DataFrame by actor name and count the number of times each actor was credited as "Himself"
top_10_himself_actors = himself_credits['name'].value_counts().head(10)

print("The 10 people most often credited as 'Himself' in film history:")
print(top_10_himself_actors)


# In[ ]:


10.Which actors or actressess appeared in the most movies in the year 1945


# In[4]:


import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('cast.csv')

# Filter rows where the year is 1945
movies_in_1945 = df[df['year'] == 1945]

# Group the DataFrame by actor/actress name and count the number of movies for each
most_appearances_1945 = movies_in_1945['name'].value_counts().reset_index()
most_appearances_1945.columns = ['Actor/Actress', 'Number of Movies']

# Find the actor/actress with the most movie appearances in 1945
top_actor_1945 = most_appearances_1945.iloc[0]

print("Actor/Actress with the most movie appearances in 1945:")
print(top_actor_1945)


# In[ ]:


11.Which actors or actressess appeared in the most movies in the year 1985?


# In[5]:


import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('cast.csv')

# Filter rows where the year is 1985
movies_in_1985 = df[df['year'] == 1985]

# Group the DataFrame by actor/actress name and count the number of movies for each
most_appearances_1985 = movies_in_1985['name'].value_counts().reset_index()
most_appearances_1985.columns = ['Actor/Actress', 'Number of Movies']

# Find the actor/actress with the most movie appearances in 1985
top_actor_1985 = most_appearances_1985.iloc[0]

print("Actor/Actress with the most movie appearances in 1985:")
print(top_actor_1985)


# In[ ]:


12.Plot how many roles Mammotty has played in each year of his carrer 


# In[6]:


import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
df = pd.read_csv('cast.csv')

# Filter rows where the name is "Mammootty"
mammootty_roles = df[df['name'] == 'Mammootty']

# Group the DataFrame by year and count the number of roles in each year
roles_by_year = mammootty_roles.groupby('year').size()

# Plot the number of roles Mammootty has played in each year of his career
plt.figure(figsize=(12, 6))
roles_by_year.plot(kind='bar', color='skyblue')
plt.title('Number of Roles Played by Mammootty Each Year')
plt.xlabel('Year')
plt.ylabel('Number of Roles')
plt.xticks(rotation=45)
plt.show()


# In[ ]:


13.What are the 10 most frequent roles that start with the phrase "Parton in"?


# In[8]:


import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('cast.csv')

# Filter rows where the character name starts with "Parton in"
parton_in_roles = df[df['character'].str.startswith('Parton in')]

# Count the occurrences of each role and get the top 10 most frequent roles
top_10_parton_in_roles = parton_in_roles['character'].value_counts().head(10)

print("The 10 most frequent roles that start with 'Parton in':")
print(top_10_parton_in_roles)


# In[ ]:


14.What are the 10 most frequent roles that start with the word "Science"?


# In[9]:


import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('cast.csv')

# Filter rows where the character name starts with "Science"
science_roles = df[df['character'].str.startswith('Science')]

# Count the occurrences of each role and get the top 10 most frequent roles
top_10_science_roles = science_roles['character'].value_counts().head(10)

print("The 10 most frequent roles that start with 'Science':")
print(top_10_science_roles)


# In[ ]:


15.Plot n-values of the roles that Judi Dench has played over her carrer.


# In[11]:


import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
df = pd.read_csv('cast.csv')

# Filter rows where the name is "Judi Dench"
judi_dench_roles = df[df['name'] == 'Judi Dench']

# Remove rows where the 'n' column is not numeric
judi_dench_roles = judi_dench_roles[pd.to_numeric(judi_dench_roles['n'], errors='coerce').notna()]

# Convert the 'n' column to numeric
judi_dench_roles['n'] = pd.to_numeric(judi_dench_roles['n'])

# Plot the n-values of the roles Judi Dench has played
plt.figure(figsize=(12, 6))
plt.scatter(judi_dench_roles['year'], judi_dench_roles['n'], color='skyblue', alpha=0.5)
plt.title('n-Values of Roles Played by Judi Dench Over Her Career')
plt.xlabel('Year')
plt.ylabel('n-Value')
plt.grid(True)
plt.show()


# In[ ]:


16.Plot the n-values of Cary Grants roles through his carrer


# In[12]:


import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
df = pd.read_csv('cast.csv')

# Filter rows where the name is "Cary Grant"
cary_grant_roles = df[df['name'] == 'Cary Grant']

# Remove rows where the 'n' column is not numeric
cary_grant_roles = cary_grant_roles[pd.to_numeric(cary_grant_roles['n'], errors='coerce').notna()]

# Convert the 'n' column to numeric
cary_grant_roles['n'] = pd.to_numeric(cary_grant_roles['n'])

# Plot the n-values of Cary Grant's roles throughout his career
plt.figure(figsize=(12, 6))
plt.scatter(cary_grant_roles['year'], cary_grant_roles['n'], color='skyblue', alpha=0.5)
plt.title('n-Values of Roles Played by Cary Grant Throughout His Career')
plt.xlabel('Year')
plt.ylabel('n-Value')
plt.grid(True)
plt.show()


# In[ ]:


17.Plot the n-values of the roles that Sidney Poitier has acted over the years.


# In[13]:


import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
df = pd.read_csv('cast.csv')

# Filter rows where the name is "Sidney Poitier"
sidney_poitier_roles = df[df['name'] == 'Sidney Poitier']

# Remove rows where the 'n' column is not numeric
sidney_poitier_roles = sidney_poitier_roles[pd.to_numeric(sidney_poitier_roles['n'], errors='coerce').notna()]

# Convert the 'n' column to numeric
sidney_poitier_roles['n'] = pd.to_numeric(sidney_poitier_roles['n'])

# Plot the n-values of Sidney Poitier's roles over the years
plt.figure(figsize=(12, 6))
plt.scatter(sidney_poitier_roles['year'], sidney_poitier_roles['n'], color='skyblue', alpha=0.5)
plt.title('n-Values of Roles Played by Sidney Poitier Over the Years')
plt.xlabel('Year')
plt.ylabel('n-Value')
plt.grid(True)
plt.show()


# In[ ]:


get_ipython().run_line_magic('pinfo', 's')


# In[14]:


import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('cast.csv')

# Filter rows where the 'year' is in the 1950s
roles_in_1950s = df[(df['year'] >= 1950) & (df['year'] <= 1959)]

# Count the number of leading roles (n==1) for actors and actresses separately
leading_roles_actors = roles_in_1950s[(roles_in_1950s['n'] == 1) & (roles_in_1950s['type'] == 'actor')]
leading_roles_actresses = roles_in_1950s[(roles_in_1950s['n'] == 1) & (roles_in_1950s['type'] == 'actress')]

# Get the counts
num_leading_roles_actors = len(leading_roles_actors)
num_leading_roles_actresses = len(leading_roles_actresses)

print(f"Number of leading roles (n==1) for actors in the 1950s: {num_leading_roles_actors}")
print(f"Number of leading roles (n==1) for actresses in the 1950s: {num_leading_roles_actresses}")


# In[ ]:


get_ipython().run_line_magic('pinfo', 's')


# In[15]:


import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('cast.csv')

# Filter rows where the 'year' is in the 1950s
roles_in_1950s = df[(df['year'] >= 1950) & (df['year'] <= 1959)]

# Count the number of supporting roles (n==2) for actors and actresses separately
supporting_roles_actors = roles_in_1950s[(roles_in_1950s['n'] == 2) & (roles_in_1950s['type'] == 'actor')]
supporting_roles_actresses = roles_in_1950s[(roles_in_1950s['n'] == 2) & (roles_in_1950s['type'] == 'actress')]

# Get the counts
num_supporting_roles_actors = len(supporting_roles_actors)
num_supporting_roles_actresses = len(supporting_roles_actresses)

print(f"Number of supporting roles (n==2) for actors in the 1950s: {num_supporting_roles_actors}")
print(f"Number of supporting roles (n==2) for actresses in the 1950s: {num_supporting_roles_actresses}")


# In[ ]:


#3


# In[ ]:


1.Using groupby(), plot the number of films that have been released each decade in the history of cinema.


# In[16]:


import pandas as pd
import matplotlib.pyplot as plt

# Replace 'titles.csv' with the actual path to your CSV file
file_path = 'titles.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Create a new column 'decade' to represent the decade for each movie
df['decade'] = (df['year'] // 10) * 10

# Group the data by 'decade' and count the number of films in each decade
film_counts_by_decade = df.groupby('decade').size().reset_index(name='film_count')

# Plot the number of films released each decade
plt.figure(figsize=(10, 6))
plt.bar(film_counts_by_decade['decade'], film_counts_by_decade['film_count'], width=8)
plt.xlabel('Decade')
plt.ylabel('Number of Films')
plt.title('Number of Films Released Each Decade')
plt.xticks(film_counts_by_decade['decade'])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# In[ ]:


2.Using groupby(), plot the number of "Hamlet" films made each decade.


# In[17]:


import pandas as pd
import matplotlib.pyplot as plt

# Replace 'titles.csv' with the actual path to your CSV file
file_path = 'titles.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Filter the DataFrame to include only "Hamlet" films
hamlet_films = df[df['title'] == 'Hamlet']

# Create a new column 'decade' to represent the decade for each "Hamlet" movie
hamlet_films['decade'] = (hamlet_films['year'] // 10) * 10

# Group the data by 'decade' and count the number of "Hamlet" films in each decade
hamlet_counts_by_decade = hamlet_films.groupby('decade').size().reset_index(name='hamlet_count')

# Plot the number of "Hamlet" films made each decade
plt.figure(figsize=(10, 6))
plt.bar(hamlet_counts_by_decade['decade'], hamlet_counts_by_decade['hamlet_count'], width=8)
plt.xlabel('Decade')
plt.ylabel('Number of "Hamlet" Films')
plt.title('Number of "Hamlet" Films Made Each Decade')
plt.xticks(hamlet_counts_by_decade['decade'])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# In[ ]:


get_ipython().run_line_magic('pinfo', 's')


# In[18]:


import pandas as pd

# Replace 'cast.csv' with the actual path to your CSV file
file_path = 'cast.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Filter the DataFrame to include only rows where the 'n' column is 1 (leading roles)
leading_roles = df[df['n'] == 1]

# Create a new column 'decade' to represent the decade for each movie
leading_roles['decade'] = (leading_roles['year'] // 10) * 10

# Filter the data to include only rows where the 'decade' column is in the 1950s
leading_roles_1950s = leading_roles[(leading_roles['decade'] >= 1950) & (leading_roles['decade'] < 1960)]

# Group the data by 'decade' and 'type' (actor/actress) and count the number of leading roles
leading_roles_by_year_1950s = leading_roles_1950s.groupby(['decade', 'type']).size().reset_index(name='count')

# Pivot the data to have 'decade' as rows, 'type' as columns, and 'count' as values
pivot_table = leading_roles_by_year_1950s.pivot(index='decade', columns='type', values='count')

# Fill NaN values with 0 (in case no leading roles of a specific type were found in a year)
pivot_table = pivot_table.fillna(0)

# Print the resulting pivot table
print(pivot_table)


# In[ ]:


4.In the 1950s decade taken as a whole, how many total roles were available to actors, and how many to actresses, for each "n" number 1 through 5?


# In[19]:


import pandas as pd

# Replace 'cast.csv' with the actual path to your CSV file
file_path = 'cast.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Filter the DataFrame to include only rows from the 1950s
roles_1950s = df[(df['year'] >= 1950) & (df['year'] < 1960)]

# Group the data by 'n' (role number) and 'type' (actor/actress) and count the number of roles
roles_by_n_and_type = roles_1950s.groupby(['n', 'type']).size().reset_index(name='count')

# Pivot the data to have 'n' as rows, 'type' as columns, and 'count' as values
pivot_table = roles_by_n_and_type.pivot(index='n', columns='type', values='count')

# Fill NaN values with 0 (in case no roles of a specific type were found for a particular "n" number)
pivot_table = pivot_table.fillna(0)

# Print the resulting pivot table
print(pivot_table)


# In[ ]:


5.Use groupby() to determine how many roles are listed for each of the Pink Panther movies


# In[21]:


import pandas as pd

# Replace 'cast.csv' with the actual path to your CSV file
file_path = 'cast.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Filter the DataFrame to include only rows where the 'title' column contains "Pink Panther"
pink_panther_movies = df[df['title'].str.contains('Pink Panther', case=False)]

# Group the data by 'title' and count the number of roles for each movie
roles_per_pink_panther_movie = pink_panther_movies.groupby('title')['character'].count().reset_index()

# Print the result
print(roles_per_pink_panther_movie)


# In[ ]:


6.List, in order by year, each of the films in which Frank Oz has played more than 1 role.


# In[22]:


import pandas as pd

# Replace 'cast.csv' with the actual path to your CSV file
file_path = 'cast.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Filter the DataFrame to include only rows where the 'name' column is "Frank Oz"
frank_oz_movies = df[df['name'] == 'Frank Oz']

# Group the data by 'year' and 'title' and count the number of roles Frank Oz played in each movie
roles_per_frank_oz_movie = frank_oz_movies.groupby(['year', 'title'])['character'].count().reset_index()

# Filter the results to include only movies where Frank Oz played more than one role
multiple_role_movies = roles_per_frank_oz_movie[roles_per_frank_oz_movie['character'] > 1]

# Sort the filtered results by 'year' in ascending order
sorted_multiple_role_movies = multiple_role_movies.sort_values(by='year')

# Print the list of movies
print(sorted_multiple_role_movies)


# In[ ]:


7.List each of the characters that Frank Oz has portrayed at least twice


# In[23]:


import pandas as pd

# Replace 'cast.csv' with the actual path to your CSV file
file_path = 'cast.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Filter the DataFrame to include only rows where the 'name' column is "Frank Oz"
frank_oz_roles = df[df['name'] == 'Frank Oz']

# Group the data by 'character' and count the number of times each character has been portrayed by Frank Oz
character_counts = frank_oz_roles.groupby('character').size().reset_index(name='count')

# Filter the results to include only characters portrayed at least twice
characters_portrayed_at_least_twice = character_counts[character_counts['count'] >= 2]

# Print the list of characters
print(characters_portrayed_at_least_twice)

