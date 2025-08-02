<img width="911" height="183" alt="image" src="https://github.com/user-attachments/assets/8fca33a2-551a-4b12-a32e-1f323d07e713" />

<img width="911" height="222" alt="image" src="https://github.com/user-attachments/assets/405a98c0-8d0c-49a4-a330-5d53f52077dc" />



# PROJECT OVERVIEW
Microsoft is a major company worth billions. Hence they wish to expand and enter the movie/film industry. 
This analysis will help to give a general overview of the industry, to ensure the stakeholders understand how the industry works and what to expect of it, which will be ideal for decison making.

# BUSINESS UNDERSTANDING
Movies/Film industry is a multi-billion dollar industry that has been growing rapidly in recent years due to various factors that have ensured a steady rise from the analog to digital eras.

We've witnessed improvements in distribution, quality, genres, acting, among other things.

One of the major factors has been the internet, which today is easily accessible to majority of the population worldwide. This has lead to the indroduction of various streaming platforms, making it easy for viewers to download any type of movies they want to watch. Hence we have migrated from eras where we had to use compact disks(CD's/DVD's) which needed a physical store to buy them, to now using streaming platforms at the comfort of your home without needing to go out and get access to a particular movie, as long as you have internet access.

Other factors have also majorly contributed to its growth, ie,

- the use of various visual effects like CGI,VFX ensuring the movie is of high quality.
- Large investments to ensure the logistics in a movie are met and the movie is finished and distributed on time.
- Changing audience preferences ie genre preferences and cultural diversity has lead to content that caters to various audience.
- Consumer spending due to strong economy can lead to high box office revenues.
- Production and creative factors ie, script quality, acting skills, marketing and promotions can generate great revenues.
However, all industries have their own risks that can impact various aspects in production like piracy, economic downturns, budget overruns, changing audience preferences, weather-related risks, casts illness/injury, poor box-office run, among other things.

Therefore it is necessary to try and mitigate all/most of these risks during the planning and production periods. One of the main ways to reduce risks is by factoring data analysis to know the current trends and target audience.

## OBJECTIVES
- Find Which genres tend to earn most revenues
- Which genres recieve the highest ratings.
- Finding out the total gross based on genres.
- Finding the ratios between foreign gross and domestic gross.
- Estimate whether a movie was a box office success or failure based on the ratings.
- Identify the most profitable genres globally based on total gross.
- Identify the movie production patterns over the years.
- Identify which genres to invest based on the total gross and average ratings.

  
#


```python
# import necessary libraries
import pandas as pd
import numpy as np
import matpotlib.pyplot as plt
import seaborn as sns
```

#


```python
# load the dataset
df1 = pd.read_csv('bom.movie_gross.csv.gz')
df2 = pd.read_csv('imdb.title.basics.csv.gz')
df3 = pd.read_csv('imdb.title.ratings.csv.gz')
```

#

After loading the dataset we can merge it together

```python
df = pd.merge(df1, df2, left_on='title', right_on='primary_title')
df = pd.merge(df, df3, on='tconst')
df
```

```python
	title            studio	    domestic_gross    foreign_gross  	year   	tconst  	primary_title      	original_title	   start_year  runtime_minutes   	genres        	         averagerating    	numvotes
0	Toy Story 3       BV	      415000000.0	    652000000	  2010	 tt0435761	Toy Story 3	         Toy Story 3	       2010	         103.0	    Adventure,Animation,Comedy	    8.3	               682218
```

*the three datasets have been merged together. df1 starts from the left with title column to the year column. on the right df2 then continues from primary title colum to genres colum. Then df 3 combines with the two datasets with average ratings and numvotes columns.*

*the 'tconst' column has been merged from df2 and df3 as they share a common column*


#

```python
df.info()
```
```python
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 3027 entries, 0 to 3026
Data columns (total 13 columns):
 #   Column           Non-Null Count  Dtype  
---  ------           --------------  -----  
 0   title            3027 non-null   object 
 1   studio           3024 non-null   object 
 2   domestic_gross   3005 non-null   float64
 3   foreign_gross    1832 non-null   object 
 4   year             3027 non-null   int64  
 5   tconst           3027 non-null   object 
 6   primary_title    3027 non-null   object 
 7   original_title   3027 non-null   object 
 8   start_year       3027 non-null   int64  
 9   runtime_minutes  2980 non-null   float64
 10  genres           3020 non-null   object 
 11  averagerating    3027 non-null   float64
 12  numvotes         3027 non-null   int64  
dtypes: float64(3), int64(3), object(7)
memory usage: 307.6+ KB
```

#


foreign_gross column needs to be converted from an object type to numeric type
```python
df['foreign_gross'] = df['foreign_gross'].astype(str).str.replace(',', '').astype(float)
df.info()
```

#


adding a new column to the dataset called total_gross which will be *(domestic_gross + foreign_gross)

```python
df = df.assign(total_gross = df['domestic_gross'] + df['foreign_gross'])
df
```


#

we need to chose the 'key columns' we are going to work with for further analysis

```python
key_columns = ['title', 'year', 'genres', 'averagerating', 'domestic_gross', 'foreign_gross', 'total_gross']
df = df[key_columns]
df
```

```python
	title	       year	     genres	             averagerating   domestic_gross foreign_gross   total_gross
0	Toy Story 3  2010	Adventure,Animation,Comedy	 8.3 	       415000000.0 	652000000.0	 1.067000e+09
1	Inception	2010    Action,Adventure,Sci-Fi	       8.8	      292600000.0	535700000.0	  8.283000e+08
```




# EDA - DATA CLEANING

Data cleaning involves various steps, for example:
- checking for missing values
- checking for duplicated values
- outliers detection and handling


#

*checking for missing values in each column*

```python
df.isna().sum()
```
```python
	0
title	0
year	0
genres	7
averagerating	0
domestic_gross	22
foreign_gross	1195
total_gross	1217

dtype: int64
```

#

*checking for the proportion of missing values*
```python
df.isna().mean()
```
```python

0
title	0.000000
year	0.000000
genres	0.002313
averagerating	0.000000
domestic_gross	0.007268
foreign_gross	0.394780
total_gross	0.402048

dtype: float64
```

*the `foreign_gross` and `total_gross` columns have ~39% and ~40% of null values. this is a significant proportion*


#

dropping the missing values

```python
df = df.dropna()
df
```


#

checking for any duplicated values

```python
df.duplicated().value_counts()
```

*there are no duplicated values*


#


OUTLIER DETECTION AND HANDLING

*checking for outliers by plotting a boxplot using seaborn*

```python
sns.boxplot(data=df[['domestic_gross', 'foreign_gross', 'total_gross']])
plt.title('Gross Revenue Boxplot')
```
<img width="547" height="435" alt="image" src="https://github.com/user-attachments/assets/63452402-3b97-46a6-a463-7c9de963d07c" />

All the gross revenue columns contain many outliers. They can be removed easily using IQR, but given that this is a movie dataset, we shall keep the outliers since they represent the real data.


#

### Feature Engineering

*finding out the performance patterns between domestic and international audiences by knowing their ratios, using the .loc[] method*

```python
df.loc[:, 'domestic_ratio'] = df['domestic_gross'] / df['total_gross']
df.loc[:, 'foreign_ratio'] = df['foreign_gross'] / df['total_gross']
df
```


#

*Next is finding out the `Rating level` used  based on the average rating. 0-5 rating indicating `poor`, 5-8 rating indicating `average` movie, 8-10 indicating the movie was a `blockbuster`*

```python
df.loc[:, 'rating_level'] = pd.cut(df['averagerating'], bins=[0, 5, 8, 10], labels=['poor', 'average', 'blockbuster'])
df
```




## Univariate Analysis

we start by checking the statistical summary

```python
df.describe()
```

```python
	year	averagerating	domestic_gross	foreign_gross	total_gross	domestic_ratio	foreign_ratio
count	1803.000000	1803.000000	1.803000e+03	1.803000e+03	1.803000e+03	1803.000000	1803.000000
mean	2013.668885	6.451636	4.971950e+07	7.911398e+07	1.288335e+08	0.405832	0.594168
std	2.570356	1.010874	8.047399e+07	1.394153e+08	2.088373e+08	0.267331	0.267331
min	2010.000000	1.600000	4.000000e+02	6.000000e+02	1.080000e+04	0.000037	0.000002
25%	2011.000000	5.900000	1.300000e+06	4.800000e+06	1.000000e+07	0.194754	0.409322
50%	2014.000000	6.500000	2.080000e+07	2.130000e+07	4.810000e+07	0.403066	0.596934
75%	2016.000000	7.200000	6.155000e+07	8.215000e+07	1.489000e+08	0.590678	0.805246
max	2018.000000	9.200000	7.001000e+08	9.464000e+08	1.405400e+09	0.999998	0.999963
```

- The earliest movie produced was in 2010, and the latest movie is 2018.
- Highest rating is a 9.2 and the least is a 1.6, the average rating is 6.4
-etc....


#

*Let's now get to know the number of each genre and sub-genre found in the dataset. Then create a `bar graph` to show the distribution of the `top 10 most popular genre categories` in the dataset*

```python
genre_counts = df['genres'].value_counts()
genre_counts
```

With the code above, one movie can have several genres ie, `Adventure, Animation, Drama` 

hence Since there are too many sub-genres which contains comma seperated values ie, `'Action,Adventure'`, we'll combine them into unique genre categories by splitting them and counting properly.

```python
# .str.get_dummies() method is used to get one column per category by seperating where there is a comma

genre_categories = df['genres'].str.get_dummies(sep=',')
```

```python
# counting the genre categories
# each count shows the number of movies which some have multiple genres, hence the sum can be more than the number of rows

genre_counts = genre_categories.sum().sort_values(ascending=False)
print(genre_counts)
```

```python
Drama          950
Comedy         604
Action         464
Adventure      368
Thriller       292
Romance        255
Crime          243
Biography      160
Horror         159
Mystery        133
Fantasy        129
Animation      123
Sci-Fi         113
Documentary     86
Family          86
History         80
Music           49
Sport           35
War             20
Western         11
Musical         10
News             1
dtype: int64
```



*Creating a bar graph for the top 10 genre categories*

```python
genre_counts.head(10).plot(kind='bar')
plt.xlabel('Genre')
plt.ylabel('Count')
plt.title('Top 10 Movie Genres')
```

<img width="571" height="513" alt="image" src="https://github.com/user-attachments/assets/4dd88ca2-73f0-4f8f-9c36-f3f9df9141d8" />


#

To know how the `average rating` is distributed across this dataset, we shall visualize using a histogram.

```python
# ploting a histogram on the distribution of average rating against number of movies
# kde=True - Kernel density estimate curve
# plt.grid - adds background

sns.histplot(df['averagerating'], bins=20, kde=True, color='red', edgecolor='black')
plt.xlabel('Average Rating')
plt.ylabel('Number of movies')
plt.grid(True)
plt.title('Distribution of Average Ratings')
```

<img width="571" height="455" alt="image" src="https://github.com/user-attachments/assets/5de9fad9-1e07-41c8-99d1-717066017aa0" />


*This is a `negatively skewed` distribution as the `mean < median`.*

*This is likely due to low-value outliers pulling the mean down*



## Bivariate Analysis

Let's now check how movie production has changed over the years, by grouping the `year column` and counting how many movies were released each year.

Then visualize using a line graph

```python
# STEP 1: grouping by year and counting the number of movies

movies_per_year = df.groupby('year')['title'].count()
movies_per_year
```

```python
# STEP 2: plotting a line chart

sns.lineplot(x=movies_per_year.index, y=movies_per_year.values)
plt.xlabel('Year')
plt.ylabel('Number of Movies')
plt.title('Number of Movies per Year')
plt.grid(True)
```

<img width="571" height="455" alt="image" src="https://github.com/user-attachments/assets/5c38c1cb-edf4-4d4e-a808-6e6c7882668b" />

Movie production showed a `steady rise` from 2010-2011 having its `peak` in 2011. 

From year 2012-2013 there was a sharp `decline` but had a slight rise again in year 2014, and by year 2015-2018 there was a `steady decline` in movie production. This may be attributed to external factors ie, economical factors. 


##

Let's analyze the relationship between `genres`(categorical) and `total_gross` (numerical) so we can find which genre tends to perform better based on total gross. Then use a bar graph to analyze 


```python
# Create a new DataFrame with one genre per row
# using .explode() method to transform genre column into multiple rows

df_exploded = df.assign(genres=df['genres'].str.split(',')).explode('genres')

# Group by genre and calculate the mean of total_gross
genre_gross = df_exploded.groupby('genres')['total_gross'].mean().sort_values(ascending=False)
genre_gross
```

```python
# plotting the relationship using a bar graph

sns.barplot(x=genre_gross.values, y=genre_gross.index, palette='viridis')
plt.xlabel('Average Total Gross')
plt.ylabel('Genre')
plt.title('Average Total Gross by Genre')
```

<img width="639" height="455" alt="image" src="https://github.com/user-attachments/assets/193338b7-1dc8-40df-9694-118eccb418ed" />


*The highest genre earners are `sci-fi, adventure, animation`, While the lowest earners are `news, romance and war` respectively*

*Although `drama and comedy` are in the top 10 movie genres produced in this dataset, their `gross revenue` is not that high*, while `sci-fi` and `animation` are not in the top 10 movie genres but their `gross revenue` are among the highest. This may be due to the target audience as some genres are targeting global audience while some domestic audience.



## Multivariate Analysis

*Let's find the correlation  in the numerical columns of the dataset, then visualize the information using a heatmap*

```python
# finding the correlation in the numerical columns

df[['averagerating', 'domestic_gross', 'foreign_gross', 'total_gross']].corr()
```

```python
# plotting a heat map for the correlation

sns.heatmap(df[['averagerating', 'domestic_gross', 'foreign_gross', 'total_gross']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
```

<img width="609" height="529" alt="image" src="https://github.com/user-attachments/assets/4e917d68-5095-43da-ab21-b11ebf86310a" />

*`1.0` - perfect positive correlation*

*`0.0` - no correlation*

*`-1.0` - perfect negative correlation*

- There is a strong positive correlation between `total_gross` and `foreign_gross` meaning they increase together. Also between `total_gross` and `domestic_gross`.

- There is `almost` No correlation between `averagerating` and the rest.



#

*Exploring the relationship between `foreign and domestic gross` across `different genres`*

```python
sns.scatterplot(data=df_exploded, x='foreign_gross', y='domestic_gross', hue='genres')
plt.xlabel('Foreign Gross')
plt.ylabel('Domestic Gross')
plt.title('Foreign vs Domestic Gross by Genre')
```

<img width="554" height="535" alt="image" src="https://github.com/user-attachments/assets/345571b0-3e25-4d96-8c13-6bb300c7b46f" />

In the code below, by Exploring linear trends for each genre, this will create small regression plots per genre for clear visualization.

```python
# Trendline regression by genre

sns.lmplot(data=df_exploded, x='foreign_gross', y='domestic_gross', hue='genres', palette='viridis')
plt.xlabel('Foreign Gross')
plt.ylabel('Domestic Gross')
plt.title('Foreign vs Domestic Gross by Genre')
```

<img width="627" height="505" alt="image" src="https://github.com/user-attachments/assets/ae532003-2b8e-44ce-9967-a9b60dbc550e" />

- A `steeper trendline` indicates that movies in that genre tend to earn more `foreign gross as domestic gross increases` for example `action, sci-fi and animationd genres`

- A `flat or weak slope` suggests foreign earnings do not scale much with domestic success. For example `comedy and drama genres`. humor may not translate globally, due to maybe language barriers or cultural differences



# RECOMMENDATIONS

- Invest more on high perfoming genres, such as:
  
   1. `Action and Animation genres` for box office success as they have high revenues.

   2. `Drama and Comedy genres` for high ratings tailor made for domestic audiences, due to cultural differences.

- Invest in marketing strategies to boost visibility of a movie.
- Make your films stand out with great scripts and unique story telling. This will help when movie productions increases hence great market saturation.
- Limit overproduction of genres with low ratings and revenues, for instance, `horror genre`.
- some genres tend to earn awards due to long term cultural values they have, like, `Documentary,Biography genres`.
- Make sure before a movie production you analyze the target audiences, so as to know how much to invest and expected revenues to be expected.


# CONCLUSION

This data has provided a rich overview of the movie/film industry, hence, this analysis has lead to some insights into the industry. For example:

- Revenue patterns can differ by market, ie, some genres have global tractions and others show strong performance domestically.
- Genres influence both ratings and revenues.
- Ratings are not always aligned with revenues, for instance, high rated films are not always bringing high revenues.

By using past data and analyzing it, stakeholders can make more informed and strategic decisions so as to maximize the reach to target audiences.

