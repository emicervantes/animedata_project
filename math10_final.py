import pandas as pd
import numpy as np
import altair as alt
import streamlit as st
from PIL import Image
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

### Cleaning data
## use the dataset 'dataanime.csv'
df = pd.read_csv("dataanime.csv")
df.insert(0, "anime_id", df.index)
# remove any NaN rows
df = df[~df.isna().any(axis = 1)]
df = df.apply(lambda x: x.replace('-',np.nan))
df = df[~df.isna().any(axis = 1)]
# create a new column 'day' that shows the day of the week in which the anime was aired
# remove row with value "Not"
df["day"] = df["Broadcast time"].map(lambda x: x.split()[0])
df = df[~(df["day"] == "Not")]
# create a new column 'time' that shows the time of the day in whcih the anime was aired
# remove row with value "Unknown"
df["time"] = df["Broadcast time"].map(lambda x: x.split()[2])
df = df.apply(lambda x: x.replace('Unknown',np.nan))
df = df[~df.isna().any(axis = 1)]
# assign index to df
df.index = range(369)
# week dictionary that has days of the week as key and number corresponding to each one as a value
week = {
    "Mondays":1,
    "Tuesdays":2,
    "Wednesdays":3,
    "Thursdays":4,
    "Fridays":5,
    "Saturdays":6,
    "Sundays":7
}
# create a new column 'day_n' that shows a corresponding value number to a key (day of the week)
df["day_n"] = df["day"].map(lambda x: week[x])
# season dictionary that has season of the year as key and number corresponding to each one as a value
season = {
    "Spring":1,
    "Fall":2,
    "Summer":3,
    "Winter":4
}
# create a new column 'season_n' that shows a corresponding value number to a key (season of the year)
df["season_n"] = df["Starting season"].map(lambda x: season[x])
# create a new column 'time_n' that changes value of df["time"] to all integers without ":"
df["time_n"] = df["time"].map(lambda x: x.replace(":",""))
df["time_n"] = df["time_n"].map(lambda x: np.int64(x))
# function that categorizes time (morning, night...) based on the value of time anime was aired
def categorize_time(time):
    if time >= 1800:
        # it is night
        x = 1
    elif time >= 1200:
        # it is midday
        x = 2
    elif time >= 600:
        # it is morning
        x = 3
    else:
        # it is midnight
        x = 4
    return x
# timeframe dictionary that has values as the timeframe for each corresponding key 
timeframe = {
    1: "6:00PM~11:59PM",
    2:"12:00PM ~ 5:59PM",
    3: "6:00AM ~ 11:59AM",
    4: "12:00AM ~ 6:00AM"
}
# values of 'time_n' column is changed based on timeframe dictionary 
df["time_n"] = df["time_n"].map(lambda x: categorize_time(x))
df["time_s"] = df["time_n"].map(lambda x: timeframe[x])
# pd.get_dummies that checks the season of each anime
# https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html
check_season = pd.get_dummies(df['Starting season']).rename(columns=lambda x:'is_' +str(x))
check_season = check_season.applymap(lambda x: np.int8(x))
# pd.get_dummies that checks the day of each anime
reversed_week = {value : key for (key, value) in week.items()}
check_day = pd.get_dummies(df['day_n']).rename(columns=lambda x:'is_' + reversed_week[x])
check_day = check_day.applymap(lambda x: np.int8(x))
# pd.get_dummies that check the brodcast time of each anime
brod_time = {
    1: "night",
    2: "midday",
    3: "morning",
    4: "midnight"
}
check_time = pd.get_dummies(df['time_n']).rename(columns=lambda x:'is_' + brod_time[x])
check_time = check_time.applymap(lambda x: np.int8(x))
# put all the new pd.get_dummies dataframes into one and combine it with df
combined_dummies = pd.concat([check_day,check_time,check_season], axis=1)
df = pd.concat([df, combined_dummies], axis=1) 

# Clean Genres column
# 
df["Genres"].map(type)
df["Genres"].map(lambda x: list(x))
df["Genres"].map(lambda x: x.replace(", ",""))
mylist = []
for i in range(len(df["Genres"])):
  mylist.append(df.iloc[i,13].split(","))
df["Genres"] = mylist
genres_list = sorted(list(set(df["Genres"].sum())))

# create functions that return title/index of anime
def get_index(title):
    return df[df.Title == title].index[0]
def get_title(index):
    return df[df.index == index]["Title"].values[0]

# anime genre dictionary
genre_dict = {

}

### Create a sidebar that contains author(myself)'s information
## contains an anime terminology that gives user a definition for each anime genre
image = Image.open('choso.jpeg')
st.sidebar.header("Welcome!")
st.sidebar.write()
st.sidebar.image(image, width = 250)
st.sidebar.write("My Github! [Click here!](https://github.com/emicervantes)")
st.sidebar.write("Dataset Link: [dataanime.csv](https://www.kaggle.com/canggih/anime-data-score-staff-synopsis-and-genre)")
st.sidebar.write("[Anime Recommendation Source](https://www.youtube.com/watch?v=XoTwndOgXBM&t=4824s)")
st.sidebar.write()
st.sidebar.write("Hello! My name is emi, and I am a huge anime fan! My favorite anime is Jujutsu Kaisen. The big question of this app is to find if the time each anime was aired determined its genre. I created a model using Scikit-learn LogisticRegression algorithm to predict the genre of anime based on its score and the time it was aired. Another feature that I added was the anime recommendation engine using Scikit-learn with a concept of cosine similarity.")


### main streamlit page
st.title("A N I M E")
st.write("Author: Emi Cervantes")
st.write()
st.write("Does the genre of anime change based on the time it was aired? We can look at the histogram to examine each distribution. The choices of factors are: Day of the Week,Season of the Year, and Time it was aired.")
st.write("Day of the Week: Sundays, Mondays, Tuesdays, Wednesdays, Thursdays, Fridays, Saturdays")
st.write("Season of the Year: Spring, Summer, Fall, Winter")
st.write("Time: 12PM-5:59PM, 6PM-11:59PM, 12AM-5:59AM, 6AM-11:59AM")
# Two columns for the selectbox where user chooses genre and predictor for histogram
col1, col2 = st.columns(2)
factor = col1.selectbox("Choose Your Genre:", genres_list)
user_x = col2.selectbox("Choose Your Predictor:", ["Day of the Week", "Season of the Year","Time"])
# Assign the variables of histogram based on user's choice
if user_x == "Day of the Week":
  order = ["Saturdays","Mondays","Tuesdays","Wednesdays","Thursdays","Fridays","Sundays"]
  X = 'day'
elif user_x == "Season of the Year":
  order = ["Winter","Summer","Fall","Spring"]
  X = 'Starting season'
else: 
  order = ['6PM~11:59PM','12PM ~ 5:59PM','6AM ~ 11:59AM','12AM ~ 6:00AM']
  X = 'time_s'
# create a new column in df, 'factor', that checks if each rows is a genre that user chooses
df[factor] = df["Genres"].map(lambda x: factor if factor in x else "Not " + factor)
# create a histogram with altair based on user's choices of factor and x
plot1 = alt.Chart(df).mark_bar(size = 40).encode(
    x = alt.X(X, sort = order, axis=alt.Axis(title=user_x)),
    y = alt.Y('count()'),
    color = factor,
    tooltip='count()',
).properties(
    width = 500,
    height = 500
).interactive()
# draw chart on Streamlit page
st.altair_chart(plot1, use_container_width=True)

### Anime genre predictor using logistic regression algorithm
### use sklearn LogisticRegression
numeric_cols = [c for c in df.columns if is_numeric_dtype(df[c])]
scaler = StandardScaler()
# create df2 that is a copy of df in which its values are standarized
df2 = df.copy()
scaler.fit(df2[numeric_cols])
df2[numeric_cols] = scaler.transform(df2[numeric_cols])
df2["anime_id"] = range(369)
st.subheader("What is your favorite anime's genre?")
# print dataframe df2 
st.dataframe(data = df2[numeric_cols])
st.write("Based on the choice of your anime, the model will use logistic regression algorithm to predict the genres of the anime using the time it was aired (week, season, time), and its score. The dataframe above contains all the columns that were used as the predictor in the model. The output is the genre of each anime. Since each anime in the dataset contains more than one genre, the model runs multiple times where each time it runs, it fits the same values of X with different Y (the genre).")
# ask user about their favorite anime
fav_anime = st.selectbox("What is your favorite anime?", (list(df["Title"])))
# logistic regression model
clf = LogisticRegression()
X = df2[["is_Mondays","is_Tuesdays","is_Wednesdays","is_Thursdays","is_Fridays","is_Saturdays","is_Sundays","is_night","is_midday","is_morning","is_midnight","is_Fall","is_Spring","is_Summer","is_Winter","Score"]]
# get index from the user's choice of anime
# function that takes an input of anime and predicts its genre using Sklearn Logistic Regression
def predict_genre(index):
    
    anime_genre_pred = []

    for i in genres_list:
        y = df2["Genres"].map(lambda genres_list: i if i in genres_list else "Not " + i)
        clf = LogisticRegression()
        clf.fit(X,y)
        pred = clf.predict(X)
        anime_genre_pred.append(pred[index])
    
    anime_genre_pred = [c for c in anime_genre_pred if "Not" not in c]
    
    return anime_genre_pred
# create separate list for predicted genres and actual genres of anime
anime_index = get_index(fav_anime)
predicted_genres_list = predict_genre(anime_index)
actual_genres_list = df.iloc[anime_index,13]
actual_genres = actual_genres_list[0]
for i in range(1,len(actual_genres_list)):
    actual_genres = actual_genres + ", " +  actual_genres_list[i]
# print both list of genres on Streamlit
# sometimes, the model will ouput no element, so put "na" if the length is zero
if len(predicted_genres_list) == 0:
    st.markdown("**Computer guessed:** NA")
else:
    predicted_genres = predicted_genres_list[0]
    for i in range(1,len(predicted_genres_list)):
        predicted_genres = predicted_genres + ", " +  predicted_genres_list[i]
    st.markdown("**_Computer guessed:_** " +  predicted_genres)

# find accuracy:
count = 0
for i in predicted_genres_list:
    if i in actual_genres_list:
        count = count + 1

st.markdown("**_Actual genres:_** " +  actual_genres)
st.write("Computer guessed " + str(count) + "/" + str(len(actual_genres_list)) + " genres!")
st.markdown("_______")

### anime recommendation using sklearn 
# https://www.youtube.com/watch?v=XoTwndOgXBM&t=4824s
# gives top 5 anime recommendations based on user's favorite anime
# Use CountVectorizer and cosine simialrity to find the similarity between the anime use chooses with other animes
# select features and combine features
features = ['Episodes','Genres','Starting season','Broadcast time','Producers','Rating','Score']
def combine_features(row):
    return str(row["Episodes"]) + " " + str(row["Genres"]) + " " + str(row["Starting season"]) + " " + str(row["Broadcast time"]) + " " + str(row["Producers"]) + " " + str(row["Rating"]) + " " + str(row["Score"])
df["combined_features"] = df.apply(combine_features, axis = 1)
# create count matrix from the combined column
cv = CountVectorizer()
count_matrix = cv.fit_transform(df["combined_features"])
# compute cosine similarity based on the count_matrix
cosine_sim = cosine_similarity(count_matrix)# get index of the anime from its title
similar_anime = list(enumerate(cosine_sim[anime_index]))

# get a list of similar animes in descending order of similarity score
sorted_similar_anime = sorted(similar_anime,key=lambda x: x[1],reverse=True)

### Streamlit anime reccomendation section
st.subheader("Anime Recommendations")
st.write("Based on your choice of favoriate anime, we've found your top five anime recommendations that you might like:")
# print top five anime recommendations
i = 0
for anime in sorted_similar_anime[1:]:
    rec = get_title(anime[0])
    index  = get_index(rec)

    rec_genres_list = df.iloc[index,13]
    rec_genres = rec_genres_list[0]

    for j in range(1,len(rec_genres_list)):
        rec_genres = rec_genres + ", " +  rec_genres_list[j]
    
    st.markdown("_______")
    st.subheader(rec)
    st.markdown("**Genres:** " + rec_genres)
    st.write("**Score:** " + str(df["Score"][index].round(decimals = 2)))
    st.write(df.iloc[index,20])

    i = i + 1
    if i == 5:
        break
