import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity #it measures how similar 2 vectors are


df_of_movies=pd.read_csv('movies.csv') #loading movies dataset
df_of_ratings=pd.read_csv('ratings.csv') #loading ratings dataset


def filtering_based_on_genre(genres,best=5):
    
    filtering_top_movies=df_of_movies[df_of_movies['genres'].str.contains(genres, case=False, na=False)] #it is not case-sensitive and missing values will not be considered
    return filtering_top_movies.head(best)

def filtering_based_on_ratings(userId, best=5):
    
    item_matrix=df_of_ratings.pivot(index='userId',columns='movieId',values='rating').fillna(0) #creating a matrix and filling any missing values with zero
    
    similarity_between_users=cosine_similarity(item_matrix) #this calculates cosine similarity between the users
    
    df_of_similarity=pd.DataFrame(similarity_between_users,index=item_matrix.index,columns=item_matrix.index) #converts matrix to dataframe
    
    similarity_between_users=df_of_similarity[userId].sort_values(ascending=False).index[1:] #extracts similar values for the user
    
    average_ratings=item_matrix.loc[similarity_between_users].mean(axis=0) #selects the desired rows
    
    movies_already_watched=item_matrix.loc[userId][item_matrix.loc[userId]>0].index #gives movies the user has already watched
    
    recommended_movies=average_ratings[~average_ratings.index.isin(movies_already_watched)] #this gives recommendations
    
    return recommended_movies.sort_values(ascending=False).head(best)

print("This is a Movie Recommendattion Mini Project!")
print("This project uses Collaborative filtering and Content-based filtering to recommend you top movies!")
print("Services provided:")
print("Press 1 to Recommend you movies by genre!")
print("Press 2to  Recommend movies through collaboration!")

print("Enter 'exit' to quit!")

while True:
    chosen_option=input("Kindly choose and option:").strip()
    
    if chosen_option=='1':
        input_genre=input("Enter your favourite genre:").strip()
        movies_recommended=filtering_based_on_genre(input_genre)
        if not movies_recommended.empty:
            print(f"Top movies in {input_genre} genre:")
            for _,row in movies_recommended.iterrows():
                print(f"-{row['title']}")
                
                
        else:
            print("Sorry, no movies found in this genre! Wanna try another one?")
            
    elif chosen_option=='2':
        userid=int(input("Enter User ID:"))
        if userid in df_of_ratings['userId'].unique():
            movies_recommended=filtering_based_on_ratings(userid)
            print(f"Top movies for you with user id:{userid}")
            
            for movieid in movies_recommended.index:
                title_of_movie = df_of_movies[df_of_movies['movieId'] == movieid]['title'].values[0]
                print(f"- {title_of_movie}")
        else:
            print(f"Sorry, no data found for User ID {userid}.")
        
    elif chosen_option == 'exit':
        print("Thank you for using the Movie Recommendation System. Goodbye!")
        break
    
    else:
        print("Invalid option chosen. Kindly choose 1 or 2.")
    print("\n")