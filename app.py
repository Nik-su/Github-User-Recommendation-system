# Core Pkg
import streamlit as st 
import pandas as pd 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from pymongo import MongoClient

def load_data_from_mongodb():
    client = MongoClient("mongodb://localhost:27017/") 
    db = client["github_users_file4"]  
    collection = db["user_collection"]
    cursor = collection.find({}, {'_id': 0})
    data = list(cursor)
    client.close()
    df = pd.DataFrame(data)
    return df

# Vectorize text data and calculate cosine similarity matrix
# def vectorize_text_to_cosine_mat(data):
#     count_vect = CountVectorizer()
#     # Concatenate the desired fields to form text data
#     text_data = (data['Starred Repositories'].fillna('') + ' ' +
#                  data['Subscriptions'].fillna('') + ' ' +
#                  data['Organizations'].fillna('') + ' ' +
#                  data['Languages'].fillna('') + ' ' +
#                  data['Total Commits'].fillna('').astype(str))
#     cv_mat = count_vect.fit_transform(text_data)
#     cosine_sim_mat = cosine_similarity(cv_mat)
#     return cosine_sim_mat
# def vectorize_text_to_cosine_mat(data):
#     count_vect = CountVectorizer(binary=True)
#     # Concatenate the desired fields to form text data
#     text_data = (data['Starred Repositories'].fillna('') + ' ' +
#                  data['Subscriptions'].fillna('') + ' ' +
#                  data['Organizations'].fillna('') + ' ' +
#                  data['Languages'].fillna('') + ' ' +
#                  data['Total Commits'].fillna('').astype(str))
#     cv_mat = count_vect.fit_transform(text_data)
#     # Convert to binary format to represent presence/absence of each field
#     binary_cv_mat = cv_mat.toarray()
#     # Calculate the sum of matching fields between profiles
#     matching_counts = binary_cv_mat.dot(binary_cv_mat.T)
#     # Normalize the matching counts to range between 0 and 1
#     max_counts = matching_counts.max()
#     cosine_sim_mat = matching_counts / max_counts
#     return cosine_sim_mat
def vectorize_text_to_cosine_mat(data):
    count_vect = CountVectorizer(binary=True)
    
    # Convert each column to string and concatenate
    text_data = (data['Starred Repositories'].apply(lambda x: str(len(x)) if isinstance(x, list) else str(x)) + ' ' +
                 data['Subscriptions'].apply(lambda x: str(len(x)) if isinstance(x, list) else str(x)) + ' ' +
                 data['Organizations'].apply(lambda x: str(len(x)) if isinstance(x, list) else str(x)) + ' ' +
                 data['Languages'].apply(lambda x: str(len(x)) if isinstance(x, dict) else str(x)) + ' ' +
                 data['Total Commits'].fillna(0).astype(int).astype(str))
    
    # Convert the text data to numpy array
    text_data_array = text_data.values
    # Transform text data into binary format
    binary_feature_matrix = count_vect.fit_transform(text_data_array)
    # Calculate the cosine similarity matrix
    cosine_sim_mat = cosine_similarity(binary_feature_matrix)
    return cosine_sim_mat

# Recommendation function
def get_recommendation(title, cosine_sim_mat, df, num_of_rec=10):
    # Assuming 'Name' field is used for titles
    course_indices = pd.Series(df.index, index=df['Login']).drop_duplicates()
    idx = course_indices[title]
    sim_scores = list(enumerate(cosine_sim_mat[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    selected_indices = [i[0] for i in sim_scores[1:num_of_rec+1]]
    return df.iloc[selected_indices]

# Main function
def main():
    st.title("GitHub Profile Recommendation App")
    menu = ["Home", "Recommend", "About"]
    choice = st.sidebar.selectbox("Menu", menu)
    if choice == "Home":
        st.subheader("Home")
        df = load_data_from_mongodb()
        st.dataframe(df)
    elif choice == "Recommend":
        st.subheader("Recommend Profiles")
        df = load_data_from_mongodb()
        cosine_sim_mat = vectorize_text_to_cosine_mat(df)
        search_term = st.text_input("Search")
        num_of_rec = st.sidebar.number_input("Number", 4, 30, 7)
        if st.button("Recommend"):
            if search_term:
                results = get_recommendation(search_term, cosine_sim_mat, df, num_of_rec)
                # Display the results in a tabular format
                st.write("Full_Name, Login, Starred Repositories, Subscriptions, Organizations, Languages, Total Commits")
                for index, row in results.iterrows():
                    full_name = row['Full_Name']
                    login = row['Login']
                    starred_repositories = row['Starred Repositories']
                    subscriptions = row['Subscriptions']
                    organizations = row['Organizations']
                    languages = row['Languages']
                    total_commits = row['Total Commits']
                    
                    # Count the occurrences of Subscriptions and Organizations
                    subscriptions_count = str(subscriptions).count(',')
                    organizations_count = str(organizations).count(',')
                    
                    # Display the row with counts
                    st.write(f"{full_name}, {login}, {starred_repositories}, {subscriptions} ({subscriptions_count}), {organizations} ({organizations_count}), {languages}, {total_commits}")

if __name__ == '__main__':
    main()
