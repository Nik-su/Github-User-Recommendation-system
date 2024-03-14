# Core Pkg
import streamlit as st 
import pandas as pd 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from pymongo import MongoClient
from pymongo.server_api import ServerApi
import os

def load_data_from_mongodb():
    uri = os.environ.get("MONGODB_URI")
    client = MongoClient(uri, server_api=ServerApi('1')) 
    db = client["Github_Profiles"]  
    collection = db["User_Collection"]
    cursor = collection.find({}, {'_id': 0})
    data = list(cursor)
    client.close()
    df = pd.DataFrame(data)
    return df

def vectorize_text_to_cosine_mat(data):
    count_vect = CountVectorizer(binary=True)
    text_data = (data['Starred Repositories'].apply(lambda x: str(len(x)) if isinstance(x, list) else str(x)) + ' ' +
                 data['Subscriptions'].apply(lambda x: str(len(x)) if isinstance(x, list) else str(x)) + ' ' +
                 data['Organizations'].apply(lambda x: str(len(x)) if isinstance(x, list) else str(x)) + ' ' +
                 data['Languages'].apply(lambda x: str(len(x)) if isinstance(x, dict) else str(x)) + ' ' +
                 data['Total Commits'].fillna(0).astype(int).astype(str))
    
    text_data_array = text_data.values
    binary_feature_matrix = count_vect.fit_transform(text_data_array)
    cosine_sim_mat = cosine_similarity(binary_feature_matrix)
    return cosine_sim_mat

def get_recommendation(title, cosine_sim_mat, df, num_of_rec=10):
    course_indices = pd.Series(df.index, index=df['Login']).drop_duplicates()
    idx = course_indices[title]
    sim_scores = list(enumerate(cosine_sim_mat[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    selected_indices = [i[0] for i in sim_scores[1:num_of_rec+1]]
    return df.iloc[selected_indices]

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
                st.write("Full_Name, Login, Starred Repositories, Subscriptions, Organizations, Languages, Total Commits")
                for index, row in results.iterrows():
                    full_name = row['Full_Name']
                    login = row['Login']
                    starred_repositories = row['Starred Repositories']
                    subscriptions = row['Subscriptions']
                    organizations = row['Organizations']
                    languages = row['Languages']
                    total_commits = row['Total Commits']
                    subscriptions_count = str(subscriptions).count(',')
                    organizations_count = str(organizations).count(',')
                    st.write(f"{full_name}, {login}, {starred_repositories}, {subscriptions} ({subscriptions_count}), {organizations} ({organizations_count}), {languages}, {total_commits}")

if __name__ == '__main__':
    main()
