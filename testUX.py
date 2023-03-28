# TEST UX for JS interface
import streamlit as st
import json
import pandas as pd
import cohere
import faiss
import numpy as np
from utils import authenticate, reddit_search

# Load the saved Faiss index
index = faiss.read_index('faiss_index.idx')

with open("econDB.json", "r") as f:
    db_json = json.load(f)
    
with open('credentials.json') as f:
    params = json.load(f)

st.title('Semantic Forum - Reddit vs Semantic Search')

reddit = authenticate(params)

# Mapping between unique indices and original post information
id_to_post = {i: post for i, post in enumerate(pd.DataFrame(db_json).to_dict('records'))}

## SIDEBAR ##
st.sidebar.title('Semantic Forum')
st.sidebar.subheader('Search for posts and comments of a given subreddit')

st.sidebar.subheader('Credentials')
cohere_api_key = st.sidebar.text_input("Cohere API Key")

st.sidebar.subheader('Search Query')
query = st.sidebar.text_input('Enter a search query')

st.sidebar.subheader('Subreddit')
subreddit = st.sidebar.selectbox('Select a subreddit', ['economics', 'technology'])

st.sidebar.write(f'Current selection: {query} -> {subreddit}')

## MAIN BODY ##
#cohere_api_key = params['cohere_api_key']

cohere_client = cohere.Client(api_key=cohere_api_key)

# Get the query's embedding
query_embed = cohere_client.embed(texts=[query], model="large").embeddings

# Convert the query_embed to a float32 NumPy array and normalize it
query_embed_np = np.array(query_embed).astype('float32')
faiss.normalize_L2(query_embed_np)

# Retrieve the nearest neighbors
D, I = index.search(query_embed_np, 5)

# Format the results
results = pd.DataFrame(data={'PostID': [id_to_post[i]['Posts']['PostID'] for i in I[0]], 
                             'PostTitle': [id_to_post[i]['Posts']['PostTitle'] for i in I[0]], 
                             'distance': D[0]})

print(f"Query: '{query}'\nNearest neighbors:")
print(results)

resultsComments = [{"PostID":db_json['Posts'][i]['PostID'], "PostComments":db_json['Posts'][i]['PostComments']} for i in range(len(db_json['Posts'])) if db_json['Posts'][i]['PostID'] in list(results.PostID)]
len(resultsComments)

resultsCommentsdf = pd.DataFrame(resultsComments)
merged = results.merge(resultsCommentsdf, left_on='PostID', right_on='PostID')

# they all start with the same comment becauase it is the moderation comment from the particular subreddit
semantic_search = [results['PostTitle'][i] for i in range(len(results['PostTitle']))]


# get subreddit of interest(s) (will be user input once interface is up)
reddit_search = reddit_search(reddit, 
                              'economics', 
                              query
                              )

with st.expander("Top 5 Reddit Posts"):
    st.header('Top 5 Reddit Posts')

    top5_reddit = reddit_search.iloc[0:5]

    for i, post in enumerate(top5_reddit['Posts']):
        st.markdown(f'### {post}')
        st.write(top5_reddit['Comments'][i])


with st.expander("Top 5 Semantic Search Posts"):
    st.header('Top 5 Semantic Search Posts')

    top5_semantic = merged[:5]

    for i, post in enumerate(top5_semantic['PostTitle']):
        st.markdown(f'### {post}')
        st.write(top5_semantic['PostComments'][i])


