# TEST UX for JS interface
import streamlit as st
import json
import cohere
import faiss
from utils import authenticate, reddit_search, embed_query, semantic_search

# Load the saved Faiss index
index = faiss.read_index('faiss_index.idx')

with open("econDB.json", "r") as f:
    db_json = json.load(f)
    
with open('credentials.json') as f:
    params = json.load(f)

st.title('Semantic Forum - Reddit vs Semantic Search')

reddit = authenticate(params)

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

if cohere_api_key == '':
    st.warning('Please enter your Cohere API key in the sidebar')
elif query == '':
    st.warning('Please enter a search query in the sidebar') 
else:
    cohere_client = cohere.Client(api_key=cohere_api_key)

    # Get the query's embedding
    query_embed = embed_query(query, cohere_client)
    
    # semantic search
    semantic_search, semantic_comments = semantic_search(query_embed, index, db_json)

    # keyword reddit search
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

        top5_semantic = semantic_comments[:5]

        for i, post in enumerate(top5_semantic['PostTitle']):
            st.markdown(f'### {post}')
            st.write(top5_semantic['PostComments'][i])


