import praw
import pandas as pd
import json
import datetime as dt
import numpy as np
import faiss

def authenticate(params):
    reddit = praw.Reddit(
        client_id=params['client_id'],
        client_secret=params['client_secret'],
        password=params['password'],
        user_agent="SemanticForum by u/DaCookieMonstav2",
        username=params['username']
    )
    return reddit

def create_post_obj(postid, title, comments, numComments):
    return {"PostID": postid, "PostTitle": title, "PostComments": comments, "NumComments": numComments}

def create_comment_obj(commentid, body):
    return {"CommentID": commentid, "CommentBody": body}

def fetch_posts_and_comments(subreddit, limit):
    econDB = {"Subreddit": "economics", "Posts": []}
    for submission in subreddit.hot(limit=limit):
        submission.comments.replace_more(limit=0)
        post_comments = [create_comment_obj(comment.id, comment.body) for comment in submission.comments]
        post = create_post_obj(submission.id, submission.title, post_comments, len(post_comments))
        econDB['Posts'].append(post)
    return econDB

def save_to_json(db, filename):
    with open(filename, "w") as f:
        json.dump(db, f, indent=4)

def calculate_stats(df):
    minComs = df.NumComments.min()
    maxComs = df.NumComments.max()
    meanComs = df.NumComments.mean()
    totalComments = df.NumComments.sum()
    return minComs, maxComs, meanComs, totalComments

def embed_query(query, cohere_client):
    query_embed = cohere_client.embed(texts=[query], model="large").embeddings
    
    # Convert the query_embed to a float32 NumPy array and normalize it
    query_embed_np = np.array(query_embed).astype('float32')
    faiss.normalize_L2(query_embed_np)

    return query_embed_np

def semantic_search(query_embed, index, db_json):
    
    # Mapping between unique indices and original post information
    id_to_post = {i: post for i, post in enumerate(pd.DataFrame(db_json).to_dict('records'))}
    
    # Retrieve the nearest neighbors
    D, I = index.search(query_embed, 5)

    # Format the results
    results = pd.DataFrame(data={'PostID': [id_to_post[i]['Posts']['PostID'] for i in I[0]], 
                                'PostTitle': [id_to_post[i]['Posts']['PostTitle'] for i in I[0]], 
                                'distance': D[0]})

    resultsComments = [{"PostID":db_json['Posts'][i]['PostID'], 
                        "PostComments":db_json['Posts'][i]['PostComments']} for i in range(len(db_json['Posts'])) if db_json['Posts'][i]['PostID'] in list(results.PostID)]
    

    resultsCommentsdf = pd.DataFrame(resultsComments)
    semantic_comments = results.merge(resultsCommentsdf, left_on='PostID', right_on='PostID')

    # they all start with the same comment becauase it is the moderation comment from the particular subreddit
    semantic_search = [results['PostTitle'][i] for i in range(len(results['PostTitle']))]
    
    return (semantic_search, semantic_comments)

def reddit_search(reddit,
                  subreddit_name, 
                  query):
    
    reddit_search_titles = []
    reddit_search_comments = []

    subreddit = reddit.subreddit(subreddit_name)

    search_obj = subreddit.search(query=query,
                                sort='hot',
                                syntax='lucene',
                                time_filter='all')

    for submission in search_obj:
        reddit_search_titles.append(submission.title)
        submission.comments.replace_more(limit=0)
        comments = [comment.body for comment in submission.comments]
        reddit_search_comments.append(comments)

    reddit_search = pd.DataFrame({"Posts":reddit_search_titles, "Comments":reddit_search_comments})
    
    return reddit_search