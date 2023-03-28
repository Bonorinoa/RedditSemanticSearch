import praw
import pandas as pd
import json
import datetime as dt

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