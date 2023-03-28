import praw
import pandas as pd
import json
import datetime as dt
from utils import authenticate, fetch_posts_and_comments, save_to_json, calculate_stats

with open('credentials.json') as f:
    params = json.load(f)

reddit = authenticate(params)

# get subreddit of interest(s) (will be user input once interface is up)
subreddit = reddit.subreddit('economics')

# get top 100 hottest posts and comments for subreddit of interest(s)
econDB = fetch_posts_and_comments(subreddit, limit=100)

save_to_json(econDB, "econDB.json")

# dictionar/json to pandas dataframe
df = pd.DataFrame(econDB["Posts"])

# some summary statistics
minComs, maxComs, meanComs, totalComments = calculate_stats(df)
print(f"The DB has {len(df)} posts with {totalComments} comments.\n The post with least comments had {minComs}," \
    + f"\n the one with most {maxComs}, \n and the overall average number of comments per post is {meanComs}")