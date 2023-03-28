import json
import numpy as np
import pandas as pd
import faiss
import cohere

with open("econDB.json", "r") as f:
    db_json = json.load(f)
    
cohere_api_key = "Ct5cRnBoK8ZVVh2S2wTnPy9rSlrgjLdPvfXmOZpz"
cohere_client = cohere.Client(api_key=cohere_api_key)
    
df = pd.DataFrame(db_json["Posts"])

postTitles = [db_json['Posts'][i]['PostTitle'] for i in range(len(db_json['Posts']))]

postsEmbed = cohere_client.embed(texts=postTitles, model='large').embeddings

# Check the dimensions of the embeddings
postsEmbed = np.array(postsEmbed)
print(postsEmbed.shape)

## Indexing with FAISS

# Step 5: Create a search index
embedding_size = postsEmbed.shape[1]
index = faiss.IndexFlatL2(embedding_size)

postsEmbed_32 = postsEmbed.astype('float32')

# Step 6: Add embeddings to the search index
faiss.normalize_L2(postsEmbed_32)
index.add(postsEmbed_32)

# Step 7: Save the search index
faiss.write_index(index, 'faiss_index.idx')