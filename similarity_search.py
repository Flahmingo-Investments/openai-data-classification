import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# load file path environment variable
embeddings_file_path = os.environ(PATH_TO_EMBEDDINGS)

# load embeddings
embeddings = pd.read_csv(embeddings_file_path)

def best_match(query_embedding):
    # define empty list for scores
    scores =[]

    # reshape query embedding
    query_embedding = np.array(query_embedding).reshape(1 ,-1)

    for i in range(len(embeddings["embedding"])):
        # get pre-computed pii embedding
        pii_embedding = embeddings.iloc[i ,2]

        # reshape pii embedding
        pii_embedding = np.array(pii_embedding).reshape(1 ,-1)


        similarity_score =cosine_similarity(pii_embedding
                                             ,query_embedding)
        # append tuple containing the original text, label and similarity score to the list
        scores.append((embeddings.iloc[i ,0] ,embeddings.iloc[i ,1] ,similarity_score))

    # sort list
    sorted_scores = sorted(scores, key=lambda x: x[2], reverse=True)

    # return best match
    return sorted_scores[:1]
