import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class DocumentRetriever:
    def __init__(self, embedded_docs):
        self.embedded_docs = embedded_docs

    def retrieve(self, query_embedding, top_k=3):
        similarities = [
            cosine_similarity(query_embedding.reshape(1, -1), 
                              doc['embedding'].reshape(1, -1))[0][0]
            for doc in self.embedded_docs
        ]
        
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return [self.embedded_docs[i] for i in top_indices]