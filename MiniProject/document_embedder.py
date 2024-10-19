from sentence_transformers import SentenceTransformer

class DocumentEmbedder:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def embed(self, text):
        return self.model.encode(text)

    def embed_documents(self, documents):
        embedded_docs = []
        for doc in documents:
            embedding = self.embed(doc['text'])
            embedded_docs.append({
                'filename': doc['filename'],
                'text': doc['text'],
                'embedding': embedding
            })
        return embedded_docs