from transformers import pipeline

class AnswerGenerator:
    def __init__(self):
        self.qa_pipeline = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")

    def generate_answer(self, question, context):
        result = self.qa_pipeline(question=question, context=context)
        return result['answer']

    def generate_answer_from_docs(self, question, relevant_docs):
        context = " ".join([doc['text'] for doc in relevant_docs])
        return self.generate_answer(question, context)