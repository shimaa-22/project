from transformers import pipeline

class PDFSummarizer:
    def __init__(self):
            self.summarization_pipeline = pipeline("summarization", model="facebook/bart-large-cnn", num_beams=5, min_length=50, max_length=200)


    def summarize(self, text, max_length=130, min_length=30, do_sample=False):
        # Split text into chunks if it's too long
        max_tokens = 1024  # Set the max tokens for the model
        chunks = []
        
        # Tokenize the text and split into manageable chunks
        current_chunk = ""
        for sentence in text.split('. '):
            if len(self.summarization_pipeline.tokenizer(current_chunk + sentence)['input_ids']) <= max_tokens:
                current_chunk += sentence + '. '
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + '. '

        # Add the last chunk if any
        if current_chunk:
            chunks.append(current_chunk.strip())

        # Summarize each chunk and combine summaries
        summaries = []
        for chunk in chunks:
            summary = self.summarization_pipeline(
                chunk,
                max_length=max_length,
                min_length=min_length,
                do_sample=do_sample
            )
            summaries.append(summary[0]['summary_text'])

        # Combine all summaries
        return " ".join(summaries)
