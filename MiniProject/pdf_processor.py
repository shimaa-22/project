import PyPDF2
import os

class PDFProcessor:
    def __init__(self, storage_dir):
        self.storage_dir = storage_dir
        if not os.path.exists(storage_dir):
            os.makedirs(storage_dir)

    def process_pdf(self, file):
        filename = file.filename
        file_path = os.path.join(self.storage_dir, filename)
        
        with open(file_path, "wb") as buffer:
            buffer.write(file.file.read())
        
        return self.extract_text(file_path)

    def extract_text(self, file_path):
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text

    def get_all_documents(self):
        documents = []
        for filename in os.listdir(self.storage_dir):
            if filename.endswith(".pdf"):
                file_path = os.path.join(self.storage_dir, filename)
                text = self.extract_text(file_path)
                documents.append({"filename": filename, "text": text})
        return documents