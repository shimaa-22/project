from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn
import webbrowser
import logging

# Import classes from other files
from pdf_processor import PDFProcessor
from text_preprocessor import TextPreprocessor
from document_embedder import DocumentEmbedder
from document_retriever import DocumentRetriever
from answer_generator import AnswerGenerator
from PDFSummarizer import PDFSummarizer
app = FastAPI()

# Initialize components
pdf_processor = PDFProcessor("./pdfs")
text_preprocessor = TextPreprocessor()
pdf_summarizer = PDFSummarizer()
document_embedder = DocumentEmbedder()
document_retriever = None  # Will be initialized after processing documents
answer_generator = AnswerGenerator()

# Jinja2 Templates directory
templates = Jinja2Templates(directory="templates")

class Query(BaseModel):
    question: str
@app.post("/summarize/")
async def summarize_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
         raise HTTPException(status_code=400, detail="Invalid file format. Please upload a PDF.")
    
    try:
        text = pdf_processor.process_pdf(file)
        summary = pdf_summarizer.summarize(text)
        return {"summary": summary}
    except Exception as e:
        logging.error("Error summarizing PDF: %s", e)
        raise HTTPException(status_code=500, detail="An error occurred while summarizing the PDF.")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/QA", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("QA.html", {"request": request})

@app.get("/summarize", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("summarize.html", {"request": request})
@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    if file.filename.endswith('.pdf'):
        text = pdf_processor.process_pdf(file)
        return {"message": "PDF uploaded and processed successfully"}
    else:
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload a PDF.")

@app.post("/process_documents/")
async def process_documents():
    global document_retriever
    documents = pdf_processor.get_all_documents()
    processed_docs = [
        {"filename": doc["filename"], "text": text_preprocessor.preprocess(doc["text"])}
        for doc in documents
    ]
    embedded_docs = document_embedder.embed_documents(processed_docs)
    document_retriever = DocumentRetriever(embedded_docs)
    return {"message": f"Processed and embedded {len(documents)} documents"}

@app.post("/query/")
async def query(query: Query):
    if document_retriever is None:
        raise HTTPException(status_code=400, detail="Documents not processed. Please process documents first.")
    
    processed_query = text_preprocessor.preprocess(query.question)
    query_embedding = document_embedder.embed(processed_query)
    relevant_docs = document_retriever.retrieve(query_embedding)
    answer = answer_generator.generate_answer_from_docs(query.question, relevant_docs)
    
    return {
        "question": query.question,
        "answer": answer,
        "relevant_documents": [doc["filename"] for doc in relevant_docs]
    }

if __name__ == "__main__":
    # Automatically open in Chrome browser
    # webbrowser.open("http://127.0.0.1:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
