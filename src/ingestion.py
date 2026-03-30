import os
import pdfplumber
import docx

def read_pdf(file_path):
    """Extracts text from a PDF file."""
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
    return text

def read_docx(file_path):
    """Extracts text from a DOCX file."""
    text = ""
    try:
        doc = docx.Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        print(f"Error reading DOCX {file_path}: {e}")
    return text

def read_txt(file_path):
    """Extracts text from a TXT file."""
    text = ""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except Exception as e:
        # Fallback encoding if utf-8 fails
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                text = f.read()
            print(f"Read TXT {file_path} using latin-1 encoding")
        except Exception as e:
            print(f"Error reading TXT {file_path}: {e}")
    return text

def read_file(file_path):
    """
    Reads a file and returns its text content based on its extension.
    Supported extensions: .pdf, .docx, .txt
    """
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    
    if ext == '.pdf':
        return read_pdf(file_path)
    elif ext == '.docx':
        return read_docx(file_path)
    elif ext == '.txt':
        return read_txt(file_path)
    else:
        print(f"Unsupported file format: {ext} for file {file_path}")
        return ""

def load_documents_from_directory(directory_path):
    """
    Loads all supported documents from a directory.
    Returns a dictionary of filename: text.
    """
    docs = {}
    if not os.path.exists(directory_path):
        print(f"Directory not found: {directory_path}")
        return docs
        
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            text = read_file(file_path)
            if text.strip():
                docs[filename] = text
            
    return docs
