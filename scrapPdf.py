import PyPDF2

def load_pdf_content(filepath):

  with open(filepath, 'rb') as f:
    pdf = PyPDF2.PdfReader(f)
    text = ''
    for page in pdf.pages:
      text += page.extract_text()

  return text