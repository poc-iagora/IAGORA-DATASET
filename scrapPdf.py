# import PyPDF2

# def load_pdf_content(filepath):

#   with open(filepath, 'rb') as f:
#     pdf = PyPDF2.PdfReader(f)
#     text = ''
#     for page in pdf.pages:
#       text += page.extract_text()

#   return text

import PyPDF2
import requests
from io import BytesIO

def load_pdf_content(url):
    response = requests.get(url)
    response.raise_for_status()  # Raise an error for bad responses

    with BytesIO(response.content) as f:
        pdf = PyPDF2.PdfReader(f)
        text = ''
        for page in pdf.pages:
            text += page.extract_text()

    return text


# import PyPDF2
# import requests
# import chardet

# def remove_null_characters(pdf_content):

#   return pdf_content.replace('\0', '')

# def load_pdf_content(filepath):

#   with open(filepath, 'rb') as f:
#     pdf = PyPDF2.PdfReader(f)
#     text = ''
#     for page in pdf.pages:
#       text += page.extract_text()

#   return text

# def load_online_pdf_content(url):

#   response = requests.get(url)
#   pdf_content = response.content

#   # Detect the encoding of the PDF file.
#   encoding = chardet.detect(pdf_content)['encoding']

#   # If the encoding is None, use iso-8859-1 as default.
#   if encoding is None:
#     encoding = 'iso-8859-1'

#   # Decode the PDF file using the detected encoding.
#   pdf_content = pdf_content.decode(encoding)

#   # Remove null characters from the PDF file.
#   pdf_content = remove_null_characters(pdf_content)

#   return load_pdf_content(pdf_content)


