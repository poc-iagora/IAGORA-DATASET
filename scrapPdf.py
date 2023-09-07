import PyPDF2

def load_pdf_content(filepath):

  with open(filepath, 'rb') as f:
    pdf = PyPDF2.PdfReader(f)
    text = ''
    for page in pdf.pages:
      text += page.extract_text()

  return text


#import requests
#import PyPDF2
#import os

#def load_pdf_content(url):

 # response = requests.get(url)

  #with open("temp.pdf", "wb") as f:
   # f.write(response.content)

  #pdf = PyPDF2.PdfReader("temp.pdf")
  #text = ''
  #for page in pdf.pages:
   # text += page.extract_text()

  #os.remove("temp.pdf")

  #return text
