import os
import json
import re
import time
import asyncio
from io import StringIO
from tqdm import tqdm
from tqdm.asyncio import tqdm
from pypdf import PdfReader, PdfWriter
import google.generativeai as genai
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from dotenv import load_dotenv
import spacy

load_dotenv()
my_api_key = os.getenv('MY_API_KEY')

def split_pdf(pdf_path, mappings):
    with tqdm(total=len(mappings), desc="Extracting Individual Documents...") as pbar:
        for docTitle, pageNumbers in mappings.items():
            split_pdf_pages(pdf_path, (pageNumbers[0], pageNumbers[-1]), docTitle)
            pbar.update(1)

def split_pdf_pages(pdf_path, pageNumbers, docTitle):
    with open(pdf_path, 'rb') as file:
        sanitized_docTitle = re.sub(r'[\\/:*?"<>|]', '_', docTitle).replace('\n', '').replace('\r', '')[:40]
        output_path = os.path.join(getOutputFolder(pdf_path), f"{sanitized_docTitle}.pdf")
        reader = PdfReader(file)
        writer = PdfWriter()
        for page_number in range(pageNumbers[0], pageNumbers[-1]+1):
            writer.add_page(reader.get_page(page_number))
        with open(output_path, 'wb') as output_file:
            writer.write(output_file)

def getOutputFolder(pdf_path):
    filename = os.path.splitext(os.path.basename(pdf_path))[0]
    sanitized_outputFolder = re.sub(r'[\\/:*?"<>|]', '_', filename)
    os.makedirs(sanitized_outputFolder, exist_ok=True)
    return sanitized_outputFolder

def pdf_to_text2(pdf_path):
    # List to store text by pages
    output_path = os.path.join(getOutputFolder(pdf_path), os.path.splitext(pdf_path)[0] + '.txt')
    nlp = spacy.load("en_core_web_sm")

    with open(pdf_path, 'rb') as pdf_file, open(output_path, 'w', encoding='utf-8') as f:
        for page_number, page in enumerate(extract_text_with_progress(pdf_path).split('')):
            text = page.strip()  # Get the text from the page and remove any leading/trailing whitespace

            #Remove names
            doc = nlp(text)
            names = set()
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    names.add(ent.text)
            for name in names:
                text = text.replace(name, "")
            # Remove Phone numbers
            phone_pattern = r'\b(?:\+?(\d{1,3}))?[-. (]*(\d{3})[-. )]*(\d{3})[-. ]*(\d{4})\b'
            phone_numbers = set()
            matches = re.finditer(phone_pattern, text)
            for m in matches:
                phone_numbers.add(m.group(0))
            for phone_number in phone_numbers:
                text = text.replace(phone_number, "")
                
            #Write page
            f.write(text)
            f.write(f"----- PAGE {page_number} -----")
            f.write("\f")


    return output_path
        

async def choose_pdf(): # Get list of PDF files in current directory 
    files = [f for f in os.listdir('.') if os.path.isfile(f) and f.lower().endswith('.pdf')]
    # Print list of PDF files
    for i, filename in enumerate(files):
        print(f"{i+1}: {filename}")

    # Get user's choice
    choice = int(input("Enter the number of the PDF you want to split into separate documents: ")) - 1

    # Check that choice is valid
    pdf_filename = None
    if 0 <= choice < len(files):
        pdf_filename = files[choice]
    else:
        print("Invalid choice.")
        return None

    if pdf_filename is not None:
        txtFilePath = pdf_to_text2(pdf_filename)
        print(f"Conversion completed. Text saved to: {txtFilePath}")
        response = await uploadPrompt(txtFilePath)
        mappings = parseJson(response)
        split_pdf(pdf_filename, mappings)
        print(f"\n\nExtraction complete, page numbers are: \n")
        summaryStr = ""
        for docTitle, pageNumbers in mappings.items():
            summaryStr += f"{docTitle} : Pg. {pageNumbers[0]} - {pageNumbers[-1]}\n"
        print(summaryStr)
    else:
        print("No PDF chosen. Exiting.")


async def uploadPrompt(txtFilePath):
    genai.configure(api_key=my_api_key)
    model = genai.GenerativeModel('gemini-1.5-pro-latest')
    
    # Ensure async calls are awaited
    sample_file =  genai.upload_file(path=txtFilePath)
    file =  genai.get_file(name=sample_file.name)
    
    #print(f"Retrieved file '{file.display_name}' as: {sample_file.uri}")
    
    prompt = (
        "Give the page bounds of the individual real estate documents contained in this file, "
        "combined when it seems they belong to the same document. Ignore lines. Return the page bounds zero indexed. If there "
        "is a gap in recognized documents, combine into the same page bounds. If there are separate documents, "
        "keep them separated. Make sure every page is accounted for. If there are unidentified documents, "
        "consider them part of the most recently encountered document. Express output as a JSON mapping from "
        "title of the document to an array containing the page ranges expressed as first page, last page."
    )
    
    # Create a task for generate_content
    generate_task = asyncio.create_task(model.generate_content_async([prompt, sample_file]))
    
    # Create an async progress bar
    print("Prompting Google AI to Split Document...")
    with tqdm(total=100, desc="Progress", bar_format='{l_bar}{bar}| {elapsed}', leave=True) as pbar:
        while not generate_task.done():
            await asyncio.sleep(0.2)
            pbar.update(1)
        pbar.n = 100
        pbar.close()
    
    # Wait for the task to complete and get the result
    response = await generate_task
    
    # Ensure async call is awaited
    genai.delete_file(sample_file.name)
    
    return response.text
    
def parseJson(response):
    _, _, result = response.partition('{')
    result, _, _ = result.rpartition('}')
    mappings = json.loads('{' + result + '}')
    return mappings

def extract_text_from_pdf_page(page, interpreter, device):
    interpreter.process_page(page)
    return device.get_text().getvalue()

def extract_text_with_progress(pdf_path):
    # Create resource manager
    rsrcmgr = PDFResourceManager()
    laparams = LAParams()
    output_string = StringIO()
    device = TextConverter(rsrcmgr, output_string, laparams=laparams)
    interpreter = PDFPageInterpreter(rsrcmgr, device)

    # Open the PDF file
    with open(pdf_path, 'rb') as fp:
        # Get the total number of pages for the progress bar
        total_pages = sum(1 for _ in PDFPage.get_pages(fp))
        fp.seek(0)  # Reset file pointer to the beginning

        # Create a progress bar with the total number of pages
        with tqdm(total=total_pages, desc="Extracting text from pages") as pbar:
            all_text = []
            for page in PDFPage.get_pages(fp):
                # Reset the output_string buffer
                output_string.truncate(0)
                output_string.seek(0)

                # Process the current page
                interpreter.process_page(page)
                
                # Get text from the buffer and add to the list
                text = output_string.getvalue()
                all_text.append(text)
                pbar.update(1)

    device.close()
    full_text = ''.join(all_text)
    return full_text

def main():
    asyncio.run(choose_pdf())
    return
    
if __name__ == "__main__":
    main()


