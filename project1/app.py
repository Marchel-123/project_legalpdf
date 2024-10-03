from pdf2image import convert_from_path
import pytesseract
import os
import pandas as pd
import openpyxl

def get_defendants(extracted_text: str) -> str:
    starting_checkpoint = ["vs.", "v"]
    end_checkpoint = "Defendants"

    # Find the first occurrence of any starting checkpoint
    start_index = None
    for start in starting_checkpoint:
        temp_index = extracted_text.find(start)
        if temp_index != -1:
            start_index = temp_index + len(start)  # Move the index to the end of "vs." or "v"
            break
    
    if start_index is None:
        return "Starting checkpoint not found"
    
    # Find the first occurrence of the end checkpoint after the starting point
    end_index = extracted_text.find(end_checkpoint, start_index)
    if end_index == -1:
        return "End checkpoint not found"
    
    # Slice the text between the start and end checkpoints
    return extracted_text[start_index:end_index].strip()


def extract_text_with_ocr(pdf_path):
    images = convert_from_path(pdf_path)
    text = ""
    for image in images:
        text += pytesseract.image_to_string(image)
    return text


# Define the directory where the PDFs are stored
folder_path = 'legal_documents'


# Iterate over all files in the folder
for file_name in os.listdir(folder_path):
    # Check if the file is a PDF
    if file_name.endswith('.pdf'):
        pdf_path = os.path.join(folder_path, file_name)
        print(f"Extracting text from: {file_name}")
        extracted_text = extract_text_with_ocr(pdf_path)
        result = get_defendants(extracted_text)
        print(result)