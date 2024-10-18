import os
import openai
import pandas as pd
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from dotenv import load_dotenv

load_dotenv()

# API Key and Model ID
openai.api_key = os.getenv('api_key')
fine_tuned_model_id = os.getenv('fine_tuned_model_id')

# Extract text
def extract_text_from_result(result):
    all_text = []
    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    all_text.append(word.value)
    return " ".join(all_text)

# Fine-tuned model
def send_to_fine_tuned_model(extracted_text):
    prompt = (
        "Please extract the defendants' names, property address, and legal description from the following text:\n"
        f"{extracted_text}\n\n"
        "Format your response as follows:\n"
        "defendant_name: [Defendant's Name]\n"
        "property_address: [Property Address]\n"
        "legal_description: [Legal Description]"
    )
    
    response = openai.ChatCompletion.create(
        model=fine_tuned_model_id,  
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response['choices'][0]['message']['content']

# Save data to CSV
def save_csv(data: list, csv_output_path: str):
    df = pd.DataFrame(data)  
    if os.path.exists(csv_output_path):
        df.to_csv(csv_output_path, mode='a', header=False, index=False) 
        print(f"Data appended for {data[0]['filename']}")
    else:
        df.to_csv(csv_output_path, index=False) 
        print(f"Creating new CSV file for {data[0]['filename']}")

# Folder Path
folder_path = 'R'
csv_output_path = 'extracted_data.csv'

for file_name in os.listdir(folder_path):
    if file_name.endswith('.pdf'):
        pdf_path = os.path.join(folder_path, file_name)
        docs = DocumentFile.from_pdf(pdf_path)

        # Load Doctr OCR model
        model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)
        
        # Get OCR results
        result = model(docs)
        print(f"Processing file: {file_name}")

        # Extract text from OCR result
        extracted_text = extract_text_from_result(result)

        # Send extracted text to fine-tuned model
        output = send_to_fine_tuned_model(extracted_text)
        print(f"Model Output:\n{output}\n")

        # Check the output format
        lines = output.split('\n')

        defendant_name = ''
        property_address = ''
        legal_description = ''
        
        try:
            for line in lines:
                if 'defendant_name:' in line:
                    defendant_name = line.split(':')[1].strip()
                elif 'property_address:' in line:
                    property_address = line.split(':')[1].strip()
                elif 'legal_description:' in line:
                    legal_description = line.split(':')[1].strip()
        except Exception as e:
            print(f"Error parsing output: {e}")

        # Prepare extracted data for the current file
        data = [{
            'filename': file_name,
            'defendant_name': defendant_name,
            'property_address': property_address,
            'legal_description': legal_description
        }]

        # Save extracted data to CSV
        save_csv(data, csv_output_path)

print(f"Processing completed. Data saved to {csv_output_path}")
