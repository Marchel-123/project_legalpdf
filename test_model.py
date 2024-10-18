import os
import openai
import pandas as pd
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from dotenv import load_dotenv

load_dotenv()

# API Key
openai.api_key = os.getenv('api_key')

def extract_text_from_result(result):
    all_text = []
    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    all_text.append(word.value)
    return " ".join(all_text)

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
        model="ft:gpt-3.5-turbo-0125:personal::AHOjGZMJ", #ft:gpt-3.5-turbo-0125:personal::AHMrDYCQ
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response['choices'][0]['message']['content']

folder_path = 'PDF Samples #2' #PDF Samples #2 legal_documents
data_list = []

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

        # Append extracted data
        data_list.append({
            'filename': file_name,
            'defendant_name': defendant_name,
            'property_address': property_address,
            'legal_description': legal_description
        })



# Save data to CSV file
csv_output_path = 'extracted_data.csv'
if os.path.exists(csv_output_path):
    df_existing = pd.read_csv(csv_output_path)
    df_new = pd.DataFrame(data_list)
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    df_combined = df_combined.drop_duplicates()
    df_combined.to_csv(csv_output_path, index=False)
    print(f"Data loaded from {csv_output_path}, new data added")
else:
    df = pd.DataFrame(data_list)
    df.to_csv(csv_output_path, index=False)
    print(f"Creating new CSV file: {csv_output_path}")

print(f"Data saved to {csv_output_path}")
