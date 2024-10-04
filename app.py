import os
from doctr.io import DocumentFile
from doctr.models import ocr_predictor


def extract_text_from_result(result):
    all_text = []
    
    # Iterate over the pages in the result (in case there are multiple pages)
    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    all_text.append(word.value)
    
    return " ".join(all_text)  


# Define the directory where the PDFs are stored
folder_path = 'legal_documents'

# Iterate over all files in the folder
for file_name in os.listdir(folder_path):

    if file_name.endswith('.pdf'):
        pdf_path = os.path.join(folder_path, file_name)
        docs = DocumentFile.from_pdf(pdf_path)
        model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)
        result = model(docs)
        print(file_name)
        extracted_text = extract_text_from_result(result)
        print(extracted_text)
