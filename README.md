# PDF Text Extraction  
This project processes the PDF files to extract the specific information such as defendants' names, property addresses and legal descriptions. It utilizes the OCR (Optical Character Recognition) to read the text from the PDF files and fine-tuned an OPENAI model to generate text.

## Instructions:
The collected pdf files will be use to fine-tune an OPENAI model with API key. For this project will utilize a `gpt-3.5-turbo-0125`. 

In this link is the complete process of fine-tuning model in the command prompt:

https://platform.openai.com/docs/guides/fine-tuning/create-a-fine-tuned-model



## Process:
- Install the python libraries which the dependencies found in `requirements.txt` file.

- Using `.env` to store the API key and the fine-tuned model ID.

  ```bash
  api_key= api_key
  fine_tuned_model_id = fine_tuned_model_id

### Here are the tasks after installation:

1. Loads the necessary python libraries.
   
2. Reads PDF files from a specified folder.
   
3. Uses an OCR model to extract text from the PDF files.

4. Sends the extracted text to a fine-tuned OpenAI model using to extract specific text information.
   
5. Saves the extracted information into a CSV file.
   

## Example Output
The output CSV file will contain the following columns:

**filename**: The name of the PDF file processed.

**defendant_name**: The name of the defendant extracted from the PDF files.

**property_address**: The property address extracted from the PDF files.

**legal_description**: The legal description extracted from the PDF files.




