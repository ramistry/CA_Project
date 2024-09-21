# Accounting_Project

## 1. Background and Motivation
<br />
The field of Chartered Accountancy (CA) requires professionals and students to navigate through complex questions related to accounting standards, taxation laws, financial management, and corporate regulations. For CA students and professionals, finding accurate answers to these questions can be challenging, especially when dealing with large volumes of regulatory material.
<br />
<br />
In this project, we aim to build a robust machine learning model that can answer questions related to Chartered Accountancy. By fine-tuning GPT-2 (Generative Pre-trained Transformer 2), we aim to develop a model that understands the intricacies of this domain and provides relevant answers to a wide range of CA-related queries. The model will be trained on a dataset containing content from official documents, accounting standards, tax regulations, and other sources, and will be capable of generating accurate, contextually appropriate answers.
<br />
In this project, we decided to build such ML model: the Urgency Classifier.

## 2. Dataset

The dataset used for training and evaluating the urgency classifier model consists of customer support tickets. 
Each ticket includes a textual description of the issue and a corresponding priority label indicating the urgency of the issue. The dataset is stored in a CSV file with the following columns:

Ticket Description: A textual description of the customer's issue or request.
Ticket Priority: The urgency level of the ticket, which can be one of the following categories: "Low", "Medium", "High", "Critical".

## 3. Structure
* `Files` urgency_classifier.py: The main script for training and evaluating the urgency classifier model. README.md: This readme file.
* `data` customer_support_tickets.csv data (csv) we used to train model

## 4. Technologies Used and Requirements for Use
* `Python`
* `PyTorch`: open-source machine learning library used for building and training the model.
* `Transformers (Hugging Face)`: A library providing pre-trained transformer models, including BERT.
* `Pandas`: data manipulation and analysis library.
* `scikit-learn`: library used for splitting the dataset into training and testing sets.
* `TQDM`: library for displaying progress bars during training.
