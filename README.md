# Accounting_Project_MSRI


## 1. Background and Motivation
<br />
The field of Chartered Accountancy (CA) requires professionals and students to navigate through complex questions related to accounting standards, taxation laws, financial management, and corporate regulations. For CA students and professionals, finding accurate answers to these questions can be challenging, especially when dealing with large volumes of regulatory material.
<br />
<br />
In this project, I am aiming to build a robust machine learning model that can answer questions related to Chartered Accountancy. By fine-tuning GPT-2 (Generative Pre-trained Transformer 2), I aim to develop a model that understands the intricacies of this domain and provides relevant answers to a wide range of CA-related queries. The model will be trained on a dataset containing content from official documents, accounting standards, tax regulations, and other sources, and will be capable of generating accurate, contextually appropriate answers.
<br />

## 2. Dataset

The dataset used for training and evaluating the Chartered Accountancy question-answering model consists of a collection of textual content related to various topics in the field of CA. The dataset is stored in a CSV file with the following columns:

Topic: The category or topic of the question (e.g., "Accounting Standards", "Taxation", "Corporate Law").
Content: The actual content or description from the dataset (e.g., accounting standard, tax law).
Question: A relevant question related to the content.
Answer: The corresponding answer, either model-generated or manually curated.

Example topics include:

Accounting Standards (e.g., IND AS)
Direct and Indirect Taxation
Corporate Laws (e.g., Companies Act)
Auditing Standards
Financial Management
Cost and Management Accounting

## 3. Structure
The project repository contains the following structure:

* `gpt2_fine_tune.py`: The main script for fine-tuning the GPT-2 model on the Chartered Accountancy dataset.
* `inference_webapp.py`: A Flask-based web application that allows users to ask questions and get answers from the fine-tuned GPT-2 model.
* `README.md`: This readme file.
* `data/`: Contains the dataset in CSV format (chartered_accountancy_dataset.csv), which is used for fine-tuning the model.
* `models/`: Directory where the fine-tuned GPT-2 model and tokenizer are saved after training.


## 4. Technologies Used and Requirements for Use
* `Python`
* `PyTorch`: open-source machine learning library used for building and training the model.
* `Transformers (Hugging Face)`: A library providing pre-trained transformer models, including BERT.
* `Pandas`: data manipulation and analysis library.
* `scikit-learn`: library used for splitting the dataset into training and testing sets.
* `TQDM`: library for displaying progress bars during training.
