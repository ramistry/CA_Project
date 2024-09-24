from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pandas as pd
import re

app = Flask(__name__)


model_sentence_reader = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
tokenizer = GPT2Tokenizer.from_pretrained('./fine_tuned_gpt2')
final_model = GPT2LMHeadModel.from_pretrained('./fine_tuned_gpt2')


df = pd.read_csv('accounting_standards_fixed.csv')


def extract_relevant_sections(df, keyword):
    keyword = keyword.lower()
    filtered_df = df[df['Name of Accounting Standard'].str.contains(keyword, case=False)]
    return filtered_df

def encode_column(column_name):
    return model_sentence_reader.encode(column_name)


df['embedding'] = df['Name of Accounting Standard'].apply(encode_column)
df['Trained Response'] = df['Trained Responses'].apply(encode_column)

def generate_response(user_query):

    keyword = re.search(r'\b(audit|tax|accounting)\b', user_query.lower())
    
    if keyword:
        relevant_df = extract_relevant_sections(df, keyword.group(0))
    else:
        relevant_df = df
    
    user_embedding = model_sentence_reader.encode(user_query)
    #might need to split lambda function and create separate one
    relevant_df['similarity'] = relevant_df['embedding'].apply(lambda x: cosine_similarity([user_embedding], [x]).flatten()[0])
    most_similar_row = relevant_df.loc[relevant_df['similarity'].idxmax()]

    #need to edit test csv for this
    prompt = f"Explain the following accounting standard: {most_similar_row['Name of Accounting Standard']}"

    
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    
    output = final_model.generate(input_ids, attention_mask=attention_mask, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return response

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/chat', methods=['POST'])
def chat():
    user_query = request.json.get('message')
    response = generate_response(user_query)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, debug=True)
