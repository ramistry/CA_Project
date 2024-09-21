from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)


model_sentence_reader = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
tokenizer = GPT2Tokenizer.from_pretrained('./fine_tuned_gpt2')
final_model = GPT2LMHeadModel.from_pretrained('./fine_tuned_gpt2')

df = pd.read_csv('accounting_standards_fixed.csv')
df['embedding'] = df['Name of Accounting Standard'].apply(lambda x: model_sentence_reader.encode(x))

def generate_response(user_query):
    user_embedding = model_sentence_reader.encode(user_query)
    df['similarity'] = df['embedding'].apply(lambda x: cosine_similarity([user_embedding], [x]).flatten()[0])
    most_similar_row = df.loc[df['similarity'].idxmax()]

    prompt = f"Explain the following accounting standard: {most_similar_row['Name of Accounting Standard']}"
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    output = final_model.generate(input_ids, attention_mask=attention_mask, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2)
    
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

@app.route('/ask', methods=['POST'])
def ask_question():
    user_query = request.json.get('query')
    response = generate_response(user_query)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
