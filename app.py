from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the model and vectorizer
with open('models/sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('models/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_input = request.form['text']
        
        # Preprocess input
        input_tfidf = vectorizer.transform([user_input])
        
        # Predict sentiment
        prediction = model.predict(input_tfidf)
        
        return render_template('index.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
