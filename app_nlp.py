from flask import Flask, request, jsonify, render_template
import pickle


app = Flask(__name__)
model = pickle.load(open('model_chatbot.pkl', 'rb'))
vocabulary = pickle.load(open('chatbot_vocabulary.pkl', 'rb'))
intent_names = pickle.load(open('response_chatbot.pkl', 'rb'))

@app.route('/')
def home():
    print('****** model : ', model)
    print('****** vocabulary : ', vocabulary)
    print('****** intent_names : ', intent_names)
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    user_utterance = [x for x in request.form.values()]
    utterance_vect = vocabulary.transform(user_utterance)
    
    output = model.predict(utterance_vect)
    prediction = intent_names[output]
    return prediction[0]


@app.route('/predict_api', methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    user_utterance = request.get_json(force=True)
    utterance_vect = vocabulary.transform([user_utterance['data']])
    output = model.predict(utterance_vect)
    prediction = intent_names[output]

    return jsonify({'data': prediction[0]})

if __name__ == "__main__":

    app.run(debug=True)