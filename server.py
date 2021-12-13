import json
from flask import Flask, request
from pre_trained_predict import predict


app = Flask(__name__)


@app.route('/')
def index():
    return 'hello world'



@app.route('/predict/<text>', methods=['GET'])
async def root(text):
    print(text)
    pred = await predict(text)
    return {"pred": pred, "text": text}



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
