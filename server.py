import json
from flask import Flask, request, render_template
from pre_trained_predict import predict, predict_with_pretrained, distil_predict


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        pre_text = request.form['pre_trained']
        our_text = request.form['our_own']
        distil_text = request.form['distil_trained']
        pred_pre = predict_with_pretrained(pre_text)
        pred_our = predict(our_text)
        pred_distil = distil_predict(distil_text)
        return render_template('index.html', 
            text=pre_text, 
            prediction=pred_pre, 
            our_text=our_text,
            our_prediction=pred_our,
            distil_text=distil_text,
            distil_prediction=pred_distil)

    elif request.method == 'GET':
        return render_template('index.html', text="", prediction="")



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
