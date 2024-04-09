import numpy as np
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
cv = pickle.load(open("models/house_model.pkl",'rb'))

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method=='POST':
        v1=int(request.form['area'])
        v2=int(request.form['bedroom'])
        v3=np.array([[v1,v2]])
        prediction = cv.predict(v3)
        return render_template('index.html', prediction=prediction)
    else:
        return render_template('index.html')
  
if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)