
from flask import Flask,render_template,request
import pickle
import numpy as np

model = pickle.load(open('model.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict_placement():
    cgpa = float(request.form.get('cgpa'))
    placement_exam_marks = int(request.form.get('placement_exam_marks'))
    
    # prediction
    result = model.predict(np.array([cgpa,placement_exam_marks]).reshape(1,2))

    if result[0] == 1:
        result = 'placed'
    else:
        result = 'not placed'
# pandas==2.1.4 numpy==1.26.4 scikit-learn==1.3.2  gunicorn 
# numpy==1.24.4  pandas==2.0.3
    return render_template('index.html',result=result)


if __name__ == '__main__':
    app.run(debug=True)