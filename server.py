from flask import Flask,render_template,request
import pickle

app = Flask(__name__)

#loading the model
try :
    model = open('model.pkl','rb')
except :
    import Model_Dumper
    exec('Model_Dumper')
    model = open('model.pkl','rb')
classifier = pickle.load(model)
model.close()


@app.route('/' , methods=["GET","POST"])

def predict():
    if request.method == "POST":
        dict = request.form
        val = list(dict.values())
        
        # Code for prediction
        inputFeatures = list(map(float,val[2:]))
        prediction = classifier.predict([inputFeatures])[0]

        #Code for greeting
        if prediction == 'bad':
            greet = 'Please find some other wine!!\nAs this wine might affect your health.'
        elif prediction == 'good' :
            greet = 'You are safe to use this wine!!\nSome other wines are also availabe with better quality than this.'
        else :
            greet = 'You are absolutely safe to use this wine!!\nDon\'t use it in high quantity as it is risky'
        greet = greet.split('\n')
        return render_template('result.html',name=val[0],prediction=prediction,greeting1=greet[0],greeting2=greet[1])
    return render_template('main_page.html')


if __name__=='__main__':
    app.run(debug=True)