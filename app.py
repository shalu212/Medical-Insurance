from flask import Flask,render_template,url_for,request,redirect
import numpy as np
import pandas as pd
import joblib
import pickle


app = Flask(__name__)

model = joblib.load('regressor.pkl')
onehot = joblib.load('OneHotee.joblib')


@app.route('/')
@app.route('/main')
def main():
	return render_template('main.html')

@app.route('/predict',methods=['POST'])
def predict():
	int_features =[[x for x in request.form.values()]]
	c = ["sex","smoker","region","age","bmi","children"]
	df = pd.DataFrame(int_features,columns=c)	
	l = onehot.transform(df.iloc[:,0:3])
	c = onehot.get_feature_names_out()
	t = pd.DataFrame(l,columns=c)
	l2 = df.iloc[:,3:]
	final =pd.concat([l2,t],axis=1)
	result = model.predict(final)
	print("The Result is :",result)


	

	return render_template("main.html",prediction_text="Expected Medical Charges in $ : {}".format(result))


if __name__ == "__main__":
    app.debug=True
    app.run(host = '0.0.0.0', port=6060, debug =True)
