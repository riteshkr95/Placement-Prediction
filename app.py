from flask import Flask ,request ,render_template ,jsonify
import joblib
import pickle
import numpy as np

# load the train model
model=joblib.load("log_reg.pkl")


app=Flask(__name__)

@app.route("/")
def home():
   return render_template("index.html")

@app.route("/predict" ,methods= ["POST"])

def predict():
       a=eval(request.form.get("cgpa"))
       b=eval(request.form.get("placement_exam_marks"))
      
    


  # make prediction

       prediction=model.predict([[a,b]])

       if prediction[0]== 0 :
            return render_template("P_0.html")
       else:
            return render_template("P_1.html")

       
if __name__ == "__main__" :
     
     app.run(debug=True  ,port=5757)
    

  
  