from flask import Flask,render_template,request
from datasetload import *
import joblib

# load trained pipeline model 
model = joblib.load(r"C:\Users\princ\OneDrive\Pictures\Final Project of linear\Bike_resell_value_calc-main\Bike_resell_value_calc-main\LinearProjectModel\model\bike_price_model.pkl")


# flask application 
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html",bike_names=BIKE_NAMES,cities=CITIES,owners = OWNERS, brands = BRANDS)


@app.route("/predict",methods = ["POST"])
def predict():
    try:
        bike_name = request.form.get("bike_name")
        city = request.form.get("cities")
        owner = request.form.get("owners")
        brand = request.form.get("brands")
        kms_driver = float(request.form.get("kms_driven"))
        age = float(request.form.get("age"))
        power = float(request.form.get("power"))
        # Basic validation 
        if not all([bike_name,city,owner,brand]):
            return render_template("index.html",bike_name=BIKE_NAMES,cities = CITIES,owners =OWNERS,brands = BRANDS,error = "Please select all dropdown feilds")
        # if your data is valid
        input_dataframe = pd.DataFrame([{
            'bike_name' : bike_name,
            'city': city,
            'owner': owner,
            'brand': brand,
            'kms_driven':kms_driver,
            'age':age,
            'power':power
        }])
        pred = model.predict(input_dataframe)
        return render_template("index.html",bike_name = BIKE_NAMES,cities = CITIES,owners =OWNERS,brands = BRANDS,prediction = pred)
    except Exception as e :
        return render_template("index.html",bike_name = BIKE_NAMES,cities = CITIES,owners =OWNERS,brands = BRANDS,error = f"prediction error {e}")
    

if __name__ == '__main__':
    app.run(debug=True)