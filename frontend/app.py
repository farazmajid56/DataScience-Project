from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

saved_model = "./model/ridge_regression_model.joblib"
saved_scaler = "./model/scaler.joblib"
model = joblib.load(saved_model)
scaler = joblib.load(saved_scaler)

@app.route("/", methods=["GET", "POST"])
def home():
    
    if request.method == "POST":
        bedrooms = request.form["bedrooms"]
        bathrooms = request.form["bathrooms"]
        waterfront = request.form["waterfront"]
        view = request.form["view"]
        sqft_living = request.form["sqft_living"]
        sqft_lot = request.form["sqft_lot"]
        floors = request.form["floors"]
        condition = request.form["condition"]
        sqft_above = request.form["sqft_above"]
        sqft_basement = request.form["sqft_basement"]
        yr_built = request.form["yr_built"]
        yr_renovated = request.form["yr_renovated"]

        if bedrooms != "":
            # Converting input values to float
            input_values = np.array([1, bedrooms, bathrooms, waterfront, view, sqft_living, sqft_lot, floors, condition, sqft_above, sqft_basement, yr_built, yr_renovated], dtype=float)
            
            # Reshaping the input for prediction
            input_values = input_values.reshape(1, -1)

            # Scaling the input using the same scaler used during training
            input_scaled = scaler.transform(input_values)

            # Making prediction
            predicted_price = model.predict(input_scaled)[0]            
            
            # Formatting the predicted price
            predicted_price = "${:,.2f}".format(predicted_price/1000)      
            
        else:
            predicted_price = "Please Enter Integer Values In All Fields"

        return render_template("index.html", price=predicted_price, metrics="R2 Score: 0.78, MAE: 126099.32, MSE: 31561219380.05, RMSE: 177654.78")

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)