import pickle
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
# Create an instance of LabelEncoder and fit
encoder = LabelEncoder()
# Load the trained ML model from the pickle file
with open('lung_ka_cancer_rf.pkl', 'rb') as file:
    model = pickle.load(file)

# Define the home page route
@app.route('/')
def home():
    return render_template('one.html')

# Define the form page route
@app.route('/form')
def form():
    return render_template('form.html')

@app.route('/results', methods=['GET', 'POST'])
def results():
    if request.method == 'POST':
        # Print out the form data for debugging
        print(request.form)

        # Extract the form data
        age= int(request.form.get('age'))
        gender = request.form.get('gender')
        if gender == 'Male': gender = 1
        else: gender = 2
        air_pollution = int(request.form.get('air_pollution'))
        alcohol_use = int(request.form.get('alcohol_use'))
        dust_allergy = int(request.form.get('dust_allergy'))
        occupational_hazards = int(request.form.get('occupational_hazards'))
        genetic_risk = int(request.form.get('genetic_risk'))
        smoking = int(request.form.get('smoking'))
        passive_smoker = int(request.form.get('passive_smoker'))
        chest_pain = int(request.form.get('chest_pain'))
        fatigue = int(request.form.get('fatigue'))
        weight_loss = int(request.form.get('weight_loss'))
        shortness_of_breath = int(request.form.get('shortness_of_breath'))
        wheezing = int(request.form.get('wheezing'))
        swallowing_difficulty = int(request.form.get('swallowing_difficulty'))
        frequent_cold = int(request.form.get('frequent_cold'))
        dry_cough = int(request.form.get('dry_cough'))
        data = np.zeros((17))
        data[0] = age
        data[1] = gender
        data[2] = air_pollution
        data[3] = alcohol_use
        data[4] = dust_allergy
        data[5] = occupational_hazards
        data[6] = genetic_risk
        data[7] = smoking
        data[8] = passive_smoker
        data[9] = chest_pain
        data[10] = fatigue
        data[11] = weight_loss
        data[12] = shortness_of_breath
        data[13] = wheezing
        data[14] = swallowing_difficulty
        data[15] = frequent_cold
        data[16] = dry_cough

        input_karo=(age,gender,air_pollution,alcohol_use,dust_allergy,occupational_hazards,genetic_risk,smoking,passive_smoker,chest_pain,fatigue,weight_loss,shortness_of_breath,wheezing,swallowing_difficulty,frequent_cold,dry_cough)
        input_array=np.asarray(input_karo)
        input_reshape = input_array.reshape(1,-1)
        pred=model.predict(input_reshape)
        # Determine the predicted outcome based on the prediction
        if pred[0] == 0:
            outcome = 'High'
        elif pred[0] == 2:
            outcome = 'Medium'
        elif pred[0]==1:
            outcome = 'Low'
        # Render the results template with the predicted outcome
        return render_template('results.html', outcome=outcome)
    else:
        # If the request method is not POST, redirect to the home page
        return redirect('/')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
