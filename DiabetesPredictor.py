#This file creates a gui that takes in user input to predict whether or not they have diabetes using a 
#logistic regression model that is trained on diabetes data from kaggle

from tkinter import *
from tkinter import ttk
import string
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#creating the model
data = pd.read_csv('diabetes_prediction_dataset.csv')
data['smoking_history'] = data['smoking_history'].replace('ever', 'never')
data.drop(data[data['smoking_history'] == 'No Info'].index, inplace=True)
data.drop(data[data['gender'] == 'Other'].index, inplace=True)
data = pd.get_dummies(data, columns=['gender', 'smoking_history'], dtype=int)
X = data.drop(['diabetes'], axis= 1)
y = data['diabetes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
model = LogisticRegression(random_state=0)
model.fit(X_train, y_train)




#creates an object of the tkinter class
gui = Tk()
gui.title("Diabetes Machine Learning Model")

#control the size of the gui window
gui.geometry("700x600")

#this creates a label for the title
name = Label(gui, text="DIABETES PREDICTOR")
name.place(x=50, y=10)

#creating the labels for the parameters
age_label = Label(gui, text="Age: ")
age_label.place(x=30,y=50)

hyp_label = Label(gui, text="Hypertension(0 or 1): ")
hyp_label.place(x=30,y=70)

heart_label = Label(gui, text="Heart Disease(0 or 1): ")
heart_label.place(x=30,y=90)

height_label = Label(gui, text="Height: ")
height_label.place(x=30,y=110)

weight_label = Label(gui, text="Weight: ")
weight_label.place(x=30,y=130)

hba_label = Label(gui, text="HBA1C: ")
hba_label.place(x=30,y=150)

glucose_label = Label(gui, text="Blood Glucose level: ")
glucose_label.place(x=30,y=170)

gender_label = Label(gui, text="Gender: ")
gender_label.place(x=30,y=195)

smoking_label = Label(gui, text="Smoking History: ")
smoking_label.place(x=30,y=220)


#creating the label entry boxes
age = Entry(gui)
age.place(x=60, y=50, width = 50, height = 23)

hyp = Entry(gui)
hyp.place(x=170, y=70, width = 50, height = 23)

heart = Entry(gui)
heart.place(x=170, y=90, width = 50, height = 23)

height = Entry(gui)
height.place(x=80, y=110, width = 50, height = 23)

weight = Entry(gui)
weight.place(x=80, y=130, width = 50, height = 23)

hba = Entry(gui)
hba.place(x=80, y=150, width = 50, height = 23)

glucose = Entry(gui)
glucose.place(x=160, y=170, width = 50, height = 23)

gender_menu = ttk.Combobox(gui, values = ["Male", "Female"])
gender_menu.place(x=80, y=195)

smoking_menu = ttk.Combobox(gui, values = ["Never", "Current", "Not Current", "Former"])
smoking_menu.place(x=150, y=220)




#creating the result labels

result_label= Label(gui, text = "")
result_label.place(x=50, y= 400)

probs_label= Label(gui, text = "")
probs_label.place(x=50, y= 440)

#creates a fucntion to calculate bmi
def bmi_calc():
    h = float(height.get())
    w = float(weight.get())
    bmi = (w /(h*h)) * 703
    return bmi


#creating ther predicting function
def predict_function():
    # Get input values
    feature1 = float(age.get())
    feature2 = float(hyp.get())
    feature3 = float(heart.get())
    feature4 = float(bmi_calc())
    feature5 = float(hba.get())
    feature6 = float(glucose.get())
    feature7 = gender_menu.get()
    feature8 = smoking_menu.get()

    # Create input array
    if feature7 == "Male":
        if feature8 == "Never":
            input_data = pd.DataFrame({"age":[feature1], "hypertension":[feature2], "heart_disease":[feature3], "bmi":[feature4], "HbA1c_level":[feature5], "blood_glucose_level":[feature6], "gender_Female":[0], "gender_Male":[1], "smoking_history_current":[0], "smoking_history_former":[0], "smoking_history_never":[1], "smoking_history_not current":[0]})
        if feature8 == "Current":
            input_data = pd.DataFrame({"age":[feature1], "hypertension":[feature2], "heart_disease":[feature3], "bmi":[feature4], "HbA1c_level":[feature5], "blood_glucose_level":[feature6], "gender_Female":[0], "gender_Male":[1], "smoking_history_current":[1], "smoking_history_former":[0], "smoking_history_never":[0], "smoking_history_not current":[0]})
        if feature8 == "Not Current":
            input_data = pd.DataFrame({"age":[feature1], "hypertension":[feature2], "heart_disease":[feature3], "bmi":[feature4], "HbA1c_level":[feature5], "blood_glucose_level":[feature6], "gender_Female":[0], "gender_Male":[1], "smoking_history_current":[0], "smoking_history_former":[0], "smoking_history_never":[0], "smoking_history_not current":[1]})
        if feature8 == "Former":
            input_data = pd.DataFrame({"age":[feature1], "hypertension":[feature2], "heart_disease":[feature3], "bmi":[feature4], "HbA1c_level":[feature5], "blood_glucose_level":[feature6], "gender_Female":[0], "gender_Male":[1], "smoking_history_current":[0], "smoking_history_former":[1], "smoking_history_never":[0], "smoking_history_not current":[0]})
    if feature7 == "Female":
        if feature8 == "Never":
            input_data = pd.DataFrame({"age":[feature1], "hypertension":[feature2], "heart_disease":[feature3], "bmi":[feature4], "HbA1c_level":[feature5], "blood_glucose_level":[feature6], "gender_Female":[1], "gender_Male":[0], "smoking_history_current":[0], "smoking_history_former":[0], "smoking_history_never":[1], "smoking_history_not current":[0]})
        if feature8 == "Current":
            input_data = pd.DataFrame({"age":[feature1], "hypertension":[feature2], "heart_disease":[feature3], "bmi":[feature4], "HbA1c_level":[feature5], "blood_glucose_level":[feature6], "gender_Female":[1], "gender_Male":[0], "smoking_history_current":[1], "smoking_history_former":[0], "smoking_history_never":[0], "smoking_history_not current":[0]})
        if feature8 == "Not Current":
            input_data = pd.DataFrame({"age":[feature1], "hypertension":[feature2], "heart_disease":[feature3], "bmi":[feature4], "HbA1c_level":[feature5], "blood_glucose_level":[feature6], "gender_Female":[1], "gender_Male":[0], "smoking_history_current":[0], "smoking_history_former":[0], "smoking_history_never":[0], "smoking_history_not current":[1]})
        if feature8 == "Former":
            input_data = pd.DataFrame({"age":[feature1], "hypertension":[feature2], "heart_disease":[feature3], "bmi":[feature4], "HbA1c_level":[feature5], "blood_glucose_level":[feature6], "gender_Female":[1], "gender_Male":[0], "smoking_history_current":[0], "smoking_history_former":[1], "smoking_history_never":[0], "smoking_history_not current":[0]})
    

    # Make prediction
    prediction = model.predict(input_data)
    prediction_probs = model.predict_proba(input_data)

    # Update output label
    result_label.config(text=prediction)
    probs_label.config(text= prediction_probs)


#this creates the predict button (add command function later)
predict = Button(gui, text="Predict", command = predict_function)
predict.place(x= 130, y=250)


#this creates a label for the title
prediction_label = Label(gui, text="A 0 would predict the patient is not diabetic and a 1 would predict the patient is diabetic")
prediction_label.place(x=30, y=380)

prob_label = Label(gui, text="This is the probability that the model predicts if the patient is not or is diabetic")
prob_label.place(x=30, y=420)


#this creates a label that gets the password and stores it in the label
name = Label(gui, text="CREATED BY NICHOLAS CIAMPAGLIO(hes a pretty cool guy)")
name.place(x=50, y=550)



gui.mainloop()