import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load the trained RandomForestClassifier model
model = pickle.load(open('grid_rf_model.pkl', 'rb'))

# Load the scaler
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Initialize the Streamlit app
st.title("Student Graduation Prediction")

#
# Define inputs for all features
Marital_Status = st.number_input('Marital Status (0 or 1)', min_value=0, max_value=1, value=0)
Application_mode = st.number_input('Application Mode (integer)', min_value=0, max_value=20, value=0)
Application_order = st.number_input('Application Order (integer)', min_value=0, max_value=10, value=0)
Course = st.number_input('Course (integer)', min_value=0, max_value=200, value=0)
Daytime_evening_attendance = st.number_input('Daytime/Evening Attendance (0 or 1)', min_value=0, max_value=1, value=0)
Previous_qualification = st.number_input('Previous Qualification (integer)', min_value=0, max_value=10, value=0)
Previous_qualification_grade = st.number_input('Previous Qualification Grade (0.0 to 20.0)', min_value=0.0, max_value=20.0, value=10.0)
Nacionality = st.number_input('Nationality (integer)', min_value=0, max_value=200, value=0)
Mother_qualification = st.number_input('Mother\'s Qualification (integer)', min_value=0, max_value=10, value=0)
Father_qualification = st.number_input('Father\'s Qualification (integer)', min_value=0, max_value=10, value=0)
Mother_occupation = st.number_input('Mother\'s Occupation (integer)', min_value=0, max_value=10, value=0)
Father_occupation = st.number_input('Father\'s Occupation (integer)', min_value=0, max_value=10, value=0)
Admission_grade = st.number_input('Admission Grade', min_value=0.0, max_value=20.0, value=10.0)
Displaced = st.number_input('Displaced (0 or 1)', min_value=0, max_value=1, value=0)
Educational_special_needs = st.number_input('Educational Special Needs (0 or 1)', min_value=0, max_value=1, value=0)
Debtor = st.number_input('Debtor (0 or 1)', min_value=0, max_value=1, value=0)
Tuition_fees_up_to_date = st.number_input('Tuition Fees Up to Date (0 or 1)', min_value=0, max_value=1, value=0)
Gender = st.number_input('Gender (0 or 1)', min_value=0, max_value=1, value=0)
Scholarship_holder = st.number_input('Scholarship Holder (0 or 1)', min_value=0, max_value=1, value=0)
Age_at_enrollment = st.number_input('Age at Enrollment', min_value=0, max_value=100, value=0)
International = st.number_input('International (0 or 1)', min_value=0, max_value=1, value=0)
Curricular_units_1st_sem_credited = st.number_input('Curricular Units 1st Sem (Credited)', min_value=0, max_value=20, value=0)
Curricular_units_1st_sem_enrolled = st.number_input('Curricular Units 1st Sem (Enrolled)', min_value=0, max_value=20, value=0)
Curricular_units_1st_sem_evaluations = st.number_input('Curricular Units 1st Sem (Evaluations)', min_value=0, max_value=20, value=0)
Curricular_units_1st_sem_approved = st.number_input('Curricular Units 1st Sem (Approved)', min_value=0, max_value=20, value=0)
Curricular_units_1st_sem_grade = st.number_input('Curricular Units 1st Sem (Grade)', min_value=0.0, max_value=20.0, value=10.0)
Curricular_units_1st_sem_without_evaluations = st.number_input('Curricular Units 1st Sem (Without Evaluations)', min_value=0, max_value=20, value=0)
Curricular_units_2nd_sem_credited = st.number_input('Curricular Units 2nd Sem (Credited)', min_value=0, max_value=20, value=0)
Curricular_units_2nd_sem_enrolled = st.number_input('Curricular Units 2nd Sem (Enrolled)', min_value=0, max_value=20, value=0)
Curricular_units_2nd_sem_evaluations = st.number_input('Curricular Units 2nd Sem (Evaluations)', min_value=0, max_value=20, value=0)
Curricular_units_2nd_sem_approved = st.number_input('Curricular Units 2nd Sem (Approved)', min_value=0, max_value=20, value=0)
Curricular_units_2nd_sem_grade = st.number_input('Curricular Units 2nd Sem (Grade)', min_value=0.0, max_value=20.0, value=10.0)
Curricular_units_2nd_sem_without_evaluations = st.number_input('Curricular Units 2nd Sem (Without Evaluations)', min_value=0, max_value=20, value=0)
Unemployment_rate = st.number_input('Unemployment Rate (%)', min_value=0.0, max_value=100.0, value=0.0)
Inflation_rate = st.number_input('Inflation Rate (%)', min_value=0.0, max_value=100.0, value=0.0)
GDP = st.number_input('GDP (in billions)', min_value=0.0, max_value=10000.0, value=0.0)

# Button for prediction
if st.button('Predict Graduation'):
    # Create an input array
    input_data = np.array([[
        Marital_Status, Application_mode, Application_order, Course,
        Daytime_evening_attendance, Previous_qualification,
        Previous_qualification_grade, Nacionality, Mother_qualification,
        Father_qualification, Mother_occupation, Father_occupation,
        Admission_grade, Displaced, Educational_special_needs, Debtor,
        Tuition_fees_up_to_date, Gender, Scholarship_holder, Age_at_enrollment,
        International, Curricular_units_1st_sem_credited, Curricular_units_1st_sem_enrolled,
        Curricular_units_1st_sem_evaluations, Curricular_units_1st_sem_approved,
        Curricular_units_1st_sem_grade, Curricular_units_1st_sem_without_evaluations,
        Curricular_units_2nd_sem_credited, Curricular_units_2nd_sem_enrolled,
        Curricular_units_2nd_sem_evaluations, Curricular_units_2nd_sem_approved,
        Curricular_units_2nd_sem_grade, Curricular_units_2nd_sem_without_evaluations,
        Unemployment_rate, Inflation_rate, GDP
    ]])

    # Apply scaling to the input data
    scaled_data = scaler.transform(input_data)

    # Predict using the pre-trained model
    prediction = model.predict(scaled_data)

    # Display the prediction result
    if prediction == 0:
        st.write("Prediction: The student is likely to drop out.")
    elif prediction == 1:
        st.write("Prediction: The student is likely to graduate.")
    else:
        st.write("Prediction: The student is still enrolled.")

# import streamlit as st
# import numpy as np
# import pickle
# from sklearn.preprocessing import StandardScaler

# # Load the trained RandomForestClassifier model (ensure you save your model as a .pkl file)
# model = pickle.load(open('grid_rf_model.pkl', 'rb'))

# # Load the scaler (if scaling is needed; otherwise, you can skip this part)
# scaler = pickle.load(open('scaler.pkl', 'rb'))

# # Initialize the Streamlit app
# st.title("Student Graduation Prediction")

# # Define inputs for the model (example with 10 features based on your dataset)
# Marital_Status = st.number_input('Marital Status', min_value=0, max_value=1, value=0)
# Application_mode = st.number_input('Application Mode', min_value=0, max_value=20, value=0)
# Application_order = st.number_input('Application Order', min_value=0, max_value=10, value=0)
# Course = st.number_input('Course', min_value=0, max_value=200, value=0)
# Daytime_evening_attendance = st.number_input('Daytime/Evening Attendance', min_value=0, max_value=1, value=0)
# Previous_qualification = st.number_input('Previous Qualification', min_value=0, max_value=10, value=0)
# Previous_qualification_grade = st.number_input('Previous Qualification Grade', min_value=0.0, max_value=20.0, value=10.0)
# Nacionality = st.number_input('Nationality', min_value=0, max_value=200, value=0)
# Mother_qualification = st.number_input('Mother\'s Qualification', min_value=0, max_value=10, value=0)
# Father_qualification = st.number_input('Father\'s Qualification', min_value=0, max_value=10, value=0)

# # Button for prediction
# if st.button('Predict Graduation'):
#     # Create an input array
#     input_data = np.array([[
#         Marital_Status, Application_mode, Application_order, Course,
#         Daytime_evening_attendance, Previous_qualification,
#         Previous_qualification_grade, Nacionality, Mother_qualification,
#         Father_qualification
#     ]])

#     # Apply scaling to the input data (if scaling is used)
#     # scaled_data = scaler.transform(input_data)

#     # Predict using the pre-trained model
#     prediction = model.predict(input_data)

#     # Display the prediction result
#     if prediction == 0:
#         st.write("Prediction: The student is likely to drop out.")
#     elif prediction == 1:
#         st.write("Prediction: The student is likely to graduate.")
#     else:
#         st.write("Prediction: The student is still enrolled.")

# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import LabelEncoder

# # Load the dataset (you should modify this to load the actual dataset from your notebook)
# data = pd.read_csv('predict_students_dropout_and_academic_success.csv')  # Replace with the actual path to your dataset

# # Make sure the dataset has a target column (e.g., 'Status') for 'Graduated' and 'Dropped'
# target_column = [0,1]  # Replace with the name of the target column in your dataset

# # Check if the target column exists
# if target_column not in data.columns:
#     st.error(f"'{target_column}' column not found in the dataset. Please check the dataset.")
#     st.stop()

# # Encode the target labels ('Graduated' -> 1, 'Dropped' -> 0)
# label_encoder = LabelEncoder()
# data[target_column] = label_encoder.fit_transform(data[target_column])

# # Select 10 random features from the dataset
# np.random.seed(42)
# features = data.columns.drop(target_column)  # Exclude the target column
# selected_features = np.random.choice(features, 10, replace=False)

# # Split the data into features (X) and target (y)
# X = data[selected_features]
# y = data[target_column]

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# # Train the RandomForest model
# clf = RandomForestClassifier(n_estimators=100, random_state=42)
# clf.fit(X_train, y_train)

# # Streamlit UI
# st.title("Student Dropout Prediction")

# st.header("Input Features for Prediction")

# # Generate input fields dynamically for the selected features
# input_data = {}
# for feature in selected_features:
#     input_data[feature] = st.number_input(f"Enter value for {feature}", value=0.0)

# # Convert input data into a DataFrame for prediction
# input_df = pd.DataFrame([input_data])

# # Make prediction when user clicks the 'Predict' button
# if st.button("Predict"):
#     prediction = clf.predict(input_df)
    
#     # Convert numeric prediction back to label
#     prediction_label = label_encoder.inverse_transform(prediction)[0]

#     # Display the result
#     if prediction_label == 'Graduated':
#         st.success("The model predicts: Graduated")
#     else:
#         st.error("The model predicts: Dropped")
