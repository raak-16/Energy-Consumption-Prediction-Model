import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score,mean_absolute_error,mean_absolute_percentage_error,confusion_matrix,mean_squared_error
from sklearn.model_selection import train_test_split
# model = load_model()

# Function to preprocess input data
# def preprocess_input(input_data):
#     # Convert input data to DataFrame
#     df = pd.DataFrame([input_data])
    
#     # Standardize numerical features (assuming you used standardization during training)
#     numerical_features = ['Lagging_Current_Reactive.Power_kVarh', 
#                          'Leading_Current_Reactive_Power_kVarh',
#                          'CO2(tCO2)', 
#                          'Lagging_Current_Power_Factor',
#                          'Leading_Current_Power_Factor',
#                          'NSM']
    
#     scaler = StandardScaler()
#     df[numerical_features] = scaler.fit_transform(df[numerical_features])
    
#     return df

# Function to make prediction
# def make_prediction(input_data):
#     processed_data = preprocess_input(input_data)
#     prediction = model.predict(processed_data)
#     return prediction[0]

def make_prediction(inputs):

    df=pd.read_csv("encoded_data.csv")
    df=df.drop("Unnamed: 0",axis=1)
    y=df.iloc[:,0]
    X=df.iloc[:,1:]

    X_train,X_test,y_train,y_test=train_test_split(X,y)


    model=DecisionTreeRegressor()

    model.fit(X_train,y_train)

    inputss=list(inputs.values())

    predicstions=model.predict(X_test)

    pred=model.predict([inputss])

    confusion=r2_score(y_test,predicstions)

    st.write(pred)



    st.write("r2 score",confusion)
    st.write("mean absolute error",mean_absolute_error(y_test,predicstions))
    st.write("mean squred error",mean_squared_error(y_test,predicstions))



# Streamlit app
def main():
    st.title('Energy Consumption Prediction Model')
    st.write('Predict Usage_kWh based on energy parameters')
    
    with st.form("prediction_form"):
        st.header("Input Parameters")
        
        # Numerical inputs
        col1, col2 = st.columns(2)
        with col1:
            lagging_reactive = st.number_input('Lagging Current Reactive Power (kVarh)', 
                                             min_value=0.0, value=10.0, step=0.1)
            leading_reactive = st.number_input('Leading Current Reactive Power (kVarh)', 
                                             min_value=0.0, value=5.0, step=0.1)
            co2 = st.number_input('CO2 Emissions (tCO2)', 
                                 min_value=0.0, value=0.5, step=0.01)
            
        with col2:
            lagging_pf = st.number_input('Lagging Current Power Factor', 
                                       min_value=0.0, value=0.85, step=0.01)
            leading_pf = st.number_input('Leading Current Power Factor', 
                                       min_value=0.0, value=0.80, step=0.01)
            nsm = st.number_input('NSM (Seconds from midnight)', 
                                min_value=0, max_value=86400, value=43200, step=100)
        
        # Week status
        week_status = st.radio("Week Status", ('Weekday', 'Weekend'))
        
        # Day of week
        day_of_week = st.selectbox("Day of Week", 
                                  ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                                   'Friday', 'Saturday', 'Sunday'])
        
        # Load type
        load_type = st.selectbox("Load Type", 
                               ['Light Load', 'Medium Load', 'Maximum Load'])
        
        # Submit button
        submitted = st.form_submit_button("Predict Usage (kWh)")
        
        if submitted:
            # Prepare input data
            input_data = {
                'Lagging_Current_Reactive.Power_kVarh': lagging_reactive,
                'Leading_Current_Reactive_Power_kVarh': leading_reactive,
                'CO2(tCO2)': co2,
                'Lagging_Current_Power_Factor': lagging_pf,
                'Leading_Current_Power_Factor': leading_pf,
                'NSM': nsm,
                'WeekStatus_Weekday': week_status == 'Weekday',
                'WeekStatus_Weekend': week_status == 'Weekend',
                'Day_of_week_Friday': day_of_week == 'Friday',
                'Day_of_week_Monday': day_of_week == 'Monday',
                'Day_of_week_Saturday': day_of_week == 'Saturday',
                'Day_of_week_Sunday': day_of_week == 'Sunday',
                'Day_of_week_Thursday': day_of_week == 'Thursday',
                'Day_of_week_Tuesday': day_of_week == 'Tuesday',
                'Day_of_week_Wednesday': day_of_week == 'Wednesday',
                'Load_Type_Light_Load': load_type == 'Light Load',
                'Load_Type_Maximum_Load': load_type == 'Maximum Load',
                'Load_Type_Medium_Load': load_type == 'Medium Load'
            }
            
            # Make prediction
            prediction = make_prediction(input_data)
            
            # Display result
            # st.success(f'Predicted Usage: {prediction:.2f} kWh')

if __name__ == '__main__':
    main()