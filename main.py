import streamlit as st
import pickle
import pandas as pd
import xgboost as xgb

# Load model and scaler
model = xgb.Booster()
model.load_model('xgb_model.json')
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("Booking Status Prediction")

# Sample inputs — change based on your dataset
lead_time  =  st.number_input("Lead Time", min_value=0.0 , step=1.0)
avg_price = st.number_input(" what is the average price  that the  guest paid ?", min_value=0.0 , step=1.0)

# booking_status =  st.number_input("Booking Status", min_value=0.0 , step=1.0 , max_value=1.0)

total_nights =  st.number_input("Total Nights", min_value=0.0 , step=1.0)
total_member =  st.number_input("Total Member", min_value=0.0 , step=1.0)
total_repeat =  st.number_input("Total Repeat", min_value=0.0 , step=1.0 )
price_per_adult = st.number_input("Price Per Adult", step=1.0)

has_special_requests = st.selectbox("Does the guest has special requests ?", options = ["Yes","No"])

has_special_requests = int(1) if has_special_requests == "Yes" else  int(0)
high_price = st.number_input("what is the high price ?", min_value=0.0 , step=1.0 )

type_of_meal_Meal_Plan_2  = st.selectbox("Type Of meal_Meal Plan 2 :" , ["True","False"] )

RoomType = st.selectbox("What is the Type of the used room ? ex:- Room Type 4" ,options= ["No Type" ," Type 4 " ," Type 5 " , " Type 6 " , " Type 7 "] )

room_type_Room_Type_4  =  True if RoomType==" Type 4 " else False
room_type_Room_Type_5  =  True if RoomType==" Type 5 "  else False
room_type_Room_Type_6  =  True if RoomType==" Type 6 " else False
room_type_Room_Type_7  =  True if RoomType==" Type 7 " else False

Market_Segment =  st.selectbox(" What is the market Segment Type ? " , ["No Type" ,"Complementary","Corporate" , "Offline" , "Online"] )

market_segment_type_Complementary =  True if Market_Segment == "Complementary" else False
market_segment_type_Corporate = True if Market_Segment == "Corporate" else False
market_segment_type_Offline  = True if Market_Segment == "Offline" else False
market_segment_type_Online   = True if Market_Segment == "Online" else False

if st.button("Predict"):
    # Construct DataFrame (must match training format!)
    input_data = pd.DataFrame([{
         "lead time" :  lead_time ,
         "average price" : avg_price ,
          "total nights":total_nights ,
         "total_member" :total_member ,
         "total_repeat" :total_repeat ,
         "price_per_adult" : price_per_adult ,
         "has_special_requests" : has_special_requests ,
         "high_price":high_price ,
         "type of meal_Meal Plan 2":type_of_meal_Meal_Plan_2 ,
         "room type_Room_Type 4":room_type_Room_Type_4 ,
         "room type_Room_Type 5":room_type_Room_Type_5 ,
         "room type_Room_Type 6":room_type_Room_Type_6 ,
         "room type_Room_Type 7":room_type_Room_Type_7 ,
         "market segment type_Complementary":market_segment_type_Complementary ,
         "market segment type_Corporate":market_segment_type_Corporate ,
         "market segment type_Offline" : market_segment_type_Offline ,
         "market segment type_Online" :market_segment_type_Online

    }])
    bool_cols = input_data.select_dtypes(include=['bool']).columns
    input_data[bool_cols] = input_data[bool_cols].astype(int)

    #  Convert 'True'/'False' strings to 1/0
    object_cols = input_data.select_dtypes(include='object').columns
    for col in object_cols:
       if set(input_data[col].unique()).issubset({'True', 'False'}):
           input_data[col] = input_data[col].map({'True': 1, 'False': 0})

    # Then scale
    input_scaled = scaler.transform(input_data)
    # prediction = model.predict(input_scaled)
    import xgboost as xgb

    dinput = xgb.DMatrix(input_scaled)
    prediction = model.predict(dinput)

    result = " Canceled" if prediction[0] == 1 else "not Canceled"
    st.success(f"The User has{result} his/her Booking with the Hotel")
else :
    input_data = pd.DataFrame([{}])
    pass
 

