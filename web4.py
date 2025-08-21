#right one
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import requests
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster
from sklearn.preprocessing import StandardScaler
import mysql.connector
import matplotlib.pyplot as plt
import seaborn as sns 

offsets = {
    "bangalore": 203,
    "delhi": 94,
    "kolkata": 628,
    "mumbai": 309
}


# Fetch real-time data from the API
def fetch_api_data(city):
    url = "https://api.data.gov.in/resource/3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69"
    api_key = "579b464db66ec23bdd000001cdd3946e44ce4aad7209ff7b23ac571b"
    format = "json"
    limit = 10
    offset = offsets.get(city, 0)
    params = {
        "api-key": api_key,
        "format": format,
        "offset": offset,
        "limit": limit
    }

    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        records = data.get("records", [])
        return pd.DataFrame(records)
    else:
        st.error(f"Failed to fetch data from API. Status code: {response.status_code}")
        return pd.DataFrame()


# Load the trained model and encoders
model = joblib.load('trained_model.pkl')
place_encoder = joblib.load('place_encoder.pkl')
station_encoder = joblib.load('station_encoder.pkl')
ordinal_encoder = joblib.load('ordinal_encoder.pkl')
scaler = joblib.load('scaler.pkl')


# Function to fetch available places and stations
def fetch_data():
    data = pd.read_csv(r"C:\Users\MuraliHebbani\OneDrive\Documents\ani\AQI-merge-tag.csv")
    data.columns = data.columns.str.strip()
    data['Place'] = data['Place'].str.strip()
    return data


# Function to get current day of the week and whether it's a weekend or not
def get_current_day_info():
    current_date = datetime.now()
    current_year = current_date.year
    day_of_week = current_date.strftime('%A')
    is_weekend = 1 if day_of_week in ["Saturday", "Sunday"] else 0
    return current_year, day_of_week, is_weekend


# Function to read AQI data from CSV file
def read_aqi_feed(selected_place, selected_station):
    aqi_feed = pd.read_csv(r"C:\Users\MuraliHebbani\Downloads\aqi-feeder - dataset-aqi.csv")
    aqi_feed.columns = aqi_feed.columns.str.strip()
    filtered_aqi_feed = aqi_feed[(aqi_feed['Place'] == selected_place) & (aqi_feed['Station'] == selected_station)]
    return filtered_aqi_feed


def get_db_connection():
    connection = mysql.connector.connect(
        host='localhost',
        user='root',
        password='ananya',
        database='aqi_db'
    )
    return connection


# Function to save user inputs to the database
def save_to_db(data):
    conn = get_db_connection()
    cursor = conn.cursor()
    insert_query = """
    INSERT INTO user_inputs (place, station, pm25, pm10, o3, no2, so2, co, year, day_of_week, is_weekend, timestamp)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    cursor.execute(insert_query, data)
    conn.commit()
    cursor.close()
    conn.close()


def main():
    st.title("AQI Prediction")

    try:
       
        data = fetch_data()

        selected_place = st.sidebar.selectbox("Select Place", data['Place'].unique())
        filtered_data = data[data['Place'] == selected_place]
        filtered_stations = data[data['Place'] == selected_place]['Station'].unique()

        selected_station = st.sidebar.selectbox("Select Station", filtered_stations)

        filtered_aqi_feed = read_aqi_feed(selected_place, selected_station)

        if not filtered_aqi_feed.empty:
            # Get current day info
            year, day_of_week, is_weekend = get_current_day_info()

            # Convert day_of_week to numeric value
            day_of_week_numeric = pd.to_datetime(day_of_week, format='%A').dayofweek

            if st.button("Predict AQI"):
                # Check if the place and station are in the label encoder's classes
                if selected_place not in place_encoder.classes_:
                    st.error(f"Place '{selected_place}' is not in the trained model's known places.")
                elif selected_station not in station_encoder.classes_:
                    st.error(f"Station '{selected_station}' is not in the trained model's known stations.")
                else:
                    # Prepare data for prediction
                    pm25 = filtered_aqi_feed['pm25'].values[0]
                    pm10 = filtered_aqi_feed['pm10'].values[0]
                    o3 = filtered_aqi_feed['o3'].values[0]
                    no2 = filtered_aqi_feed['no2'].values[0]
                    so2 = filtered_aqi_feed['so2'].values[0]
                    co = filtered_aqi_feed['co'].values[0]

                    if np.isnan([pm25, pm10, o3, no2, so2, co]).all():
                        st.error("Insufficient data for the selected place and station.")
                    else:
                        selected_place_encoded = place_encoder.transform([selected_place])[0]
                        selected_station_encoded = station_encoder.transform([selected_station])[0]

                        # Ensure the data is in the same order and format as during training
                        prediction_data = pd.DataFrame({
                            'Place': [selected_place_encoded],
                            'Station': [selected_station_encoded],
                            'pm25': [pm25],
                            'pm10': [pm10],
                            'o3': [o3],
                            'no2': [no2],
                            'so2': [so2],
                            'co': [co],
                            'year': [year],
                            'day_of_week': [day_of_week_numeric],
                            'is_weekend': [is_weekend]
                        })

                        # Scale the features
                        prediction_data[['pm25', 'pm10', 'o3', 'no2', 'so2', 'co']] = scaler.transform(prediction_data[['pm25', 'pm10', 'o3', 'no2', 'so2', 'co']])

                        # Predict AQI
                        prediction = model.predict(prediction_data)
                        predicted_label = ordinal_encoder.inverse_transform(prediction.reshape(-1, 1))

                        # Get current timestamp
                        timestamp = datetime.now()

                        # Save user inputs to database
                        save_to_db((selected_place, selected_station, pm25, pm10, o3, no2, so2, co, year, day_of_week_numeric, is_weekend, timestamp))

                        st.success(f"Predicted AQI Bucket: {predicted_label[0][0]}")

        else:
            st.error("Insufficient data for selected place and station.")

        # Fetch real-time AQI data from the API
        api_data = fetch_api_data(selected_place)

        if not api_data.empty:
            st.write("Real-time AQI data:")
            st.write(api_data)

            city_lat = api_data['latitude'].astype(float).mean()
            city_lon = api_data['longitude'].astype(float).mean()

            m = folium.Map(location=[city_lat, city_lon], zoom_start=12)

            marker_cluster = MarkerCluster().add_to(m)

            for _, row in api_data.iterrows():
                aqi_value = pd.to_numeric(row['avg_value'], errors='coerce')
                if not np.isnan(aqi_value):
                    aqi_color = 'red' if aqi_value > 100 else 'green'
                    tooltip = f"AQI: {aqi_value}"
                    folium.CircleMarker(
                        location=[row['latitude'], row['longitude']],
                        radius=10,
                        color=aqi_color,
                        fill=True,
                        fill_color=aqi_color,
                        tooltip=folium.Tooltip(tooltip, permanent=True, direction='top', opacity=0.8, sticky=False, offset=(0, 0), style="font-size: 14px;")
                    ).add_to(marker_cluster)

            st_folium(m, width=700, height=500)

            # Plot histogram for highest pollutant
            if not filtered_aqi_feed.empty:
                st.write("Pollutant Percentages:")

                # Calculate total sum of pollutants for percentage calculation
                total_pm25 = filtered_aqi_feed['pm25'].sum()
                total_pm10 = filtered_aqi_feed['pm10'].sum()
                total_o3 = filtered_aqi_feed['o3'].sum()
                total_no2 = filtered_aqi_feed['no2'].sum()
                total_so2 = filtered_aqi_feed['so2'].sum()
                total_co = filtered_aqi_feed['co'].sum()
                
                # plotting pie chart
                if total_pm25 > 0 or total_pm10 > 0 or total_o3 > 0 or total_no2 > 0 or total_so2 > 0 or total_co > 0:
                    percentages = []
                    labels = []
                    colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'orange', 'lightgreen']
                    explode = (0.1, 0, 0, 0, 0, 0) 
                    if total_pm25 > 0:
                        percentages.append(filtered_aqi_feed['pm25'].sum() / (total_pm25 + total_pm10 + total_o3 + total_no2 + total_so2 + total_co) * 100)
                        labels.append('pm25')

                    if total_pm10 > 0:
                        percentages.append(filtered_aqi_feed['pm10'].sum() / (total_pm25 + total_pm10 + total_o3 + total_no2 + total_so2 + total_co) * 100)
                        labels.append('pm10')

                    if total_o3 > 0:
                        percentages.append(filtered_aqi_feed['o3'].sum() / (total_pm25 + total_pm10 + total_o3 + total_no2 + total_so2 + total_co) * 100)
                        labels.append('o3')

                    if total_no2 > 0:
                        percentages.append(filtered_aqi_feed['no2'].sum() / (total_pm25 + total_pm10 + total_o3 + total_no2 + total_so2 + total_co) * 100)
                        labels.append('no2')

                    if total_so2 > 0:
                        percentages.append(filtered_aqi_feed['so2'].sum() / (total_pm25 + total_pm10 + total_o3 + total_no2 + total_so2 + total_co) * 100)
                        labels.append('so2')

                    if total_co > 0:
                        percentages.append(filtered_aqi_feed['co'].sum() / (total_pm25 + total_pm10 + total_o3 + total_no2 + total_so2 + total_co) * 100)
                        labels.append('co')

                    # Plotting the bar chart
                    plt.figure(figsize=(12, 10))
                    explode = (0.1, 0.1, 0.1, 0.1, 0.1, 0.1)  
                    plt.pie(percentages, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=120)
                    plt.title(f'Pollutant Percentages in {selected_place}', pad=20) 
                    st.pyplot(plt)
                    colors = {
                        'Good': 'lightgreen',
                        'Moderate': 'darkgreen',
                        'Satisfactory': 'yellow',
                        'Poor': 'orange',
                        'Very Poor': 'red',
                        'Severe': 'maroon'
                    }
                    aqi_counts = filtered_data['AQI_bucket_calculated'].value_counts().reset_index()
                    aqi_counts.columns = ['AQI_bucket', 'Count']
                    aqi_counts['AQI_bucket'] = pd.Categorical(aqi_counts['AQI_bucket'], categories=['Good', 'Moderate', 'Satisfactory', 'Poor', 'Very Poor', 'Severe'], ordered=True)
                    plt.figure(figsize=(10, 6))
                    sns.barplot(x='Count', y='AQI_bucket', data=aqi_counts, orient='h', palette=colors.values())
                    plt.title(f'Distribution of AQI Bucket Classes for {selected_place}')
                    plt.xlabel('Count')
                    plt.ylabel('AQI Bucket Class')
                    plt.grid(True)

                    # Displaying plot in Streamlit
                    st.pyplot(plt)
                else:
                    st.error("No valid pollutant data available for pie chart plotting.")
        else:
            st.error("No real-time AQI data available.")

    except Exception as e:
        st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
