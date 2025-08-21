import mysql.connector
import csv
from datetime import datetime

# Establish database connection
mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="ananya",
    database="aqi_database"  # Use your database name
)

mycursor = mydb.cursor()

# Read CSV and insert data into MySQL
with open("C:\\Users\\MuraliHebbani\\OneDrive\\Documents\\ani\\AQI-merge-tag.csv", 'r') as csvfile:
    csv_data = csv.reader(csvfile)
    header = next(csv_data)  # Skip header row

    for row in csv_data:
        try:
            # Parse date and ensure proper data types (convert to float where necessary)
            date_str = row[2]  # Assuming the date is in the third column
            date_obj = datetime.strptime(date_str, '%d-%m-%Y').strftime('%Y-%m-%d') if date_str else None

            # Convert numerical columns to float
            row[3] = float(row[3]) if row[3] else None  # pm25
            row[4] = float(row[4]) if row[4] else None  # pm10
            row[5] = float(row[5]) if row[5] else None  # o3
            row[6] = float(row[6]) if row[6] else None  # no2
            row[7] = float(row[7]) if row[7] else None  # so2
            row[8] = float(row[8]) if row[8] else None  # co

            # Convert AQI_calculated to int
            row[10] = int(row[10]) if row[10] else None  # AQI_calculated

            # Rearrange row order to match table columns
            data_to_insert = (
                row[0],  # Place
                row[1],  # Station
                date_obj,  # date
                row[3],  # pm25
                row[4],  # pm10
                row[5],  # o3
                row[6],  # no2
                row[7],  # so2
                row[8],  # co
                row[10],  # AQI_calculated
                row[9]   # AQI_bucket_calculated
            )

            # Insert the row into the database
            mycursor.execute('''
                INSERT INTO aqsql (Place, Station, date, pm25, pm10, o3, no2, so2, co, AQI_calculated, AQI_bucket_calculated) 
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ''', data_to_insert)
        except ValueError as ve:
            print(f"Error converting data: {ve}")
            print(f"Row with error: {row}")
        except mysql.connector.errors.DatabaseError as db_err:
            print(f"Database error: {db_err}")
            print(f"Row with error: {row}")

# Commit changes and close the connection
mydb.commit()
mydb.close()
