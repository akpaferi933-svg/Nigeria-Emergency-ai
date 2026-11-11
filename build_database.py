import pandas as pd
import sqlite3

def create_master_database():
    print("Loading 530,000+ rows from CSV dataset...")
    # Ensure your Kaggle CSV file is in this same folder and named '911.csv'
    df = pd.read_csv("911.csv") 
    
    # Feature Engineering
    df['timeStamp'] = pd.to_datetime(df['timeStamp'])
    df['Year'] = df['timeStamp'].dt.year
    df['Month'] = df['timeStamp'].dt.month
    df['Hour'] = df['timeStamp'].dt.hour
    
    # Extract Reason from title
    temp_split = df['title'].str.split(':', n=1, expand=True)
    df['Reason'] = temp_split[0].str.strip()
    
    # Connect and save to SQLite database
    conn = sqlite3.connect("nigeria_emergency_master.db")
    print("Writing records to SQLite database...")
    df.to_sql("incidents_master", conn, if_exists="replace", index=False)
    
    # Create Indexes for lightning-fast queries across 530k rows
    cursor = conn.cursor()
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_reason ON incidents_master(Reason);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_year ON incidents_master(Year);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_lat_lng ON incidents_master(lat, lng);")
    conn.commit()
    conn.close()
    print("Database built and indexed successfully!")

if __name__ == "__main__":
    create_master_database()