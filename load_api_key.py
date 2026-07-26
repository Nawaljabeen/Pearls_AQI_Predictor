import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_AQI_API_KEY")

if api_key: 
    print("api key loaded successfully")
else:
    print("couldnt find aqi api key :(, check your dotenv!")
    