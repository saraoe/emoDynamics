"""
This script creates a .json file with twitter credentials 
"""
import json
import tweepy
import sys

# Authenticate
CONSUMER_KEY = "INxb3675f0VUk1hY57rtAMgpn" #@param {type:"string"}
CONSUMER_SECRET_KEY = "6bzAJAF92WoPG1GdQdeqQHOcpJX4uiBB8asz9uc55DomDzwEat" #@param {type:"string"}
ACCESS_TOKEN_KEY = "1194585720-pmecXsV8B9Y0jIJ4LqiE2Qk2fVsMwZJeoYYnbqy" #@param {type:"string"}
ACCESS_TOKEN_SECRET_KEY = "Oi5ccDrf2uAiSljOMSXQkwEZgvIr0NfrF0a70dwoJfALP" #@param {type:"string"}

#Creates a JSON Files with the API credentials
with open('api_keys.json', 'w') as outfile:
    json.dump({
    "consumer_key":CONSUMER_KEY,
    "consumer_secret":CONSUMER_SECRET_KEY,
    "access_token":ACCESS_TOKEN_KEY,
    "access_token_secret": ACCESS_TOKEN_SECRET_KEY
     }, outfile)

#The lines below are just to test if the twitter credentials are correct
# Authenticate
auth = tweepy.AppAuthHandler(CONSUMER_KEY, CONSUMER_SECRET_KEY)

api = tweepy.API(auth, wait_on_rate_limit=True)
				   # wait_on_rate_limit_notify=True)
 
if (not api):
   print ("Can't Authenticate")
   sys.exit(-1)

from IPython.display import clear_output

# Run in terminal
# wget https://raw.githubusercontent.com/thepanacealab/SMMT/master/data_acquisition/get_metadata.py -O get_metadata.py

clear_output()
