import pandas as pd
import pymysql as mdb
import numpy as np
import urllib
import json
import re
import pprint
import csv

#The functions takes an address, queries the google api with the address and returns a json

def googleGeocoding(address):
    """This function takes an address and returns the latitude and longitude from the Google geocoding API."""
    baseURL = 'http://maps.googleapis.com/maps/api/geocode/json?'
    geocodeURL = baseURL + 'address=' + address + '&components=administrative_area:MA|country:US'
    geocode = json.loads(urllib.urlopen(geocodeURL).read())
    return geocode
  
    