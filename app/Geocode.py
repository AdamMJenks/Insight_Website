import pandas as pd
import pymysql as mdb
import numpy as np
import urllib
import json
import re
import pprint
import csv


def googleGeocoding(address):
    """This function takes an address and returns the latitude and longitude from the Google geocoding API."""
    baseURL = 'http://maps.googleapis.com/maps/api/geocode/json?'
    geocodeURL = baseURL + 'address=' + address + '&components=administrative_area:MA|country:US'
    geocode = json.loads(urllib.urlopen(geocodeURL).read())
    return geocode
  
    Local_json = googleGeocoding('1695 commonwealth avenue, Boston,MA')
  
    Local_jsonStr = str(Local_json)
    
    StartLoccoord = Local_jsonStr.find("u'location'")
    EndLoccoord = Local_jsonStr.find("'address_components'")
    
    Coordstring = Local_jsonStr[StartLoccoord:EndLoccoord]
    
    Coordinates = re.findall(r"[-+]?\d*\.\d+|\d+",Coordstring)
    
    Latcoord_Final = Coordinates[0]
    Longcoord_Final = Coordinates[1]