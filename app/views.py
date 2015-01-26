from flask import render_template, request
from app import app
import pymysql as mdb
from Geocode import googleGeocoding
import pandas as pd
import pymysql as mdb
import numpy as np
import urllib
import json
import re
import pprint
import csv


con = mdb.connect('localhost', 'root', 'sqlbacon', 'Crime_data')


@app.route('/')
@app.route('/index')
def index():
	return render_template("index.html",
        title = 'Home',
        )


@app.route('/input')
def cities_input():
  return render_template("input.html")

@app.route('/output')


def cities_output():
    Address = request.args.get('Address')
    Radius = request.args.get('Radius')
    Crime = str(request.args.get('crime'))
    Year = str(request.args.get('year'))
    Month = str(request.args.get('month'))
    
    Addresspassed = googleGeocoding(Address)
    

    Local_jsonStr = str(Addresspassed)
    StartLoccoord = Local_jsonStr.find("u'location'")
    EndLoccoord = Local_jsonStr.find("'address_components'")

    Coordstring = Local_jsonStr[StartLoccoord:EndLoccoord]

    Coordinates = re.findall(r"[-+]?\d*\.\d+|\d+",Coordstring)

    Latitude = str(Coordinates[0])
    Longitude = str(Coordinates[1])
       
    cur = con.cursor()
    CrimeSelected =pd.read_sql("""SELECT *
      FROM (
     SELECT z.*,
            p.radius,
            p.distance_unit
                     * DEGREES(ACOS(COS(RADIANS(p.latpoint))
                     * COS(RADIANS(z.lat))
                     * COS(RADIANS(p.longpoint - z.lng))
                     + SIN(RADIANS(p.latpoint))
                     * SIN(RADIANS(z.lat)))) AS distance
      FROM crime_new AS z
      JOIN (   /* these are the query parameters */
            SELECT  """+ Latitude+ """ AS latpoint, """+Longitude+""" AS longpoint,"""
                    +Radius+""" AS radius,      111.045 AS distance_unit
        ) AS p 
      WHERE z.lat
         BETWEEN p.latpoint  - (p.radius / p.distance_unit)
             AND p.latpoint  + (p.radius / p.distance_unit)
        AND z.lng
         BETWEEN p.longpoint - (p.radius / (p.distance_unit * COS(RADIANS(p.latpoint))))
             AND p.longpoint + (p.radius / (p.distance_unit * COS(RADIANS(p.latpoint))))
     ) AS d
     WHERE distance <= radius and Incident = '"""+Crime+"""'
     ORDER BY distance;""",con)
    
    Crimelen = len(CrimeSelected.index)
    
    Crimetoanalyze= CrimeSelected[['Incident','date']].set_index('date')
    Selectedcrime = CrimeSelected.groupby(['Incident'])
    crimecounts = Selectedcrime.count(['ReptDist'])
    
    Crimecount = crimecounts.iloc[0]['ReptDist']
    

    Normalizingnumber = len(Crimetoanalyze.index)
    
    grouped = Crimetoanalyze.groupby([lambda x: x.year,lambda x: x.month])
    
    Reindex_counts = grouped.count().reset_index()
    
    yes = np.array(range(1,len(Reindex_counts)+1))
    
    Reindex_counts['Year_made'] = yes
    
    Reindex_counts['Norm_incidents'] = Reindex_counts["Incident"] / Normalizingnumber
    
    
    x = np.atleast_2d(Reindex_counts['Year_made'])
    y = np.atleast_2d(Reindex_counts['Norm_incidents'])
    
    X_train = np.transpose(x)
    y_train = np.transpose(y)
    
    
    from sklearn.gaussian_process import GaussianProcess
    import matplotlib.pyplot as plt
    
    
    G = GaussianProcess(theta0=1e-1,
                         thetaL=1e-3, thetaU=1, nugget=0.0000005,corr='cubic')
    G.fit(X_train,y_train)
    
    
    X_pred = np.linspace(X_train.min(), X_train.max())[:, None]
    
    y_pred, MSE = G.predict(X_pred, eval_MSE=True)
    sigma = np.sqrt(MSE)
      
    fig = plt.figure() 
    plt.plot(X_train, y_train, 'r.', markersize=6, label='Observations')
    plt.plot(X_pred, y_pred, 'k:', label=u'Prediction')
    plt.fill(np.concatenate([X_pred, X_pred[::-1]]),
            np.concatenate([y_pred - 1.9600 * sigma,
                           (y_pred + 1.9600 * sigma)[::-1]]),
            alpha=.3, fc='k', ec='None', label='95% confidence interval')
    plt.xlabel('Month')
    plt.ylabel('Relative Number of Incidents')
  
    import os.path
    if os.path.exists("/Users/Jenks/Desktop/Insight_Website/app/static/img/Position_model_image.png"):
        os.remove("/Users/Jenks/Desktop/Insight_Website/app/static/img/Position_model_image.png")
    

    fig.savefig("/Users/Jenks/Desktop/Insight_Website/app/static/img/Position_model_image.png")
    
   
    #cities = []
    #for result in query_results:
    #  cities.append(dict(name=result[0], country=result[1], population=result[2]))
    #
    ####call a function from a_Model package. note we are only pulling one result in the query
    #pop_input = cities[0]['population']
    #the_result = ModelIt(city, pop_input)
    
    import time
    time.sleep(5)
    return render_template("output.html",longitude = Longitude,Length=Crimelen,Radius = Radius,
                           Crimecount = Crimecount, Crime = Crime)