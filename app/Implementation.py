#!/usr/bin/env python
def Implement(Address,Radius,Crime,Year,Month):
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
    from Whole_sets import Monetary, Vice_Crimes, Violent, Nonviolent, Vehicular, Theft
    from Implementation import Implement
    from Watchout import Watchout
    
    con = mdb.connect('localhost', 'root', 'sqlbacon', 'Crime_data')
    
    if Crime == "Vice_Crimes":
	Crimetopass ="Vice Crime"
    else:
	Crimetopass = Crime+' Crime'
    
    if Year == "2012":
	Month = float(Month)
    elif Year == "2013":
	Month = float(Month)
	Month = Month + 12
    elif Year == "2014":
	Month = float(Month)
	Month = Month + 24
		
    if Crime == "Violent":
	WholeX_train,WholeY_train,wholexrand = Violent()
    if Crime == "Nonviolent":
	WholeX_train,WholeY_train,wholexrand = Nonviolent()
    if Crime == "Theft":
	WholeX_train,WholeY_train,wholexrand = Theft()
    if Crime == "Monetary":
	WholeX_train,WholeY_train,wholexrand = Monetary()
    if Crime == "Vice_Crimes":
	WholeX_train,WholeY_train,wholexrand = Vice_Crimes()
    if Crime == "Vehicular":
	WholeX_train,WholeY_train,wholexrand = Vehicular()
	
    Monthdict = {0:'December 2011',1:'January 2012',2:'February 2012',3:'March 2012',
		 4:'April 2012',5:'May 2012',6:'June 2012',7:'July 2012',8:'August 2012',
		 9:'September 2012',10:'October 2012',11:'November 2012',12:'December 2012',
		 13:'January 2013',14:'February 2013',15:'March 2013',16:'April 2013',
		 17:'May 2013',18:'June 2013',19:'July 2013',20:'August 2013',21:'September 2013',
		 22:'October 2013',23:'November 2013',24:'December 2013',25:'January 2014',
		 26:'February 2014',27:'March 2014',28:'April 2014',29:'May 2014',30:'June 2014',
		 31:'July 2014',32:'August 2014',33:'September 2014',34:'October 2014',
		 35:'November 2014',36:'December 2014',37:'January 2015'}
    
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
    
    Latlong = CrimeSelected[['Incident','lat','lng']]
   
    LatLongList = map(list, Latlong.values)
    
    
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
    
    xrand = np.atleast_2d(np.linspace(0, len(Reindex_counts), 1000)).T
    
    
    from sklearn.gaussian_process import GaussianProcess
    import matplotlib.pyplot as plt
    
    
    RadiusG = GaussianProcess(theta0=1e-1,thetaL=1e-3, thetaU=1,nugget = 0.00000005,corr='cubic')
    RadiusG.fit(X_train,y_train) 
    modely_pred, modelMSE = RadiusG.predict(xrand, eval_MSE=True)
    sigma = np.sqrt(modelMSE)
    
    AllG = GaussianProcess(theta0=1e-1,thetaL=1e-3, thetaU=1,nugget = 0.00000005,corr='cubic')
    AllG.fit(WholeX_train,WholeY_train) 
    WholeY_pred, wholeMSE = AllG.predict(wholexrand, eval_MSE=True)
    wholesigma = np.sqrt(wholeMSE)
	
   
    fig = plt.figure()
    plt.plot(WholeX_train, WholeY_train, 'k.',markerfacecolor='none', markersize=8, label='Observations')
    plt.fill(np.concatenate([wholexrand, wholexrand[::-1]]),
            np.concatenate([WholeY_pred - wholesigma,
                           (WholeY_pred + wholesigma)[::-1]]),
            alpha=.01, facecolor = 'red',ec='None', label='95% confidence interval')
   
   
    plt.plot(X_train, y_train, 'k.', markersize=8, label='Observations')
    plt.fill(np.concatenate([xrand, xrand[::-1]]),
            np.concatenate([modely_pred - sigma,
                           (modely_pred + sigma)[::-1]]),
            alpha=.01, facecolor = 'blue', ec='None', label='95% confidence interval')
    
    xmarkers = [Month,Month+1,Month+2,Month+3,Month+4,Month+5,Month+6,Month+7,Month+8,Month+9,Month+10,Month+11,Month+12]
    labels = ['0','1','2','3','4','5','6','7','8','9','10','11','12']
    plt.xlabel("Month's After Implementation")
    plt.xlim(Month,Month+12)
    plt.xticks(xmarkers,labels)
    plt.ylabel('Relative Number of Incidents')
    
    
    monthtoprint = Month
    monthslower = []
    monthshigher = []
    
    Dicthigher = []
    Dictlower = []
    
    for i in range(int(Month),37):
	Radpred, Radmse = RadiusG.predict(i,eval_MSE=True)
	Allpred, Allmse = AllG.predict(i,eval_MSE=True)
	if (Radpred - 1.96 * np.sqrt(Radmse)) > (Allpred + 1.96 * np.sqrt(Allmse)):
		monthshigher.append(i)
	if (Radpred + 1.96 * np.sqrt(Radmse)) < (Allpred - 1.06 * np.sqrt(Allmse)):
		monthslower.append(i)
		
    for i in range (0,len(monthshigher)):
	Dicthigher.append(Monthdict.get(monthshigher[i]))	

    for i in range (0,len(monthslower)):
	Dictlower.append(Monthdict.get(monthslower[i]))
    
    if not monthslower:
	monthslower.append("None: Your implementation was ineffective")
    
    if not monthshigher:
	monthshigher.append("Crime has not been higher than Boston's average rate")
    
    


  
    import os.path
    if os.path.exists("Position_model_image.png"):
        os.remove("Position_model_image.png")
    

    fig.savefig("Position_model_image.png")
    
    return render_template("output_implementation.html",Longitude = Longitude,Latitude = Latitude,
	Length=Crimelen,Radius = Radius, Crimecount = Crimecount, Crime = Crimetopass,
	Higher = Dicthigher,Lower = Dictlower, LatLongList = LatLongList)
