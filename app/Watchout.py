### Entire file is a function which is called by the input selected "What to Watch out For" by the user in the starter page of TopCop website ###

#Function takes the address radius and crime that the user inputs
def Watchout(Address,Radius,Crime):
    from flask import render_template, request
    from app import app
    import pymysql as mdb			#to access SQL database
    from Geocode import googleGeocoding		#to get latlng from address entered
    import pandas as pd
    import numpy as np
    import urllib
    import json
    import re
    import os			# for deleting of files if exist
    import random		# for figure naming
    import csv
    from Whole_sets import Monetary, Vice_Crimes, Violent, Nonviolent, Vehicular, Theft
    from Implementation import Implement
    from Watchout import Watchout
    
    # variable to connect to SQL database    
    con = mdb.connect('localhost', 'root', 'sqlbacon', 'Crime_data')
    
    # to define the name that will be passed to the output file, have the right adjectives/nouns etc.
    if Crime == "Vice_Crimes":
	Crimetopass ="Vice Crime"
    else:
	Crimetopass = Crime+' Crime'
    
    # name variable crime so that it knows which features to select from the sql database
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
	
    # dictionary containing all of the months so that it knows where each one stands in an index type fashion
    Monthdict = {0:'December 2011',1:'January 2012',2:'February 2012',3:'March 2012',
		 4:'April 2012',5:'May 2012',6:'June 2012',7:'July 2012',8:'August 2012',
		 9:'September 2012',10:'October 2012',11:'November 2012',12:'December 2012',
		 13:'January 2013',14:'February 2013',15:'March 2013',16:'April 2013',
		 17:'May 2013',18:'June 2013',19:'July 2013',20:'August 2013',21:'September 2013',
		 22:'October 2013',23:'November 2013',24:'December 2013',25:'January 2014',
		 26:'February 2014',27:'March 2014',28:'April 2014',29:'May 2014',30:'June 2014',
		 31:'July 2014',32:'August 2014',33:'September 2014',34:'October 2014',
		 35:'November 2014',36:'December 2014',37:'January 2015'}
    
    # address  from the google api for lat long
    Addresspassed = googleGeocoding(Address)
    
    
    Local_jsonStr = str(Addresspassed)				# Addresspassed turn string (JSON was different every time, dirty method)
    StartLoccoord = Local_jsonStr.find("u'location'")		# location of start of coordinates in string
    EndLoccoord = Local_jsonStr.find("'address_components'")	# location of end of coordinates in string
    
    Coordstring = Local_jsonStr[StartLoccoord:EndLoccoord]	# string of just that latitue and longitude
    
    Coordinates = re.findall(r"[-+]?\d*\.\d+|\d+",Coordstring)	# make a list of the lat and long
    
    # Pass latitiude and longitude to their own variables to be used for sql query
    Latitude = str(Coordinates[0])
    Longitude = str(Coordinates[1])
       
    # using the database
    cur = con.cursor()
    
    ### MySQL query that pulls out the events within the radius given of the addresses latitude and longitude (uses curvature of the earth, not flat form)
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
    
       
    # index the crime selection by the date and pull out only the incident
    Crimetoanalyze= CrimeSelected[['Incident','date']].set_index('date')
    
    # create list of lat/lngs to be passed to the output map
    Latlong = CrimeSelected[['Incident','lat','lng']]
    LatLongList = map(list, Latlong.values)
    
    # pass total number of crimes (over all years) to variable, used later to create relative crime incident counts
    Normalizingnumber = len(Crimetoanalyze.index)
    
    # Group the crimes by month and year so you have the counts and reindex the table so it's a normal table, not pandas grouped format
    grouped = Crimetoanalyze.groupby([lambda x: x.year,lambda x: x.month])
    Reindex_counts = grouped.count().reset_index()
    
    # make new column that is a continuous number based on the month number after the event and create normalized crime column
    Countsarray = np.array(range(1,len(Reindex_counts)+1))
    Reindex_counts['Year_made'] = Countsarray
    Reindex_counts['Norm_incidents'] = Reindex_counts["Incident"] / Normalizingnumber
    
    # make training set for the model, array has to be 2d as per scikit learns request
    x = np.atleast_2d(Reindex_counts['Year_made'])
    y = np.atleast_2d(Reindex_counts['Norm_incidents'])
    X_train = np.transpose(x)
    y_train = np.transpose(y)
    
    # random values to make line on graph of prediction
    xrand = np.atleast_2d(np.linspace(0, len(Reindex_counts), 1000)).T
    
    # import sklearn for modeling and matplotlib for graphing
    from sklearn.gaussian_process import GaussianProcess
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    # Gaussian modeling of the training set of the crimes requested using the radius given by the user
    RadiusG = GaussianProcess(theta0=1e-1,thetaL=1e-3, thetaU=1,nugget = 0.00000005,corr='cubic')
    RadiusG.fit(X_train,y_train) 
    modely_pred, modelMSE = RadiusG.predict(xrand, eval_MSE=True)
    sigma = np.sqrt(modelMSE)
    
    # Gaussian modeling of the training set of all crimes across boston of that crime
    AllG = GaussianProcess(theta0=1e-1,thetaL=1e-3, thetaU=1,nugget = 0.00000005,corr='cubic')
    AllG.fit(WholeX_train,WholeY_train) 
    WholeY_pred, wholeMSE = AllG.predict(wholexrand, eval_MSE=True)
    wholesigma = np.sqrt(wholeMSE)

    # graph the lines and confidence intervals of the radius set and all of boston sets 
    fig = plt.figure()
    plt.plot(WholeX_train, WholeY_train, 'k.',markerfacecolor='none', markersize=8, label='Observations')
    plt.fill(np.concatenate([wholexrand, wholexrand[::-1]]),
            np.concatenate([WholeY_pred - wholesigma,
                           (WholeY_pred + wholesigma)[::-1]]),
            alpha=.5, facecolor = 'grey',ec='None', label='95% confidence interval')
    plt.plot(X_train, y_train, 'k.', markersize=8, label='Observations')
    plt.fill(np.concatenate([xrand, xrand[::-1]]),
            np.concatenate([modely_pred - sigma,
                           (modely_pred + sigma)[::-1]]),
            alpha=.5, facecolor = 'blue', ec='None', label='95% confidence interval')
    plt.xlabel("Month's After January 2012")
    plt.xlim(1,36)
    plt.ylabel('Relative Number of Incidents')
    
    #give the figure name a random integer name to be saved as (will pass another way in the future, dirty way)
    Figurenumber = str(random.randint(1,200))
    Figurename = Figurenumber + ".png"
    
    # delete image if already exists, then save and clear the fig.clf so that I can graph differences using matplotlib
    if os.path.exists(os.path.join("./app/static/img/", Figurename)):
        os.remove(os.path.join("./app/static/img/",Figurename))
    fig.savefig(os.path.join("./app/static/img/", Figurename))
    fig.clf()

    #initiate lists to be used (brute strength way to do it)
    monthslower = []
    monthshigher = []
    
    Dicthigher = []
    Dictlower = []     
    
    # find the months in which the radius selected is lower or higher than bostons, based on 1.5X difference from std. dev.
    for i in range(1,36):
	Radpred, Radmse = RadiusG.predict(i,eval_MSE=True)
	Allpred, Allmse = AllG.predict(i,eval_MSE=True)
	if (Radpred - 1.5 * np.sqrt(Radmse)) > (Allpred + 1.5 * np.sqrt(Allmse)):
		monthshigher.append(i)
	if (Radpred + 1.5 * np.sqrt(Radmse)) < (Allpred - 1.5 * np.sqrt(Allmse)):
		monthslower.append(i)
    
    # initiate difference lengths and xaxis to be used for differences graph	
    Difference= []
    Xaxiscount = range(1,36)
    
    # put differences of mean between radius and all of boston crimes into the previously initiated list "Difference"
    for i in range(1,36):
	Radpred = RadiusG.predict(i)
	Allpred = AllG.predict(i)
	Difference.append((Radpred)/(Allpred))
	
    # plot the bar graph of the differences for each month with a red dashed line at the 1:1 ratio mark
    plt.bar(Xaxiscount,Difference,color='black')
    plt.xlim(1,36)
    plt.ylim(0.5,1.5)
    plt.axhline(1,color='red',linestyle='dashed',linewidth=3)
    plt.xlabel("Months After January, 2012")
    plt.ylabel("Ratio of Crime Within Radius To Boston Crime")
    
    # make figure with randomized number (dirty way to do it, will pass with a better way in the future)
    Figurenumber = str(random.randint(201,400))
    Figurename2 = Figurenumber + ".png"
    
    # make a new figure of the bar graph, remove if already there
    if os.path.exists(os.path.join("./app/static/img/", Figurename2)):
        os.remove(os.path.join("./app/static/img/",Figurename2))
    fig.savefig(os.path.join("./app/static/img/", Figurename2))
    
    # lists for binning of the seasons
    Spring = [3,4,5,15,16,17,27,28,29]
    Summer = [6,7,8,18,19,20,30,31,32]
    Fall = [9,10,11,21,22,23,33,34,35]
    Winter = [0,1,2,12,13,14,24,25,26,36]
    
    # initialize scores for each season
    FallScore = 0
    SummerScore = 0
    SpringScore = 0
    WinterScore = 0
    
    # lists of which seasons have the radius has higher or lower seasonal rates
    HigherSeasons = []
    LowerSeasons = []
    
    # loop through months with higher rates and change season score accordingly    
    for i in range (0,len(monthshigher)):
	if monthshigher[i] in (Spring):
	    SpringScore += 1
	if monthshigher[i] in (Fall):
	    FallScore += 1
	if monthshigher[i] in (Summer):
	    SummerScore += 1
	if monthshigher[i] in (Winter):
	    WinterScore += 1
    
    # loop through months with lower rates and change season score accordingly
    for i in range (0,len(monthslower)):
	if monthslower[i] in (Spring):
	    SpringScore += -1
	if monthslower[i] in (Fall):
	    FallScore += -1
	if monthslower[i] in (Summer):
	    SummerScore += -1
	if monthslower[i] in (Winter):
	    WinterScore += -1
    
    # if statements that add season strings to Higher seasons list to be passed to output
    if SpringScore < 0:
	LowerSeasons.append('Spring')
    if FallScore < 0:
	LowerSeasons.append('Fall')
    if SummerScore < 0:
	LowerSeasons.append('Summer')
    if WinterScore < 0:
	LowerSeasons.append('Winter')
    
    # if statements that add season strings to Higher seasons list to be passed to output
    if SpringScore > 0:
	HigherSeasons.append('Spring')
    if FallScore > 0:
	HigherSeasons.append('Fall')
    if SummerScore > 0:
	HigherSeasons.append('Summer')
    if WinterScore > 0:
	HigherSeasons.append('Winter')

    # if higher or lower seasons list is empty, add the following strings to be passed off to the output
    if not LowerSeasons:
	LowerSeasons.append("Average rate is no lower than Boston's Average")    
    if not HigherSeasons:
	HigherSeasons.append("Average rate is no higher than Boston's Average")
    
    # render the output file and pass variables to the html to use for output
    return render_template("output_watchout.html",Longitude = Longitude,Latitude=Latitude,Radius = Radius, Crime = Crimetopass,
	Higher = Dicthigher, Lower = Dictlower,	HigherSeasons = HigherSeasons,
	LowerSeasons = LowerSeasons, FallScore = FallScore, SpringScore = SpringScore,
	SummerScore = SummerScore, WinterScore = WinterScore, LatLongList=LatLongList,
	Figurename = Figurename, Figurename2=Figurename2)