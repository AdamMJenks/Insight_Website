#!/usr/bin/env python
import pandas as pd
import pymysql as mdb
import numpy as np
import urllib
import json
import re
import pprint
import csv

###Each of the following functions prepares training sets for all of boston's crime data, not within a radius
### The name of each function is the kind of crime that it calls from the sql database
def Nonviolent():
    con = mdb.connect('localhost', 'root', 'sqlbacon', 'Crime_data') #host, user, password, #database
    cur = con.cursor()
    CrimeSelected =pd.read_sql("""SELECT * FROM crime_new where Incident = 'Nonviolent'""",con)
    con.close()
    cur.close()

    Crimetoanalyze= CrimeSelected[['Incident','date']].set_index('date')
    Normalizingnumber = len(Crimetoanalyze.index)
    grouped = Crimetoanalyze.groupby([lambda x: x.year,lambda x: x.month])
    Reindex_counts = grouped.count().reset_index()
    yes = np.array(range(1,len(Reindex_counts)+1))
    Reindex_counts['Year_made'] = yes
    Reindex_counts['Norm_incidents'] = Reindex_counts["Incident"] / Normalizingnumber

    x = np.atleast_2d(Reindex_counts['Year_made'])
    y = np.atleast_2d(Reindex_counts['Norm_incidents'])

    WholeX_train = np.transpose(x)
    WholeY_train = np.transpose(y)
    
    xrand = np.atleast_2d(np.linspace(0, len(Reindex_counts), 1000)).T

    return WholeX_train,WholeY_train, xrand

def Theft():
    con = mdb.connect('localhost', 'root', 'sqlbacon', 'Crime_data') #host, user, password, #database
    cur = con.cursor()
    CrimeSelected =pd.read_sql("""SELECT * FROM crime_new where Incident = 'Theft'""",con)
    con.close()
    cur.close()

    Crimetoanalyze= CrimeSelected[['Incident','date']].set_index('date')
    Normalizingnumber = len(Crimetoanalyze.index)
    grouped = Crimetoanalyze.groupby([lambda x: x.year,lambda x: x.month])
    Reindex_counts = grouped.count().reset_index()
    yes = np.array(range(1,len(Reindex_counts)+1))
    Reindex_counts['Year_made'] = yes
    Reindex_counts['Norm_incidents'] = Reindex_counts["Incident"] / Normalizingnumber

    x = np.atleast_2d(Reindex_counts['Year_made'])
    y = np.atleast_2d(Reindex_counts['Norm_incidents'])

    WholeX_train = np.transpose(x)
    WholeY_train = np.transpose(y)
    
    xrand = np.atleast_2d(np.linspace(0, len(Reindex_counts), 1000)).T
    
    return WholeX_train,WholeY_train, xrand

def Violent():
    con = mdb.connect('localhost', 'root', 'sqlbacon', 'Crime_data') #host, user, password, #database
    cur = con.cursor()
    CrimeSelected =pd.read_sql("""SELECT * FROM crime_new where Incident = 'Violent'""",con)
    con.close()
    cur.close()

    Crimetoanalyze= CrimeSelected[['Incident','date']].set_index('date')
    Normalizingnumber = len(Crimetoanalyze.index)
    grouped = Crimetoanalyze.groupby([lambda x: x.year,lambda x: x.month])
    Reindex_counts = grouped.count().reset_index()
    yes = np.array(range(1,len(Reindex_counts)+1))
    Reindex_counts['Year_made'] = yes
    Reindex_counts['Norm_incidents'] = Reindex_counts["Incident"] / Normalizingnumber

    x = np.atleast_2d(Reindex_counts['Year_made'])
    y = np.atleast_2d(Reindex_counts['Norm_incidents'])

    WholeX_train = np.transpose(x)
    WholeY_train = np.transpose(y)
    
    xrand = np.atleast_2d(np.linspace(0, len(Reindex_counts), 1000)).T

    return WholeX_train,WholeY_train,xrand

def Vehicular():
    con = mdb.connect('localhost', 'root', 'sqlbacon', 'Crime_data') #host, user, password, #database
    cur = con.cursor()
    CrimeSelected =pd.read_sql("""SELECT * FROM crime_new where Incident = 'Vehicular'""",con)
    con.close()
    cur.close()

    Crimetoanalyze= CrimeSelected[['Incident','date']].set_index('date')
    Normalizingnumber = len(Crimetoanalyze.index)
    grouped = Crimetoanalyze.groupby([lambda x: x.year,lambda x: x.month])
    Reindex_counts = grouped.count().reset_index()
    yes = np.array(range(1,len(Reindex_counts)+1))
    Reindex_counts['Year_made'] = yes
    Reindex_counts['Norm_incidents'] = Reindex_counts["Incident"] / Normalizingnumber

    x = np.atleast_2d(Reindex_counts['Year_made'])
    y = np.atleast_2d(Reindex_counts['Norm_incidents'])

    WholeX_train = np.transpose(x)
    WholeY_train = np.transpose(y)
    
    xrand = np.atleast_2d(np.linspace(0, len(Reindex_counts), 1000)).T

    return WholeX_train,WholeY_train,xrand

def Vice_Crimes():
    con = mdb.connect('localhost', 'root', 'sqlbacon', 'Crime_data') #host, user, password, #database
    cur = con.cursor()
    CrimeSelected =pd.read_sql("""SELECT * FROM crime_new where Incident = 'Nonviolent'""",con)
    con.close()
    cur.close()

    Crimetoanalyze= CrimeSelected[['Incident','date']].set_index('date')
    Normalizingnumber = len(Crimetoanalyze.index)
    grouped = Crimetoanalyze.groupby([lambda x: x.year,lambda x: x.month])
    Reindex_counts = grouped.count().reset_index()
    yes = np.array(range(1,len(Reindex_counts)+1))
    Reindex_counts['Year_made'] = yes
    Reindex_counts['Norm_incidents'] = Reindex_counts["Incident"] / Normalizingnumber

    x = np.atleast_2d(Reindex_counts['Year_made'])
    y = np.atleast_2d(Reindex_counts['Norm_incidents'])

    WholeX_train = np.transpose(x)
    WholeY_train = np.transpose(y)
    
    xrand = np.atleast_2d(np.linspace(0, len(Reindex_counts), 1000)).T

    return WholeX_train,WholeY_train,xrand

def Monetary():
    con = mdb.connect('localhost', 'root', 'sqlbacon', 'Crime_data') #host, user, password, #database
    cur = con.cursor()
    CrimeSelected =pd.read_sql("""SELECT * FROM crime_new where Incident = 'Monetary'""",con)
    con.close()
    cur.close()

    Crimetoanalyze= CrimeSelected[['Incident','date']].set_index('date')
    Normalizingnumber = len(Crimetoanalyze.index)
    grouped = Crimetoanalyze.groupby([lambda x: x.year,lambda x: x.month])
    Reindex_counts = grouped.count().reset_index()
    yes = np.array(range(1,len(Reindex_counts)+1))
    Reindex_counts['Year_made'] = yes
    Reindex_counts['Norm_incidents'] = Reindex_counts["Incident"] / Normalizingnumber

    x = np.atleast_2d(Reindex_counts['Year_made'])
    y = np.atleast_2d(Reindex_counts['Norm_incidents'])

    WholeX_train = np.transpose(x)
    WholeY_train = np.transpose(y)
    
    xrand = np.atleast_2d(np.linspace(0, len(Reindex_counts), 1000)).T

    return WholeX_train,WholeY_train,xrand