from flask import render_template, request, make_response
from app import app
import pymysql as mdb
from Geocode import googleGeocoding

from Implementation import Implement
from Watchout import Watchout

@app.route('/input')
def full_input():
  return render_template("input.html")

@app.route('/mapimages/<path:filename>')
def return_image (filename):
    response = make_response(app.send_static_file(filename))
    response.cache_control.max_age = 0
    return response

@app.route('/output')
def which_output():
	
	Address = request.args.get('Address')
	Radius = request.args.get('Radius')
	Crime = str(request.args.get('crime'))
	Year = str(request.args.get('year'))
	Month = str(request.args.get('month'))
	implementation = str(request.args.get('implementation'))

	if implementation == '2':
		Output = Implement(Address,Radius,Crime,Year,Month)
	elif implementation == '1':
		Output = Watchout(Address,Radius,Crime,Year,Month)
		
	return Output