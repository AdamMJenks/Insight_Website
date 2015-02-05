from flask import render_template, request, make_response
from app import app
import pymysql as mdb
from Geocode import googleGeocoding
from Implementation import Implement
from Watchout import Watchout



@app.route('/')
@app.route('/starter')
def start_page():
	return render_template("starter.html")
	

@app.route('/mapimages/<path:filename>')
def return_image (filename):
    response = make_response(app.send_static_file(filename))
    response.cache_control.max_age = 0
    return response

@app.route('/ask_page')
def next_page():
	
	Whichone = request.args.get('Whichone')
	
	if Whichone == '1':
		return render_template("watchout.html")
	if Whichone == '2':
		return render_template("implement.html")

@app.route('/implement')
def imple():
	Address = request.args.get('Address')
	Radius = request.args.get('Radius')
	Crime = str(request.args.get('crime'))
	Year = str(request.args.get('year'))
	Month = str(request.args.get('month'))
	implementation = str(request.args.get('implementation'))
	
	Output = Implement(Address,Radius,Crime,Year,Month)
	
	return Output

@app.route('/watchout')
def watch():
	Address = request.args.get('Address')
	Radius = request.args.get('Radius')
	Crime = str(request.args.get('crime'))
	
	Output = Watchout(Address,Radius,Crime)
	
	return Output

