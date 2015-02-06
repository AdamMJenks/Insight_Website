from flask import render_template, request, make_response
from app import app
import pymysql as mdb
from Geocode import googleGeocoding
from Implementation import Implement
from Watchout import Watchout


###Route to start without having append starter on the end ###
@app.route('/')
@app.route('/starter')
def start_page():
	return render_template("starter.html")
	
###Define the image route so that the files are not cached in the browser###
###Makes it so that there is no need to refresh ####
@app.route('/mapimages/<path:filename>')
def return_image (filename):
    response = make_response(app.send_static_file(filename))
    response.cache_control.max_age = 0	# make cache store zero 
    return response

### Call the page depending on what the user inputs as their usage###
@app.route('/ask_page')
def next_page():
	
	Whichone = request.args.get('Whichone')
	
	if Whichone == '1':
		return render_template("watchout.html")   # what to watch out for in an area
	if Whichone == '2':
		return render_template("implement.html")  # whether their crime reduction policy worked

# Passing variables to the the implement output from the implement.html page and running the output_implementation.html
@app.route('/implement')
def imple():
	Address = request.args.get('Address')        # centralized address
	Radius = request.args.get('Radius')	     # radius user inputs
	Crime = str(request.args.get('crime'))	     # type of crime user inputs
	Year = str(request.args.get('year'))	     # year of crime reduction strategy
	Month = str(request.args.get('month'))	     # month of crime reduction strategy

	Output = Implement(Address,Radius,Crime,Year,Month)
	return Output

# Passing variable to the watch out output from the watchout.html page and running the output_watchout.html
@app.route('/watchout')
def watch():
	Address = request.args.get('Address')		# centralized address
	Radius = request.args.get('Radius')		# radius user inputs
	Crime = str(request.args.get('crime'))		# type of crime user inputs
	
	Output = Watchout(Address,Radius,Crime)
	return Output

# Path to the about page when "about" clicked by a user
@app.route('/about')
def about_me():
	return render_template("about.html")

