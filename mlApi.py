#!/usr/bin/env python2.7
import os

from flask import Flask, request
from flask_restplus import Resource, Api
from flask_restplus import fields
from sklearn.externals import joblib

import mlModel

app = Flask(__name__)
api = Api(
	app,
	default = 'ML Prediction API',
	title = 'ML Prediction API',
	description = 'This is an API that enables users get predictions on whether' +
				  'they will like or dislike a film, given their id and a list of ' +
				  'will-watch movies.'
)

# The following is a model of the prediction input.
predictionInputModel = api.model('MLPredictions', {
    'userId': fields.String,
    'moviesList': fields.List,
})

@api.route('/mlPredictions')
class MLPredictions(Resource):

	@api.response(200, 'Successful')
	@api.response(400, 'Validation error: Prediction input not provided')
	@api.expect(predictionInputModel)
	@api.doc(description='Get ML predictions for a user, given a userId and list of will-watch movies')
	def get(self):
		if 'userId' not in request.json or 'moviesList' not in request.json:
			api.abort(400, 'Validation error: Prediction input not provided')
		
		userId = api.payload['userId']
		moviesList = api.payload['moviesList']

		premadeClassifier = None
		premadeMovieStats = None
		filename = "./models/" + userId
		if os.path.exists(filename):
			premadeClassifier = joblib.load(filename) 
			premadeMovieStats = mlModel.loadMovieStats(filename + "-stats")
		result = mlModel.predictIfUserLikesMovies(userId, moviesList, premadeClassifier, premadeMovieStats)
		return result, 200


if __name__ == '__main__':
	# Create a models folder if one does not exist.
	directory = "./models"
	if not os.path.exists(directory):
		os.makedirs(directory)
	app.run(debug=True)

















