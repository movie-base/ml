#!/usr/bin/env python2.7
import json
import os
import requests

from sklearn import preprocessing
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier

INTERACTIONS_API_ENDPOINT = "http://45.63.27.74:8080/interactions/user"
MOVIES_API_ENDPOINT = "http://45.63.27.74:8080/movies"
TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjViYzEzOGE5MTdjOTViMDhiZjAyOWY3YSIsImlhdCI6MTUzOTU3NDM3OX0.ZjUTTbQe5BMCRFK14nkRi13ZCA4zyhQ7bIIOs2DMfd4"

"""
	getInteractions takes in a userId and retrieves all interactions from the
	INTERACTIONS_API_ENDPOINT. A dictionary containing all of the user interactions is returned.
"""
def getInteractions(userId):
	return requests.get(
		INTERACTIONS_API_ENDPOINT + "/" + userId,
		headers = {"Authorization":"Bearer " + TOKEN},
	).json()

"""
	getMovie takes in a movieId and obtains movie details from the MOVIES_API_ENDPOINT. 
	A dictionary containing movie details is returned.
"""
def getMovie(movieId):
	return requests.get(
		MOVIES_API_ENDPOINT + "/" + movieId,
		headers = {"Authorization":"Bearer " + TOKEN},
	).json()

"""
	loadMovieStats takes in a filename and loads stats into a dictionary.
"""
def loadMovieStats(filename):
	movieStats = {}
	f = open(filename, "r")
	for line in f:
		key, value = line.split(":")
		if key != "mostFrequentCountry":
			value = float(value)
		movieStats[key] = value
	return movieStats

"""
	writeMovieStats takes in a filename and writes stats into the corresponding file.
"""
def writeMovieStats(filename, movieStats):
	f = open(filename, "w+")
	for movieStat, value in movieStats.items():
		f.write(movieStat + ":" + str(value) + "\n")
	f.close()

"""
	Given a formattedToWatchMoviesList, getGenreList gets a list of genres from the list.
"""
def getGenreList(formattedToWatchMoviesList):
	genreSet = set()
	for formattedToWatchMovie in formattedToWatchMoviesList:
		for genre in formattedToWatchMovie[0:3]:
			genreSet.add(genre)
	return list(genreSet)

"""
	Given a formattedToWatchMoviesList, getPeopleList gets a list of people from the list.
"""
def getPeopleList(formattedToWatchMoviesList):
	peopleSet = set()
	for formattedToWatchMovie in formattedToWatchMoviesList:
		for person in formattedToWatchMovie[3:7]:
			peopleSet.add(person)
	return list(peopleSet)

"""
	Given a formattedToWatchMoviesList, getCountryList gets a list of countries from the list.
"""
def getCountryList(formattedToWatchMoviesList):
	countrySet = set()
	for formattedToWatchMovie in formattedToWatchMoviesList:
		countrySet.add(formattedToWatchMovie[7])
	return list(countrySet)

"""
	convertBooleanListToIntList converts a list of booleans to a list of integers, where
	all Trues are represented by 1s and all Falses are represented by 0s.
"""
def convertBooleanListToIntList(booleanList):
	intList = []
	for b in booleanList:
		if b == True:
			intList.append(1)
		else:
			intList.append(0)
	return intList

"""
	convertIntListToBooleanList converts a list of integers to a list of booleans, where
	all 1s are represented by Trues and all 0s are represented by Falses.
"""
def convertIntListToBooleanList(intList):
	booleanList = []
	for i in intList:
		if i == 1:
			booleanList.append(True)
		else:
			booleanList.append(False)
	return booleanList


"""
	applyLabelEncoderToMovies allows for the string elements of the 
	formattedToWatchMoviesList to be converted to integers, which will
	be an acceptable type for ML models.
"""
def applyLabelEncoderToMovies(formattedToWatchMoviesList):
	# Encode genres
	genreEncoder = preprocessing.LabelEncoder()
	genreList = getGenreList(formattedToWatchMoviesList)
	genreEncoder.fit(genreList)

	# Encode people (directors, writers, actors)
	peopleEncoder = preprocessing.LabelEncoder()
	peopleList = getPeopleList(formattedToWatchMoviesList)
	peopleEncoder.fit(peopleList)

	# Encode countries
	countryEncoder = preprocessing.LabelEncoder()
	countryList = getCountryList(formattedToWatchMoviesList)
	countryEncoder.fit(countryList)

	encodedMoviesList = []
	for formattedToWatchMovie in formattedToWatchMoviesList:
		encodedGenre = genreEncoder.transform(formattedToWatchMovie[0:3]).tolist()
		encodedPeople = peopleEncoder.transform(formattedToWatchMovie[3:7]).tolist()
		encodedCountry = countryEncoder.transform([formattedToWatchMovie[7]]).tolist()
		encodedList = encodedGenre + encodedPeople + encodedCountry + formattedToWatchMovie[8:]
		encodedMoviesList.append(encodedList)

	return encodedMoviesList

"""
	Given a movieList containing details of each movie (ratings, box office),
	return a dictionary of stats such as: averageImdbRating, mostCommonCountry.
"""
def getMovieStats(movieList):
	statFrequencies = {
		"rottenTomatoesRating": 0,
		"metascore": 0,
		"imdbRating": 0,
		"imdbVotes": 0,
		"boxOffice": 0,
	}
	statSums = {
		"rottenTomatoesRating": 0,
		"metascore": 0,
		"imdbRating": 0,
		"imdbVotes": 0,
		"boxOffice": 0,
	}
	countryFrequencies = {}
	for movie in movieList:
		if movie["rottenTomatoesRating"] is not None:
			statFrequencies["rottenTomatoesRating"] += 1
			statSums["rottenTomatoesRating"] += float(movie["rottenTomatoesRating"])
		if movie["metascore"] is not None:
			statFrequencies["metascore"] += 1
			statSums["metascore"] += float(movie["metascore"])
		if movie["imdbRating"] is not None:
			statFrequencies["imdbRating"] += 1
			statSums["imdbRating"] += float(movie["imdbRating"])
		if movie["imdbVotes"] is not None:
			statFrequencies["imdbVotes"] += 1
			statSums["imdbVotes"] += float(movie["imdbVotes"])
		if movie["boxOffice"] is not None:
			statFrequencies["boxOffice"] += 1
			statSums["boxOffice"] += float(movie["boxOffice"])
		
		country = movie["country"]
		if country is not None:
			if country not in countryFrequencies:
				countryFrequencies[country] = 0
			countryFrequencies[country] += 1

	sortedCountries = sorted(countryFrequencies.items(), key=lambda x:x[1], reverse=True)

	return {
		"averageRottenTomatoesRating": float(statSums["rottenTomatoesRating"])/statFrequencies["rottenTomatoesRating"] if statFrequencies["rottenTomatoesRating"] else 0,
		"averageMetascore": float(statSums["metascore"])/statFrequencies["metascore"] if statFrequencies["metascore"] else 0,
		"averageImdbRating": float(statSums["imdbRating"])/statFrequencies["imdbRating"] if statFrequencies["imdbRating"] else 0,
		"averageImdbVotes": float(statSums["imdbVotes"])/statFrequencies["imdbVotes"] if statFrequencies["imdbVotes"] else 0,
		"averageBoxOffice": float(statSums["boxOffice"])/statFrequencies["boxOffice"] if statFrequencies["boxOffice"] else 0,
		"mostFrequentCountry": (sortedCountries[0])[0] if sortedCountries else "None"
	}

"""
	formatMovie takes in a dictionary of movieDetails and returns a list of
	feature values to provide the ML model.
"""
def formatMovie(movieDetails, movieStats):
	genres = movieDetails["genres"] or []
	directors = movieDetails["directors"] or []
	writers = movieDetails["writers"] or []
	actors = movieDetails["actors"] or []

	return [
		genres[0] if len(genres) >= 1 else "None",
		genres[1] if len(genres) >= 2 else "None",
		genres[2] if len(genres) >= 3 else "None",
		directors[0] if len(directors) >= 1 else "None",
		writers[0] if len(writers) >= 1 else "None",
		actors[0] if len(actors) >= 1 else "None",
		actors[1] if len(actors) >= 2 else "None",
		movieDetails["country"] if movieDetails["country"] != None else movieStats["mostFrequentCountry"],
		float(movieDetails["rottenTomatoesRating"]) if movieDetails["rottenTomatoesRating"] != None else movieStats["averageRottenTomatoesRating"],
		float(movieDetails["metascore"]) if movieDetails["metascore"] != None else movieStats["averageMetascore"],
		float(movieDetails["imdbRating"]) if movieDetails["imdbRating"] != None else movieStats["averageImdbRating"],
		int(movieDetails["imdbVotes"]) if movieDetails["imdbVotes"] != None else movieStats["averageImdbVotes"],
		float(movieDetails["boxOffice"]) if movieDetails["boxOffice"] != None else movieStats["averageBoxOffice"],
	]

"""
	predictIfUserLikesMovies takes in a userId and movieIds list of length K and returns
	a list of length K, where each element is a corresponding prediction.

	This is achieved by retrieving all the movies that a user has watched, formatting it
	and creating a machine learning model that uses the movies + a corresponding hasLiked
	list that indicates if the user has liked each of the watched movies.
"""
def predictIfUserLikesMovies(userId, movieIds, premadeClassifier=None, premadeMovieStats=None):
	classifier = None
	movieStats = None
	if premadeClassifier is not None:
		classifier = premadeClassifier
		movieStats = premadeMovieStats
	else:
		# Generate formatted lists of watchedMovies and hasLiked variables
		interactions = getInteractions(userId)
		watchedMovieIds = [
			interaction["movie"]
			for interaction in interactions
			if interaction["hasWatched"] == True
		]
		hasLikedList = [
			interaction["hasLiked"]
			for interaction in interactions
			if interaction["hasWatched"] == True
		]

		watchedMoviesList = [ 
			getMovie(watchedMovieId)
			for watchedMovieId in watchedMovieIds
		]

		movieStats = getMovieStats(watchedMoviesList)
		formattedWatchedMoviesList = [
			formatMovie(watchedMovie, movieStats)
			for watchedMovie in watchedMoviesList
		]

		# Converts string attributes to numerical attributes
		encodedWatchedMoviesList = applyLabelEncoderToMovies(formattedWatchedMoviesList) 
		encodedLikedList = convertBooleanListToIntList(hasLikedList)

		# Create new file to hold stats for future use and dump stats in file.
		writeMovieStats("./models/" + userId + "-stats", movieStats)

		# Create new file to hold classifier for future use and dump classifier in file.
		filename = "./models/" + userId
		f = open(filename, "w+")
		f.close()

		classifier = KNeighborsClassifier()
		classifier.fit(encodedWatchedMoviesList, encodedLikedList) 
		# print watchedMovieIds
		# print encodedLikedList
		joblib.dump(classifier, filename) 

	# Format toWatchMovieList for prediction.
	toWatchMoviesList = [
		getMovie(movieId)
		for movieId in movieIds
	]
	formattedToWatchMoviesList = [
		formatMovie(toWatchMovie, movieStats)
		for toWatchMovie in toWatchMoviesList
	]
	encodedToWatchMoviesList = applyLabelEncoderToMovies(formattedToWatchMoviesList) 

	# Perform predictions and convert back numerical encoded likedList to boolean values.
	predictions = classifier.predict(encodedToWatchMoviesList)
	return convertIntListToBooleanList(predictions)









