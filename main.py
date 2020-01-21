import json
import pandas as pd
import numpy as np

from flask import Flask, request, make_response, jsonify

from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import numbers

import random
import string

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns

import io
from google.cloud import storage

#connect to cloud storage and specify bucket for storing images
client = storage.Client(project='my-chatbot-test-86850')
bucket = client.bucket('my-chatbot-test-86850.appspot.com')


app = Flask(__name__)
log = app.logger

df = pd.read_csv('Airline.csv')

#correlation plot here is only used for generating content before interaction with the bot
THRESHOLD = 0.7

measurements = df.select_dtypes(include=[np.number])
exclusions_set = {'key', 'sku', 'phone', 'ssn', 'upc', 'age','zipcode','longitude','latitude'}
columns_to_exclude = set()

for column in measurements:
	if column.lower() in exclusions_set:
		columns_to_exclude.add(column)

measurements_filtered = measurements.drop(columns_to_exclude, axis=1)
list(measurements_filtered.columns)
measurements_stat_params = measurements_filtered.describe()
std_row = measurements_stat_params.loc['std', :]
inclusion_list = []
measurements_stat_params = measurements_filtered.describe()
mean_row = measurements_stat_params.loc['mean', :]
std_row = measurements_stat_params.loc['std', :]
for col, value in mean_row.items():
	tmp = col, value
	if value/std_row[col] < 2:
		inclusion_list.append(col)
measurements_for_corr = measurements.filter(inclusion_list)
list(measurements_for_corr.columns)

df_model1_sample = measurements_for_corr.sample(frac=0.3, random_state=42)

column_list = list(df_model1_sample.columns)
column_geo_include = []

for column in column_list:
	if column.find('longitude') == -1 and column.find('latitude') == -1 and column.find('zipcode') == -1:
		column_geo_include.append(column)


df_model1_sample_filtered = df_model1_sample.filter(column_geo_include).columns

df_model1_corr = round(df_model1_sample.filter(column_geo_include).corr(), 6).fillna(0)

n_columns = df_model1_corr.shape[1]
column_list = list(df_model1_corr.columns)

corr_matrix  = df_model1_corr.values
triu_corr_matrix = np.triu(corr_matrix, k=1)

i = 0
corr_ranking = []
for i in range(n_columns):
	for j in range(i):
		if abs(triu_corr_matrix[j][i]) > THRESHOLD and abs(triu_corr_matrix[j][i]) <= 1:
			temp = triu_corr_matrix, column_list[i], column_list[j]
			corr_ranking.append(temp)

corr_ranking.sort(reverse=True)

l = 0
corr_ranking_smallest = []
for l in range(n_columns):
	for k in range(l):
		if abs(triu_corr_matrix[k][l]) < THRESHOLD and abs(triu_corr_matrix[k][l]) >= 0:
			temp = triu_corr_matrix[k][l], column_list[l], column_list[k]
			corr_ranking_smallest.append(temp)

sorted(corr_ranking_smallest, key = lambda item: abs(item[0]))
if len(corr_ranking_smallest) == 0:
	corr_ranking_smallest = corr_ranking[:]

length = len(corr_ranking)
if length <= 10:
	top_10_corr = corr_ranking.copy()
else:
	top_10_corr = []
	for i in range(10):
		if corr_ranking[i][0] >= abs(corr_ranking[length-1-i][0]):
			top_10_corr.append(corr_ranking[i])
		else:
			top_10_corr.append(corr_ranking[length-1-i])

col_corr = []
for i in range(len(df_model1_corr.columns)):
	for j in range(len(df_model1_corr.columns)):
		if (abs(df_model1_corr.iloc[j, i]) > 0.5) and (abs(df_model1_corr.iloc[j, i]) < 0.9) and (df_model1_corr.columns[i] not in col_corr) and i != j:
			colname = df_model1_corr.columns[i]
			col_corr.append(colname)

top_ten_set = set()
for item in top_10_corr:
	for i in range(1, len(item)):
		top_ten_set.add(item[i])

if len(top_ten_set) > 0:
	df_model1_filtered = df.filter(top_ten_set)
else:
	df_model1_filtered = df

sns.set()
sns.set(font_scale=1.4)

corr = df_model1_filtered.corr()
fig = plt.subplots(figsize=(15,15))
cmap = sns.diverging_palette(0, 359, as_cmap=True)

ax = sns.heatmap(corr, square=True, cbar_kws={'shrink': 0.82}, annot=True, annot_kws={'size': 14})
labels_list = df_model1_filtered.columns
ax.set_yticklabels(labels_list, rotation=0, va='center')
ax.set_xticklabels(labels_list, rotation=45, va='top')
ax.collections[0].colorbar.set_label('Absolute value of the correlation', rotation=-90, va='bottom')
buf = io.BytesIO()
plt.savefig(buf, format='png')
blob = bucket.blob('Correlation heatmap')
blob.upload_from_string(
	buf.getvalue(),
	content_type='image/png')
buf.close()
blob.make_public()
corr_url = blob.public_url

def checkFile(fileName):
	blobs = client.list_blobs('my-chatbot-test-86850.appspot.com')
	files = []
	for file in blobs:
		files.append(file.name)

	if fileName in files:
		return True
	else:
		return False
#return the shape of the dataset
def get_shape():
	return "The shape of this data set is: " + str(df.shape)

#return the number of missing cells
def get_unknown_count():
	return "There are " + str(df.isnull().sum().sum()) + " missing cells."

#calculate the average for a column
def get_average(columnName):
	return "The average " + columnName + " is: " + str(df[columnName].mean())

#list all the columns with numeric data
def get_numeric_columns():
	measurements = df.select_dtypes(include=[np.number])
	numeric_columns = list(measurements.columns)
	return "This is the list of all numeric columns: " + str(numeric_columns)

#list all the non numeric columns
def get_categories():
	categories = df.select_dtypes(exclude=[np.number])
	other_columns = list(categories.columns)
	return "These are all the non numeric columns: " + str(other_columns)

#predict the amount of clusters required using the elbow method
def get_clusters(columnName1, columnName2):
	df_model1 = df.dropna()
	first_list = [i for i in list(df_model1[columnName1]) if isinstance(i, numbers.Number) and i<10**20]
	second_list = [i for i in list(df_model1[columnName2]) if isinstance(i, numbers.Number) and i<10**20]
	zipped_list = list(zip(first_list, second_list))

	X = np.array(zipped_list).reshape(len(zipped_list), 2)
	distortions = []
	K = range(1, 10)
	#compute distortions for each k in range(1, 10)
	for k in K:
		kmeanModel = KMeans(n_clusters=k).fit(X)
		kmeanModel.fit(X)
		distortions.append(np.sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1))/X.shape[0])

	#compute first order differences
	delta1 = []
	for i in range(0, len(distortions) - 1):
		delta1.append(distortions[i] - distortions[i+1])

	#compute second order differences
	delta2 = []
	for i in range(0, len(delta1) - 1):
		delta2.append(delta1[i] - delta1[i+1])

	strength = {}

	for i in range(0, 7):
		if(delta2[i] - delta1[i+1] > 0):
			strength.update({ i+2: delta2[i] - delta1[i+1] })

	#the max gradient would be identified as the "elbow"
	k = max(strength, key=strength.get)
	return k


def doKmeans(X, nclust=2):
	model = KMeans(nclust)
	model.fit(X)
	clust_labels = model.predict(X)
	cent = model.cluster_centers_
	return(clust_labels, cent)

#plot the clustered data and upload it to cloud storage
def plot_clustered(columnName1, columnName2):
	fileName = columnName1 + '-' + columnName2
	#check if file is already present and return url if it is
	# blobs = client.list_blobs('keen-hope-253318.appspot.com')
	# files = []
	# for file in blobs:
	# 	files.append(file.name)
	# if fileName in files:
	if checkFile(fileName) == True:
		blob = bucket.blob(fileName)
		url = blob.public_url
		return url
	else:
		#else perform clustering and upload the data to storage
		k = get_clusters(columnName1, columnName2)
		num_cols = df.select_dtypes(include=[np.number])
		num_cols.dropna()
		df_cluster = num_cols[[columnName1, columnName2]]
		clust_labels, cent = doKmeans(df_cluster, k)
		kmeans = pd.DataFrame(clust_labels)
		df.insert((df.shape[1]), 'kmeans', kmeans, allow_duplicates=True)
		fig = plt.figure()
		ax = fig.add_subplot(111)
		scatter = ax.scatter(df[columnName1], df[columnName2], c=kmeans[0], cmap='plasma', s=50)
		ax.set_title('K-Means Clustering')
		ax.set_xlabel(columnName1)
		ax.set_ylabel(columnName2)
		plt.colorbar(scatter)
		buf = io.BytesIO()
		plt.savefig(buf, format='png')
		# fileName = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(6)])
		blob = bucket.blob(fileName)
		blob.upload_from_string(
			buf.getvalue(),
			content_type='image/png')

		buf.close()
		blob.make_public()
		url = blob.public_url
		return url

def validate_columns(columnNames, desired_length):
	numeric_columns = get_numeric_columns()
	if len(columnNames) != desired_length:
		return "bad-length"

	for columnName in columnNames:
		if columnName not in numeric_columns:
			return "not-number"
	return "valid"

def selection(choice):
	flag = False
	if str(choice) == "[1.0]":
		res = get_numeric_columns()
	elif str(choice) == "[2.0]":
		res = get_unknown_count()
	elif str(choice) == "[3.0]":
		res = get_categories()
	elif str(choice) == "[4.0]":
		res = get_shape()
	elif str(choice) == "[5.0]" and checkFile('Correlation heatmap'):
		res = corr_url
	elif not checkFile('Correlation heatmap'):
		res = "Please pick a value between 1 and 4"
	else:
		res = "Please pick a value between 1 and 5"
	return res

@app.route('/webhook', methods=['POST'])
def webhook():
	#get fulfillment request
	req = request.get_json(silent=True, force=True)

	#get the intent from json
	try:
		action = req.get('queryResult').get('intent').get('displayName')
	except AttributeError:
		return 'json error'

	if action == 'data.average':
		columnName = req.get('queryResult').get('parameters').get('columnName')
		#validate information to provide appropriate response
		if validate_columns(columnName, 1) == "not-number":
			res = "Please specify a numeric column. To see the list of numeric columns, try asking: What numeric columns are present in the data?"
		elif validate_columns(columnName, 1) == "bad-length":
			res = "Please specify only one column for this operation."
		elif validate_columns(columnName, 1) == "valid":
			try:
				columnName[0] = columnName[0].strip('\'')
				columnName[0] = columnName[0].strip('"')

				res = get_average(columnName[0])
			except:
				res = "Something went wrong, please try again."
	elif action == 'data.missing':
		res = get_unknown_count()
	elif action == 'data.dimensions':
		res = get_shape()
	elif action == 'data.clusters.num':
		columns = req.get('queryResult').get('parameters').get('columnNames')
		#validate the columns to generate appropriate error message
		if validate_columns(columns, 2) == "not-number":
			res = "Please specify a numeric column. To see the list of numeric columns, try asking: What numeric columns are present in the data?"
		elif validate_columns(columns, 2) == "bad-length":
			res = "You have specified an invalid number of columns for this operations. Please specify exactly 2 columns."
		else:
			columnName1 = columns[0]
			columnName2 = columns[1]
			#clean up column names before performing any operations
			#remove inverted commas if present in the column name
			columnName1 = columnName1.strip('\'')
			columnName1 = columnName1.strip('"')

			columnName2 = columnName2.strip('\'')
			columnName2 = columnName2.strip('"')

			try:
				k = get_clusters(columnName1, columnName2)
				res = "You need " + str(k) +" clusters"
			except:
				res = "Something went wrong, please try again."
	elif action == 'data.information.numeric':
		res = get_numeric_columns()
	elif action == 'data.information.categories':
		res = get_categories()
	elif action == 'data.clusters.plot':
		columns = req.get('queryResult').get('parameters').get('columnName')

		if validate_columns(columns, 2) == "not-number":
			res = "Please specify a numeric column. To see the list of numeric columns, try asking: What numeric columns are present in the data?"
		elif validate_columns(columns, 2) == "bad-length":
			res = "You have specified an invalid number of columns for this operations. Please specify exactly 2 columns."
		else:
			#clean up column names
			columnName1 = columns[0]
			columnName2 = columns[1]

			columnName1 = columnName1.strip('\'')
			columnName1 = columnName1.strip('"')

			columnName2 = columnName2.strip('\'')
			columnName2 = columnName2.strip('"')

			res = plot_clustered(columnName1, columnName2)
	elif action == 'Default Welcome Intent':
		#if content has been generated then return this as an additional option
		if checkFile('Correlation heatmap') == True:
			res = "Hello! My name is Data Explorer Bot. I can assist you with the following: \n1. Find the numeric columns in the data\n2. Find how many cells are empty\n3. Find non-numeric categories are present\n4. Find the shape of the data set\n5. Get a correlation heatmap.\nFor column based operations(e.g: finding average value), you can enter any of these commands like this:\n- Find the number of clusters for Column 0 and birthdateid\n- Find the average for any numeric column\n- Plot clustered data for BaseFareAmt and TotalDocAmt\nEnter your selection(1-5):"
		# else return normal set of options
		else:
			res = "Hello! My name is Data Explorer Bot. I can assist you with the following: \n1. Find the numeric columns in the data\n2. Find how many cells are empty\n3. Find non-numeric categories are present\n4. Find the shape of the data set.\nFor column based operations(e.g: finding average value), you can enter any of these commands like this:\n- Find the number of clusters for Column 0 and birthdateid\n- Find the average for any numeric column\n- Plot clustered data for BaseFareAmt and TotalDocAmt\nEnter your selection(1-4):"
	elif action == 'list.options':
		if checkFile('Correlation heatmap') == True:
			res = "Hello! My name is Data Explorer Bot. I can assist you with the following: \n1. Find the numeric columns in the data\n2. Find how many cells are empty\n3. Find non-numeric categories are present\n4. Find the shape of the data set\n5. Get a correlation heatmap.\nFor column based operations(e.g: finding average value), you can enter any of these commands like this:\n- Find the number of clusters for Column 0 and birthdateid\n- Find the average for any numeric column\n- Plot clustered data for BaseFareAmt and TotalDocAmt\nEnter your selection(1-5):"
		else:
			res = "Hello! My name is Data Explorer Bot. I can assist you with the following: \n1. Find the numeric columns in the data\n2. Find how many cells are empty\n3. Find non-numeric categories are present\n4. Find the shape of the data set.\nFor column based operations(e.g: finding average value), you can enter any of these commands like this:\n- Find the number of clusters for Column 0 and birthdateid\n- Find the average for any numeric column\n- Plot clustered data for BaseFareAmt and TotalDocAmt\nEnter your selection(1-4):"
	#perform operation based on selection
	elif action == 'select.number':
		choice = req.get('queryResult').get('parameters').get('number')
		res = selection(choice)
	elif action == 'list.options - select.number':
		choice = req.get('queryResult').get('parameters').get('number')
		res = selection(choice)
	elif action == 'default.number.select':
		choice = req.get('queryResult').get('parameters').get('number')
		res = selection(choice)
	else:
		res = req.get('webhookStatus').get('message')
		log.error('Unexpected action.')

	# webhookStatus = req.get('webhookStatus').get('message')
	# if webhookStatus != 'Webhook execution successful':
	# 	res = webhookStatus

	return make_response(jsonify({'fulfillmentText': res}))

if __name__ == '__main__':
	PORT = 8080
	app.run(
		debug=True,
		port=PORT,
		host='0.0.0.0'
	)
