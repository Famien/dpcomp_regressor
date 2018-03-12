import os
from os import listdir
from os.path import isfile, join
import numpy as np
import math
from sklearn.metrics import r2_score
from sklearn.externals import joblib
import pickle

# mypath = join(os.getcwd(), "datafiles/1D")
# data_files = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]


errors = [float(x)/100 for x in range(1,50)]
# # 2. get data vectors from files

# dataset_vectors = []

# for data_file in data_files:
# 	data_from_file = np.load(data_file)
# 	dataset_vectors.append(data_from_file)

dataset_vectors = np.load("DATA3.npy")[21:121]

# dataset_vectors = data3[0:21] + data3[19679:19700] + data3[25231:25252] + data3[12232:12332


# 5309:5330]

dataset_stats = []

for dataset in dataset_vectors:
	scale = sum(dataset)
	domain_size = len(dataset)
	data_range = max(dataset) - min(dataset)
	std_dev = math.sqrt(np.var(dataset))
	uniform_distance = 0
	for error in errors:
		dataset_stats.append([scale, domain_size, error, data_range, std_dev, uniform_distance])


models = joblib.load("models.pkl")

all_results = []

for dataset_stat in dataset_stats:
	results  = {}
	results["dataset_stat"] = dataset_stat
	for model_name in models:
		model = models[model_name]
		results[model_name] = model.predict([dataset_stat])
	all_results.append(results)

pickle.dump(all_results, open("model_predictions.p", "wb"))