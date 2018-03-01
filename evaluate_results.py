import pickle

predictions = pickle.load( open( "model_predictions.p", "rb" ) )

exp_results = pickle.load( open( "/home/famien/Code/dpcomp_core/experiment_results.p", "rb" ) )

for i in range(len(predictions)):
	print "prediction: ", predictions[i]
	print "results: ", exp_results[i]