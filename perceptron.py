import numpy as np 
import pandas as pd 
import pickle as pkl 
import math
from random import randint
import matplotlib.pyplot as plt
import sys

__set_file = "beds.csv"

__training_percent = 75

__params = ["Height", "Width", "Thickness"]

__output = ["Output"]

__perceptron_model = "perceptron_model.pickle"

__perceptron = {
	"number_of_neurons": 1,
	"number_of_inputs": 3
}

def sigmoid(input_vec):
	return 1/(1 + np.exp(-input_vec))

def sigmoid_diff(input_vec):
	return sigmoid(input_vec) * sigmoid(1 - input_vec)

def summation(weights, input_vec, bias_vec):
	intermediate_vec = np.matmul(weights, input_vec) + bias_vec
	return intermediate_vec

def initialize_weights(number_of_inputs, neurons_count):
	return np.random.randn(neurons_count, number_of_inputs)

def initialize_bias(neurons_count):
	return np.random.randn(neurons_count, 1)

def get_last_index(length_of_df):
	return math.floor(length_of_df * (__training_percent/100))

def calculate_error(actual_output, desired_output, last_neuron_layer_count):
	diff_error = actual_output - desired_output
	squared_error = diff_error ** 2
	squared_sum_error = np.sum(squared_error)
	avg_squared_sum_error = squared_sum_error / last_neuron_layer_count
	return avg_squared_sum_error

def error_derivative(actual_output, desired_output, last_neuron_layer_count):
	return ((actual_output - desired_output)/last_neuron_layer_count)

def store_model(model_perceptron):
	with open(__perceptron_model, 'wb') as handle:
		pkl.dump(model_perceptron, handle, protocol=pkl.HIGHEST_PROTOCOL)

def load_model():
	with open(__perceptron_model, 'rb') as handle:
		model_perceptron = pkl.load(handle)
	return model_perceptron

def get_data_set(set_type="train"):
	dataframe = pd.read_csv(__set_file)
	dataframe["Output"] = np.where(
		dataframe["Bed"] == "Small Bed",
		0,
		1
	)
	# dataframe = dataframe.sample(frac=1).reset_index(drop=True)

	__last_idx = get_last_index(len(dataframe))

	if set_type == "train":
		chuncked_dataframe = dataframe[:__last_idx]
	elif set_type == "test":
		chuncked_dataframe = dataframe[__last_idx:]

	return chuncked_dataframe[__params].as_matrix(), chuncked_dataframe[__output].as_matrix()

def confusion_matrix(actual_output, desired_output):
	y_actu = pd.Series(actual_output, name='Actual')
	y_pred = pd.Series(desired_output, name='Predicted')
	df_confusion = pd.crosstab(y_actu, y_pred)

	return df_confusion

def get_opposite(output_value):
	if output_value == 1:
		return 0
	else:
		return 1

def get_output_value(output_vector):
	summed_value = np.sum(output_vector)
	return round(summed_value)

def train_nn():
	input_params, desired_output = get_data_set("train")
	weights = initialize_weights(__perceptron.get("number_of_inputs"), __perceptron.get("number_of_neurons"))
	bias = initialize_bias(__perceptron.get("number_of_neurons"))

	__learning_rate = 0.2
	N_times = 333
	__batch_size = 150
	__iteration_times = N_times * __batch_size
	__total_dataset_size = input_params.shape[0]

	__start_idx = 0
	__end_idx = __start_idx + __batch_size - 1

	__errors = []

	while(__end_idx < __total_dataset_size):
		for each_row in range(__iteration_times):
			random_idx = randint(__start_idx, __end_idx)
			input_vec = np.array([input_params[random_idx]])
			desired_output_vec = np.array([desired_output[random_idx]])
			net_output = summation(weights, input_vec.T, bias)
			synaptic_output = sigmoid(net_output)

			# MINIMIZING ERROR
			calculated_error = calculate_error(synaptic_output, desired_output_vec, __perceptron.get("number_of_neurons"))
			derivative_of_error = error_derivative(synaptic_output, desired_output_vec, __perceptron.get("number_of_neurons"))
			derivative_of_sig = sigmoid_diff(net_output)

			recurrsive_delta = np.multiply(derivative_of_error, derivative_of_sig)
			weights_delta = np.matmul(input_vec.T, recurrsive_delta.T).T
			bais_delta = recurrsive_delta

			bias -= __learning_rate * bais_delta
			weights -= __learning_rate * weights_delta

			__errors.append(calculated_error)
		__start_idx = __end_idx + 1
		__end_idx = __end_idx + __batch_size

	# plt.plot(__errors)
	# plt.ylabel("Error plotting")
	# plt.show()

	__perceptron["weights"] = weights
	__perceptron["bias"] = bias

	return __perceptron

def test_nn(model_perceptron):
	__error_threshold = 0.01

	test_params, desired_output = get_data_set("test")

	test_dataset_size = len(test_params)
	weights = model_perceptron.get("weights")
	bias = model_perceptron.get("bias")

	calculated_output = []
	desired_output_arr = []
	correct_classification = 0

	for idx in range(test_dataset_size):
		input_vec = np.array([test_params[idx]])
		desired_output_vec = np.array([desired_output[idx]])

		net_output = summation(weights, input_vec.T, bias)
		synaptic_output = sigmoid(net_output)

		calculated_error = calculate_error(synaptic_output, desired_output_vec, model_perceptron.get("number_of_neurons"))
		output_gross_value = get_output_value(desired_output_vec)

		calculated_output.append(output_gross_value)

		if calculated_error <= __error_threshold:
			correct_classification += 1
			desired_output_arr.append(output_gross_value)
		else:
			desired_output_arr.append(get_opposite(output_gross_value))

	accuracy = (correct_classification / test_dataset_size) * 100.0
	print("Model accuracy: ",accuracy)

	return confusion_matrix(calculated_output, desired_output_arr)

if __name__ == "__main__":
	try:
		command = sys.argv[1]
	except Exception as ex:
		command = "empty"

	if command == "train":
		model = train_nn()
		store_model(model)
	elif command == "test":
		model = load_model()
		result = test_nn(model)
		print("-- Confusion matrix --")
		print(result)
	else:
		print("Please provide command")
		print("Example: python perceptron.py train")
		print("Example: python perceptron.py test")
