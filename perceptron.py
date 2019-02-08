import numpy as np 
import pandas as pd 
import pickle as pkl 
import math
from random import randint
import matplotlib.pyplot as plt

__set_file = "beds.csv"

__training_percent = 75

__params = ["Height", "Width", "Thickness"]

__output = ["Output"]


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

def get_data_set(set_type="train"):
	dataframe = pd.read_csv(__set_file)
	dataframe["Output"] = np.where(
		dataframe["Bed"] == "Small Bed",
		0,
		1
	)
	dataframe = dataframe.sample(frac=1).reset_index(drop=True)

	__last_idx = get_last_index(len(dataframe))

	if set_type == "train":
		chuncked_dataframe = dataframe[:__last_idx]
	elif set_type == "test":
		chuncked_dataframe = dataframe[__last_idx:]

	return chuncked_dataframe[__params].as_matrix(), chuncked_dataframe[__output].as_matrix()

def train_nn():
	input_params, desired_output = get_data_set("train")
	weights = initialize_weights(__perceptron.get("number_of_inputs"), __perceptron.get("number_of_neurons"))
	bias = initialize_bias(__perceptron.get("number_of_neurons"))

	__learning_rate = 0.1
	N_times = 15
	__batch_size = 50
	__iteration_times = N_times * __batch_size
	__total_dataset_size = input_params.shape[0]

	__start_idx = 0
	__end_idx = __start_idx + __batch_size

	__errors = []

	print(weights, bias)

	while(__end_idx < __total_dataset_size):
		print(__end_idx)
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
		__start_idx = __end_idx
		__end_idx = __end_idx + __batch_size

	print(len(__errors))
	print(weights, bias)

	plt.plot(__errors)
	plt.ylabel("Error plotting")
	plt.show()

def test_nn():
	pass


if __name__ == "__main__":
	a = train_nn()