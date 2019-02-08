import numpy as np 
import pandas as pd 
import pickle as pkl 
import math

__set_file = "beds.csv"

__training_percent = 75

__params = ["Height", "Width", "Thickness"]

__output = ["Bed"]


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

def get_data_set(set_type="train"):
	dataframe = pd.read_csv(__set_file)

	__last_idx = get_last_index(len(dataframe))

	if set_type == "train":
		chuncked_dataframe = dataframe[:__last_idx]
	elif set_type == "test":
		chuncked_dataframe = dataframe[__last_idx:]

	return chuncked_dataframe[__params].as_matrix(), chuncked_dataframe[__output].as_matrix()

def train_nn():
	input_params, actual_output = get_data_set("train")
	weights = initialize_weights(__perceptron.get("number_of_inputs"), __perceptron.get("number_of_neurons"))
	bias = initialize_bias(__perceptron.get("number_of_neurons"))

	__batch_size = 50
	__total_dataset_size = input_params.shape[0]

	net_output = summation(weights, input_vec, bias)
	synaptic_output = sigmoid(net_output)


def test_nn():
	pass



