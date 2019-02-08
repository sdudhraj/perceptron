import numpy as np 
import pandas as pd 

__perceptron = {
	"number_of_neurons": 1,
	"input": 3
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

def get_training_set():
	pass

def get_test_set():
	pass

def ():



