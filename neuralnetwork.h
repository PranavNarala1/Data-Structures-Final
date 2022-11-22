#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include "arraylist.h" // ArrayList code from class


enum LossFunction{categorical_crossentropy, sparse_categorical_crossentropy}

typedef struct neural_network{
    ArrayList layers;
    LossFunction loss_function;
} NeuralNetwork;

NeuralNetwork create_neural_network(ArrayList layers, LossFunction loss_function);
double get_loss(double *input, double *expected);
double *forward_propagate(double *input);
double ***back_propagate_weights(double *prediction, double *expected); //Gets weights gradient for each layer for a single training example
double **back_propagate_biases(double *prediction, double *expected); //Gets bias gradient for each layer for a single training example
void train_neural_network(double **training_inputs, double **training_expected, double **validation_inputs, double **validation_expected, size_t iterations, size_t mini_batch_size, size_t evaluation_frequency); //prints out accuracy after each evaluation_frequency iterations
void test_neural_network(double **testing_inputs, double **expected);
void print_model_architecture();
double **make_predictions(double **inputs);