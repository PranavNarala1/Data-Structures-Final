#include <stdio.h>
#include <stdlib.h>

enum ActivationFunction{
    IDENTITY, //y = x
    RELU,
    SIGMOID,
    SOFTMAX,
    TANH
};

typedef struct perceptron{
    double *weights;
    int num_weights;
    double bias;
    ActivationFunction activation_function;
} Perceptron;

Perceptron create_perceptron(); // randomely initialize weights and bias
double get_output(double *input);
void set_weight(double value, size_t pos);
void set_bias(double value);