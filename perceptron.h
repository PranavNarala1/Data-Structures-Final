#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef enum activation_function{
    IDENTITY, //y = x
    RELU,
    SIGMOID,
    TANH
} ActivationFunction;

typedef struct perceptron{
    double *weights;
    size_t num_weights;
    double bias;
    ActivationFunction activation_function;
} Perceptron;

Perceptron *create_perceptron(ActivationFunction activation_function, size_t num_weights);
double get_output(Perceptron *perceptron, double *input);

//Allocates space for a perceptron, randomly initializes its weights and biases, and returns it.
Perceptron *create_perceptron(ActivationFunction activation_function, size_t num_weights){
    Perceptron *new_perceptron = malloc(sizeof(Perceptron));
    new_perceptron->num_weights = num_weights;
    new_perceptron->activation_function = activation_function;
    double bias_sign = -1;
    if(rand() % 2 == 0){
        bias_sign = 0;
    }
    new_perceptron->bias = bias_sign * (rand() * 1.0) / RAND_MAX;
    new_perceptron->weights = malloc(sizeof(double)*num_weights);
    for(size_t i = 0; i < num_weights; ++i){
        double weight_sign = -1;
        if(rand() % 2 == 0){
            weight_sign = 0;
        }
        new_perceptron->weights[i] = weight_sign * (rand() * 1.0) / RAND_MAX;
    }
    
    return new_perceptron;
}

double get_output(Perceptron *perceptron, double *input){
    double result = 0;
    for(size_t i = 0; i < perceptron->num_weights; ++i){
        result += input[i] * perceptron->weights[i];
    }
    result += perceptron->bias;

    if(perceptron->activation_function == IDENTITY){
        return result;
    } else if(perceptron->activation_function == RELU){
        if(result <= 0)
            return 0;
        else
            return result;
    } else if(perceptron->activation_function == SIGMOID){
        return 1 / (1 + exp(-1 * result));
    } else if(perceptron->activation_function == TANH){
        return (exp(result) - exp(-1 * result))/ (exp(result) + exp(-1 * result));
    }
    return result;
}