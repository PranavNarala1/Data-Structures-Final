#include <stdio.h>
#include <stdlib.h>
#include "perceptron.h"


typedef struct layer{
    Perceptron **perceptrons;
    size_t num_perceptrons;
    ActivationFunction activation_function;
} Layer;

Layer *create_layer(size_t num_perceptrons, ActivationFunction activation_function, size_t num_weights);
double *get_layer_output(Layer *layer, double *input);
Perceptron *get_perceptrons(Layer *layer);
void print_layer(Layer *layer); // print out layer information

Layer *create_layer(size_t num_perceptrons, ActivationFunction activation_function, size_t num_weights){
    Layer *new_layer = malloc(sizeof(Layer));
    new_layer->num_perceptrons = num_perceptrons;
    new_layer->activation_function = activation_function;
    for(size_t i = 0; i < num_perceptrons; ++i)
        new_layer->perceptrons[i] = create_perceptron(activation_function, num_weights);
    
    return new_layer;
}

double *get_layer_output(Layer *layer, double *input){ //look for memory leaks
    double *output = malloc(sizeof(double) * layer->num_perceptrons);
    for(int i = 0; i < layer->num_perceptrons; ++i)
        output[i] = get_output(layer->perceptrons[i], input);
    
    return output;
}

void print_layer(Layer *layer){
    printf("Dense layer with %zu perceptrons and ", layer->num_perceptrons);
    if(layer->activation_function == IDENTITY)
        puts("identity activation function.");
    if(layer->activation_function == RELU)
        puts("relu activation function.");
    if(layer->activation_function == SIGMOID)
        puts("sigmoid activation function.");
    if(layer->activation_function == TANH)
        puts("tanh activation function.");
}