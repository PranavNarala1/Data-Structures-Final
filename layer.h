#include <stdio.h>
#include <stdlib.h>
#include "perceptron.h"

/*
Layers will be defined through the use of structs. It will contain the following:
Each layer will have an array of perceptron pointers (a double pointer array is used since the perceptron's weight and bias values need to be modified while training and applying stochastic gradient descent)
A size variable for the number of the perceptrons (length of the perceptrons array) to use when iterating through perceptrons to get the layer output
The corresponding activation function for a layer to pass in as a parameter to the get_output function for perceptrons.
*/
typedef struct layer
{
    Perceptron **perceptrons;
    size_t num_perceptrons;
    ActivationFunction activation_function;
} Layer;

//Function prototypes
void destroy_layer(Layer *layer);
Layer *create_layer(size_t num_perceptrons, ActivationFunction activation_function, size_t num_weights);
double *get_layer_output(Layer *layer, double *input);
Perceptron *get_perceptrons(Layer *layer);
void print_layer(Layer *layer);

/// @brief Function to allocate and initialize layer
/// @param num_perceptrons The number of perceptrons the layer will have (this is needed to know what size to allocate the perceptrons array to)
/// @param activation_function The activation function the layer will use (needed to know what activation function to pass into the get_output function for perceptrons)
/// @param num_weights The number of weights each perceptron will have (needed to pass in to initialize perceptrons using the create_perceptron function)
/// @return Returns the allocated and initialized layer
Layer *create_layer(size_t num_perceptrons, ActivationFunction activation_function, size_t num_weights)
{
    //Allocating layer and initializing its struct variables based on user input
    Layer *new_layer = (Layer *)malloc(sizeof(Layer));
    new_layer->num_perceptrons = num_perceptrons;
    new_layer->activation_function = activation_function;
    new_layer->perceptrons = (Perceptron **)calloc(num_perceptrons, sizeof(Perceptron *));

    //Creating all the perceptrons for the layer using the create_perceptron function and storing those allocated perceptrons in its perceptron pointer array.
    for (size_t i = 0; i < num_perceptrons; ++i)
        new_layer->perceptrons[i] = create_perceptron(activation_function, num_weights);

    //returning the initialized layer
    return new_layer;
}

/// @brief Function to get layer output
/// @param layer Pointer to the layer instance
/// @param input Input value to pass into each perceptron to get output
/// @return The output of the layer
double *get_layer_output(Layer *layer, double *input)
{
    //Allocating output vector
    double *output = (double *)calloc(layer->num_perceptrons, sizeof(double));

    //Populating output vector through iterating through each perceptron and getting its output.
    for (int i = 0; i < layer->num_perceptrons; ++i)
        output[i] = get_output(layer->perceptrons[i], input);

    return output;
}

/// @brief Prints out information about the layer, including the number of perceptrons it has and its activation function.
/// @param layer Pointer to the layer instance 
void print_layer(Layer *layer)
{
    printf("Dense layer with %zu perceptrons and ", layer->num_perceptrons);
    if (layer->activation_function == IDENTITY)
        puts("identity activation function.");
    if (layer->activation_function == RELU)
        puts("relu activation function.");
    if (layer->activation_function == SIGMOID)
        puts("sigmoid activation function.");
    if (layer->activation_function == TANH)
        puts("tanh activation function.");
}

/// @brief Function to destroy layer by deallocating each of its perceptrons, the perceptron array pointer, and the layer pointer itself to delete the the other variables it is storing
/// @param layer A pointer to the layer instance to delete
void destroy_layer(Layer *layer)
{
    for (int i = 0; i < layer->num_perceptrons; ++i)
        delete_perceptron(layer->perceptrons[i]);

    free(layer->perceptrons);
    free(layer);
}