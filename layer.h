#include <stdio.h>
#include <stdlib.h>
#include "perceptron.h"

typedef struct layer{
    Perceptron *perceptrons;
    int num_perceptrons;
} Layer;

Layer create_layer(size_t num_perceptrons, ActivationFunction activation_function);
double *get_layer_output(double *input);
Perceptron *get_perceptrons(void);
char *print_layer(); // print out layer information