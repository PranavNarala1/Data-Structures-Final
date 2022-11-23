#include <stdio.h>
#include <stdlib.h>
#include "neuralnetwork.h"

int main(int argc, char *argv[]) {
    //Pass in queue of training examples (more efficient since you dont need to resize and only need first example)
    //We can create binary decision tree to represent prediction survival rate under certain conditions
    srand(42);


    ArrayList *arrayListPtr = create_al();

    ActivationFunction relu = RELU;
    ActivationFunction tanh = TANH;
    ActivationFunction sigmoid = SIGMOID;

    

    Layer *layer1Ptr = create_layer(50, relu, 2);
    
    append_al (*layer1Ptr, arrayListPtr);

    Layer *layer2Ptr = create_layer(20, tanh, 50);
    append_al (*layer2Ptr, arrayListPtr);


    Layer *layer3Ptr = create_layer(1, sigmoid, 20);
    append_al (*layer3Ptr, arrayListPtr);

    NeuralNetwork *neural_network = create_neural_network(*arrayListPtr, 1, 0.05);

    double *inputs = malloc(sizeof(double) * 2);
    inputs[0] = 10;
    inputs[1] = 26;
    double **results = make_predictions(neural_network, &inputs, 1);
    for(int i = 0; i < 2; ++i){
        printf("%f, ", results[0][i]);
    }
    puts("");

    // destroy_neural_network(neural_network);

    double *result = forward_propagate(neural_network, inputs);
    for(int i = 0; i < 2; ++i){
        printf("%f, ", result[i]);
    }
    puts("");

    return 0;
}