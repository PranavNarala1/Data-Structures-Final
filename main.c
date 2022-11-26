#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "neuralnetwork.h"

int main(int argc, char *argv[]) {
    //Pass in queue of training examples (more efficient since you dont need to resize and only need first example)
    //We can create binary decision tree to represent prediction survival rate under certain conditions
    srand(42);


    ArrayList *arrayListPtr = create_al();

    ActivationFunction relu = RELU;
    ActivationFunction tanh = TANH;
    ActivationFunction sigmoid = SIGMOID;
    ActivationFunction identity = IDENTITY;

    

    Layer *layer1Ptr = create_layer(20, relu, 2);
    
    append_al (layer1Ptr, arrayListPtr);

    Layer *layer2Ptr = create_layer(5, relu, 20);
    append_al (layer2Ptr, arrayListPtr);


    Layer *layer3Ptr = create_layer(1, identity, 5);
    append_al (layer3Ptr, arrayListPtr);

    NeuralNetwork *neural_network = create_neural_network(*arrayListPtr, 1, 0.1);

    double *inputs = (double *) malloc(sizeof(double) * 2);
    inputs[0] = 2;
    inputs[1] = 2;
    double **results = make_predictions(neural_network, &inputs, 1);
    printf("%f, ", results[0][0]);
    puts("");

    // destroy_neural_network(neural_network);

    // [1, 1] -> 2, [2,2] -> 4, [3,3] -> 6


    double **training_inputs = (double **) malloc(sizeof(double *) * 100);
    for(int i = 0; i < 100; ++i){
        training_inputs[i] = (double *) malloc(sizeof(double) * 2);
        training_inputs[i][0] = i + 1;
        training_inputs[i][1] = i + 1;
    }

    double **training_expected = (double **) malloc(sizeof(double *) * 100);
    for(int i = 0; i < 100; ++i){
        training_expected[i] = (double *) malloc(sizeof(double));
        training_expected[i][0] = (i + 1) * 2;
    }
    double **validation_inputs = (double **) malloc(sizeof(double *) * 10);
    for(int i = -10; i < 0; ++i){
        validation_inputs[i] = (double *) malloc(sizeof(double) * 2);
        validation_inputs[i][0] = i + 1;
        validation_inputs[i][1] = i + 1;
    }

    double **validation_expected = (double **) malloc(sizeof(double *) * 10);
    for(int i = -10; i < 0; ++i){
        validation_expected[i] = (double *) malloc(sizeof(double));
        validation_expected[i][0] = (i + 1) * 2;
    }



    train_neural_network(neural_network, training_inputs, training_expected, training_inputs, training_expected, 100, 100, 100, 100);


    double **new_results = make_predictions(neural_network, &inputs, 1);
    printf("New Results: %f, ", new_results[0][0]);
    puts("");

    return 0;
}