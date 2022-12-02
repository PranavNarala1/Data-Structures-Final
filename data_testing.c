#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "neuralnetwork.h"

int main(int argc, char *argv[])
{
    srand(42);

    double **inputLabels = (double **) calloc(10000, sizeof(double *));
    double **inputData = (double **) calloc(10000, sizeof(double *));
    // todo split into training

    FILE *stream = fopen("data/citrus.csv", "r");

    if (stream == NULL)
        printf("Could not open file");

    char line[1024];
    size_t counter = 0;
    while (fgets(line, 1024, stream))
    {
        char *tmp = strdup(line);
        double *tempArr = (double *) calloc(5, sizeof(double));

        char *label = strtok(tmp, ",");
        inputLabels[counter] = (double *) malloc(sizeof(double));
        inputLabels[counter][0] = strcmp(label, "grapefruit") == 0 ? 1.0 : 0.0;

        for (size_t i = 0; i < 5; ++i)
        {
            char *tempData = strtok(NULL, ",");
            char *ptr;
            tempArr[i] = strtod(tempData, &ptr);
        }

        inputData[counter] = tempArr;

        ++counter;
    }

    /*for(size_t i = 0; i < 10000; ++i){
        for(size_t j = 0; j < 5; ++j){
            printf("%f, ", inputData[i][j]);
        }
        puts("");
    } */

    /*for(size_t i = 0; i < 10000; ++i){
        printf("%f, ", inputLabels[i][0]);
        puts("");
    } */












    ArrayList *arrayListPtr = create_al();

    ActivationFunction relu = RELU;
    ActivationFunction tanh = TANH;
    ActivationFunction sigmoid = SIGMOID;
    ActivationFunction identity = IDENTITY;

    

    Layer *layer1Ptr = create_layer(20, identity, 5);
    
    append_al (layer1Ptr, arrayListPtr);

    Layer *layer2Ptr = create_layer(10, sigmoid, 20);
    append_al (layer2Ptr, arrayListPtr);


    Layer *layer3Ptr = create_layer(1, sigmoid, 10);
    append_al (layer3Ptr, arrayListPtr);

    NeuralNetwork *neural_network = create_neural_network(*arrayListPtr, 1, 0.01);

    /*double *test_input = malloc(sizeof(double)*5);
    test_input[0] = 1;
    test_input[1] = 1;
    test_input[2] = 1;
    test_input[3] = 1;
    test_input[4] = 1;

    double *result = forward_propagate(neural_network, test_input);
    printf("Result: %f\n", result[0]);*/

    train_neural_network(neural_network, inputData, inputLabels, inputData, inputLabels, 10000, 10000, 1000, 500);


    return 0;
}

//clang++ -g main.c -o prog
//lldb ./prog
//run
//quit