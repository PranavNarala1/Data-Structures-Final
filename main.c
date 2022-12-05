#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "neuralnetwork.h"

int main(int argc, char *argv[])
{
    srand(42);

    double **inputLabels = (double **)calloc(10000, sizeof(double *));
    double **inputData = (double **)calloc(10000, sizeof(double *));
    // todo split into training

    FILE *stream = fopen("data/citrus.csv", "r");

    if (stream == NULL)
        printf("Could not open file");

    char line[1024];
    size_t counter = 0;
    while (fgets(line, 1024, stream))
    {
        char *tmp = strdup(line);
        double *tempArr = (double *)calloc(5, sizeof(double));
        double *labelArr = (double *)calloc(1, sizeof(double));

        char *label = strtok(tmp, ",");
        labelArr[0] = strcmp(label, "grapefruit") == 0 ? 1.0 : 0.0;

        for (size_t i = 0; i < 5; ++i)
        {
            char *tempData = strtok(NULL, ",");
            char *ptr;
            tempArr[i] = strtod(tempData, &ptr);
        }

        inputData[counter] = tempArr;
        inputLabels[counter] = labelArr;
        free(tmp);
        ++counter;
    }

    fclose(stream);

    // split input data into testing and training arrays
    double **trainingData = (double **)calloc(8000, sizeof(double *));
    double **trainingLabels = (double **)calloc(8000, sizeof(double *));

    double **testingData = (double **)calloc(1000, sizeof(double *));
    double **testingLabels = (double **)calloc(1000, sizeof(double *));

    double **validationData = (double **)calloc(1000, sizeof(double *));
    double **validationLabels = (double **)calloc(1000, sizeof(double *));

    size_t otherIdx = 0;
    size_t trainIdx = 0;

    for (size_t i = 0; i < counter / 10; i++)
    {
        size_t startIndex = i * 10;

        for (size_t a = 0; a < 8; ++a)
        {
            trainingData[trainIdx] = inputData[startIndex + a];
            trainingLabels[trainIdx] = inputLabels[startIndex + a];
            ++trainIdx;
        }

        testingData[otherIdx] = inputData[startIndex + 8];
        testingLabels[otherIdx] = inputLabels[startIndex + 8];

        validationData[otherIdx] = inputData[startIndex + 9];
        validationLabels[otherIdx] = inputLabels[startIndex + 9];

        ++otherIdx;
    }

    ArrayList *arrayListPtr = create_al();

    ActivationFunction relu = RELU;
    ActivationFunction tanh = TANH;
    ActivationFunction sigmoid = SIGMOID;
    ActivationFunction identity = IDENTITY;

    Layer *layer1Ptr = create_layer(30, identity, 5);

    append_al(layer1Ptr, arrayListPtr);

    Layer *layer2Ptr = create_layer(10, sigmoid, 30);
    append_al(layer2Ptr, arrayListPtr);

    Layer *layer3Ptr = create_layer(1, sigmoid, 10);
    append_al(layer3Ptr, arrayListPtr);

    NeuralNetwork *neural_network = create_neural_network(*arrayListPtr, 1, 0.05);

    train_neural_network(neural_network, trainingData, trainingLabels, validationData, validationLabels, 8000, 1000, 100, 500);

    test_neural_network(neural_network, testingData, testingLabels, 1000);

    destroy_neural_network(neural_network);
    free(arrayListPtr);

    for (size_t i = 0; i < counter; ++i)
    {
        free(inputData[i]);
        free(inputLabels[i]);
    }

    free(inputData);
    free(inputLabels);

    // we can free these because these are only references to the data in inputData and inputLabels, which we already freed
    free(trainingData);
    free(trainingLabels);
    free(testingData);
    free(testingLabels);
    free(validationData);
    free(validationLabels);

    return 0;
}
