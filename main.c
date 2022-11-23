#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// #include "neuralnetwork.h"

int main(int argc, char *argv[])
{
    srand(42);

    FILE *stream = fopen("data/citrus.csv", "r");

    size_t rows = 0;
    char tempStr[1024];
    while (fgets(tempStr, 1024, stream))
        rows++;

    printf("There are %lu rows\n", rows);

    if (stream == NULL)
        printf("Could not open file");

    int *inputLabels = calloc(rows, sizeof(int));
    double **inputData = calloc(rows, sizeof(double *));
    // todo split into training
    rewind(stream);

    size_t counter = 0;

    char line[1024];
    char* temp;
    double* tempArr;

    while (fgets(line, 1024, stream))
    {
        temp = strdup(line);
        tempArr = calloc(5, sizeof(double));

        char *label = strtok(temp, ",");
        inputLabels[counter] = strcmp(label, "grapefruit") == 0 ? 1 : 0;

        for (size_t i = 0; i < 5; ++i)
        {
            char *tempData = strtok(NULL, ",");
            char *ptr;
            tempArr[i] = strtod(tempData, &ptr);
        }

        inputData[counter] = tempArr;

        ++counter;
    }

    double percentageForTesting = 20.0;
    int testRows = percentageForTesting * rows;

    int *testInputLabels = calloc(testRows, sizeof(int));
    double **testInputData = calloc(testRows, sizeof(double *));

    for (int i=0; i<testRows; ++i) {
        testInputData[i] = inputData[rand()%rows];
        testInputLabels[i] = inputLabels[rand()%rows];
    }

    return 0;
}