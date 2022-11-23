#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "neuralnetwork.h"

int main(int argc, char *argv[])
{
    srand(42);

    int *inputLabels = calloc(10000, sizeof(int));
    double **inputData = calloc(10000, sizeof(double *));
    // todo split into training

    FILE *stream = fopen("data/citrus.csv", "r");

    if (stream == NULL)
        printf("Could not open file");

    char line[1024];
    size_t counter = 0;
    while (fgets(line, 1024, stream))
    {
        char *tmp = strdup(line);
        double *tempArr = calloc(5, sizeof(double));

        char *label = strtok(tmp, ",");
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

    return 0;
}