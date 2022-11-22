#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "neuralnetwork.h"

double* readInput(char* filename, int* size);

int main(int argc, char *argv[]) {
    srand(42);

    FILE *stream = fopen("data/citrus.csv", "r");
    if (stream == NULL)
        printf("Error: Could not open file")

    char line[1024];
    while (fgets(line, 1024, stream)) {
        char* tmp = strdup(line);
        printf("%s\n", tmp);
    }

    return 0;
}