#include <stdio.h>
#include <stdlib.h>
#include "layer.h"

size_t INITIAL_CAPACITY = 8;

typedef struct arraylist{
    Layer *layers;
    size_t capacity; //Total amount it can contain ()
    size_t length; // Current size
} ArrayList;

ArrayList *create_al();
void destroy_al(ArrayList *listPtr);
void print_al(ArrayList list);
void append_al (Layer layer, ArrayList *listPtr);
void resizeiffull(ArrayList *listPtr);
//delete and insert functions are not needed for this use case

ArrayList *create_al(){
    ArrayList *new = malloc(sizeof(ArrayList));
    new->length = 0;
    new->capacity = 8;
    // new.array = malloc(sizeof(Layer) * new.capacity);
    new->layers = calloc(new->capacity, sizeof(Layer));
    return new;
}

void destroy_al(ArrayList *listPtr){
    free(listPtr->layers);
    free(listPtr);
}

void print_al(ArrayList list){
    for(size_t i = 0; i < list.length; ++i){
        print_layer(&(list.layers[i]));
    }
}

void append_al (Layer layer, ArrayList *listPtr){
    resizeiffull(listPtr);
    listPtr->layers[listPtr->length] = layer;
    listPtr->length++;
}

void resizeiffull(ArrayList *listPtr){
    if(listPtr->length >= listPtr->capacity){
        listPtr->capacity *= 2;
        listPtr->layers = realloc(listPtr->layers, listPtr->capacity*sizeof(Layer));
    }
}
