#include <stdio.h>
#include <stdlib.h>
#include "layer.h"
//This code is from the arraylist code written in class and was modified to store layers instead of integers. Unnecessary functions for this specific use case were removed.

// The initial capacity of the dynamically size array in the arraylist (this will be resized accordingly as values are added)
size_t INITIAL_CAPACITY = 8;


/*
The arraylist will be stored as a struct with the following variables:
An array of layer pointers (double pointer to a layer). Layer pointers and not layers themselves are stored to make it easier to delete the layers in the destroy function.
The capacity of the arraylist, which is the total amount of values the array can store before it needs to be resized.
The length of the arraylist, which is the current amount of values it is storing.
*/
typedef struct arraylist
{
    Layer **layers;
    size_t capacity; // Total amount it can contain ()
    size_t length;   // Current size
} ArrayList;

//Function prototypes
ArrayList *create_al();
void destroy_al(ArrayList *listPtr);
void print_al(ArrayList list);
void append_al(Layer *layer, ArrayList *listPtr);
void resizeiffull(ArrayList *listPtr);
// delete and insert functions are not needed for this use case

/// @brief Function to allocate and initialize an arraylist
/// @return Returns the allocated and initialized arraylist
ArrayList *create_al()
{
    ArrayList *new_al = (ArrayList *)malloc(sizeof(ArrayList));
    new_al->length = 0;
    new_al->capacity = 8;
    new_al->layers = (Layer **)calloc(new_al->capacity, sizeof(Layer *));
    return new_al;
}

/// @brief Function to destroy an arraylist by deallocating each layer in the layers array using the destroy_layer function and deallocating the arrraylist itself to delete other variables in the struct.
/// @param listPtr 
void destroy_al(ArrayList *listPtr)
{
    for (size_t i = 0; i < listPtr->length; ++i)
        destroy_layer(listPtr->layers[i]);

    free(listPtr->layers);
}

/// @brief Function to print all the layers stored by calling the print_layer function on each function in the layers array.
/// @param list The arraylist instance to print out
void print_al(ArrayList list)
{
    for (size_t i = 0; i < list.length; ++i)
        print_layer(list.layers[i]);
}

/// @brief Function to append a layer onto an arraylist
/// @param layer The layer instance to append
/// @param listPtr The arraylist instance to append to its specific layers array
void append_al(Layer *layer, ArrayList *listPtr)
{
    //Checking to see if the array is full and if so resize before appending.
    resizeiffull(listPtr);
    listPtr->layers[listPtr->length] = layer;
    listPtr->length++;
}

/// @brief Function to dynamically resize the array to double its current size whenever it gets full when the append function is called
/// @param listPtr A poiner to the arraylist instance to append to
void resizeiffull(ArrayList *listPtr)
{
    if (listPtr->length >= listPtr->capacity)
    {
        listPtr->capacity *= 2;
        //Reallocating the array to double its current size and changing the arraylist capacity variable accordingly.
        listPtr->layers = (Layer **)realloc(listPtr->layers, listPtr->capacity * sizeof(Layer *));
    }
}
