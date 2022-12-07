#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// The activation function is stored as an enum.
typedef enum activation_function{
    IDENTITY, //y = x
    RELU,
    SIGMOID,
    TANH
} ActivationFunction;

//Perceptrons are stored as structs with an array of weights corresponding to each input value, a variable for the number of weights, a double value for the bias, and an activation enum value.
typedef struct perceptron{
    double *weights;
    size_t num_weights;
    double bias;
    ActivationFunction activation_function;
} Perceptron;

//Function prototypes
void delete_perceptron(Perceptron *perceptron);
Perceptron *create_perceptron(ActivationFunction activation_function, size_t num_weights);
double get_output(Perceptron *perceptron, double *input);

/// @brief Function to allocate and initialize perceptron. Its bias and weights values are randomely initialized to values between -1 and 1.
/// @param activation_function Activation function to use for the perceptron
/// @param num_weights The number of weights the perceptron has, which corresponds to its input size
/// @return Returns the allocated and initialized perceptron.
Perceptron *create_perceptron(ActivationFunction activation_function, size_t num_weights){
    //Allocating space for the perceptron
    Perceptron *new_perceptron = (Perceptron *) malloc(sizeof(Perceptron));
    
    //Setting the activation function
    new_perceptron->activation_function = activation_function;
    
    //Randomly initializing weights and bias values to be between -1 and 1
    new_perceptron->num_weights = num_weights;
    double bias_sign = -1;
    if(rand() % 2 == 0){
        bias_sign = 1;
    }
    new_perceptron->bias = bias_sign * ((double) rand()) / (double) RAND_MAX;
    new_perceptron->weights = (double *) calloc(num_weights, sizeof(double));
    for(size_t i = 0; i < num_weights; ++i){
        double weight_sign = -1;
        if(rand() % 2 == 0){
            weight_sign = 1;
        }
        new_perceptron->weights[i] = weight_sign * ((double) rand()) / (double) RAND_MAX;
    }

    //Returning a pointer to the created perceptron
    return new_perceptron;
}



/// @brief Gets the out of a perceptron through taking the dot product between the input array and weights, adding the bias value to that result, and finally applying the activation function. 
/// @param perceptron Pointer to perceptron instance to get the output of based on its specific weight and bias values
/// @param input The input array to pass into the perceptron
/// @return Returns a double that is the output value of the perceptron.
double get_output(Perceptron *perceptron, double *input){
    //Taking the dot product between the input array and weights
    double result = 0;
    for(size_t i = 0; i < perceptron->num_weights; ++i){
        result += input[i] * perceptron->weights[i];
    }
    //Adding bias value
    result += perceptron->bias;

    //Applying activation function
    if(perceptron->activation_function == IDENTITY){
        return result;
    } else if(perceptron->activation_function == RELU){
        if(result <= 0)
            return 0;
        else
            return result;
    } else if(perceptron->activation_function == SIGMOID){
        return 1.0 / (1 + exp(-1 * result));
    } else if(perceptron->activation_function == TANH){
        return (exp(result) - exp(-1 * result))/ (exp(result) + exp(-1 * result));
    }
    return result;
}

/// @brief Function to delete perceptron by deallocating its weights array and a pointer to itself to delete the other variables it's storing
/// @param perceptron Pointer to the perceptron instance to delete
void delete_perceptron(Perceptron *perceptron) {
    free(perceptron->weights);
    free(perceptron);
}