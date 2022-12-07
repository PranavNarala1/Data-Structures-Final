#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include "arraylist.h" // ArrayList code from class

/*
The neural network will be implementing using a struct with the following variables:
An arraylist storing layer structs for each layer in the network
A size variable storing the output size of the network for evaluating the loss of the network
A learning rate variable, which is a constant that controls how fast the network learns (how much it adjusts its weights and biases to improve accuracy during training based on gradient values calculated using stochastic gradient descent)
*/
typedef struct neural_network
{ // Using mean squared error
    ArrayList layers;
    size_t output_size;
    double learning_rate;
} NeuralNetwork;

//Function prototypes
NeuralNetwork *create_neural_network(ArrayList layers, size_t output_size, double learning_rate);
double get_loss(double *prediction, double *expected, size_t size);
double *forward_propagate(NeuralNetwork *neural_network, double *input);
double ***back_propagate_weights(NeuralNetwork *neural_network, double *input, double *expected);
double **back_propagate_biases(NeuralNetwork *neural_network, double *input, double *expected);
void train_neural_network(NeuralNetwork *neural_network, double **training_inputs, double **training_expected, double **validation_inputs, double **validation_expected, size_t training_size, size_t validation_size, size_t iterations, size_t mini_batch_size);
void test_neural_network(NeuralNetwork *neural_network, double **testing_inputs, double **expected, size_t size);
void print_model_architecture(NeuralNetwork *neural_network);
double **make_predictions(NeuralNetwork *neural_network, double **inputs, size_t size);
void destroy_neural_network(NeuralNetwork *neural_network);

/// @brief Function to allocate and initialize neural network
/// @param layers An arraylist of layers to use as the layers for the neural network
/// @param output_size A size variable for the output array size of the network
/// @param learning_rate The learning rate to use for the neural network (a constant for how much the model adjusts its weights and biases to improve accuracy during training based on gradient values calculated using stochastic gradient descent)
/// @return Returns a pointer to the allocated and initialized neural network
NeuralNetwork *create_neural_network(ArrayList layers, size_t output_size, double learning_rate)
{
    NeuralNetwork *new_neural_network = (NeuralNetwork *)malloc(sizeof(NeuralNetwork));
    new_neural_network->layers = layers;
    new_neural_network->learning_rate = learning_rate;
    new_neural_network->output_size = output_size;
    return new_neural_network;
}

/// @brief The loss of a model is a measure of how inaccurate its predictions are. The loss function used is mean squared error, which is the mean of the errors (predicted - expected values) squared for each output value.
/// @param prediction An array of doubles with the model's prediction
/// @param expected An array of doubles containing the label (what the output should actually be for a correct prediction)
/// @param size The output size of the model
/// @return Returns the mean squared error of the model's prediction
double get_loss(double *prediction, double *expected, size_t size)
{
    double loss = 0;
    for (size_t i = 0; i < size; ++i)
        loss += (prediction[i] - expected[i]) * (prediction[i] - expected[i]);

    return loss / size;
}

/// @brief Forward propagation is one passthrough of an input data case through the neural network (getting the model's prediction for one input case through passing it into the first layer then passing that layer's output as input into the next layer and so on until you get the output of the last layer)
/// @param neural_network Pointer to the neural network instance
/// @param input Input value to get the model prediction for
/// @return The model's output for that input value.
double *forward_propagate(NeuralNetwork *neural_network, double *input)
{
    //Allocating array containing the outputs for each layer to use to pass into the next layer. The first value stored is the input.
    double **layer_outputs = (double **)calloc(1 + neural_network->layers.length, sizeof(double *));
    layer_outputs[0] = input;

    //Populating the layer output array through iterating through each layer and getting the output for that specific layer by passing in the output of the previous layer as input.
    for (size_t i = 0; i < neural_network->layers.length; ++i)
        layer_outputs[i + 1] = get_layer_output((neural_network->layers.layers[i]), layer_outputs[i]);

    //Freeing all the layer predictions except for the last one since that is returned
    for (size_t i = 1; i < neural_network->layers.length; ++i)
        free(layer_outputs[i]);

    //Storing the output prediction separately and freeing the last value so that the array itself can also be freed
    double *out = malloc(sizeof(double) * neural_network->output_size);
    memcpy(out, layer_outputs[neural_network->layers.length], sizeof(double) * neural_network->output_size);
    free(layer_outputs[neural_network->layers.length]);
    free(layer_outputs);

    //Returning the model's prediction.
    return out;
}

/// @brief Function for weight back propagation, which, in this case for our optimizer, is calculating the partial derivative of the loss function with respect to each weight and storing that in a gradient matrix (3-D since there is an array for each perceptron, array of those for each layer, and array of those for each layer). 
/// @param neural_network The neural network pointer instance to use
/// @param input The input value to back propagate for
/// @param expected The label for that input (expected output)
/// @return Returns the weights gradient array
double ***back_propagate_weights(NeuralNetwork *neural_network, double *input, double *expected)
{
    //Allocating gradients matrix (3-D since there is an array for each perceptron, array of those for each layer, and array of those for each layer)
    double ***weights_gradient = (double ***)calloc(neural_network->layers.length, sizeof(double **));
    //Iterating through every layer
    for (size_t layer = 0; layer < neural_network->layers.length; ++layer)
    {
        //Iterating through every perceptron and creating array for that layer's gradient to add to the weights gradient
        double **layer_gradient = (double **)calloc(neural_network->layers.layers[layer]->num_perceptrons, sizeof(double *));
        for (size_t perceptron = 0; perceptron < neural_network->layers.layers[layer]->num_perceptrons; ++perceptron)
        {
            //Iterating through every weight and creating a gradient for each perceptron to add to the layer gradient array
            double *perceptron_gradient = (double *)calloc(neural_network->layers.layers[layer]->perceptrons[perceptron]->num_weights, sizeof(double));
            for (size_t current_weight = 0; current_weight < neural_network->layers.layers[layer]->perceptrons[perceptron]->num_weights; ++current_weight)
            {
                //Getting loss values for before and after the weight is slightly adjusted to calculate numerical partial derivative
                double *prediction = forward_propagate(neural_network, input);
                double current_loss = get_loss(prediction, expected, neural_network->output_size);
                neural_network->layers.layers[layer]->perceptrons[perceptron]->weights[current_weight] += 0.5;
                double *new_prediction = forward_propagate(neural_network, input);
                double new_loss = get_loss(new_prediction, expected, neural_network->output_size);
                double partial_derivative = (new_loss - current_loss) / 0.5;
                neural_network->layers.layers[layer]->perceptrons[perceptron]->weights[current_weight] -= 0.5;
                perceptron_gradient[current_weight] = partial_derivative;
                free(prediction);
                free(new_prediction);
            }
            layer_gradient[perceptron] = perceptron_gradient;
        }
        weights_gradient[layer] = layer_gradient;
    }

    return weights_gradient;
}
/// @brief Function for bias back propagation, which, in this case for our optimizer, is calculating the partial derivative of the loss function with respect to each bias and storing that in a gradient matrix (2-D since there is 1 bias for each perceptron, 1-D array of those for each layer, and 2-D array of those for each layer). 
/// @param neural_network The neural network instance to use.
/// @param input The input value to back propagate for
/// @param expected The data label for that input (expected output value)
/// @return Returns the bias gradient array
double **back_propagate_biases(NeuralNetwork *neural_network, double *input, double *expected)
{
    //Allocating bias gradient 
    double **bias_gradient = (double **)calloc(neural_network->layers.length, sizeof(double *));
    //Iterating through each layer
    for (size_t layer = 0; layer < neural_network->layers.length; ++layer)
    {
        //Allocating gradient array for that layer to add to the gradient array
        double *layer_gradient = (double *)calloc(neural_network->layers.layers[layer]->num_perceptrons, sizeof(double));
        //Iteration through each perceptron
        for (size_t perceptron = 0; perceptron < neural_network->layers.layers[layer]->num_perceptrons; ++perceptron)
        {
            //Getting loss values for before and after the bias is slightly adjusted to calculate numerical partial derivative
            double bias_gradient = 0;
            double *prediction = forward_propagate(neural_network, input);
            double current_loss = get_loss(prediction, expected, neural_network->output_size);
            neural_network->layers.layers[layer]->perceptrons[perceptron]->bias += 0.5;
            double *new_prediction = forward_propagate(neural_network, input);
            double new_loss = get_loss(new_prediction, expected, neural_network->output_size);
            double partial_derivative = (new_loss - current_loss) / 0.5;
            neural_network->layers.layers[layer]->perceptrons[perceptron]->bias -= 0.5;
            layer_gradient[perceptron] = partial_derivative;
            free(prediction);
            free(new_prediction);
        }
        bias_gradient[layer] = layer_gradient;
    }

    return bias_gradient;
}

/// @brief Function to train neural network using stochastic gradient descent. This optimization algorithm works by randomly picking a mini-batch of values (the exact amount is taken in as a parameter), calculating the gradient of the loss function for each training example in the batch with respect to model parameters (weights and biases), calculating the average gradient matrix, and updating each parameter by subtracting the learning rate times the corresponding average gradient value to it. This entire process is 1 iteration and is performed for the amount of iterations passed in. 
/// @param neural_network The neural network instance to use
/// @param training_inputs An array of inputs to use for training
/// @param training_expected The corresponding labels for the array of inputs
/// @param validation_inputs An array of validation examples to use
/// @param validation_expected The corresponding labels for the validation array
/// @param training_size The number of training example passed in
/// @param validation_size The number of validation examples passed in
/// @param iterations The number of iterations to perform the process described. The model loss and accuracy is evaluated after every iteration.
/// @param mini_batch_size The amount of examples to use for each mini-batch.
void train_neural_network(NeuralNetwork *neural_network, double **training_inputs, double **training_expected, double **validation_inputs, double **validation_expected, size_t training_size, size_t validation_size, size_t iterations, size_t mini_batch_size)
{   
    //Allocating arrays to store the weights and bias gradients.
    double ****weights_gradients = (double ****)calloc(mini_batch_size, sizeof(double ***));
    double ***bias_gradients = (double ***)calloc(mini_batch_size, sizeof(double **));

    //Performing the process described in brief for the amount of iterations there are.
    for (size_t iteration = 0; iteration < iterations; ++iteration)
    {
        //Getting gradient for each training example and adding it to the weights and bias gradients arrays
        for (size_t i = 0; i < mini_batch_size; ++i)
        {
            //Randomly picking example to use
            size_t row_to_use = rand() % training_size;
            weights_gradients[i] = back_propagate_weights(neural_network, training_inputs[row_to_use], training_expected[row_to_use]);
            bias_gradients[i] = back_propagate_biases(neural_network, training_inputs[row_to_use], training_expected[row_to_use]);
        }
            //Adjusting weights by subtracting the learning rate times the average calculated gradient
        for (size_t layer = 0; layer < neural_network->layers.length; ++layer)
        {
            for (size_t perceptron = 0; perceptron < neural_network->layers.layers[layer]->num_perceptrons; ++perceptron)
            {
                for (size_t current_weight = 0; current_weight < neural_network->layers.layers[layer]->perceptrons[perceptron]->num_weights; ++current_weight)
                {
                    double average = 0;
                    for (size_t i = 0; i < mini_batch_size; ++i)
                    {
                        average += weights_gradients[i][layer][perceptron][current_weight];
                    }
                    average /= mini_batch_size;
                    neural_network->layers.layers[layer]->perceptrons[perceptron]->weights[current_weight] -= neural_network->learning_rate * average;
                }
            }
        }
        //Adjusting biases by subtracting the learning rate times the average calculated gradient
        for (size_t layer = 0; layer < neural_network->layers.length; ++layer)
        {
            for (size_t perceptron = 0; perceptron < neural_network->layers.layers[layer]->num_perceptrons; ++perceptron)
            {
                double average = 0;
                for (size_t i = 0; i < mini_batch_size; ++i)
                {
                    average += bias_gradients[i][layer][perceptron];
                }
                average /= mini_batch_size;
                neural_network->layers.layers[layer]->perceptrons[perceptron]->bias -= neural_network->learning_rate * average;
            }
        }
        //Calculating model loss and accuracy on validation data after every iteration.
        test_neural_network(neural_network, validation_inputs, validation_expected, validation_size);

        //Freeing weights and bias gradients matrix
        for (size_t gradient = 0; gradient < mini_batch_size; ++gradient)
        {
            for (size_t layer = 0; layer < neural_network->layers.length; ++layer)
            {
                for (size_t perceptron = 0; perceptron < neural_network->layers.layers[layer]->num_perceptrons; ++perceptron)
                {
                    free(weights_gradients[gradient][layer][perceptron]);
                }
                free(weights_gradients[gradient][layer]);
                free(bias_gradients[gradient][layer]);
            }

            free(weights_gradients[gradient]);
            free(bias_gradients[gradient]);
        }
    }

    free(weights_gradients);
    free(bias_gradients);
}


/// @brief Function to calculate model accuracy and loss on testing data
/// @param neural_network Neural network pointer instance to use
/// @param testing_inputs The amount of testing inputs
/// @param expected The labels for the testing data
/// @param size The amount of testing examples
void test_neural_network(NeuralNetwork *neural_network, double **testing_inputs, double **expected, size_t size)
{
    double accuracy = 0;
    double loss = 0;
    //Calculating loss
    for (size_t i = 0; i < size; ++i)
    {
        double *result = forward_propagate(neural_network, testing_inputs[i]);
        loss += get_loss(result, expected[i], neural_network->output_size);
        free(result);
    }
    loss = loss / size;
    printf("Average Loss: %f\n", loss);

    //Calculating accuracy
    for (size_t i = 0; i < size; ++i)
    {

        double *results = forward_propagate(neural_network, testing_inputs[i]);
        if (results[0] >= 0.5 && expected[i][0] == 1.0)
            accuracy += 1;

        else if (results[0] < 0.5 && expected[i][0] == 0.0)
            accuracy += 1;
        free(results);
    }
    accuracy = (accuracy / size) * 100;
    printf("Accuracy: %f\n", accuracy); // Accuracy of classification (if it is a binary classification model and has one output between 0-1 representing the model's confidence that it is one class or the other). The model's output prediction value is rounded to 0 or 1 to interpret the result and compare it to the actual label.
    //If the model's prediction is >= 0.5, it is considered to have predicted class 1 and otherwise (if the prediction is < 0.5), the prediction is class 0.
}

/// @brief Function to print out the model's architecture by iterating through each layer and calling the print_layer function on it.
/// @param neural_network The neural network instance to use
void print_model_architecture(NeuralNetwork *neural_network)
{
    for (size_t i = 0; i < neural_network->layers.length; ++i)
        print_layer(neural_network->layers.layers[i]);
}

/// @brief Function to use the model to make predictions after it is finished training
/// @param neural_network The neura network instance to use
/// @param inputs An array of input examples to perform predictions on
/// @param size The number of input prediction examples
/// @return A double array containing the model's predictions
double **make_predictions(NeuralNetwork *neural_network, double **inputs, size_t size)
{
    double **results = (double **)calloc(size, sizeof(double *));
    for (size_t i = 0; i < size; ++i)
        results[i] = forward_propagate(neural_network, inputs[i]);

    return results;
}

/// @brief Function to destroy the neural network by deallocting its layers using destroy_al and freeing the neural network pointer as well to delete other variables within the struct
/// @param neural_network The neural network instance to use
void destroy_neural_network(NeuralNetwork *neural_network)
{
    destroy_al(&(neural_network->layers));
    free(neural_network);
}
