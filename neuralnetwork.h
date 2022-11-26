#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include "arraylist.h" // ArrayList code from class



typedef struct neural_network{ //Using mean squared error
    ArrayList layers;
    size_t output_size;
    double learning_rate;
} NeuralNetwork;


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


NeuralNetwork *create_neural_network(ArrayList layers, size_t output_size, double learning_rate){
    NeuralNetwork *new_neural_network = (NeuralNetwork *) malloc(sizeof(NeuralNetwork));
    new_neural_network->layers = layers;
    new_neural_network->learning_rate = learning_rate;
    new_neural_network->output_size = output_size;
    return new_neural_network;
}
double get_loss(double *prediction, double *expected, size_t size){ //issue with size being 2000
    double loss = 0;
    //printf("size: %d\n", size);
    for(size_t i = 0; i < size; ++i){
        loss += (prediction[i] - expected[i]) * (prediction[i] - expected[i]);
    }
    
    return loss / size;
}
double *forward_propagate(NeuralNetwork *neural_network, double *input){
    double **layer_outputs = (double **) calloc(1 + neural_network->layers.length, sizeof(double *));
    layer_outputs[0] = input;
    //double *current_values = input;
    //printf("Number of Perceptrons in Layer 1: %d\n", neural_network->layers.layers[0]->num_perceptrons);
    //printf("Number of Perceptrons in Layer 2: %d\n", neural_network->layers.layers[1]->num_perceptrons);
    //printf("Number of Perceptrons in Layer 3: %d\n", neural_network->layers.layers[2]->num_perceptrons);


    for(size_t i = 0; i < neural_network->layers.length; ++i){
        layer_outputs[i + 1] = (double *) calloc(neural_network->layers.layers[i]->num_perceptrons, sizeof(double));
        
        //printf("Layer Output %d: %f\n", i, get_layer_output(neural_network->layers.layers[i], layer_outputs[i])[0]);
        //double *result = get_layer_output((neural_network->layers.layers[i]), layer_outputs[i]);
        //for(int j = 0; j < neural_network->layers.layers[i]->num_perceptrons; ++j){
        //    printf("%f, ", result[j]);
        //}
        //puts("");


        layer_outputs[i + 1] = get_layer_output((neural_network->layers.layers[i]), layer_outputs[i]);
        // current_values = get_layer_output(&(neural_network->layers.layers[i]), current_values);
    }

    return layer_outputs[neural_network->layers.length];
}

double ***back_propagate_weights(NeuralNetwork *neural_network, double *input, double *expected){
    double ***weights_gradient = (double ***) calloc(neural_network->layers.length, sizeof(double**));
    for(size_t layer = 0; layer < neural_network->layers.length; ++layer){
        double **layer_gradient = (double **) calloc(neural_network->layers.layers[layer]->num_perceptrons, sizeof(double*));
        for(size_t perceptron = 0; perceptron < neural_network->layers.layers[layer]->num_perceptrons; ++perceptron){
            double *perceptron_gradient = (double *) calloc(neural_network->layers.layers[layer]->perceptrons[perceptron]->num_weights, sizeof(double));
            for(size_t current_weight = 0; current_weight < neural_network->layers.layers[layer]->perceptrons[perceptron]->num_weights; ++current_weight){
                double *prediction = forward_propagate(neural_network, input);
                //puts("got here 4");
                double current_loss = get_loss(prediction, expected, neural_network->output_size);
                neural_network->layers.layers[layer]->perceptrons[perceptron]->weights[current_weight] += 0.5;
                double *new_prediction = forward_propagate(neural_network, input);
                double new_loss = get_loss(new_prediction, expected, neural_network->output_size); //prediction value is wrong
                //puts("got here 5");
                double partial_derivative = (new_loss - current_loss) / 0.5;
                //printf("Derivative: %f \n", partial_derivative);
                neural_network->layers.layers[layer]->perceptrons[perceptron]->weights[current_weight] -= 0.5;
                //printf("Changing layer %d %f %f %f %f\n", layer, prediction[0], new_prediction[0], expected[0], current_loss);
                perceptron_gradient[current_weight] = partial_derivative;
            }
            layer_gradient[perceptron] = perceptron_gradient;
        }
        weights_gradient[layer] = layer_gradient;
    }

    return weights_gradient;
} //Gets weights gradient for each layer for a single training example

double **back_propagate_biases(NeuralNetwork *neural_network, double *input, double *expected){
    double **bias_gradient = (double **) calloc(neural_network->layers.length, sizeof(double*));
    for(size_t layer = 0; layer < neural_network->layers.length; ++layer){
        double *layer_gradient = (double *) calloc(neural_network->layers.layers[layer]->num_perceptrons, sizeof(double));
        for(size_t perceptron = 0; perceptron < neural_network->layers.layers[layer]->num_perceptrons; ++perceptron){
            double bias_gradient = 0;
            double *prediction = forward_propagate(neural_network, input);
            double current_loss = get_loss(prediction, expected, neural_network->output_size);
            neural_network->layers.layers[layer]->perceptrons[perceptron]->bias += 0.5;
            double *new_prediction = forward_propagate(neural_network, input);
            double new_loss = get_loss(new_prediction, expected, neural_network->output_size);
            double partial_derivative = (new_loss - current_loss) / 0.5;
            neural_network->layers.layers[layer]->perceptrons[perceptron]->bias -= 0.5;
            layer_gradient[perceptron] = partial_derivative;
        }
        bias_gradient[layer] = layer_gradient;
    }

    return bias_gradient;
} //Gets bias gradient for each layer for a single training example

void train_neural_network(NeuralNetwork *neural_network, double **training_inputs, double **training_expected, double **validation_inputs, double **validation_expected, size_t training_size, size_t validation_size, size_t iterations, size_t mini_batch_size){
    //Uses stochastic gradient descent
    
    //double ****weights_gradients = malloc((sizeof(double**) * neural_network->layers.length) * mini_batch_size);
    double ****weights_gradients = (double ****) calloc(mini_batch_size, sizeof(double***));
    //double ***bias_gradients = malloc((sizeof(double*) * neural_network->layers.length) * mini_batch_size);
    double ***bias_gradients = (double ***) calloc(mini_batch_size, sizeof(double**));
    
    //do this for the amount of iterations there are
    for(size_t iteration = 0; iteration < iterations; ++iteration){

        //puts("got here 1");

        for(size_t i = 0; i < mini_batch_size; ++i){
            size_t row_to_use = rand() % training_size;
            weights_gradients[i] = back_propagate_weights(neural_network, training_inputs[row_to_use], training_expected[row_to_use]);
            bias_gradients[i] = back_propagate_biases(neural_network, training_inputs[row_to_use], training_expected[row_to_use]);
        }

        for(size_t layer = 0; layer < neural_network->layers.length; ++layer){
            for(size_t perceptron = 0; perceptron < neural_network->layers.layers[layer]->num_perceptrons; ++perceptron){
                for(size_t current_weight = 0; current_weight < neural_network->layers.layers[layer]->perceptrons[perceptron]->num_weights; ++current_weight){
                    double average = 0;
                    for(size_t i = 0; i < mini_batch_size; ++i){
                        average += weights_gradients[i][layer][perceptron][current_weight];
                    }
                    average /= mini_batch_size;
                    neural_network->layers.layers[layer]->perceptrons[perceptron]->weights[current_weight] -= neural_network->learning_rate * average;

                }
            }
        }
        

        for(size_t layer = 0; layer < neural_network->layers.length; ++layer){
            for(size_t perceptron = 0; perceptron < neural_network->layers.layers[layer]->num_perceptrons; ++perceptron){
                    double average = 0;
                    for(size_t i = 0; i < mini_batch_size; ++i){
                        average += bias_gradients[i][layer][perceptron];
                    }
                    average /= mini_batch_size;
                    neural_network->layers.layers[layer]->perceptrons[perceptron]->bias -= neural_network->learning_rate * average;
            }
        }

        

        //puts("got here 2");
        test_neural_network(neural_network, validation_inputs, validation_expected, validation_size);
        
    }
}

void test_neural_network(NeuralNetwork *neural_network, double **testing_inputs, double **expected, size_t size){
    double accuracy = 0;
    double loss = 0;
    for(size_t i = 0; i < size; ++i){
        loss += get_loss(forward_propagate(neural_network, testing_inputs[i]), expected[i], neural_network->output_size);
    }
    loss = loss / size;
    printf("Average Loss: %f\n", loss);


    for(size_t i = 0; i < size; ++i){ //assuming there is one output where value >= 0.5 is one classification while the other classification is < 0.5.
        double *results = forward_propagate(neural_network, testing_inputs[i]);
        //printf("Output: %f\n", testing_inputs[i][1]);
        //if(expected[i][0] == 0.0){
        //    puts("Reached 0");
        //}
        if(results[0] >= 0.5 && expected[i][0] == 1.0){
            accuracy += 1;
        } else if(results[0] < 0.5 && expected[i][0] == 0.0){
            accuracy += 1;
        }
    }
    accuracy = (accuracy / size) * 100;
    printf("Accuracy: %f\n", accuracy); //Accuracy of classification (if it is a classification model)
}

void print_model_architecture(NeuralNetwork *neural_network){
    for(size_t i = 0; i < neural_network->layers.length; ++i){
        print_layer(neural_network->layers.layers[i]);
    }
}
double **make_predictions(NeuralNetwork *neural_network, double **inputs, size_t size){
    double **results = (double **) calloc(size, sizeof(double *));
    for(size_t i = 0; i < size; ++i){
        results[i] = forward_propagate(neural_network, inputs[i]);
    }
    return results;
}

void destroy_neural_network(NeuralNetwork *neural_network){
    destroy_al(&(neural_network->layers));
    free(neural_network);
}