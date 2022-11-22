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
double *forward_propagate(double *input);
double ***back_propagate_weights(NeuralNetwork *neural_network, double *prediction, double *expected);
double **back_propagate_biases(NeuralNetwork *neural_network, double *prediction, double *expected);
void train_neural_network(NeuralNetwork *neural_network, double **training_inputs, double **training_expected, double **validation_inputs, double **validation_expected, size_t training_size, size_t validation_size, size_t iterations, size_t mini_batch_size);
void test_neural_network(double **testing_inputs, double **expected, size_t size);
void print_model_architecture(NeuralNetwork *neural_network);
double **make_predictions(double **inputs, size_t size);
void destroy_neural_network(NeuralNetwork *neural_network);


NeuralNetwork *create_neural_network(ArrayList layers, size_t output_size, double learning_rate){
    NeuralNetwork *new_neural_network = malloc(sizeof(NeuralNetwork));
    new_neural_network->layers = layers;
    new_neural_network->learning_rate = learning_rate;
    new_neural_network->output_size = output_size;
    return new_neural_network;
}
double get_loss(double *prediction, double *expected, size_t size){
    double loss = 0;
    for(size_t i = 0; i < size; ++i)
        loss += (prediction[i] - expected[i]) * (prediction[i] - expected[i]);
    
    return loss / size;
}
double *forward_propagate(double *input){
    double *current_values = input;
    for(size_t i = 0; i < layers.length; ++i){
        current_values = get_layer_output(&layers.layers[i], current_values);
    }
    return current_values;
}

double ***back_propagate_weights(NeuralNetwork *neural_network, double *prediction, double *expected){
    double ***weights_gradient = malloc(sizeof(double**) * neural_network->layers.length);
    for(size_t layer = 0; layer < neural_network->layers.length; ++layer){
        double **layer_gradient = malloc(sizeof(double*) * neural_network->layers.layers[layer].num_perceptrons);
        for(size_t perceptron = 0; perceptron < neural_network->layers.layers[layer].num_perceptrons; ++perceptron){
            double *perceptron_gradient = malloc(sizeof(double) * neural_network->layers.layers[layer].perceptrons[perceptron].num_weights);
            for(size_t current_weight = 0; current_weight < neural_network->layers.layers[layer].perceptrons[perceptron].num_weights; ++current_weight){
                double current_loss = get_loss(prediction, expected, neural_network->output_size);
                neural_network->layers.layers[layer].perceptrons[perceptron]->weights[current_weight] += 0.05;
                double new_loss = get_loss(prediction, expected, neural_network->output_size);
                double partial_derivative = (new_loss - current_loss) / 0.05;
                neural_network->layers.layers[layer].perceptrons[perceptron]->weights[current_weight] -= 0.05;
                perceptron_gradient[current_weight] = partial_derivative;
            }
            layer_gradient[perceptron] = perceptron_gradient;
        }
        weights_gradient[layer] = layer_gradient;
    }

    return weights_gradient;
} //Gets weights gradient for each layer for a single training example

double **back_propagate_biases(NeuralNetwork *neural_network, double *prediction, double *expected){
    double **bias_gradient = malloc(sizeof(double*) * neural_network->layers.length);
    for(size_t layer = 0; layer < neural_network->layers.length; ++layer){
        double *layer_gradient = malloc(sizeof(double) * neural_network->layers.layers[layer].num_perceptrons);
        for(size_t perceptron = 0; perceptron < neural_network->layers.layers[layer].num_perceptrons; ++perceptron){
            double bias_gradient = 0;
            double current_loss = get_loss(prediction, expected, neural_network->output_size);
            neural_network->layers.layers[layer].perceptrons[perceptron]->bias += 0.05;
            double new_loss = get_loss(prediction, expected, neural_network->output_size);
            double partial_derivative = (new_loss - current_loss) / 0.05;
            neural_network->layers.layers[layer].perceptrons[perceptron]->bias -= 0.05;
            layer_gradient[perceptron] = partial_derivative;
        }
        bias_gradient[layer] = layer_gradient;
    }

    return bias_gradient;
} //Gets bias gradient for each layer for a single training example

void train_neural_network(NeuralNetwork *neural_network, double **training_inputs, double **training_expected, double **validation_inputs, double **validation_expected, size_t training_size, size_t validation_size, size_t iterations, size_t mini_batch_size){
    double ****weights_gradients = malloc((sizeof(double**) * neural_network->layers.length) * mini_batch_size);
    double ***bias_gradients = malloc((sizeof(double*) * neural_network->layers.length) * mini_batch_size);
    
    //do this for the amount of iterations there are
    for(size_t iteration = 0; iteration < iterations; ++iteration){

        for(size_t i = 0; i < mini_batch_size; ++i){
            size_t row_to_use = rand() % training_size;
            weights_gradients[i] = back_propagate_weights(neural_network, forward_propagate(training_inputs[row_to_use]), training_expected[row_to_use]);
            bias_gradients[i] = back_propagate_biases(neural_network, forward_propagate(training_inputs[row_to_use]), training_expected[row_to_use]);
        }

        for(size_t layer = 0; layer < neural_network->layers.length; ++layer){
            for(size_t perceptron = 0; perceptron < neural_network->layers.layers[layer].num_perceptrons; ++perceptron){
                for(size_t current_weight = 0; current_weight < neural_network->layers.layers[layer].perceptrons[perceptron].num_weights; ++current_weight){
                    double average = 0;
                    for(size_t i = 0; i < mini_batch_size; ++i){
                        average += weights_gradients[i][layer][perceptron][current_weight];
                    }
                    average /= mini_batch_size;
                    neural_network->layers.layers[layer].perceptrons[perceptron]->weights[current_weight] -= learning_rate * average;

                }
            }
        }

        for(size_t layer = 0; layer < neural_network->layers.length; ++layer){
            for(size_t perceptron = 0; perceptron < neural_network->layers.layers[layer].num_perceptrons; ++perceptron){
                    double average = 0;
                    for(size_t i = 0; i < mini_batch_size; ++i){
                        average += bias_gradients[i][layer][perceptron];
                    }
                    average /= mini_batch_size;
                    neural_network->layers.layers[layer].perceptrons[perceptron]->bias -= learning_rate * average;
            }
        }


        test_neural_network(validation_inputs, validation_expected, validation_size);
    }
} //prints out accuracy after each evaluation_frequency iterations

void test_neural_network(double **testing_inputs, double **expected, size_t size){
    double accuracy = 0;
    double loss = 0;
    for(size_t i = 0; i < size; ++i){
        loss += get_loss(forward_propagate(testing_inputs[i]), expected[i], size);
    }
    loss = loss / size;
    printf("Average Loss: %d\n", loss);

    for(size_t i = 0; i < size; ++i){ //assuming there is one output where value >= 0.5 is one classification while the other classification is < 0.5.
        results = forward_propagate(testing_inputs[i]);
        if(results[0] >= 0.5 && expected[i][0] == 1){
            accuracy += 1;
        } else if(results[0] < 0.5 && expected[i][0] == 0){
            accuracy += 1;
        }
    }
    accuracy = (accuracy / size) * 100;
    printf("Accuracy: %d\n", accuracy);
}

void print_model_architecture(NeuralNetwork *neural_network){
    for(size_t i = 0; i < neural_network->layers.length; ++i){
        print_layer(&neural_network->layers->layers[i]);
    }
}
double **make_predictions(double **inputs, size_t size){
    double **results = malloc(sizeof(double *) * size);
    for(size_t i = 0; i < size; ++i){
        results[i] = forward_propagate(inputs[i]);
    }
    return results;
}

void destroy_neural_network(NeuralNetwork *neural_network){
    destroy_al(&(neural_network->layers));
    free(neural_network);
}