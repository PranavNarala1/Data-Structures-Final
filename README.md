# Data-Structures-Final

Made by Pranav Narala and Eric Ge
 

For our project, are planning on implementing an artificial neural network (ANN) from scratch (no external libraries) in C. ANNs are built upon the concept of a perceptron, which is a node that takes in an input, multiplies a weight to it, adds a bias to it, and applies an activation function to it to get output. A combination of these nodes form layers, and layers are stacked to form networks. The ANN will be a struct that contains an ArrayList of pointers to layer structs. Each fully connected layer struct will contain an array of perceptron pointers and the number of perceptrons in that layer. Each perceptron will be a struct with weight, bias, and activation function variables. Functions will be created to define the model architecture, train the model using stochastic gradient descent with mini-batches of data that are in array format, evaluate the accuracy of the model on testing data, and use the model. To test the functionality of the ANN, we will apply it to a titanic survival prediction dataset. However, this project will focus more on the algorithms and data structures used to implement the model rather than applying it to datasets. Since arrays will be fed into the model, matrix operations functions for arrays will be defined to assist with forward and back propagation. We will apply the concepts learned in our Data Structures with C course to implement this project.

 

Referenced Information:

https://towardsdatascience.com/introduction-to-neural-networks-advantages-and-applications-96851bd1a207

https://www.analyticsvidhya.com/blog/2021/04/artificial-neural-network-its-inspiration-and-the-working-mechanism/

https://reader.elsevier.com/reader/sd/pii/B9780444636232000074?token=F2263E933FA36E7BEEF4F9AC3432341CEE78E404DE1549CB9F06E9A3A3D9AF44CE4D45F2CA5EC5FC566DD6DB0FE4B0F8&originRegion=us-east-1&originCreation=20221111045342

https://dotnettutorials.net/lesson/how-artificial-neural-network-work/

https://towardsdatascience.com/stochastic-gradient-descent-clearly-explained-53d239905d31

 

Dataset:

https://www.kaggle.com/competitions/titanic/data
