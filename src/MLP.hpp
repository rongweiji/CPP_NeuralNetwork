#pragma once
#include <vector>
#include <iostream>
#include <random>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <time.h>

class Perceptron {
public:
    std::vector<double> weights;
    double bias;
    Perceptron(size_t input,double bias=1.0);
    double run(std::vector<double> x);
    void set_weights(std::vector<double> w_init);
    double sigmoid(double x);
};


class MultiLayerPerceptron {
public:
    MultiLayerPerceptron(const std::vector<size_t>& layers, double bias=1.0, double learning_rate=0.1);

    std::vector<size_t> layers; // a vector holding the number of neurons in each layer,{2,2,1} for 2 input neurons, 2 hidden neurons, and 1 output neuron
    double bias;
    double learning_rate;
    std::vector<std::vector<Perceptron>> network; // 2D vector to hold layers of perceptrons
    std::vector<std::vector<double>> values;// output of layer.
    std::vector<std::vector<double>> deltas;// delta values for backpropagation.error terms 

    void set_weights(std::vector<std::vector<std::vector<double>>> w_init);
    void print_weights();
    std::vector<double> run(std::vector<double> x);
};

