#include "MLP.hpp"

// generate a random double number between -1.0 and 1.0
double frand(){
    return (2.0*(double)rand()/RAND_MAX)-1.0;
}

// return a new perception object with the specific number of inpu 
Perceptron::Perceptron(size_t input,double bias){
    this->bias=bias;
    weights.resize(input+1); 
    generate(weights.begin(),weights.end(),frand); // initialize weights randomly
}

// run the perceptron with input x, x is vector of double
double Perceptron::run(std::vector<double> x){
    // x=x1,x2,...,xn,
    x.push_back(bias); // add bias to the input vector
    double sum=inner_product(x.begin(),x.end(),weights.begin(),(double)0.0);
    // sum=w1*x1+w2*x2+...+wn*xn+bias*1
    // put all the sum into the sigmoid function 1/(1+exp(-sum))
    return sigmoid(sum);
}


// set_weights function to set the weights of the perceptron
void Perceptron::set_weights(std::vector<double> w_init){
    if(w_init.size()!=weights.size()){
        std::cerr<<"Error: weights size mismatch!"<<std::endl;
        return;
    }
    weights=w_init;
}

// sigmoid activation function
double Perceptron::sigmoid(double x){
    return 1.0/(1.0+exp(-x));
}





// set weight for multi-layer perceptron
void MultiLayerPerceptron::set_weights(std::vector<std::vector<std::vector<double>>> w_init){
    for(size_t i=0;i<w_init.size();i++){
        for(size_t j=0;j<w_init[i].size();j++){
            network[i+1][j].set_weights(w_init[i][j]);
        }
    }
}

void MultiLayerPerceptron::print_weights(){
    for(size_t i=1;i<network.size();i++){
        std::cout<<"Layer "<<i<<":"<<std::endl;
        for(size_t j=0;j<network[i].size();j++){
            std::cout<<" Neuron "<<j<<": ";
            for(auto w:network[i][j].weights){
                std::cout<<w<<" ";
            }
            std::cout<<std::endl;
        }
    }
}



// MultiLayerPerceptron constructor
MultiLayerPerceptron::MultiLayerPerceptron(const std::vector<size_t>& layers, double bias, double learning_rate){
    this->layers=layers;
    this->bias=bias;
    this->learning_rate=learning_rate;

    // initialize network
    for(size_t i=0;i<layers.size();i++){
        // for each layer
        values.push_back(std::vector<double>(layers[i],0.0));// initialize output values
        network.push_back(std::vector<Perceptron>()); // initialize perceptron layer
        if (i>0){ //newwork[0] is input layer,
            // for every neuron in this layer
            for(size_t j=0;j<layers[i];j++){
                network[i].push_back(Perceptron(layers[i-1],bias)); // create perceptron with number of inputs equal to number of neurons in previous layer
            }
        }
    }
}

// run function is for forward propagation
std::vector<double>  MultiLayerPerceptron::run(std::vector<double> x){
    values[0]=x; // set input layer values
    for(size_t i=1;i<network.size();i++){
        // for each layer
        for(size_t j=0;j<network[i].size();j++){
            // for each neuron in this layer
            values[i][j]=network[i][j].run(values[i-1]); // run the perceptron with input from previous layer
        }
    }


    return values.back(); // return output layer values
}