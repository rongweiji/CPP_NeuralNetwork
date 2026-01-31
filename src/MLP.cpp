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



// MultiLayerPerceptron constructor, intialize values: the output of the all the neurons, and intialize network: the perceptrons in each layer
MultiLayerPerceptron::MultiLayerPerceptron(const std::vector<size_t>& layers, double bias, double learning_rate){
    this->layers=layers;// {2,2,1} means the count of neurons in each layer
    this->bias=bias;
    this->learning_rate=learning_rate;

    // initialize network
    for(size_t i=0;i<layers.size();i++){
        // for each layer
        values.push_back(std::vector<double>(layers[i],0.0));// layers[i] count of neurons in this layer, 0.0 initial value for each neuron
        deltas.push_back(std::vector<double>(layers[i],0.0));// layers[i] count of neurons in this layer, 0.0 initial delta for each neuron
        network.push_back(std::vector<Perceptron>()); // newwork add a new layer(the list of neurons in layer[i]) 
        if (i>0){ //newwork[0] is input layer,
            for(size_t j=0;j<layers[i];j++){  // for every neuron in this layer
                network[i].push_back(Perceptron(layers[i-1],bias)); // for current layer i, each neuron has layers[i-1] inputs
                // each neruon is fij(x)=sigmoid(w1*x1+w2*x2+...+wn*xn+bias*1), so perception size is based previous layer neuron count so layers[i-1]
            }
        }
    }
}


// backpropagation function to train the MLP, return the total error
double MultiLayerPerceptron::bp(std::vector<double> x, std::vector<double>y){
    // step1: feed a sample to network
    std::vector<double> output=run(x); // based on current weights, run the network to get output

    // step2: caculate MSE 
    double MSE=0.0;
    std::vector<double> error;
    for(size_t i=0;i<y.size();i++){
        error.push_back(y[i]-output[i]);
        MSE+=error[i]*error[i];
    }
    MSE/=layers.back(); // layer.back() is output layer neuron count summse/n
    // mse=1/n*sum((target-output)^2)

    // step3: caculate output layer error terms 
    for(size_t i=0;i<output.size();i++){
        deltas.back()[i]=output[i]*(1.0-output[i])*error[i]; // derivative of sigmoid function f'(x)=f(x)*(1-f(x))
    }
    // based on sigmoid activate function 1/(1+exp(-x)), the derivative f'(x)=f(x)*(1-f(x))
    

    // step4: calculate the error term of each unit on each layer 
    for(size_t i=network.size()-2;i>0;i--){ //itereate each layer
        for(size_t j=0;j<network[i].size();j++){ // iterate each neuron in this layer
            double fwd_error=0.0;
            for(size_t k=0;k<layers[i+1];k++){
                fwd_error+=deltas[i+1][k]*network[i+1][k].weights[j]; // sum of delta of next layer * weight between current neuron and next layer neuron
            }
            deltas[i][j]=values[i][j]*(1.0-values[i][j])*fwd_error; // delta = f'(net)*error , f'(x)=f(x)*(1-f(x)) for sigmoid function
        }
    }

    // step5 &6 , calculate detal and update weights
    for(size_t i=1;i<network.size();i++){
        for(size_t j=0;j<layers[i];j++){
            for (size_t k=0; k<layers[i-1]+1;k++){
                double delta;
                if(k==layers[i-1]){ // two options: 1. use bias as input 2. separate bias weight update
                    delta=learning_rate*deltas[i][j]*bias; // 
                }else{
                    delta=learning_rate*deltas[i][j]*values[i-1][k];
                }
                network[i][j].weights[k]+=delta;
            }
        }

    }

    return MSE;

}



// run function is for forward propagation , update all the values: the output of each neuron in the network
std::vector<double>  MultiLayerPerceptron::run(std::vector<double> x){
    values[0]=x; // set input layer values
    for(size_t i=1;i<network.size();i++){
        // for each layer
        for(size_t j=0;j<network[i].size();j++){
            // for each neuron in this layer, using output from previous layer as input
            // network[i][j] is the j-th neuron in layer which is the perceptron object will have neruon run function 
            values[i][j]=network[i][j].run(values[i-1]); // run the perceptron with input from previous layer， update all neuron output values in this layer
        }
    }


    return values.back(); // return output layer values
}