#include "MLP.hpp"  

int main(){
    srand(time(NULL)); // seed for random number generation
    rand(); // warm up rand()
    std::cout<<"\n\n--------Logic Gate Example --------\n\n";
    Perceptron p(2); // create a perceptron with 2 inputs

    // AND: 10,10,-15
    // OR:15,15,-10
    p.set_weights({15,15,-10}); // set weights for OR gate



    std::cout<<"Gate: "<<std::endl;
    std::cout<<"0 OR 0 = "<<p.run({0,0})<<std::endl;
    std::cout<<"0 OR 1 = "<<p.run({0,1})<<std::endl;
    std::cout<<"1 OR 0 = "<<p.run({1,0})<<std::endl;
    std::cout<<"1 OR 1 = "<<p.run({1,1})<<std::endl;

    std::cout<<"\n\n--------MLP Example --------\n\n";
    MultiLayerPerceptron mlp({2,2,1}); // create a MLP with 2 input neurons, 2 hidden neurons, and 1 output neuron
    mlp.set_weights({
        {   // weights between input layer and hidden layer
            {-10,-10,15}, // AND gate weights for hidden neuron 1
            {15,15,-10}  // OR gate weights for hidden neuron 2
        },
        {   // weights between hidden layer and output layer
            {10,10,-15} // 
        }
    });

    std::cout<<"Hardcoded weights"<<std::endl;
    mlp.print_weights();

    std::cout<<"MLP Output for XOR Gate: "<<std::endl;
    std::cout<<"0 XOR 0 = "<<mlp.run({0,0})[0]<<std::endl;
    std::cout<<"0 XOR 1 = "<<mlp.run({0,1})[0]<<std::endl;
    std::cout<<"1 XOR 0 = "<<mlp.run({1,0})[0]<<std::endl;
    std::cout<<"1 XOR 1 = "<<mlp.run({1,1})[0]<<std::endl;

// Training MLP for XOR gate using backpropagation
    std::cout<<"\n-------BP Example ------------------"<<std::endl;
    mlp=MultiLayerPerceptron({2,2,1});
    double MSE;
    for(int i=0;i<3000;i++){
        MSE=0.0;
        MSE+=mlp.bp({0,0},{0});
        MSE+=mlp.bp({0,1},{1});
        MSE+=mlp.bp({1,0},{1});
        MSE+=mlp.bp({1,1},{0});
        MSE/=4.0;
        if(i%100==0){
            std::cout<<" Epoch "<<i<<", MSE: "<<MSE<<std::endl;
        }
    }
    std::cout<<"Trained MLP Weights:"<<std::endl;
    mlp.print_weights();
    std::cout<<"MLP Output for XOR Gate after training: "<<std::endl;
    std::cout<<"0 XOR 0 = "<<mlp.run({0,0})[0]<<std::endl;
    std::cout<<"0 XOR 1 = "<<mlp.run({0,1})[0]<<std::endl;
    std::cout<<"1 XOR 0 = "<<mlp.run({1,0})[0]<<std::endl;
    std::cout<<"1 XOR 1 = "<<mlp.run({1,1})[0]<<std::endl;
    


    return 0;

// train for simple SDR 7-segment to know the number 
// 'a' 'b' 'c' 'd' 'e' 'f' 'g' means the display segments for number 0-9
// 0: 1 1 1 1 1 1 0 
// 1: 0 1 1 0 0 0 0 


// 3 parttern to train the 7segemet SDR 
// 7 to 1  put the value 7 dig get the number resuylt of the vale 
// 7 to 10  7digs get the 0-9 number result  
// 7 to 7 7digs get the 7digs result


}