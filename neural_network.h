#include <stdlib.h>
#include <stdio.h>


typedef struct neuron Neuron;
struct neuron 
{
    int N;
    
    // bias
    float b;

    // weights
    float* w;

    // inputs
    float* x;

    // inactivated output
    float z;

    // activated output
    float s;
};

void ReLU (Neuron* neuron);
float derivative_ReLU (float z);

void sigmoid (Neuron* neuron);
float derivative_sigmoid (float z);

void Leaky_ReLU (Neuron* neuron);
float derivative_Leaky_ReLU (float z);

void free_neuron (Neuron* neuron);
void init_neuron_weights (Neuron* neuron, int num_inputs, float min, float max);
void update_neuron_weights (Neuron* neuron, float* gradientes, float learning_rate);
