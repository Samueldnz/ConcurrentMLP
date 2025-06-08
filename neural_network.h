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

    // unactivated output
    float z;

    // activated output
    float s;
};

typedef struct layer Layer;
struct layer
{
    // number of neurons
    int N;

    // number of inputs
    int input_size;
    void (*activation)(Neuron*);
    float (*activation_derivative)(float);

    // neurons vector
    Neuron* neurons;
};

void init_layer(Layer* layer, int N, int input_size, void (*activation)(Neuron*), float (*activation_derivative)(float));
void set_input_layer(Layer* layer, float* input);
void backprop_layer (Layer* current_layer, float* grad_next_layer, Layer* next_layer, float learning_rate, float* grad_current_layer);
void forward_layer (Layer* layer, Layer* input);

void ReLU (Neuron* neuron);
float derivative_ReLU (float s);

void sigmoid (Neuron* neuron);
float derivative_sigmoid (float s);

void Leaky_ReLU (Neuron* neuron);
float derivative_Leaky_ReLU (float s);

void free_neuron (Neuron* neuron);
void init_neuron_weights (Neuron* neuron, int num_inputs, float min, float max);
void update_neuron_weights (Neuron* neuron, float* gradientes, float learning_rate);
