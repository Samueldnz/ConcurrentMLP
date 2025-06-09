#include <stdlib.h>
#include <stdio.h>


typedef struct neuron Neuron;
struct neuron 
{
    // numero de entradas
    int N;
    
    // bias
    float d_b;
    float b;

    // vetor de pesos
    float* d_w;
    float* w;

    // vetor de entradas
    float* x;

    // saida inativada
    float z;

    // saida ativada
    float d_s;
    float s;
};

typedef struct layer Layer;
struct layer 
{
    // numero de neuronios
    int n;

    // activation function
    void (*activation)(Neuron*);
    float (*activation_derivative)(float);

    // vetor de neuronios
    Neuron* neurons;
};

void init_layer (Layer* layer, int N, int n, void (*activation)(Neuron*), float (*activation_derivative)(float));
void free_layer (Layer* layer);

void set_input_layer (Layer* i, float* input);
void forward_pass (Layer* layer, Layer* input);
void update_neuron_weights(Neuron* neuron, float learning_rate);
void model_metrics(float** data, int samples, Layer* layers[], int num_layers, double total);

void h_backward_pass (Layer* h, Layer* n, float eta);
void o_backward_pass (Layer* o, float eta, float* target, float* total_error);

void ReLU (Neuron* neuron);
float derivative_ReLU (float s);

void sigmoid (Neuron* neuron);
float derivative_sigmoid (float s);

void Leaky_ReLU (Neuron* neuron);
float derivative_Leaky_ReLU (float s);