#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

typedef struct {
    int N;         // Número de entradas
    float *w;      // Pesos
    float *x;      // Entradas
    float b;       // Bias
    float z;       // Soma ponderada
    float s;       // Saída ativada
} Neuron;

void init_neuron_weights(Neuron* neuron, int num_inputs, float min, float max);
void update_neuron_weights(Neuron* neuron, float* gradientes, float learning_rate);
void free_neuron(Neuron* neuron);

void sigmoid(Neuron* neuron);
float derivative_sigmoid(float s);

void ReLU(Neuron* neuron);
float derivative_ReLU(float z);

void Leaky_ReLU(Neuron* neuron);
float derivative_Leaky_ReLU(float z);

#endif