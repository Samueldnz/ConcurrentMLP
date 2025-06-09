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

#define ROWS 2000
#define COLS 6
#define INPUT_SIZE 5
#define HIDDEN_SIZE 100
#define OUTPUT_SIZE 1
#define EPOCHS 1000
#define LEARNING_RATE 0.1f
#define SAMPLES 2000

void init_neuron_weights(Neuron* neuron, int num_inputs, float min, float max);
void update_neuron_weights(Neuron* neuron, float* gradientes, float learning_rate);
void free_neuron(Neuron* neuron);

void sigmoid(Neuron* neuron);
float derivative_sigmoid(float s);

void ReLU(Neuron* neuron);
float derivative_ReLU(float z);

void Leaky_ReLU(Neuron* neuron);
float derivative_Leaky_ReLU(float z);

float forward_pass(float input[INPUT_SIZE], Neuron hidden[HIDDEN_SIZE], Neuron *output);
void backpropagation(float input[INPUT_SIZE], float target, Neuron hidden[HIDDEN_SIZE], Neuron *output);

#endif