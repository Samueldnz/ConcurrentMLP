#include "neural_network.h"
#include <stdlib.h>
#include <math.h>

void free_neuron(Neuron* neuron) {
    free(neuron->w);
    free(neuron->x);
}

void init_neuron_weights(Neuron* neuron, int num_inputs, float min, float max) {
    neuron->N = num_inputs;
    neuron->w = (float*) malloc(sizeof(float) * num_inputs);
    neuron->x = (float*) malloc(sizeof(float) * num_inputs);
    for (int i = 0; i < num_inputs; i++) {
        neuron->w[i] = ((float) rand() / RAND_MAX) * (max - min) + min;
    }
    neuron->b = 0;
}

void update_neuron_weights(Neuron* neuron, float* gradientes, float learning_rate) {
    for (int i = 0; i < neuron->N; i++) {
        neuron->w[i] -= learning_rate * gradientes[i];
    }
    neuron->b -= learning_rate * gradientes[neuron->N];
}

static void calc_z(Neuron* neuron) {
    float soma = neuron->b;
    for (int i = 0; i < neuron->N; i++) {
        soma += neuron->x[i] * neuron->w[i];
    }
    neuron->z = soma;
}

void sigmoid(Neuron* neuron) {
    calc_z(neuron);
    neuron->s = 1.0f / (1.0f + expf(-neuron->z));
}

float derivative_sigmoid(float s) {
    return s * (1.0f - s);
}

void ReLU(Neuron* neuron) {
    calc_z(neuron);
    neuron->s = neuron->z > 0 ? neuron->z : 0.0f;
}

float derivative_ReLU(float z) {
    return z > 0 ? 1.0f : 0.0f;
}

void Leaky_ReLU(Neuron* neuron) {
    calc_z(neuron);
    neuron->s = neuron->z > 0 ? neuron->z : 0.01f * neuron->z;
}

float derivative_Leaky_ReLU(float z) {
    return z > 0 ? 1.0f : 0.01f;
}