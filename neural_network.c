#include "neural_network.h"
#include <math.h>


void free_neuron (Neuron* neuron) 
{
    free(neuron->w);
    free(neuron->x);
}

void init_neuron_weights (Neuron* neuron, int num_inputs, float min, float max) 
{
    neuron->N = num_inputs;
    neuron->w = (float*) malloc(sizeof(float) * num_inputs);
    neuron->x = (float*) malloc(sizeof(float) * num_inputs);

    for (int i = 0; i < num_inputs; i++) 
    {
        neuron->w[i] = ((float) rand() / RAND_MAX) * (max - min) + min;
    }
    neuron->b = 0;
}

void init_layer (Layer* layer, int N, int input_size, void (*activation)(Neuron*), float (*activation_derivative)(float)) 
{
    layer->N = N;
    layer->input_size = input_size;
    layer->activation = activation;
    layer->activation_derivative = activation_derivative;

    layer->neurons = (Neuron*) malloc(sizeof(Neuron) * N);
    for (int i = 0; i < N; i++) 
    {
        init_neuron_weights(&layer->neurons[i], input_size, -1.0f, 1.0f);
    }
}

void set_input_layer(Layer* layer, float* input) 
{
    for (int i = 0; i < layer->N; i++) 
    {
        layer->neurons[i].s = input[i];
    }
}

void backprop_layer (Layer* current_layer, float* grad_next_layer, Layer* next_layer, float learning_rate, float* grad_current_layer) 
{
    for (int i = 0; i < current_layer->N; i++) 
    {
        float sum = 0.0f;
        if (next_layer != NULL) 
        {
            for (int k = 0; k < next_layer->N; k++) 
            {
                sum += grad_next_layer[k] * next_layer->neurons[k].w[i];
            }
        } else {
            sum = grad_next_layer[i];
        }

        // derivada da ativação para o neurônio atual
        float delta = sum * current_layer->activation_derivative(current_layer->neurons[i].s);

        // atualizar pesos da camada atual
        float grad_weights[current_layer->input_size + 1]; // +1 para bias
        for (int j = 0; j < current_layer->input_size; j++) {
            grad_weights[j] = delta * current_layer->neurons[i].x[j];
        }
        grad_weights[current_layer->input_size] = delta; // bias

        update_neuron_weights(&current_layer->neurons[i], grad_weights, learning_rate);

        // armazenar gradientes para camada anterior
        for (int j = 0; j < current_layer->input_size; j++) {
            grad_current_layer[j] += delta * current_layer->neurons[i].w[j];
        }
    }
}

void forward_pass (Layer* layer, Layer* input) 
{
    for (int i = 0; i < layer->N; i++) 
    {
        for (int j = 0; j < layer->input_size; j++) 
        {
            layer->neurons[i].x[j] = input->neurons[j].s;
        }
        layer->activation(&layer->neurons[i]);
    }
}

void free_layer (Layer* layer) 
{
    for (int i = 0; i < layer->N; i++) 
    {
        free_neuron(&layer->neurons[i]);
    }
    free(layer->neurons);
}

void update_neuron_weights (Neuron* neuron, float* gradientes, float learning_rate) 
{
    for (int i = 0; i < neuron->N; i++) 
    {
        neuron->w[i] -= learning_rate * gradientes[i];
    }
    neuron->b -= learning_rate * gradientes[neuron->N];
}

static void calc_z (Neuron* neuron) 
{
    float soma = neuron->b;

    for (int i = 0; i < neuron->N; i++)
    {
        soma += neuron->x[i] * neuron->w[i];
    }

    neuron->z = soma;
}

// thread-safe
void ReLU (Neuron* neuron) 
{
    calc_z(neuron);

    neuron->s = neuron->z > 0 ? neuron->z : 0.0f;
}
float derivative_ReLU (float s) 
{
    return (s > 0) ? 1.0f : 0.0f;
}

// thread-safe
void sigmoid (Neuron* neuron) 
{
    calc_z(neuron);

    neuron->s = 1.0f / (1.0f + expf(-neuron->z));
}
float derivative_sigmoid (float s)
{
    return s * (1.0f - s);
}

// thread-safe
void Leaky_ReLU (Neuron* neuron) 
{
    calc_z(neuron);

    neuron->s = neuron->z > 0 ? neuron->z : 0.01f * neuron->z;
}
float derivative_Leaky_ReLU (float s) 
{
    return (s > 0) ? 1.0f : 0.01f;
}