#include "neural_network.h"
#include <math.h>


static void calc_z (Neuron* neuron) 
{
    float soma = neuron->b;

    for (int i = 0; i < neuron->N; i++)
    {
        soma += neuron->x[i] * neuron->w[i];
    }

    neuron->z = soma;
}

static void init_neuron (Neuron* neuron, int N, float min, float max) 
{
    neuron->N = N;
    neuron->x = (float*) malloc(sizeof(float) * N);
    neuron->w = (float*) malloc(sizeof(float) * N);
    neuron->d_w = (float*) malloc(sizeof(float) * N);
    
    for (int i = 0; i < N; i++) 
    {
        neuron->w[i] = ((float) rand() / RAND_MAX) * (max - min) + min;
    }
    neuron->b = 0;
}
static void free_neuron (Neuron* neuron) 
{
    free(neuron->w);
    free(neuron->x);
}

void init_layer (Layer* layer, int n, int N, void (*activation)(Neuron*), float (*activation_derivative)(float)) 
{
    layer->n = n;
    layer->activation = activation;
    layer->activation_derivative = activation_derivative;
    
    layer->neurons = (Neuron*) malloc(sizeof(Neuron) * n);
    for (int i = 0; i < n; i++) 
    {
        init_neuron(&layer->neurons[i], N, -1.0f, 1.0f);
    }
}
void free_layer (Layer* layer) 
{
    if (layer->neurons) 
    {
        for (int i = 0; i < layer->n; i++) 
        {
            free_neuron(&layer->neurons[i]);
        }
        free(layer->neurons);
    }
}


void set_input_layer (Layer* i, float* input) 
{
    for (int j = 0; j < i->n; j++) 
    {
        i->neurons[j].s = input[j];
    }
}

void forward_pass (Layer* layer, Layer* input) 
{
    for (int i = 0; i < layer->n; i++) 
    {
        for (int j = 0; j < layer->neurons[i].N; j++) 
        {
            layer->neurons[i].x[j] = input->neurons[j].s;
        }
        layer->activation(&layer->neurons[i]);
    }
}

void update_neuron_weights (Neuron* neuron, float learning_rate) 
{
    for (int i = 0; i < neuron->N; i++) 
    {
        neuron->w[i] -= learning_rate * neuron->d_w[i];
    }
    neuron->b -= learning_rate * neuron->d_b;
}

static void backward_pass (Layer* layer, float eta) 
{
    for (int i = 0; i < layer->n; i++) 
    {
        for (int j = 0; j < layer->neurons[i].N; j++) 
        {
            layer->neurons[i].d_w[j] = layer->neurons[i].d_s * layer->neurons[i].x[j];
        }
        layer->neurons[i].d_b = layer->neurons[i].d_s;

        update_neuron_weights(&layer->neurons[i], eta);
    }
}
void h_backward_pass (Layer* h, Layer* n, float eta) 
{
    for (int i = 0; i < h->n; i++) 
    {
        float sum = 0.0f;
        for (int j = 0; j < n->n; j++) 
        {
            sum += n->neurons[j].d_s * n->neurons[j].w[i];
        }
        h->neurons[i].d_s = sum * h->activation_derivative(h->neurons[i].s);
    }
    backward_pass(h, eta);
}
void o_backward_pass (Layer* o, float eta, float* target, float* total_error) 
{
    for (int i = 0; i < o->n; i++) 
    {
        float error = target[i] - o->neurons[i].s;
        o->neurons[i].d_s = -1 * error * derivative_sigmoid(o->neurons[i].s);
        *total_error += (error * error) * 0.5f;
    }

    backward_pass(o, eta);
}

void model_metrics (float** data, int samples, Layer* layers[], int num_layers, double total) 
{
    int TP = 0, TN = 0, FP = 0, FN = 0;

    for (int s = 0; s < samples; s++) 
    {
        set_input_layer(layers[0], data[s]);

        for (int i = 1; i < num_layers; i++) 
        {
            forward_pass(layers[i], layers[i - 1]);
        }

        Layer* output = layers[num_layers - 1];
        float output_value = output->neurons[0].s;

        int pred = (output_value >= 0.55f) ? 1 : 0;
        int real = (int)(data[s][layers[0]->n]);

        if (pred == 1 && real == 1) TP++;
        else if (pred == 0 && real == 0) TN++;
        else if (pred == 1 && real == 0) FP++;
        else if (pred == 0 && real == 1) FN++;
    }

    float accuracy = (float)(TP + TN) / samples;
    float precision = TP + FP == 0 ? 0 : (float)TP / (TP + FP);
    float recall = TP + FN == 0 ? 0 : (float)TP / (TP + FN);
    float f1 = (precision + recall) == 0 ? 0 : 2 * (precision * recall) / (precision + recall);

    printf("Métricas do modelo:\n");
    printf("Acurácia: %.8f\n", accuracy);
    printf("Precisão: %.8f\n", precision);
    printf("Recall: %.8f\n", recall);
    printf("F1 Score: %.8f\n", f1);
    printf("Tempo de Treinamento: %.8f segundos\n", total);
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