#include "activation_function.h"
#include <math.h>


static float calc_z (Neuron* neuron) 
{
    float soma = neuron->bias;
    for (int i = 0; i < neuron->num_entradas; i++)
    {
        soma += neuron->entradas[i] * neuron->pesos[i];
    }
    neuron->z = soma;
    return soma;
}

float ReLU (Neuron* neuron) 
{
    float z = calc_z(neuron);
    neuron->saida = z > 0 ? z : 0.0f;
    return neuron->saida;
}

float sigmoid (Neuron* neuron) 
{
    float z = calc_z(neuron);
    neuron->saida = 1.0f / (1.0f + expf(-z));
    return neuron->saida;
}

float Leaky_ReLU (Neuron* neuron) 
{
    float z = calc_z(neuron);
    neuron->saida = z > 0 ? z : 0.01f * z;
    return neuron->saida;
}