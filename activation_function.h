#include <stdlib.h>
#include <stdio.h>

typedef struct neuron Neuron;
struct neuron {
    int num_entradas;
    
    float bias;        // bias
    float* pesos;      // vetor de pesos: w1, w2, ..., wn
    float* entradas;   // vetor de entradas: x1, x2, ..., xn
    
    float z;           // saida do neuronio sem ativacao
    float saida;       // saída do neurônio com ativacao
};

// 
float ReLU (Neuron* neuron);

//
float sigmoid (Neuron* neuron);

//
float Leaky_ReLU (Neuron* neuron);
