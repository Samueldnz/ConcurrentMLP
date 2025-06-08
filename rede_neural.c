#include <stdio.h>
#include <stdlib.h>
#include "activation_function.h"

int main() {
    Neuron n;
    n.num_entradas = 3;
    n.bias = 1.0;

    n.entradas = malloc(sizeof(float) * n.num_entradas);
    n.pesos = malloc(sizeof(float) * n.num_entradas);

    n.entradas[0] = 1.0;
    n.entradas[1] = 2.0;
    n.entradas[2] = -1.0;

    n.pesos[0] = 0.5;
    n.pesos[1] = -1.0;
    n.pesos[2] = 2.0;

    sigmoid(&n);
    printf("Saída (Sigmoid): %f\n", n.saida);

    ReLU(&n);
    printf("Saída (ReLU): %f\n", n.saida);

    Leaky_ReLU(&n);
    printf("Saída (Leaky ReLU): %f\n", n.saida);

    free(n.entradas);
    free(n.pesos);

    return 0;
}