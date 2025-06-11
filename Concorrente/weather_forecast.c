#include "neural_network.h"
#include "matrix.h"
#include "timer.h"

#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>


#define INPUT_SIZE 5
#define HIDDEN_SIZE_1 1024
#define OUTPUT_SIZE 1

#define EPOCHS 200
#define TEST_SAMPLES 500
#define TRAIN_SAMPLES 2000
#define LEARNING_RATE 0.01f


int main (int argc, char* argv[]) 
{   
    //
    int dummy = 0;
    srand((unsigned int)(time(NULL) ^ (uintptr_t)&dummy));

    // 
    float** data = read_csv("weather_train.csv");
    double ini, fim, ini_2, fim_2;

    //
    Layer i;
    Layer h1;
    Layer o;

    init_layer(&i, INPUT_SIZE, 0, NULL, NULL);
    init_layer(&h1, HIDDEN_SIZE_1, INPUT_SIZE, Leaky_ReLU, derivative_Leaky_ReLU);
    init_layer(&o, OUTPUT_SIZE, HIDDEN_SIZE_1, sigmoid, derivative_sigmoid);

    GET_TIME(ini);
    for (int epoch = 0; epoch < EPOCHS; epoch++) 
    {
        float total_error = 0.0f;

        for (int s = 0; s < TRAIN_SAMPLES; s++) 
        {
            // resetando as entradas
            set_input_layer(&i, data[s]);

            // FORWARD_PASS FUNCTIONS
            forward_pass(&h1, &i);
            forward_pass(&o, &h1);

            // BACKPROPAGATION FUNCTIONS
            o_backward_pass(&o, LEARNING_RATE, &data[s][INPUT_SIZE], &total_error);
            h_backward_pass(&h1, &o, LEARNING_RATE);
        }
    }
    GET_TIME(fim);

    // calculando as metricas do modelo
    Layer* layers[] = { &i, &h1, &o };

    GET_TIME(ini_2);
    model_metrics(read_csv("weather_test.csv"), TEST_SAMPLES, layers, 3, fim - ini);
    GET_TIME(fim_2);
    printf("TS: %f\n\n", fim_2 - ini_2); 

    GET_TIME(ini_2);
    model_metrics_conc(read_csv("weather_test.csv"), TEST_SAMPLES, layers, 3, fim - ini);
    GET_TIME(fim_2);
    printf("TC: %f\n", fim_2 - ini_2); 

    // desalocando memoria das camadas
    free_layer(&i);
    free_layer(&h1);
    free_layer(&o);

    // desalocando memoria do csv de treino
    for (int s = 0; s < TRAIN_SAMPLES; s++) free(data[s]);
    free(data);

    return 0;
}