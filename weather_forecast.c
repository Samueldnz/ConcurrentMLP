#include "neural_network.h"
#include "matrix.h"

#include <stdlib.h>
#include <stdio.h>
#include <time.h>


#define INPUT_SIZE 5
#define HIDDEN_SIZE 8
#define OUTPUT_SIZE 1

#define EPOCHS 1000
#define SAMPLES 2000
#define LEARNING_RATE 0.1f


int main (int argc, char* argv[]) 
{
    //
    srand((unsigned int)time(NULL));

    // 
    float** data = read_csv("weather_train.csv");

    // 
    Neuron hidden_layer[HIDDEN_SIZE];
    for (int i = 0; i < HIDDEN_SIZE; i++) 
    {
        init_neuron_weights(&hidden_layer[i], INPUT_SIZE, -1.0f, 1.0f);
    }

    //
    Neuron output_neuron;
    init_neuron_weights(&output_neuron, HIDDEN_SIZE, -1.0f, 1.0f);

    for (int epoch = 0; epoch < EPOCHS; epoch++) 
    {
        float total_error = 0.0f;

        for (int s = 0; s < SAMPLES; s++) 
        {
            //
            for (int i = 0; i < HIDDEN_SIZE; i++) 
            {
                for (int j = 0; j < INPUT_SIZE; j++) 
                {
                    hidden_layer[i].x[j] = data[s][j];
                }
            }
            for (int i = 0; i < HIDDEN_SIZE; i++) sigmoid(&hidden_layer[i]);

            //
            for (int i = 0; i < HIDDEN_SIZE; i++) 
            {
                output_neuron.x[i] = hidden_layer[i].s;
            }
            sigmoid(&output_neuron);
            
            // MSE
            float error = data[s][INPUT_SIZE] - output_neuron.s;
            total_error += error * error;

            // 
            float d_z = error * derivative_sigmoid(output_neuron.s);
            float grad_output[HIDDEN_SIZE + 1];
            for (int i = 0; i < HIDDEN_SIZE; i++) 
            {
                grad_output[i] = d_z * output_neuron.x[i];
            }
            grad_output[HIDDEN_SIZE] = d_z;
            update_neuron_weights(&output_neuron, grad_output, LEARNING_RATE);

            //
            for (int i = 0; i < HIDDEN_SIZE; i++) 
            {
                float d_hidden = d_z * output_neuron.w[i] * derivative_sigmoid(hidden_layer[i].s);

                float grad_hidden[INPUT_SIZE + 1];
                for (int j = 0; j < INPUT_SIZE; j++) 
                {
                    grad_hidden[j] = d_hidden * hidden_layer[i].x[j];
                }
                grad_hidden[INPUT_SIZE] = d_hidden;

                update_neuron_weights(&hidden_layer[i], grad_hidden, LEARNING_RATE);
            }
        }
        if (epoch % 100 == 0) 
        {
            printf("Epoch %d - MSE: %.9f\n", epoch, total_error / SAMPLES);
        }
    }

    //
    for (int i = 0; i < HIDDEN_SIZE; i++) free_neuron(&hidden_layer[i]);
    free_neuron(&output_neuron);
    free_matrix(data, 2000);

    return 0;
}