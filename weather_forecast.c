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
#define LEARNING_RATE 0.05f


int main (int argc, char* argv[]) 
{
    //
    srand((unsigned int)time(NULL));

    // 
    float** data = read_csv("weather_train.csv");

    //
    Layer i;
    Layer h1;
    Layer o;

    init_layer(&i, INPUT_SIZE, 0, NULL, NULL);
    init_layer(&h1, HIDDEN_SIZE, INPUT_SIZE, Leaky_ReLU, derivative_Leaky_ReLU);
    init_layer(&o, OUTPUT_SIZE, HIDDEN_SIZE, sigmoid, derivative_sigmoid);

    for (int epoch = 0; epoch < EPOCHS; epoch++) 
    {
        float total_error = 0.0f;

        for (int s = 0; s < SAMPLES; s++) 
        {
            set_input_layer(&i, data[s]);

            forward_layer(&h1, &i);

            forward_layer(&o, &h1);

            //
            float d_z[o.N];
            for (int i = 0; i < o.N; i++) 
            {
                float error = data[s][INPUT_SIZE + i] - o.neurons[i].s;
                d_z[i] = -1 * error * derivative_sigmoid(o.neurons[i].s);
                total_error += (error * error) * 0.5f;
            }

            for (int n = 0; n < o.N; n++) 
            {
                float grad_output[HIDDEN_SIZE + 1];
                
                // Gradientes
                for (int j = 0; j < HIDDEN_SIZE; j++) 
                {
                    grad_output[j] = d_z[n] * o.neurons[n].x[j];
                }
                grad_output[HIDDEN_SIZE] = d_z[n];

                // Atualiza os pesos do neurônio de saída n
                update_neuron_weights(&o.neurons[n], grad_output, LEARNING_RATE);
            }

            //
            for (int i = 0; i < HIDDEN_SIZE; i++) 
            {
                float sum = 0.0f;
                for (int j = 0; j < OUTPUT_SIZE; j++) 
                {
                    sum += d_z[j] * o.neurons[j].w[i];
                }

                float d_hidden = sum * derivative_Leaky_ReLU(h1.neurons[i].s);

                float grad_hidden[INPUT_SIZE + 1];

                // Gradientes
                for (int j = 0; j < INPUT_SIZE; j++) 
                {
                    grad_hidden[j] = d_hidden * h1.neurons[i].x[j];
                }
                grad_hidden[INPUT_SIZE] = d_hidden;

                update_neuron_weights(&h1.neurons[i], grad_hidden, LEARNING_RATE);
            }
        }
        if (epoch % 100 == 0) 
        {
            printf("Epoch %d - MSE: %.9f\n", epoch, total_error / SAMPLES);
        }
    }

    // liberando memoria

    return 0;
}