#include "neural_network.h"
#include "matrix.h"
#include "timer.h"

#include <stdlib.h>
#include <stdio.h>
#include <time.h>


#define INPUT_SIZE 5
#define HIDDEN_SIZE 8
#define OUTPUT_SIZE 1

#define EPOCHS 200
#define SAMPLES 2000
#define LEARNING_RATE 0.05f

void calculate_metrics(float** data, int samples, Layer* i, Layer* h1, Layer* o) {
    int TP = 0, TN = 0, FP = 0, FN = 0;

    for (int s = 0; s < samples; s++) 
    {
        // Passa os dados pela rede para predição
        set_input_layer(i, data[s]);
        forward_pass(h1, i);
        forward_pass(o, h1);

        // Saída da rede
        float output = o->neurons[0].s;

        // Classe prevista (0 ou 1)
        int pred = (output >= 0.75f) ? 1 : 0;

        // Classe real
        int real = (int)(data[s][INPUT_SIZE]);  // considerando rótulo na coluna INPUT_SIZE

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
    printf("Acurácia: %.4f\n", accuracy);
    printf("Precisão: %.4f\n", precision);
    printf("Recall: %.4f\n", recall);
    printf("F1 Score: %.4f\n", f1);
}

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

    //
    for (int epoch = 0; epoch < EPOCHS; epoch++) 
    {
        float total_error = 0.0f;

        for (int s = 0; s < SAMPLES; s++) 
        {
            set_input_layer(&i, data[s]);

            forward_pass(&h1, &i);
            forward_pass(&o, &h1);

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
    }

    float** data_test = read_csv("weather_test.csv");
    calculate_metrics(data_test, 500, &i, &h1, &o);

    // liberando memoria

    return 0;
}