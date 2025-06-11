#include <stdlib.h>
#include <stdio.h>

typedef struct layer Layer;
struct layer 
{
    // Numero de entradas
    int N;

    // numero de neuronios
    int n;
    
    // Neuronios
    float** d_w;
    float* d_b;
    float* d_s;
    float** w;
    float* s;
    float* z;
    float* x;
    float* b;

    // funcao de ativacao
    void (*activation)(Layer*);
    float (*activation_derivative)(float);
};

void init_layer (Layer* layer, int N, int n, void (*activation)(Layer*), float (*activation_derivative)(float));
void free_layer (Layer* layer);

void set_input_layer (Layer* i, float* input);
void forward_pass (Layer* layer, Layer* input);
void update_layer_parameters (Layer* layer, float eta);

void model_metrics (float** data, int samples, Layer* layers[], int num_layers, double total);
void h_backward_pass (Layer* h, Layer* n, float eta);
void o_backward_pass (Layer* o, float eta, float* target, float* total_error);

void ReLU (Layer* layer);
float derivative_ReLU (float s);
static void* process_samples_thread (void* arg);
void model_metrics_conc(float** data, int samples, Layer* layers[], int num_layers, double total);
void sigmoid (Layer* layer);
float derivative_sigmoid (float s);

void Leaky_ReLU (Layer* layer);
float derivative_Leaky_ReLU (float s);