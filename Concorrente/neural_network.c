#include "neural_network.h"
#include <pthread.h>
#include <math.h>

pthread_mutex_t nn_mutex;
#define NUM_THREADS 16


void init_layer (Layer* layer, int n, int N, void (*activation)(Layer*), float (*activation_derivative)(float))
{
    layer->N = N;
    layer->n = n;

    layer->activation = activation;
    layer->activation_derivative = activation_derivative;

    layer->d_w = malloc(n * sizeof(float*));
    // Alocação e inicialização de w com valores aleatórios em [-1, 1]
    layer->w = malloc(n * sizeof(float*)); // n neurônios (saídas)

    for (int i = 0; i < n; ++i) {
        layer->w[i] = malloc(N * sizeof(float)); // cada neurônio recebe N entradas
        layer->d_w[i] = calloc(N, sizeof(float));
        for (int j = 0; j < N; ++j) {
            layer->w[i][j] = ((float) rand() / RAND_MAX) * 2.0f - 1.0f;
        }
    }

    layer->b = calloc(n, sizeof(float));
    layer->d_b = calloc(n, sizeof(float));
    layer->d_s = calloc(n, sizeof(float));
    layer->s = calloc(n, sizeof(float));
    layer->z = calloc(n, sizeof(float));
    layer->x = calloc(N, sizeof(float)); // deve ser N, não n

    if (!layer->b || !layer->d_b || !layer->d_s || !layer->s || !layer->z || !layer->x) exit(1);
}
void free_layer (Layer* layer) 
{
    if (layer->w) 
    {
        for (int i = 0; i < layer->n; i++) free(layer->w[i]);
        free(layer->w);
    }
    if (layer->d_w) 
    {
        for (int i = 0; i < layer->n; i++)free(layer->d_w[i]);
        free(layer->d_w);
    }

    if (layer->b) free(layer->b);
    if (layer->d_b) free(layer->d_b);

    if (layer->z) free(layer->z);
    if (layer->s) free(layer->s);
    if (layer->d_s) free(layer->d_s);
}

static void calc_z (Layer* layer) 
{
    for (int i = 0; i < layer->n; i++) 
    {
        float soma = layer->b[i];
        for (int j = 0; j < layer->N; j++) 
        {
            soma += layer->w[i][j] * layer->x[j];
        }
        layer->z[i] = soma;
    }
}


void set_input_layer (Layer* i, float* input) 
{
    for (int k = 0; k < i->n; k++) 
    {
        i->s[k] = input[k];
    }
}

void forward_pass (Layer* layer, Layer* input) 
{
    for (int i = 0; i < layer->N; i++) 
    {
        layer->x[i] = input->s[i];
    }
    calc_z(layer);
    layer->activation(layer);
}

void update_layer_parameters (Layer* layer, float eta)
{
    int num_inputs = layer->N;
    int num_neurons = layer->n;

    for (int i = 0; i < num_neurons; i++)
    {
        for (int j = 0; j < num_inputs; j++)
        {
            layer->w[i][j] -= eta * layer->d_w[i][j];
        }
        layer->b[i] -= eta * layer->d_b[i];
    }
}

static void backward_pass (Layer* layer, float eta) 
{
    for (int i = 0; i < layer->n; i++) 
    {
        for (int j = 0; j < layer->N; j++) 
        {
            layer->d_w[i][j] = layer->d_s[i] * layer->x[j];
        }
        layer->d_b[i] = layer->d_s[i];

    }
    update_layer_parameters(layer, eta);
}
void h_backward_pass (Layer* h, Layer* layer_ahead, float eta) 
{
    for (int i = 0; i < h->n; i++) 
    {
        float sum = 0.0f;
        for (int j = 0; j < layer_ahead->n; j++) 
        {
            sum += layer_ahead->d_s[j] * layer_ahead->w[j][i];
        }
        h->d_s[i] = sum * h->activation_derivative(h->s[i]);
    }
    backward_pass(h, eta);
}
void o_backward_pass (Layer* o, float eta, float* target, float* total_error) 
{
    for (int i = 0; i < o->n; i++) 
    {
        float error = target[i] - o->s[i];
        o->d_s[i] = -1 * error * derivative_sigmoid(o->s[i]);
        *total_error += (error * error) * 0.5f;
    }

    backward_pass(o, eta);
}


typedef struct {
    int start_index;
    int end_index;
    float** data;
    Layer** layers;
    int num_layers;
    int TP, TN, FP, FN;
} ThreadData;

void* process_samples_thread (void* arg) 
{
    ThreadData* t_data = (ThreadData*) arg;

    t_data->TP = 0;
    t_data->TN = 0;
    t_data->FP = 0;
    t_data->FN = 0;

    for (int s = t_data->start_index; s < t_data->end_index; s++) 
    {
        pthread_mutex_lock(&nn_mutex);
        set_input_layer(t_data->layers[0], t_data->data[s]);

        for (int i = 1; i < t_data->num_layers; i++) 
        {
            forward_pass(t_data->layers[i], t_data->layers[i - 1]);
        }

        Layer* output_layer = t_data->layers[t_data->num_layers - 1];
        float output_value = output_layer->s[0];  // CORRETO!
        pthread_mutex_unlock(&nn_mutex);

        int pred = (output_value >= 0.55f) ? 1 : 0;
        int real = (int)(t_data->data[s][t_data->layers[0]->n]);

        if (pred == 1 && real == 1) t_data->TP++;
        else if (pred == 0 && real == 0) t_data->TN++;
        else if (pred == 1 && real == 0) t_data->FP++;
        else if (pred == 0 && real == 1) t_data->FN++;
    }

    pthread_exit(NULL);
}

void model_metrics_conc (float** data, int samples, Layer* layers[], int num_layers, double total)
{
    pthread_t tid[NUM_THREADS];
    ThreadData thread_args[NUM_THREADS];
    pthread_mutex_init(&nn_mutex, NULL);

    int samples_per_thread = samples / NUM_THREADS;
    int remaining_samples = samples % NUM_THREADS;
    int current_sample_index = 0;

    for (int i = 0; i < NUM_THREADS; i++)
    {
        thread_args[i].TP = 0;
        thread_args[i].TN = 0;
        thread_args[i].FP = 0;
        thread_args[i].FN = 0;

        thread_args[i].data = data;
        thread_args[i].layers = layers;
        thread_args[i].num_layers = num_layers;

        thread_args[i].start_index = current_sample_index;
        int num_samples_for_this_thread = samples_per_thread + (i < remaining_samples ? 1 : 0);
        thread_args[i].end_index = current_sample_index + num_samples_for_this_thread;
        current_sample_index = thread_args[i].end_index;

        pthread_create(&tid[i], NULL, process_samples_thread, &thread_args[i]);
    }
    int TP = 0, TN = 0, FP = 0, FN = 0;
    for (int i = 0; i < NUM_THREADS; i++) 
    {
        pthread_join(tid[i], NULL);
        TP += thread_args[i].TP;
        TN += thread_args[i].TN;
        FP += thread_args[i].FP;
        FN += thread_args[i].FN;
    }

    float accuracy = (float)(TP + TN) / samples;
    float precision = TP + FP == 0 ? 0 : (float)TP / (TP + FP);
    float recall = TP + FN == 0 ? 0 : (float)TP / (TP + FN);
    float f1 = (precision + recall) == 0 ? 0 : 2 * (precision * recall) / (precision + recall);

    printf("%.8f,%.8f,%.8f,%.8f,%.8f\n", accuracy, precision, recall, f1, total);
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
        float output_value = output->s[0];

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

    printf("%.8f,%.8f,%.8f,%.8f,%.8f\n", accuracy, precision, recall, f1, total);
}


// thread-safe
void ReLU (Layer* layer) 
{
    for (int i = 0; i < layer->n; i++)
    {
        layer->s[i] = layer->z[i] > 0 ? layer->z[i] : 0.0f;
    }
}
float derivative_ReLU (float s) 
{
    return (s > 0) ? 1.0f : 0.0f;
}

// thread-safe
void sigmoid (Layer* layer) 
{
    for (int i = 0; i < layer->n; i++)
    {
        layer->s[i] = 1.0f / (1.0f + expf(-layer->z[i]));
    }
}
float derivative_sigmoid (float s)
{
    return s * (1.0f - s);
}

// thread-safe
void Leaky_ReLU (Layer* layer) 
{
    for (int i = 0; i < layer->n; i++)
    {
        layer->s[i] = layer->z[i] > 0 ? layer->z[i] : 0.01f * layer->z[i];
    }
}
float derivative_Leaky_ReLU (float s) 
{
    return (s > 0) ? 1.0f : 0.01f;
}