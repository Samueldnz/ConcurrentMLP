#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include "matrix.h"


#define MAX_LEN 1024

static int rows_count (const char* file_name) 
{
    FILE* file = fopen(file_name, "r");
    if (!file) 
    {
        perror("Erro ao abrir arquivo");
        exit(1);
    }

    char row[MAX_LEN];
    int rows = 0;
    while (fgets(row, sizeof(row), file)) rows++;

    fclose(file);
    
    return rows;
}


float** malloc_matrix (int rows, int columns) 
{
    float** matriz = malloc(rows * sizeof(float*));
    if (!matriz) 
    {
        perror("Erro: malloc() de matriz");
        exit(1);
    }

    for (int i = 0; i < rows; i++) 
    {
        matriz[i] = malloc(columns * sizeof(float));
        if (!matriz[i]) 
        {
            perror("Erro: malloc() da coluna");
            exit(1);
        }
    }

    return matriz;
}

void free_matrix (float** matriz, int linhas) 
{
    for (int i = 0; i < linhas; i++) free(matriz[i]);

    free(matriz);
}

float** read_csv (const char* file_name, int columns) 
{
    int rows = rows_count(file_name);

    float** matriz = alocar_matriz(rows, columns);

    FILE* file = fopen(file_name, "r");
    if (!file) 
    {
        perror("Erro: fopen() do arquivo");
        exit(1);
    }

    int row = 0;
    char tam[MAX_LEN];
    while (fgets(tam, sizeof(tam), file) && row < rows) 
    {
        // tokenizando as colunas de cada linha
        char* token = strtok(tam, ",\n");
        int col = 0;

        while (token && col < columns) 
        {
            // atof: converte str em float
            matriz[row][col] = atof(token);

            // pegando o proximo token
            token = strtok(NULL, ",\n");
            col++;
        }

        if (col != columns) 
        {
            fprintf(stderr, "Erro: linha %d tem %d colunas (esperado %d)\n", row + 1, col, columns);
            exit(2);
        }

        row++;
    }
    fclose(file);

    return matriz;
}