#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#define MAX_LINE_LEN 1024
#define COLS 5


float** alocar_matriz (int linhas, int colunas) {
    float** matriz = malloc(linhas * sizeof(float*));

    for (int i = 0; i < linhas; i++) {
        matriz[i] = malloc(colunas * sizeof(float));
    }

    return matriz;
}

int contar_linhas (char* nome_arquivo) {
    char line[MAX_LINE_LEN];
    int linhas = 0;

    FILE* file = fopen(nome_arquivo, "r");
    if (!file) {
        perror("ERROR: fopen()");
        exit(1);
    }
    while (fgets(line, sizeof(line), file)) {
        linhas++;
    }
    fclose(file);
    
    return linhas;
}

int main (int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Entradas do programa: %s <nome_do_arquivo>\n", argv[0]);
        return 1;
    }

    int ROWS = contar_linhas(argv[1]);
    float** data = alocar_matriz(ROWS, COLS);
    
    FILE* file = fopen(argv[1], "r");
    if (!file) {
        perror("ERROR: fopen()");
        return 1;
    }

    int row = 0;
    char line[MAX_LINE_LEN];
    while (fgets(line, sizeof(line), file) && (row < ROWS)) {
        // tokenizando as colunas de cada linha
        char* token = strtok(line, ",\n");
        int col = 0;
        while (token && (col < COLS)) {
            // atof: converte str em float
            data[row][col] = atof(token);

            // pegando o proximo token
            token = strtok(NULL, ",\n");
            col++;
        }
        if (col != COLS) {
            fprintf(stderr, "ERROR: linha %d tem %d colunas (esperado %d)\n", row + 1, col, COLS);
            return 2;
        }
        row++;
    }

    // fim da leitura
    fclose(file);

    printf("Leitura completa: %d linhas lidas.\n", row - 1);

    // liberar memoria

    return 0;
}