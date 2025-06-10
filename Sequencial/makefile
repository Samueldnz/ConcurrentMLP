# Nome do executavel final
EXEC = weather_forecast

# Lista dos arquivos necessarios no compilamento 
SRCS = weather_forecast.c neural_network.c matrix.c

# Nome dos executaveis com mesmo nome dos arquivos-geradores
OBJS = $(SRCS:.c=.o)

# flags
CC = gcc
CFLAGS = -Wall -g

# Regra padrão
all: $(EXEC)


# Comando: make
$(EXEC): $(OBJS)
	$(CC) $(OBJS) -o $(EXEC) -lm

# Comando: make clean
clean:
	rm -f $(OBJS) $(EXEC)