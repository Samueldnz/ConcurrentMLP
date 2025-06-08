# Nome do executavel final
EXEC = rede_neural

# Lista dos arquivos necessarios no compilamento 
SRCS = rede_neural.c activation_function.c

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