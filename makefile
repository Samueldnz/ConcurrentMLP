# Nome do executável final
TARGET = ml_program

# Compilador e flags
CC = gcc
CFLAGS = -Wall
LDFLAGS = -lm

# Arquivos fonte
SRCS = main.c data.c network.c

# Arquivos objeto
OBJS = $(SRCS:.c=.o)

# Regra principal
all: $(TARGET)

# Como compilar o executável
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Regra para remover os arquivos compilados
clean:
	rm -f $(OBJS) $(TARGET)

# Regra de compilação para arquivos .c -> .o
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

.PHONY: all clean
