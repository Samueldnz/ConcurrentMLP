CC = gcc
CFLAGS = -Wall
OBJ = main.o neural_network.o
TARGET = neural_net

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ -lm

main.o: main.c neural_network.h
	$(CC) $(CFLAGS) -c main.c

neural_network.o: neural_network.c neural_network.h
	$(CC) $(CFLAGS) -c neural_network.c

clean:
	rm -f *.o $(TARGET)
