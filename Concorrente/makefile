CC = gcc
CFLAGS = -Wall -std=c11
LDFLAGS = -lm -lpthread

OBJ = main.o neural_network.o

TARGET = neural_net

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJ) $(LDFLAGS)

main.o: main.c neural_network.h
	$(CC) $(CFLAGS) -c main.c

neural_network.o: neural_network.c neural_network.h
	$(CC) $(CFLAGS) -c neural_network.c

clean:
	rm -f $(OBJ) $(TARGET)
