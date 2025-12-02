CC = gcc
CFLAGS = -Wall -Wextra -O3 -pthread
TARGET = countermachine

all: $(TARGET)

$(TARGET): main.c
	$(CC) $(CFLAGS) -o $(TARGET) main.c

clean:
	rm -f $(TARGET)

.PHONY: all clean
