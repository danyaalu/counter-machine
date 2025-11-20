CC = gcc
CFLAGS = -Wall -Wextra -O3 -pthread
TARGET = busy_beaver

all: $(TARGET)

$(TARGET): busy_beaver.c
	$(CC) $(CFLAGS) -o $(TARGET) busy_beaver.c

clean:
	rm -f $(TARGET)

.PHONY: all clean
