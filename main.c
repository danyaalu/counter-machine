#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

// Enum for instructions
typedef enum {
    OP_IF,
    OP_DEC,
    OP_INC,
    OP_COPY,
    OP_GOTO,
    OP_CLR,
    OP_HALT
} Op;

// Struct to store operation and register index
typedef struct
{
    Op op;
    int a, b;
} Instr;

// Struct for register file
typedef struct
{
    uint64_t *r;    // array of register values
    size_t size;    // how many registers are allocated
} RegFile;

