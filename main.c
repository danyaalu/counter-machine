#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

// Enum for instructions
typedef enum
{
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
    uint64_t *r; // array of register values
    size_t size; // how many registers are allocated
} RegFile;

// Ensure register index exists and expanding if need be
void ensure_register(RegFile *regFile, size_t index)
{
    if (index < regFile->size)
        return;

    size_t new_size = index + 1;
    regFile->r = realloc(regFile->r, new_size * sizeof(uint64_t));

    // Set new registers to 0
    for (size_t i = regFile->size; i < new_size; i++)
        regFile->r[i] = 0;

    regFile->size = new_size;
}

// Convert program.txt pneuomics to code
Op op_from_string(const char *s)
{
    if (strcmp(s, "IF") == 0)
        return OP_IF;
    if (strcmp(s, "DEC") == 0)
        return OP_DEC;
    if (strcmp(s, "INC") == 0)
        return OP_INC;
    if (strcmp(s, "COPY") == 0)
        return OP_COPY;
    if (strcmp(s, "GOTO") == 0)
        return OP_GOTO;
    if (strcmp(s, "CLR") == 0)
        return OP_CLR;
    if (strcmp(s, "HALT") == 0)
        return OP_HALT;

    fprintf(stderr, "Unknown opcode: %s\n", s);
    exit(1);
}

Instr *load_program(const char *filename, size_t *out_instruction_count)
{
    FILE *file = fopen(filename, "r");
    if (!file)
    {
        perror("Error opening program file");
        exit(1);
    }

    Instr *instruction_list = NULL;
    size_t allowed_capacity = 0;
    size_t instruction_count = 0;

    char opcode_text[32];
    int arg1, arg2;
    char line_buffer[128];

    while (fgets(line_buffer, sizeof(line_buffer), file))
    {
        // Skip blank lines and comments
        if (line_buffer[0] == '\n' || line_buffer[0] == '#' || line_buffer[0] == '\0')
            continue;

        // Expand the array if needed
        if (instruction_count >= allowed_capacity)
        {
            allowed_capacity = (allowed_capacity == 0) ? 16 : allowed_capacity * 2;
            instruction_list = realloc(instruction_list, allowed_capacity * sizeof(Instr));
        }

        // Try to parse up to two integer args
        int parsed_fields = sscanf(line_buffer, "%31s %d %d", opcode_text, &arg1, &arg2);

        Instr instruction;
        instruction.op = op_from_string(opcode_text);

        if (instruction.op == OP_HALT)
        {
            instruction.a = 0;
            instruction.b = 0;
        }
        else if (instruction.op == OP_GOTO)
        {
            instruction.a = arg1 - 1; // jump target, convert to 0-based
            instruction.b = 0;
        }
        else if (parsed_fields == 3)
        {
            instruction.a = arg1 - 1; // convert to 0-based
            instruction.b = arg2 - 1; // convert to 0-based
        }
        else if (parsed_fields == 2)
        {
            instruction.a = arg1 - 1; // single-argument instruction
            instruction.b = 0;
        }
        else
        {
            fprintf(stderr, "Invalid line in program file: %s\n", line_buffer);
            exit(1);
        }

        instruction_list[instruction_count++] = instruction;
    }

    fclose(file);
    *out_instruction_count = instruction_count;
    return instruction_list;
}

RegFile load_registers(const char *filename)
{
    FILE *file = fopen(filename, "r");
    if (!file)
    {
        perror("Error opening register file");
        exit(1);
    }

    RegFile regFile;
    regFile.r = NULL;
    regFile.size = 0;

    uint64_t value;
    size_t allocated_capacity = 0;

    while (fscanf(file, "%llu", &value) == 1)
    {
        if (regFile.size >= allocated_capacity)
        {
            allocated_capacity = (allocated_capacity == 0) ? 8 : allocated_capacity * 2;
            regFile.r = realloc(regFile.r, allocated_capacity * sizeof(uint64_t));
        }

        regFile.r[regFile.size++] = value;
    }

    fclose(file);
    return regFile;
}

// Helper function to print instruction name
const char *op_to_string(Op op)
{
    switch (op)
    {
    case OP_IF:
        return "IF";
    case OP_DEC:
        return "DEC";
    case OP_INC:
        return "INC";
    case OP_COPY:
        return "COPY";
    case OP_GOTO:
        return "GOTO";
    case OP_CLR:
        return "CLR";
    case OP_HALT:
        return "HALT";
    default:
        return "UNKNOWN";
    }
}

// Helper function to print register values
void print_registers(const RegFile *register_file)
{
    printf("  Registers: ");
    if (register_file->size == 0)
    {
        printf("(empty)\n");
        return;
    }
    for (size_t i = 0; i < register_file->size; i++)
    {
        printf("R%zu=%llu", i + 1, (unsigned long long)register_file->r[i]);
        if (i < register_file->size - 1)
            printf(", ");
    }
    printf("\n");
}

// Helper function to print instruction details
void print_instruction(const Instr *instr, size_t pc)
{
    printf("  PC=%zu: %s", pc + 1, op_to_string(instr->op));
    
    switch (instr->op)
    {
    case OP_IF:
        printf(" R%d (jump to %d if zero)", instr->a + 1, instr->b + 1);
        break;
    case OP_DEC:
    case OP_INC:
    case OP_CLR:
        printf(" R%d", instr->a + 1);
        break;
    case OP_COPY:
        printf(" R%d R%d (copy R%d to R%d)", instr->a + 1, instr->b + 1, instr->a + 1, instr->b + 1);
        break;
    case OP_GOTO:
        printf(" %d", instr->a + 1);
        break;
    case OP_HALT:
        // no arguments
        break;
    }
    printf("\n");
}

void run_program(const Instr *instruction_list,
                 size_t instruction_count,
                 RegFile *register_file,
                 int debug_mode,
                 uint64_t max_steps)
{
    size_t program_counter = 0; // which instruction we are executing
    uint64_t step_count = 0;    // number of executed steps (for debugging)

    // Main execution loop
    while (program_counter < instruction_count)
    {
        Instr current_instruction = instruction_list[program_counter];
        step_count++;
        
        // Check if we've exceeded the step limit
        if (max_steps > 0 && step_count > max_steps)
        {
            printf("\nProgram exceeded maximum step limit (%llu steps).\n",
                   (unsigned long long)max_steps);
            return;
        }

        // Debug mode: print current state before executing
        if (debug_mode)
        {
            printf("%llu [", (unsigned long long)step_count);
            for (size_t i = 0; i < register_file->size; i++)
            {
                printf("%llu", (unsigned long long)register_file->r[i]);
                if (i < register_file->size - 1)
                    printf(", ");
            }
            printf("]\n");
        }

        switch (current_instruction.op)
        {
        case OP_IF:
        {
            size_t test_register = current_instruction.a;
            size_t jump_target = current_instruction.b;

            ensure_register(register_file, test_register);

            if (register_file->r[test_register] == 0)
                program_counter = jump_target;
            else
                program_counter++;

            break;
        }

        case OP_DEC:
        {
            size_t target_register = current_instruction.a;
            ensure_register(register_file, target_register);

            if (register_file->r[target_register] > 0)
                register_file->r[target_register]--;

            program_counter++;
            break;
        }

        case OP_INC:
        {
            size_t target_register = current_instruction.a;
            ensure_register(register_file, target_register);

            register_file->r[target_register]++;
            program_counter++;
            break;
        }

        case OP_COPY:
        {
            size_t source_register = current_instruction.a;
            size_t destination_register = current_instruction.b;

            ensure_register(register_file, source_register);
            ensure_register(register_file, destination_register);

            register_file->r[destination_register] =
                register_file->r[source_register];

            program_counter++;
            break;
        }

        case OP_GOTO:
        {
            size_t jump_target = current_instruction.a;
            program_counter = jump_target;
            break;
        }

        case OP_CLR:
        {
            size_t target_register = current_instruction.a;
            ensure_register(register_file, target_register);

            register_file->r[target_register] = 0;

            program_counter++;
            break;
        }

        case OP_HALT:
        {
            printf("\nProgram halted after %llu steps.\n",
                   (unsigned long long)step_count);
            return;
        }
        }
    }

    printf("\nProgram terminated because PC went out of range.\n");
}

int main(int argc, char *argv[])
{
    // Check for debug flag and max steps
    int debug_mode = 0;
    uint64_t max_steps = 0; // 0 means no limit
    
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-d") == 0)
        {
            debug_mode = 1;
            printf("Debug mode enabled.\n");
        }
        else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc)
        {
            max_steps = strtoull(argv[i + 1], NULL, 10);
            printf("Maximum steps: %llu\n", (unsigned long long)max_steps);
            i++; // Skip the next argument
        }
    }

    size_t program_length;
    Instr *program = load_program("program.txt", &program_length);
    RegFile regFile = load_registers("registers.txt");

    // Run program
    run_program(program, program_length, &regFile, debug_mode, max_steps);

    // Print final registers
    printf("\nFinal register values:\n");
    for (size_t i = 0; i < regFile.size; i++)
        printf("R%zu = %llu\n", i + 1, (unsigned long long)regFile.r[i]);

    free(program);
    free(regFile.r);
    return 0;
}