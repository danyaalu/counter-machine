#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <time.h>
#include <pthread.h>
#include <signal.h>

#ifdef __unix__
#include <unistd.h>  // For sysconf(_SC_NPROCESSORS_ONLN)
#endif

// Available opcodes (excluding HALT which is always last)
#define NUM_OPCODES 6
#define QUEUE_SIZE 1000  // Buffer size for work queue (can hold 1000 programs)
#define MAX_REGISTERS 64 // Maximum number of registers for inline execution

// Opcode enum matching Python's encoding
typedef enum {
    OP_IF = 0,
    OP_DEC = 1,
    OP_INC = 2,
    OP_COPY = 3,
    OP_GOTO = 4,
    OP_CLR = 5,
    OP_HALT = 6
} Opcode;

// Encoded instruction type (packed into 32-bit int)
typedef uint32_t EncodedInstruction;

// Program structure
typedef struct {
    EncodedInstruction *instructions;
    int length;
} Program;

// Result structure
typedef struct {
    int steps;
    bool exceeded_limit;
    char error[256];
} ExecutionResult;

// Forward declarations
typedef struct ProgramGenerator ProgramGenerator;
typedef struct SharedGenerator SharedGenerator;
typedef struct WorkQueue WorkQueue;

// Work queue for producer-consumer model
struct WorkQueue {
    Program *programs;      // Ring buffer of programs
    int capacity;           // Max size of queue
    int size;              // Current number of items
    int head;              // Read position
    int tail;              // Write position
    pthread_mutex_t lock;
    pthread_cond_t not_empty;
    pthread_cond_t not_full;
    bool producer_done;    // Producer finished generating
};

// Best program tracking (shared across threads)
typedef struct {
    Program program;
    int steps;
    pthread_mutex_t lock;
} BestProgram;

// Shared program generator state for multi-threading
struct SharedGenerator {
    ProgramGenerator *generator;
    pthread_mutex_t lock;
    bool exhausted;
};

// Worker thread data
typedef struct {
    int worker_id;
    int max_steps;
    int max_register;
    int num_instructions;
    BestProgram *best;
    WorkQueue *work_queue;
    
    // Statistics
    unsigned long programs_tested;
    unsigned long programs_timed_out;
    unsigned long programs_out_of_range;
    unsigned long programs_errored;
    
    // Inline execution: pre-allocated register array
    uint64_t registers[MAX_REGISTERS];
} WorkerContext;

// Producer thread data
typedef struct {
    ProgramGenerator *generator;
    WorkQueue *work_queue;
    int num_instructions;
    int max_register;
} ProducerContext;

// Global statistics
typedef struct {
    unsigned long total_programs;
    unsigned long programs_tested;
    unsigned long programs_timed_out;
    unsigned long programs_out_of_range;
    unsigned long programs_errored;
    pthread_mutex_t lock;
} Statistics;

static Statistics global_stats = {0};
static volatile sig_atomic_t interrupted = 0;

// Signal handler for Ctrl+C
void signal_handler(int sig) {
    (void)sig;  // Unused parameter
    interrupted = 1;
}

// Encode instruction into 32-bit integer
EncodedInstruction encode_instruction(const char *instruction) {
    char opcode_str[32];
    int arg1 = 0, arg2 = 0;
    
    sscanf(instruction, "%s %d %d", opcode_str, &arg1, &arg2);
    
    Opcode opcode;
    if (strcmp(opcode_str, "IF") == 0) opcode = OP_IF;
    else if (strcmp(opcode_str, "DEC") == 0) opcode = OP_DEC;
    else if (strcmp(opcode_str, "INC") == 0) opcode = OP_INC;
    else if (strcmp(opcode_str, "COPY") == 0) opcode = OP_COPY;
    else if (strcmp(opcode_str, "GOTO") == 0) opcode = OP_GOTO;
    else if (strcmp(opcode_str, "CLR") == 0) opcode = OP_CLR;
    else if (strcmp(opcode_str, "HALT") == 0) opcode = OP_HALT;
    else {
        fprintf(stderr, "Unknown opcode: %s\n", opcode_str);
        exit(1);
    }
    
    // Pack into single integer: opcode | arg1 | arg2
    return (opcode << 16) | (arg1 << 8) | arg2;
}

// Decode instruction from 32-bit integer
void decode_instruction(EncodedInstruction encoded, char *output, int max_len) {
    Opcode opcode = (encoded >> 16) & 0x7;
    int arg1 = (encoded >> 8) & 0xFF;
    int arg2 = encoded & 0xFF;
    
    const char *opcode_names[] = {"IF", "DEC", "INC", "COPY", "GOTO", "CLR", "HALT"};
    
    if (opcode == OP_HALT) {
        snprintf(output, max_len, "HALT");
    } else if (opcode == OP_IF || opcode == OP_COPY) {
        snprintf(output, max_len, "%s %d %d", opcode_names[opcode], arg1, arg2);
    } else {
        snprintf(output, max_len, "%s %d", opcode_names[opcode], arg1);
    }
}

// Generate all possible instructions for a given position
typedef struct {
    char **instructions;
    int count;
} InstructionList;

InstructionList generate_instructions_for_position(int line_num, int max_register, int total_lines) {
    (void)line_num;  // Unused parameter - kept for API consistency
    InstructionList list;
    list.instructions = malloc(1000 * sizeof(char*)); // Generous allocation
    list.count = 0;
    
    // IF instructions
    for (int reg = 1; reg <= max_register; reg++) {
        for (int jump = 1; jump <= total_lines; jump++) {
            list.instructions[list.count] = malloc(32);
            snprintf(list.instructions[list.count], 32, "IF %d %d", reg, jump);
            list.count++;
        }
    }
    
    // DEC instructions
    for (int reg = 1; reg <= max_register; reg++) {
        list.instructions[list.count] = malloc(32);
        snprintf(list.instructions[list.count], 32, "DEC %d", reg);
        list.count++;
    }
    
    // INC instructions
    for (int reg = 1; reg <= max_register; reg++) {
        list.instructions[list.count] = malloc(32);
        snprintf(list.instructions[list.count], 32, "INC %d", reg);
        list.count++;
    }
    
    // COPY instructions
    for (int src = 1; src <= max_register; src++) {
        for (int dst = 1; dst <= max_register; dst++) {
            if (src != dst) {
                list.instructions[list.count] = malloc(32);
                snprintf(list.instructions[list.count], 32, "COPY %d %d", src, dst);
                list.count++;
            }
        }
    }
    
    // GOTO instructions
    for (int jump = 1; jump <= total_lines; jump++) {
        list.instructions[list.count] = malloc(32);
        snprintf(list.instructions[list.count], 32, "GOTO %d", jump);
        list.count++;
    }
    
    // CLR instructions
    for (int reg = 1; reg <= max_register; reg++) {
        list.instructions[list.count] = malloc(32);
        snprintf(list.instructions[list.count], 32, "CLR %d", reg);
        list.count++;
    }
    
    return list;
}

void free_instruction_list(InstructionList *list) {
    for (int i = 0; i < list->count; i++) {
        free(list->instructions[i]);
    }
    free(list->instructions);
}

// Check if HALT is reachable (conservative check)
bool is_halt_reachable(Program *program) {
    if (program->length < 2) return true;
    
    // Get the last instruction before HALT
    char last_instr[64];
    decode_instruction(program->instructions[program->length - 2], last_instr, sizeof(last_instr));
    
    char opcode[32];
    int arg1 = 0;
    sscanf(last_instr, "%s %d", opcode, &arg1);
    
    // Only reject if GOTO jumps to itself
    if (strcmp(opcode, "GOTO") == 0) {
        int goto_line = program->length - 1;
        if (arg1 == goto_line) {
            return false;  // GOTO to itself = infinite loop
        }
    }
    
    return true;
}

// Check for obvious infinite loops
bool has_obvious_infinite_loop(Program *program) {
    for (int i = 0; i < program->length - 1; i++) {
        char instr[64];
        decode_instruction(program->instructions[i], instr, sizeof(instr));
        
        char opcode[32];
        int target = 0;
        sscanf(instr, "%s %d", opcode, &target);
        
        int current_line = i + 1;
        
        // Unconditional backward GOTO = definite infinite loop
        if (strcmp(opcode, "GOTO") == 0 && target <= current_line) {
            return true;
        }
    }
    
    return false;
}

// Inline counter machine execution - no subprocess overhead!
// Directly executes the encoded program and returns result
ExecutionResult run_program_inline(WorkerContext *ctx, Program *program) {
    ExecutionResult result = {0};
    
    // Clear registers (use ctx's pre-allocated buffer)
    memset(ctx->registers, 0, MAX_REGISTERS * sizeof(uint64_t));
    
    int pc = 0;  // Program counter (0-based)
    uint64_t step_count = 0;
    int program_length = program->length;
    uint64_t max_steps = (uint64_t)ctx->max_steps;
    
    // Main execution loop
    while (pc >= 0 && pc < program_length) {
        step_count++;
        
        // Check step limit
        if (step_count > max_steps) {
            result.steps = 0;
            result.exceeded_limit = true;
            strcpy(result.error, "Step limit exceeded");
            return result;
        }
        
        // Decode instruction inline (avoid function call overhead)
        EncodedInstruction encoded = program->instructions[pc];
        Opcode opcode = (encoded >> 16) & 0x7;
        int arg1 = (encoded >> 8) & 0xFF;
        int arg2 = encoded & 0xFF;
        
        // Convert 1-based args to 0-based register indices
        int reg_a = arg1 - 1;
        int reg_b = arg2 - 1;
        
        switch (opcode) {
            case OP_IF: {
                // IF reg_a (jump to arg2 if zero)
                // Note: arg2 is already the 1-based line number, convert to 0-based
                if (reg_a >= 0 && reg_a < MAX_REGISTERS && ctx->registers[reg_a] == 0) {
                    pc = arg2 - 1;  // arg2 is 1-based line number
                } else {
                    pc++;
                }
                break;
            }
            
            case OP_DEC: {
                if (reg_a >= 0 && reg_a < MAX_REGISTERS && ctx->registers[reg_a] > 0) {
                    ctx->registers[reg_a]--;
                }
                pc++;
                break;
            }
            
            case OP_INC: {
                if (reg_a >= 0 && reg_a < MAX_REGISTERS) {
                    ctx->registers[reg_a]++;
                }
                pc++;
                break;
            }
            
            case OP_COPY: {
                if (reg_a >= 0 && reg_a < MAX_REGISTERS && 
                    reg_b >= 0 && reg_b < MAX_REGISTERS) {
                    ctx->registers[reg_b] = ctx->registers[reg_a];
                }
                pc++;
                break;
            }
            
            case OP_GOTO: {
                // GOTO arg1 (1-based line number)
                pc = arg1 - 1;
                break;
            }
            
            case OP_CLR: {
                if (reg_a >= 0 && reg_a < MAX_REGISTERS) {
                    ctx->registers[reg_a] = 0;
                }
                pc++;
                break;
            }
            
            case OP_HALT: {
                result.steps = (int)step_count;
                result.exceeded_limit = false;
                result.error[0] = '\0';
                return result;
            }
            
            default: {
                // Unknown opcode - should not happen with valid programs
                result.steps = 0;
                result.exceeded_limit = false;
                strcpy(result.error, "Unknown opcode");
                return result;
            }
        }
    }
    
    // PC went out of range
    result.steps = 0;
    result.exceeded_limit = false;
    strcpy(result.error, "Out of range termination");
    return result;
}

// Format program for display
void format_program(Program *program, char *output, int max_len) {
    int offset = 0;
    for (int i = 0; i < program->length; i++) {
        char instr[64];
        decode_instruction(program->instructions[i], instr, sizeof(instr));
        offset += snprintf(output + offset, max_len - offset, "%d: %s\n", i + 1, instr);
    }
}

// Copy program
void copy_program(Program *dest, Program *src) {
    dest->length = src->length;
    dest->instructions = malloc(dest->length * sizeof(EncodedInstruction));
    memcpy(dest->instructions, src->instructions, dest->length * sizeof(EncodedInstruction));
}

// Free program
void free_program(Program *program) {
    if (program->instructions) {
        free(program->instructions);
        program->instructions = NULL;
    }
}

// Program generator structure
struct ProgramGenerator {
    InstructionList *position_instructions;
    int num_positions;
    int *indices;
    bool first;
};

ProgramGenerator* create_program_generator(int num_instructions, int max_register) {
    ProgramGenerator *gen = malloc(sizeof(ProgramGenerator));
    gen->num_positions = num_instructions;
    gen->position_instructions = malloc(num_instructions * sizeof(InstructionList));
    gen->indices = calloc(num_instructions, sizeof(int));
    gen->first = true;
    
    int total_lines = num_instructions + 1;
    for (int i = 0; i < num_instructions; i++) {
        gen->position_instructions[i] = generate_instructions_for_position(i + 1, max_register, total_lines);
    }
    
    // Calculate total programs
    unsigned long long total = 1;
    for (int i = 0; i < num_instructions; i++) {
        total *= gen->position_instructions[i].count;
    }
    
    printf("Total programs to generate: %llu\n", total);
    printf("Mode: ITERATIVE\n");
    printf("Filters: ENABLED\n\n");
    
    return gen;
}

bool next_program(ProgramGenerator *gen, Program *program, bool use_optimization) {
    if (interrupted) return false;
    
    // First call
    if (gen->first) {
        gen->first = false;
    } else {
        // Increment indices (like an odometer)
        int pos = gen->num_positions - 1;
        while (pos >= 0) {
            gen->indices[pos]++;
            if (gen->indices[pos] < gen->position_instructions[pos].count) {
                break;
            }
            gen->indices[pos] = 0;
            pos--;
        }
        
        // If we carried past the first position, we're done
        if (pos < 0) {
            return false;
        }
    }
    
    // Generate program from current indices (reuse existing buffer)
    do {
        // Program buffer should already be allocated by caller
        // program->length should already be set
        // Just fill in the instructions
        
        for (int i = 0; i < gen->num_positions; i++) {
            const char *instr = gen->position_instructions[i].instructions[gen->indices[i]];
            program->instructions[i] = encode_instruction(instr);
        }
        program->instructions[gen->num_positions] = encode_instruction("HALT");
        
        // Apply filters if optimization is enabled
        if (!use_optimization) {
            return true;
        }
        
        if (!is_halt_reachable(program)) {
            // Don't free - we're reusing the buffer
            
            // Increment for next iteration
            int pos = gen->num_positions - 1;
            while (pos >= 0) {
                gen->indices[pos]++;
                if (gen->indices[pos] < gen->position_instructions[pos].count) {
                    break;
                }
                gen->indices[pos] = 0;
                pos--;
            }
            if (pos < 0) return false;
            continue;
        }
        
        if (has_obvious_infinite_loop(program)) {
            // Don't free - we're reusing the buffer
            
            // Increment for next iteration
            int pos = gen->num_positions - 1;
            while (pos >= 0) {
                gen->indices[pos]++;
                if (gen->indices[pos] < gen->position_instructions[pos].count) {
                    break;
                }
                gen->indices[pos] = 0;
                pos--;
            }
            if (pos < 0) return false;
            continue;
        }
        
        return true;
    } while (1);
}

void free_program_generator(ProgramGenerator *gen) {
    for (int i = 0; i < gen->num_positions; i++) {
        free_instruction_list(&gen->position_instructions[i]);
    }
    free(gen->position_instructions);
    free(gen->indices);
    free(gen);
}

// Work queue functions
WorkQueue* create_work_queue(int capacity, int program_length) {
    WorkQueue *queue = malloc(sizeof(WorkQueue));
    queue->capacity = capacity;
    queue->size = 0;
    queue->head = 0;
    queue->tail = 0;
    queue->producer_done = false;
    
    pthread_mutex_init(&queue->lock, NULL);
    pthread_cond_init(&queue->not_empty, NULL);
    pthread_cond_init(&queue->not_full, NULL);
    
    // Pre-allocate all program buffers
    queue->programs = malloc(capacity * sizeof(Program));
    for (int i = 0; i < capacity; i++) {
        queue->programs[i].length = program_length;
        queue->programs[i].instructions = malloc(program_length * sizeof(EncodedInstruction));
    }
    
    return queue;
}

void free_work_queue(WorkQueue *queue) {
    for (int i = 0; i < queue->capacity; i++) {
        free(queue->programs[i].instructions);
    }
    free(queue->programs);
    pthread_mutex_destroy(&queue->lock);
    pthread_cond_destroy(&queue->not_empty);
    pthread_cond_destroy(&queue->not_full);
    free(queue);
}

bool queue_push(WorkQueue *queue, Program *program) {
    pthread_mutex_lock(&queue->lock);
    
    // Wait while queue is full
    while (queue->size >= queue->capacity && !interrupted) {
        pthread_cond_wait(&queue->not_full, &queue->lock);
    }
    
    if (interrupted) {
        pthread_mutex_unlock(&queue->lock);
        return false;
    }
    
    // Copy program into queue buffer
    Program *dest = &queue->programs[queue->tail];
    memcpy(dest->instructions, program->instructions, 
           program->length * sizeof(EncodedInstruction));
    
    queue->tail = (queue->tail + 1) % queue->capacity;
    queue->size++;
    
    pthread_cond_signal(&queue->not_empty);
    pthread_mutex_unlock(&queue->lock);
    
    return true;
}

bool queue_pop(WorkQueue *queue, Program *program) {
    pthread_mutex_lock(&queue->lock);
    
    // Wait while queue is empty and producer not done
    while (queue->size == 0 && !queue->producer_done && !interrupted) {
        pthread_cond_wait(&queue->not_empty, &queue->lock);
    }
    
    if (interrupted || (queue->size == 0 && queue->producer_done)) {
        pthread_mutex_unlock(&queue->lock);
        return false;
    }
    
    // Copy program from queue
    Program *src = &queue->programs[queue->head];
    memcpy(program->instructions, src->instructions,
           program->length * sizeof(EncodedInstruction));
    
    queue->head = (queue->head + 1) % queue->capacity;
    queue->size--;
    
    pthread_cond_signal(&queue->not_full);
    pthread_mutex_unlock(&queue->lock);
    
    return true;
}

void queue_mark_done(WorkQueue *queue) {
    pthread_mutex_lock(&queue->lock);
    queue->producer_done = true;
    pthread_cond_broadcast(&queue->not_empty);  // Wake all waiting consumers
    pthread_mutex_unlock(&queue->lock);
}

// Producer thread function
void* producer_thread(void *arg) {
    ProducerContext *ctx = (ProducerContext*)arg;
    
    Program program;
    program.length = ctx->num_instructions + 1;  // +1 for HALT
    program.instructions = malloc(program.length * sizeof(EncodedInstruction));
    
    while (next_program(ctx->generator, &program, true) && !interrupted) {
        if (!queue_push(ctx->work_queue, &program)) {
            break;
        }
    }
    
    free(program.instructions);
    queue_mark_done(ctx->work_queue);
    
    return NULL;
}

// Worker thread function
void* worker_thread(void *arg) {
    WorkerContext *ctx = (WorkerContext*)arg;
    
    // Allocate program buffer
    Program program;
    program.length = ctx->num_instructions + 1;  // +1 for HALT
    program.instructions = malloc(program.length * sizeof(EncodedInstruction));
    
    // Process programs from work queue
    while (queue_pop(ctx->work_queue, &program)) {
        if (interrupted) break;
        
        // Run program with inline execution (no subprocess overhead!)
        ExecutionResult result = run_program_inline(ctx, &program);
        
        ctx->programs_tested++;
        
        if (result.exceeded_limit) {
            ctx->programs_timed_out++;
        } else if (strcmp(result.error, "Out of range termination") == 0) {
            ctx->programs_out_of_range++;
        } else if (result.error[0] != '\0') {
            ctx->programs_errored++;
        }
        
        // Check if this is a new best
        if (result.steps > 0) {
            pthread_mutex_lock(&ctx->best->lock);
            if (result.steps > ctx->best->steps) {
                if (ctx->best->program.instructions) {
                    free_program(&ctx->best->program);
                }
                copy_program(&ctx->best->program, &program);
                ctx->best->steps = result.steps;
                
                printf("\n**********************************************************************\n");
                printf("NEW BEST FOUND! Steps: %d\n", result.steps);
                printf("**********************************************************************\n");
                
                char formatted[4096];
                format_program(&program, formatted, sizeof(formatted));
                printf("%s", formatted);
                printf("**********************************************************************\n\n");
            }
            pthread_mutex_unlock(&ctx->best->lock);
        }
        
        // Update global statistics periodically
        if (ctx->programs_tested % 100 == 0) {
            pthread_mutex_lock(&global_stats.lock);
            global_stats.programs_tested += 100;
            
            if (global_stats.programs_tested % 1000 == 0) {
                pthread_mutex_lock(&ctx->work_queue->lock);
                int queue_size = ctx->work_queue->size;
                pthread_mutex_unlock(&ctx->work_queue->lock);
                printf("Progress: %lu tested - best: %d steps - queue: %d\n", 
                       global_stats.programs_tested, ctx->best->steps, queue_size);
            }
            pthread_mutex_unlock(&global_stats.lock);
        }
        
        // Note: Do NOT free_program(&program) here - we're reusing the buffer
    }
    
    // Cleanup
    free_program(&program);  // Free the reused buffer once at the end
    // No server process to stop - we use inline execution!
    
    return NULL;
}

int main(int argc, char *argv[]) {
    // Parse arguments
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <num_instructions> [--max-register N] [--max-steps N] [--workers N]\n", argv[0]);
        return 1;
    }
    
    int num_instructions = atoi(argv[1]);
    int max_register = 3;
    int max_steps = 100000;
    int num_workers = 4;  // Default to 4 workers
    
    // Try to get CPU count at runtime
    #ifdef _SC_NPROCESSORS_ONLN
    num_workers = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (num_workers < 1) num_workers = 4;
    #endif
    
    // Parse optional arguments
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--max-register") == 0 && i + 1 < argc) {
            max_register = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--max-steps") == 0 && i + 1 < argc) {
            max_steps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--workers") == 0 && i + 1 < argc) {
            num_workers = atoi(argv[++i]);
        }
    }
    
    // Validate
    if (num_instructions < 1) {
        fprintf(stderr, "Error: Number of instructions must be at least 1\n");
        return 1;
    }
    
    // Setup signal handler
    signal(SIGINT, signal_handler);
    
    // Initialize random seed
    srand(time(NULL));
    
    // Initialize mutex
    pthread_mutex_init(&global_stats.lock, NULL);
    
    // Initialize best program tracking
    BestProgram best;
    best.program.instructions = NULL;
    best.program.length = 0;
    best.steps = 0;
    pthread_mutex_init(&best.lock, NULL);
    
    printf("======================================================================\n");
    printf("BUSY BEAVER BRUTE FORCE SEARCH (C VERSION - INLINE EXECUTION)\n");
    printf("======================================================================\n");
    printf("Instructions (excluding HALT): %d\n", num_instructions);
    printf("Max register: %d\n", max_register);
    printf("Max steps per program: %d\n", max_steps);
    printf("Parallel workers: %d\n", num_workers);
    printf("Work queue size: %d\n", QUEUE_SIZE);
    printf("Execution: INLINE (no subprocess overhead)\n");
    printf("======================================================================\n\n");
    
    // Create work queue
    WorkQueue *queue = create_work_queue(QUEUE_SIZE, num_instructions + 1);
    if (!queue) {
        fprintf(stderr, "Failed to create work queue\n");
        return 1;
    }
    
    // Create producer context and thread
    ProducerContext prod_ctx;
    prod_ctx.work_queue = queue;
    prod_ctx.generator = create_program_generator(num_instructions, max_register);
    prod_ctx.num_instructions = num_instructions;
    
    pthread_t producer;
    if (pthread_create(&producer, NULL, producer_thread, &prod_ctx) != 0) {
        fprintf(stderr, "Failed to create producer thread\n");
        free_work_queue(queue);
        free_program_generator(prod_ctx.generator);
        return 1;
    }
    
    // Create worker contexts and threads
    WorkerContext *workers = malloc(num_workers * sizeof(WorkerContext));
    pthread_t *threads = malloc(num_workers * sizeof(pthread_t));
    
    for (int i = 0; i < num_workers; i++) {
        workers[i].worker_id = i;
        workers[i].max_steps = max_steps;
        workers[i].max_register = max_register;
        workers[i].num_instructions = num_instructions;
        workers[i].best = &best;
        workers[i].work_queue = queue;
        workers[i].programs_tested = 0;
        workers[i].programs_timed_out = 0;
        workers[i].programs_out_of_range = 0;
        workers[i].programs_errored = 0;
        // Pre-clear registers array (will be cleared again per-program)
        memset(workers[i].registers, 0, sizeof(workers[i].registers));
        
        if (pthread_create(&threads[i], NULL, worker_thread, &workers[i]) != 0) {
            fprintf(stderr, "Failed to create worker thread %d\n", i);
            // Clean up and exit
            for (int j = 0; j < i; j++) {
                pthread_cancel(threads[j]);
            }
            pthread_cancel(producer);
            free(workers);
            free(threads);
            free_work_queue(queue);
            free_program_generator(prod_ctx.generator);
            return 1;
        }
    }
    
    // Wait for producer to finish
    pthread_join(producer, NULL);
    
    // Wait for all workers to complete
    for (int i = 0; i < num_workers; i++) {
        pthread_join(threads[i], NULL);
    }
    
    // Aggregate statistics from workers
    unsigned long total_tested = 0;
    unsigned long total_timed_out = 0;
    unsigned long total_out_of_range = 0;
    unsigned long total_errored = 0;
    
    for (int i = 0; i < num_workers; i++) {
        total_tested += workers[i].programs_tested;
        total_timed_out += workers[i].programs_timed_out;
        total_out_of_range += workers[i].programs_out_of_range;
        total_errored += workers[i].programs_errored;
    }
    
    // Print final results
    printf("\n======================================================================\n");
    printf("SEARCH COMPLETE\n");
    printf("======================================================================\n");
    printf("Programs tested: %lu\n", total_tested);
    printf("Programs exceeded step limit: %lu\n", total_timed_out);
    printf("Programs went out of range: %lu\n", total_out_of_range);
    printf("Programs with errors: %lu\n", total_errored);
    printf("\nLONGEST RUNNING PROGRAM (%d steps):\n", best.steps);
    printf("======================================================================\n");
    
    if (best.program.instructions) {
        char formatted[4096];
        format_program(&best.program, formatted, sizeof(formatted));
        printf("%s", formatted);
        free_program(&best.program);
    } else {
        printf("No successful programs found.\n");
    }
    
    printf("======================================================================\n");
    
    // Cleanup
    free(workers);
    free(threads);
    free_program_generator(prod_ctx.generator);
    free_work_queue(queue);
    pthread_mutex_destroy(&best.lock);
    pthread_mutex_destroy(&global_stats.lock);
    
    return 0;
}
