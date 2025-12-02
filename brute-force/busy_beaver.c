#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <time.h>
#include <pthread.h>
#include <signal.h>
#include <errno.h>

// Available opcodes (excluding HALT which is always last)
#define NUM_OPCODES 6
#define QUEUE_SIZE 1000  // Buffer size for work queue (can hold 1000 programs)

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
    char counter_machine_path[512];
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
    
    // Server process info
    pid_t server_pid;
    int stdin_pipe[2];
    int stdout_pipe[2];
    FILE *stdin_stream;
    FILE *stdout_stream;
    int programs_since_restart;
    int restart_threshold;
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
    unsigned long programs_filtered_unreachable;  // Filtered by is_halt_reachable
    unsigned long programs_filtered_infinite;     // Filtered by has_obvious_infinite_loop
    unsigned long programs_filtered_cfg;          // Filtered by cfg_reaches_halt
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

// CFG-based reachability analysis: DFS helper for forward reachability
void dfs_forward(Program *program, int line, bool *visited) {
    // Line numbers are 1-indexed, array is 0-indexed
    int idx = line - 1;
    
    if (idx < 0 || idx >= program->length || visited[idx]) {
        return;
    }
    
    visited[idx] = true;
    
    // If this is HALT, stop here
    char instr[64];
    decode_instruction(program->instructions[idx], instr, sizeof(instr));
    
    char opcode[32];
    int arg1 = 0, arg2 = 0;
    sscanf(instr, "%s %d %d", opcode, &arg1, &arg2);
    
    if (strcmp(opcode, "HALT") == 0) {
        return;
    }
    
    // Build edges based on instruction type
    if (strcmp(opcode, "GOTO") == 0) {
        // Unconditional jump: only edge is to target
        dfs_forward(program, arg1, visited);
    } else if (strcmp(opcode, "IF") == 0) {
        // Conditional: both branches possible
        dfs_forward(program, arg2, visited);      // Jump target
        dfs_forward(program, line + 1, visited);  // Fall through
    } else {
        // Normal instructions (INC/DEC/CLR/COPY): fall through to next
        dfs_forward(program, line + 1, visited);
    }
}

// CFG-based reachability analysis: DFS helper for backward reachability
void dfs_backward(Program *program, int target_line, bool *can_reach, bool *visited) {
    // target_line is what we're checking can reach from
    int idx = target_line - 1;
    
    if (idx < 0 || idx >= program->length || visited[idx]) {
        return;
    }
    
    visited[idx] = true;
    
    // This line can reach the target
    can_reach[idx] = true;
    
    // Find all lines that can jump to this line
    for (int line = 1; line <= program->length; line++) {
        int i = line - 1;
        
        char instr[64];
        decode_instruction(program->instructions[i], instr, sizeof(instr));
        
        char opcode[32];
        int arg1 = 0, arg2 = 0;
        sscanf(instr, "%s %d %d", opcode, &arg1, &arg2);
        
        bool has_edge_to_target = false;
        
        if (strcmp(opcode, "HALT") == 0) {
            // HALT has no outgoing edges
            continue;
        } else if (strcmp(opcode, "GOTO") == 0) {
            // GOTO only goes to its target
            has_edge_to_target = (arg1 == target_line);
        } else if (strcmp(opcode, "IF") == 0) {
            // IF has two edges
            has_edge_to_target = (arg2 == target_line || line + 1 == target_line);
        } else {
            // Normal instructions fall through
            has_edge_to_target = (line + 1 == target_line);
        }
        
        if (has_edge_to_target) {
            dfs_backward(program, line, can_reach, visited);
        }
    }
}

// Check if HALT is reachable from all reachable code (CFG analysis)
bool cfg_reaches_halt(Program *program) {
    if (program->length <= 1) return true;
    
    // Allocate reachability arrays (stack allocation for small programs)
    bool forward_reach[32] = {false};   // Reachable from line 1
    bool backward_reach[32] = {false};  // Can reach HALT
    bool visited[32] = {false};
    
    if (program->length > 32) {
        // Shouldn't happen in practice, but safety check
        return true;
    }
    
    // Compute forward reachability from line 1
    dfs_forward(program, 1, forward_reach);
    
    // Compute backward reachability from HALT (last line)
    dfs_backward(program, program->length, backward_reach, visited);
    
    // Check if line 1 can reach HALT
    if (!backward_reach[0]) {
        return false;  // HALT is unreachable from start
    }
    
    // Check for dead code: any non-HALT line that is either:
    // 1. Not reachable from start (dead code), OR
    // 2. Reachable but cannot reach HALT (leads nowhere useful)
    for (int i = 0; i < program->length - 1; i++) {  // Exclude HALT itself
        if (!forward_reach[i]) {
            return false;  // Dead code: unreachable from start
        }
        if (forward_reach[i] && !backward_reach[i]) {
            return false;  // Reachable but cannot reach HALT
        }
    }
    
    return true;
}

// Start server process for a worker
bool start_server_process(WorkerContext *ctx) {
    // Create pipes for stdin and stdout
    if (pipe(ctx->stdin_pipe) == -1 || pipe(ctx->stdout_pipe) == -1) {
        perror("pipe");
        return false;
    }
    
    pid_t pid = fork();
    if (pid == -1) {
        perror("fork");
        return false;
    }
    
    if (pid == 0) {
        // Child process
        close(ctx->stdin_pipe[1]);  // Close write end
        close(ctx->stdout_pipe[0]); // Close read end
        
        // Redirect stdin and stdout
        dup2(ctx->stdin_pipe[0], STDIN_FILENO);
        dup2(ctx->stdout_pipe[1], STDOUT_FILENO);
        
        // Close unused file descriptors
        close(ctx->stdin_pipe[0]);
        close(ctx->stdout_pipe[1]);
        
        // Execute counter machine in server mode
        char max_steps_str[32];
        snprintf(max_steps_str, sizeof(max_steps_str), "%d", ctx->max_steps);
        
        execl(ctx->counter_machine_path, ctx->counter_machine_path, 
              "--server", "-s", max_steps_str, NULL);
        
        // If exec fails
        perror("execl");
        exit(1);
    }
    
    // Parent process
    close(ctx->stdin_pipe[0]);  // Close read end
    close(ctx->stdout_pipe[1]); // Close write end
    
    ctx->server_pid = pid;
    ctx->programs_since_restart = 0;
    
    // Convert file descriptors to FILE* streams for buffering
    ctx->stdin_stream = fdopen(ctx->stdin_pipe[1], "w");
    ctx->stdout_stream = fdopen(ctx->stdout_pipe[0], "r");
    
    if (!ctx->stdin_stream || !ctx->stdout_stream) {
        perror("fdopen");
        return false;
    }
    
    // Set line buffering
    setvbuf(ctx->stdin_stream, NULL, _IOLBF, 0);
    setvbuf(ctx->stdout_stream, NULL, _IOLBF, 0);
    
    // Set new restart threshold with jitter
    double jitter = 0.8 + (rand() / (double)RAND_MAX) * 0.4;  // 0.8 to 1.2
    ctx->restart_threshold = (int)(5000 * jitter);
    
    // Give the server a moment to start up
    usleep(10000);  // 10ms
    
    return true;
}

// Stop server process
void stop_server_process(WorkerContext *ctx) {
    if (ctx->server_pid > 0) {
        // Close streams first
        if (ctx->stdin_stream) {
            fclose(ctx->stdin_stream);
            ctx->stdin_stream = NULL;
        }
        if (ctx->stdout_stream) {
            fclose(ctx->stdout_stream);
            ctx->stdout_stream = NULL;
        }
        
        kill(ctx->server_pid, SIGKILL);
        waitpid(ctx->server_pid, NULL, 0);
        ctx->server_pid = 0;
    }
}

// Send program to server and get result
ExecutionResult run_program_on_server(WorkerContext *ctx, Program *program) {
    ExecutionResult result = {0};
    
    // Restart server periodically to prevent hangs
    if (ctx->programs_since_restart >= ctx->restart_threshold) {
        stop_server_process(ctx);
        if (!start_server_process(ctx)) {
            strcpy(result.error, "Failed to restart server");
            return result;
        }
    }
    
    // Send program to server
    // Write number of instructions
    fprintf(ctx->stdin_stream, "%d\n", program->length);
    
    // Write each instruction
    for (int i = 0; i < program->length; i++) {
        char instr[64];
        decode_instruction(program->instructions[i], instr, sizeof(instr));
        fprintf(ctx->stdin_stream, "%s\n", instr);
    }
    
    // Flush to ensure data is sent
    fflush(ctx->stdin_stream);
    
    // Read result from server
    char output[128];
    if (!fgets(output, sizeof(output), ctx->stdout_stream)) {
        strcpy(result.error, "Failed to read from server");
        return result;
    }
    
    // Parse result
    int result_value;
    if (sscanf(output, "%d", &result_value) != 1) {
        snprintf(result.error, sizeof(result.error), "Invalid output: %s", output);
        return result;
    }
    
    ctx->programs_since_restart++;
    
    if (result_value > 0) {
        // Halted successfully
        result.steps = result_value;
        result.exceeded_limit = false;
    } else if (result_value == -1) {
        // Exceeded step limit
        result.steps = 0;
        result.exceeded_limit = true;
        strcpy(result.error, "Step limit exceeded");
    } else if (result_value == -2) {
        // Out of range
        result.steps = 0;
        result.exceeded_limit = false;
        strcpy(result.error, "Out of range termination");
    } else {
        result.steps = 0;
        result.exceeded_limit = false;
        snprintf(result.error, sizeof(result.error), "Unknown result code: %d", result_value);
    }
    
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
            // Track filtered programs
            pthread_mutex_lock(&global_stats.lock);
            global_stats.programs_filtered_unreachable++;
            pthread_mutex_unlock(&global_stats.lock);
            
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
            // Track filtered programs
            pthread_mutex_lock(&global_stats.lock);
            global_stats.programs_filtered_infinite++;
            pthread_mutex_unlock(&global_stats.lock);
            
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
        
        // CFG-based reachability check
        if (!cfg_reaches_halt(program)) {
            // Track filtered programs
            pthread_mutex_lock(&global_stats.lock);
            global_stats.programs_filtered_cfg++;
            pthread_mutex_unlock(&global_stats.lock);
            
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
    
    // Start server process
    if (!start_server_process(ctx)) {
        fprintf(stderr, "Worker %d: Failed to start server\n", ctx->worker_id);
        return NULL;
    }
    
    // Allocate program buffer
    Program program;
    program.length = ctx->num_instructions + 1;  // +1 for HALT
    program.instructions = malloc(program.length * sizeof(EncodedInstruction));
    
    // Process programs from work queue
    while (queue_pop(ctx->work_queue, &program)) {
        if (interrupted) break;
        
        // Run program
        ExecutionResult result = run_program_on_server(ctx, &program);
        
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
                printf("Progress: %lu tested - best: %d steps\n", 
                       global_stats.programs_tested, ctx->best->steps);
            }
            pthread_mutex_unlock(&global_stats.lock);
        }
        
        // Note: Do NOT free_program(&program) here - we're reusing the buffer
    }
    
    // Cleanup
    free_program(&program);  // Free the reused buffer once at the end
    stop_server_process(ctx);
    
    return NULL;
}

int main(int argc, char *argv[]) {
    // Parse arguments
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <num_instructions> [--max-register N] [--max-steps N] [--counter-machine PATH]\n", argv[0]);
        return 1;
    }
    
    int num_instructions = atoi(argv[1]);
    int max_register = 3;
    int max_steps = 100000;
    char counter_machine_path[512] = "../countermachine";
    int num_workers = sysconf(_SC_NPROCESSORS_ONLN);
    
    // Parse optional arguments
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--max-register") == 0 && i + 1 < argc) {
            max_register = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--max-steps") == 0 && i + 1 < argc) {
            max_steps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--counter-machine") == 0 && i + 1 < argc) {
            strncpy(counter_machine_path, argv[++i], sizeof(counter_machine_path) - 1);
        } else if (strcmp(argv[i], "--workers") == 0 && i + 1 < argc) {
            num_workers = atoi(argv[++i]);
        }
    }
    
    // Validate
    if (num_instructions < 1) {
        fprintf(stderr, "Error: Number of instructions must be at least 1\n");
        return 1;
    }
    
    if (access(counter_machine_path, X_OK) != 0) {
        fprintf(stderr, "Error: Counter machine executable not found at %s\n", counter_machine_path);
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
    printf("BUSY BEAVER BRUTE FORCE SEARCH (C VERSION - PRODUCER-CONSUMER)\n");
    printf("======================================================================\n");
    printf("Instructions (excluding HALT): %d\n", num_instructions);
    printf("Max register: %d\n", max_register);
    printf("Max steps per program: %d\n", max_steps);
    printf("Parallel workers: %d\n", num_workers);
    printf("Work queue size: %d\n", QUEUE_SIZE);
    printf("Counter machine: %s\n", counter_machine_path);
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
        strncpy(workers[i].counter_machine_path, counter_machine_path, 
                sizeof(workers[i].counter_machine_path));
        workers[i].max_steps = max_steps;
        workers[i].max_register = max_register;
        workers[i].num_instructions = num_instructions;
        workers[i].best = &best;
        workers[i].work_queue = queue;
        workers[i].programs_tested = 0;
        workers[i].programs_timed_out = 0;
        workers[i].programs_out_of_range = 0;
        workers[i].programs_errored = 0;
        workers[i].server_pid = 0;
        workers[i].programs_since_restart = 0;
        
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
    
    // Print filter statistics
    pthread_mutex_lock(&global_stats.lock);
    unsigned long filtered_unreachable = global_stats.programs_filtered_unreachable;
    unsigned long filtered_infinite = global_stats.programs_filtered_infinite;
    unsigned long filtered_cfg = global_stats.programs_filtered_cfg;
    pthread_mutex_unlock(&global_stats.lock);
    
    unsigned long total_filtered = filtered_unreachable + filtered_infinite + filtered_cfg;
    printf("\nPrograms filtered (not tested):\n");
    printf("  - Unreachable HALT: %lu\n", filtered_unreachable);
    printf("  - Obvious infinite loops: %lu\n", filtered_infinite);
    printf("  - CFG dead code/unreachable: %lu\n", filtered_cfg);
    printf("  - Total filtered: %lu\n", total_filtered);
    
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
