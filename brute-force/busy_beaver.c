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
// static const char *OPCODES[] = {"IF", "DEC", "INC", "COPY", "GOTO", "CLR"};  // Not used in C version

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

// Best program tracking (shared across threads)
typedef struct {
    Program program;
    int steps;
    pthread_mutex_t lock;
} BestProgram;

// Worker thread data
typedef struct {
    int worker_id;
    char counter_machine_path[512];
    int max_steps;
    int max_register;
    int num_instructions;
    BestProgram *best;
    
    // Statistics
    unsigned long programs_tested;
    unsigned long programs_timed_out;
    unsigned long programs_out_of_range;
    unsigned long programs_errored;
    
    // Server process info
    pid_t server_pid;
    int stdin_pipe[2];
    int stdout_pipe[2];
    int programs_since_restart;
    int restart_threshold;
} WorkerContext;

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
    
    // Set new restart threshold with jitter
    double jitter = 0.8 + (rand() / (double)RAND_MAX) * 0.4;  // 0.8 to 1.2
    ctx->restart_threshold = (int)(5000 * jitter);
    
    return true;
}

// Stop server process
void stop_server_process(WorkerContext *ctx) {
    if (ctx->server_pid > 0) {
        kill(ctx->server_pid, SIGKILL);
        waitpid(ctx->server_pid, NULL, 0);
        close(ctx->stdin_pipe[1]);
        close(ctx->stdout_pipe[0]);
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
    char buffer[4096];
    int offset = 0;
    
    // Write number of instructions
    offset += snprintf(buffer + offset, sizeof(buffer) - offset, "%d\n", program->length);
    
    // Write each instruction
    for (int i = 0; i < program->length; i++) {
        char instr[64];
        decode_instruction(program->instructions[i], instr, sizeof(instr));
        offset += snprintf(buffer + offset, sizeof(buffer) - offset, "%s\n", instr);
    }
    
    offset += snprintf(buffer + offset, sizeof(buffer) - offset, "END\n");
    
    // Write to server
    ssize_t written = write(ctx->stdin_pipe[1], buffer, offset);
    if (written != offset) {
        strcpy(result.error, "Failed to write to server");
        return result;
    }
    
    // Read result from server
    char output[128];
    ssize_t bytes_read = read(ctx->stdout_pipe[0], output, sizeof(output) - 1);
    if (bytes_read <= 0) {
        strcpy(result.error, "Failed to read from server");
        return result;
    }
    
    output[bytes_read] = '\0';
    
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

// Generate next program (returns false when done)
typedef struct {
    InstructionList *position_instructions;
    int num_positions;
    int *indices;
    bool first;
} ProgramGenerator;

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
    
    // Generate program from current indices
    do {
        program->length = gen->num_positions + 1;  // +1 for HALT
        program->instructions = malloc(program->length * sizeof(EncodedInstruction));
        
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
            free_program(program);
            
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
            free_program(program);
            
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

// Worker thread function
void* worker_thread(void *arg) {
    WorkerContext *ctx = (WorkerContext*)arg;
    
    // Start server process
    if (!start_server_process(ctx)) {
        fprintf(stderr, "Worker %d: Failed to start server\n", ctx->worker_id);
        return NULL;
    }
    
    // Create program generator for this worker
    ProgramGenerator *gen = create_program_generator(ctx->num_instructions, ctx->max_register);
    
    Program program;
    while (next_program(gen, &program, true)) {
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
        
        free_program(&program);
    }
    
    // Cleanup
    stop_server_process(ctx);
    free_program_generator(gen);
    
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
    printf("BUSY BEAVER BRUTE FORCE SEARCH (C VERSION)\n");
    printf("======================================================================\n");
    printf("Instructions (excluding HALT): %d\n", num_instructions);
    printf("Max register: %d\n", max_register);
    printf("Max steps per program: %d\n", max_steps);
    printf("Parallel workers: %d\n", num_workers);
    printf("Counter machine: %s\n", counter_machine_path);
    printf("======================================================================\n\n");
    
    // Note: For simplicity, using single-threaded generation
    // Multi-threading would require work-stealing or chunked distribution
    WorkerContext ctx;
    ctx.worker_id = 0;
    strncpy(ctx.counter_machine_path, counter_machine_path, sizeof(ctx.counter_machine_path));
    ctx.max_steps = max_steps;
    ctx.max_register = max_register;
    ctx.num_instructions = num_instructions;
    ctx.best = &best;
    ctx.programs_tested = 0;
    ctx.programs_timed_out = 0;
    ctx.programs_out_of_range = 0;
    ctx.programs_errored = 0;
    ctx.server_pid = 0;
    ctx.programs_since_restart = 0;
    
    // Run single-threaded for now
    worker_thread(&ctx);
    
    // Print final results
    printf("\n======================================================================\n");
    printf("SEARCH COMPLETE\n");
    printf("======================================================================\n");
    printf("Programs tested: %lu\n", ctx.programs_tested);
    printf("Programs exceeded step limit: %lu\n", ctx.programs_timed_out);
    printf("Programs went out of range: %lu\n", ctx.programs_out_of_range);
    printf("Programs with errors: %lu\n", ctx.programs_errored);
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
    pthread_mutex_destroy(&best.lock);
    pthread_mutex_destroy(&global_stats.lock);
    
    return 0;
}
