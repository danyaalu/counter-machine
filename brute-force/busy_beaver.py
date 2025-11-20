#!/usr/bin/env python3
import subprocess
import itertools
import os
import sys
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count
import tempfile
import shutil
import threading
import time
import signal
import random

# Available opcodes (excluding HALT which is always last)
OPCODES = ['IF', 'DEC', 'INC', 'COPY', 'GOTO', 'CLR']

# Global worker variables - each worker keeps its own server process
_worker_counter_machine = None
_worker_work_dir = None
_worker_max_steps = None
_worker_server_process = None
_worker_programs_run = 0
_worker_restart_interval = 5000  # Base restart interval
_worker_restart_threshold = None  # Actual threshold with jitter (set per worker)

# Encoding/Decoding functions for memory efficiency
def encode_instruction(instruction):
    parts = instruction.split()
    opcode = parts[0]
    
    opcode_map = {'IF': 0, 'DEC': 1, 'INC': 2, 'COPY': 3, 'GOTO': 4, 'CLR': 5, 'HALT': 6}
    opcode_bits = opcode_map[opcode]
    
    arg1 = int(parts[1]) if len(parts) > 1 else 0
    arg2 = int(parts[2]) if len(parts) > 2 else 0
    
    # Pack into single integer: opcode | arg1 | arg2
    return (opcode_bits << 16) | (arg1 << 8) | arg2

def decode_instruction(encoded):
    opcode_map = ['IF', 'DEC', 'INC', 'COPY', 'GOTO', 'CLR', 'HALT']
    
    opcode_bits = (encoded >> 16) & 0x7
    arg1 = (encoded >> 8) & 0xFF
    arg2 = encoded & 0xFF
    
    opcode = opcode_map[opcode_bits]
    
    if opcode == 'HALT':
        return 'HALT'
    elif opcode in ['IF', 'COPY']:
        return f"{opcode} {arg1} {arg2}"
    elif opcode in ['DEC', 'INC', 'CLR']:
        return f"{opcode} {arg1}"
    elif opcode == 'GOTO':
        return f"{opcode} {arg1}"

def encode_program(program):
    return tuple(encode_instruction(instr) for instr in program)

def decode_program(encoded_program):
    return [decode_instruction(enc) for enc in encoded_program]

def generate_instruction(opcode, line_num, max_register, total_lines):
    instructions = []
    
    if opcode == 'IF':
        # IF <register> <line_to_jump_to>
        for reg in range(1, max_register + 1):
            for jump_line in range(1, total_lines + 1):
                instructions.append(f"IF {reg} {jump_line}")
    
    elif opcode == 'DEC':
        # DEC <register>
        for reg in range(1, max_register + 1):
            instructions.append(f"DEC {reg}")
    
    elif opcode == 'INC':
        # INC <register>
        for reg in range(1, max_register + 1):
            instructions.append(f"INC {reg}")
    
    elif opcode == 'COPY':
        # COPY <source_register> <dest_register>
        for src in range(1, max_register + 1):
            for dst in range(1, max_register + 1):
                if src != dst:  # Don't copy to itself
                    instructions.append(f"COPY {src} {dst}")
    
    elif opcode == 'GOTO':
        # GOTO <line>
        for jump_line in range(1, total_lines + 1):
            instructions.append(f"GOTO {jump_line}")
    
    elif opcode == 'CLR':
        # CLR <register>
        for reg in range(1, max_register + 1):
            instructions.append(f"CLR {reg}")
    
    return instructions

def is_halt_reachable(program):
    """
    Check if HALT is reachable in the program.
    
    VERY conservative - only catches the most obvious case:
    - The last instruction before HALT is a GOTO that jumps away
    
    Note: This doesn't catch all unreachable HALT cases (like complex control flow),
    but it avoids false positives. Static analysis of reachability is hard!
    """
    if len(program) < 2:  # Need at least one instruction before HALT
        return True
    
    # Get the last instruction before HALT
    last_instruction = program[-2] if program[-1] == 'HALT' else program[-1]
    
    # Parse the instruction
    parts = last_instruction.split()
    opcode = parts[0]
    
    # Only check if the instruction right before HALT is GOTO
    # Even then, it might be reachable from earlier in the program
    # So we only reject if GOTO jumps to itself (creating a 1-instruction loop)
    if opcode == 'GOTO':
        jump_target = int(parts[1])
        goto_line = len(program) - 1  # Line number of this GOTO
        halt_line = len(program)
        
        # Only reject if GOTO jumps to itself (infinite 1-instruction loop)
        # or jumps to an earlier line (but not HALT)
        if jump_target == goto_line:
            return False  # GOTO to itself = infinite loop
        # If GOTO jumps to HALT, that's fine
        if jump_target == halt_line:
            return True
        # If GOTO jumps forward or backward (but not to itself), 
        # HALT might still be reachable from other paths
        # So we'll be conservative and say it's reachable
    
    return True

def has_obvious_infinite_loop(program):
    """
    Detect obvious infinite loops through static analysis.
    Returns True if the program definitely has an infinite loop.
    
    This is now VERY conservative - only catches the most obvious cases:
    1. Unconditional backward GOTO (always loops)
    2. Self-jumping GOTO (GOTO to itself)
    
    We avoid catching:
    - IF loops (they may terminate depending on register values)
    - Loops with INC/DEC (complex to analyze correctly)
    """
    
    num_instructions = len(program) - 1  # Exclude HALT
    
    # Only catch unconditional backward GOTO
    for i, instr in enumerate(program[:-1]):
        parts = instr.split()
        opcode = parts[0]
        current_line = i + 1
        
        if opcode == 'GOTO':
            target = int(parts[1])
            # GOTO to same line or earlier = definite infinite loop
            if target <= current_line:
                return True

    return False

def normalize_program(program, max_register):
    register_map = {}
    next_register = 1
    normalized = []
    
    for instruction in program:
        if instruction == 'HALT':
            normalized.append('HALT')
            continue
            
        parts = instruction.split()
        opcode = parts[0]
        
        # Process register references and map them to canonical names
        if opcode in ['IF', 'DEC', 'INC', 'CLR']:
            reg = int(parts[1])
            if reg not in register_map:
                register_map[reg] = next_register
                next_register += 1
            
            if opcode == 'IF':
                # IF <reg> <line>
                normalized.append(f"{opcode} {register_map[reg]} {parts[2]}")
            else:
                # DEC/INC/CLR <reg>
                normalized.append(f"{opcode} {register_map[reg]}")
                
        elif opcode == 'COPY':
            # COPY <src> <dst>
            src = int(parts[1])
            dst = int(parts[2])
            
            if src not in register_map:
                register_map[src] = next_register
                next_register += 1
            if dst not in register_map:
                register_map[dst] = next_register
                next_register += 1
                
            normalized.append(f"{opcode} {register_map[src]} {register_map[dst]}")
            
        elif opcode == 'GOTO':
            # GOTO <line> - no register references
            normalized.append(instruction)
    
    return tuple(normalized)

def filter_program(encoded_program):
    """
    Filter function for parallel optimization.
    Returns (encoded_program, should_keep, skip_reason) tuple.
    """
    # Decode for filtering
    program = decode_program(encoded_program)
    
    # Skip if HALT is unreachable
    if not is_halt_reachable(program):
        return (encoded_program, False, 'unreachable')
    
    # Skip if program has obvious infinite loop
    if has_obvious_infinite_loop(program):
        return (encoded_program, False, 'infinite')
    
    return (encoded_program, True, None)

def generate_all_programs(num_instructions, max_register, use_optimization=True, workers=None):
    total_lines = num_instructions + 1  # +1 for HALT
    
    # Generate all possible instructions for each position
    all_position_instructions = []
    
    for line_num in range(1, num_instructions + 1):
        position_instructions = []
        for opcode in OPCODES:
            position_instructions.extend(
                generate_instruction(opcode, line_num, max_register, total_lines)
            )
        all_position_instructions.append(position_instructions)
    
    # Generate all combinations (cartesian product)
    total_programs = 1
    for pos_instructions in all_position_instructions:
        total_programs *= len(pos_instructions)
    
    print(f"Total programs to test: {total_programs:,}")
    
    if not use_optimization:
        print(f"Optimizations DISABLED - generating all programs...")
        all_programs = []
        for program_tuple in itertools.product(*all_position_instructions):
            program = list(program_tuple)
            program.append('HALT')
            # Encode before storing
            encoded = encode_program(program)
            all_programs.append(encoded)
        print(f"Programs to test (no optimization): {len(all_programs):,}\n")
        return all_programs
    
    # Apply optimizations in parallel
    print(f"Applying optimizations in parallel with {workers or cpu_count()} workers...")
    
    # Generate all programs first
    print("Generating program combinations...")
    all_programs_raw = []
    for program_tuple in itertools.product(*all_position_instructions):
        program = list(program_tuple)
        program.append('HALT')
        # Encode before storing
        encoded = encode_program(program)
        all_programs_raw.append(encoded)
    
    print(f"Generated {len(all_programs_raw):,} programs, filtering in parallel...")
    
    # Filter in parallel
    programs_kept = []
    programs_skipped_unreachable = 0
    programs_skipped_infinite = 0
    
    with Pool(processes=workers) as pool:
        # Use imap_unordered for better performance with progress tracking
        chunk_size = 1000
        processed = 0
        
        for encoded_program, should_keep, skip_reason in pool.imap_unordered(
            filter_program, all_programs_raw, chunksize=chunk_size
        ):
            processed += 1
            
            if should_keep:
                programs_kept.append(encoded_program)
            elif skip_reason == 'unreachable':
                programs_skipped_unreachable += 1
            elif skip_reason == 'infinite':
                programs_skipped_infinite += 1
            
            # Progress update every 50,000 programs
            if processed % 50000 == 0:
                percent = (processed / len(all_programs_raw)) * 100
                print(f"  Filtering progress: {processed:,}/{len(all_programs_raw):,} ({percent:.1f}%) - "
                      f"kept: {len(programs_kept):,}")
    
    print(f"Programs after optimization: {len(programs_kept):,}")
    print(f"  - Skipped {programs_skipped_unreachable:,} with unreachable HALT")
    print(f"  - Skipped {programs_skipped_infinite:,} with obvious infinite loops")
    print()
    
    return programs_kept

def create_empty_registers(filepath):
    with open(filepath, 'w') as f:
        f.write('0\n')

def init_worker_pool(counter_machine_path, temp_dir_base, max_steps):
    """
    Initialize each worker with persistent server process.
    Each worker starts a counter machine in server mode and communicates via stdin/stdout.
    """
    global _worker_counter_machine, _worker_work_dir, _worker_max_steps, _worker_server_process, _worker_programs_run, _worker_restart_threshold
    
    _worker_counter_machine = counter_machine_path
    _worker_max_steps = max_steps
    _worker_programs_run = 0
    
    # Add jitter to restart interval (±20%) to prevent all workers restarting at once
    jitter = random.uniform(0.8, 1.2)
    _worker_restart_threshold = int(_worker_restart_interval * jitter)
    
    # Each worker gets its own persistent work directory
    _worker_work_dir = Path(tempfile.mkdtemp(dir=temp_dir_base, prefix='worker_'))
    
    # Start the server process
    _start_server_process()

def _start_server_process():
    """Start or restart the server process"""
    global _worker_server_process, _worker_programs_run, _worker_restart_threshold
    
    # Kill existing process if any
    if _worker_server_process is not None:
        try:
            _worker_server_process.kill()
            _worker_server_process.wait(timeout=1)
        except:
            pass
    
    # Start new server process
    _worker_server_process = subprocess.Popen(
        [str(_worker_counter_machine), '--server', '-s', str(_worker_max_steps)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,  # Line buffered
        cwd=_worker_work_dir
    )
    _worker_programs_run = 0
    
    # Reset threshold with new jitter for next restart
    jitter = random.uniform(0.8, 1.2)
    _worker_restart_threshold = int(_worker_restart_interval * jitter)

def run_program_server(encoded_program):
    """
    Run program using the worker's persistent server process.
    Periodically restarts server to prevent hangs (with jitter to avoid synchronized restarts).
    """
    global _worker_programs_run
    
    # Restart server periodically to prevent hangs (using jittered threshold)
    if _worker_programs_run >= _worker_restart_threshold:
        try:
            _start_server_process()
        except Exception as e:
            return (encoded_program, (0, False, f"Server restart failed: {e}"))
    
    program = decode_program(encoded_program)
    
    try:
        # Write program to server with timeout
        result = _send_program_to_server(program)
        _worker_programs_run += 1
        return (encoded_program, result)
    
    except Exception as e:
        # If anything goes wrong, restart server and try once more
        try:
            _start_server_process()
            result = _send_program_to_server(program)
            _worker_programs_run += 1
            return (encoded_program, result)
        except Exception as e2:
            return (encoded_program, (0, False, f"Server error: {e2}"))

def _send_program_to_server(program, timeout=5.0):
    """
    Send a program to the server and get result with timeout.
    Returns: (steps, exceeded_limit, error)
    """
    global _worker_server_process
    
    # Check if server is still alive
    if _worker_server_process.poll() is not None:
        raise Exception("Server process died")
    
    # Prepare program data
    program_data = f"{len(program)}\n"
    for instr in program:
        program_data += f"{instr}\n"
    program_data += "END\n"
    
    # Use threading to implement timeout
    result_container = [None]
    error_container = [None]
    
    def write_and_read():
        try:
            # Write program
            _worker_server_process.stdin.write(program_data)
            _worker_server_process.stdin.flush()
            
            # Read result
            output_line = _worker_server_process.stdout.readline()
            if not output_line:
                error_container[0] = "No output from server"
                return
            
            result_container[0] = output_line.strip()
        except Exception as e:
            error_container[0] = str(e)
    
    # Run with timeout
    thread = threading.Thread(target=write_and_read)
    thread.daemon = True
    thread.start()
    thread.join(timeout=timeout)
    
    if thread.is_alive():
        # Timeout - kill the server process
        try:
            _worker_server_process.kill()
        except:
            pass
        raise Exception(f"Server timeout after {timeout}s")
    
    if error_container[0]:
        raise Exception(error_container[0])
    
    if result_container[0] is None:
        raise Exception("No result received")
    
    # Parse numeric result
    try:
        result_value = int(result_container[0])
        
        if result_value > 0:
            # Halted successfully
            return (result_value, False, None)
        elif result_value == -1:
            # Exceeded step limit
            return (0, True, "Step limit exceeded")
        elif result_value == -2:
            # Out of range
            return (0, False, "Out of range termination")
        else:
            return (0, False, f"Unknown result code: {result_value}")
    
    except ValueError:
        return (0, False, f"Invalid output: {result_container[0]}")


def cleanup_worker():
    """Cleanup worker resources on exit"""
    global _worker_work_dir, _worker_server_process
    
    # Kill server process
    if _worker_server_process is not None:
        try:
            _worker_server_process.kill()
            _worker_server_process.wait(timeout=1)
        except:
            pass
    
    # Remove work directory
    if _worker_work_dir and _worker_work_dir.exists():
        shutil.rmtree(_worker_work_dir, ignore_errors=True)

def run_program(encoded_program, counter_machine_path, work_dir, max_steps=1000000):
    program_file = work_dir / 'program.txt'
    registers_file = work_dir / 'registers.txt'
    
    # Decode program before writing to file
    program = decode_program(encoded_program)
    
    # Write program to file
    with open(program_file, 'w') as f:
        f.write('\n'.join(program) + '\n')
    
    # Create empty registers file
    create_empty_registers(registers_file)
    
    try:
        # Run the counter machine with step limit (no timeout)
        result = subprocess.run(
            [str(counter_machine_path), '-s', str(max_steps)],
            cwd=work_dir,
            capture_output=True,
            text=True
        )
    
        # Parse output for step count
        output = result.stdout
        
        # Look for "Program halted after X steps" or "RESULT: halted after X steps"
        if "halted after" in output:
            parts = output.split("halted after")[1].split("steps")[0].strip()
            steps = int(parts)
            return steps, False, None
        elif "exceeded limit" in output:
            # Program hit the step limit - treat as infinite loop
            return 0, True, "Step limit exceeded"
        elif "out of range" in output:
            # Count this as termination, try to extract step count
            # The program doesn't print steps for out-of-range termination
            # We'll treat this as 0 or minimal steps
            return 0, False, "Out of range termination"
        else:
            return 0, False, "Unknown output format"
    
    except Exception as e:
        return 0, False, str(e)

def run_program_wrapper(args):
    """
    Wrapper function for multiprocessing.
    Creates a unique temporary directory for each worker process.
    """
    program, counter_machine_path, temp_dir_base, max_steps = args
    
    # Create a unique temporary directory for this process
    work_dir = Path(tempfile.mkdtemp(dir=temp_dir_base, prefix='worker_'))
    
    try:
        result = run_program(program, counter_machine_path, work_dir, max_steps)
        return (program, result)
    finally:
        # Cleanup temporary directory
        shutil.rmtree(work_dir, ignore_errors=True)

def format_program(encoded_program):
    program = decode_program(encoded_program)
    return '\n'.join(f"{i+1}: {instr}" for i, instr in enumerate(program))

def main():
    parser = argparse.ArgumentParser(
        description='Brute force search for busy beaver programs in counter machine'
    )
    parser.add_argument('num_instructions', type=int,
                        help='Number of instructions (excluding HALT)')
    parser.add_argument('--max-register', type=int, default=3,
                        help='Maximum register number to use (default: 3)')
    parser.add_argument('--max-steps', type=int, default=100000,
                        help='Maximum steps before considering infinite loop (default: 100000)')
    parser.add_argument('--counter-machine', type=str, default='../countermachine',
                        help='Path to counter machine executable (default: ../countermachine)')
    parser.add_argument('--workers', type=int, default=cpu_count(),
                        help=f'Number of parallel workers (default: {cpu_count()} - all CPU cores)')
    parser.add_argument('--no-optimization', action='store_true',
                        help='Disable all optimizations (test all possible programs)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.num_instructions < 1:
        print("Error: Number of instructions must be at least 1")
        sys.exit(1)
    
    # Setup paths
    work_dir = Path(__file__).parent
    counter_machine_path = (work_dir / args.counter_machine).resolve()
    
    if not counter_machine_path.exists():
        print(f"Error: Counter machine executable not found at {counter_machine_path}")
        print("Please compile the counter machine first with: gcc main.c -o countermachine")
        sys.exit(1)
    
    # Create temporary directory for worker processes
    temp_dir_base = work_dir / 'temp_work_dirs'
    temp_dir_base.mkdir(exist_ok=True)
    
    print(f"{'='*70}")
    print(f"BUSY BEAVER BRUTE FORCE SEARCH (SERVER MODE)")
    print(f"{'='*70}")
    print(f"Instructions (excluding HALT): {args.num_instructions}")
    print(f"Max register: {args.max_register}")
    print(f"Max steps per program: {args.max_steps:,}")
    print(f"Parallel workers: {args.workers}")
    print(f"Server restart interval: ~{_worker_restart_interval:,} programs (with jitter)")
    print(f"Optimizations: {'DISABLED' if args.no_optimization else 'ENABLED'}")
    print(f"Counter machine: {counter_machine_path}")
    print(f"{'='*70}\n")
    
    # Track the best program
    best_program = None
    best_steps = 0
    programs_tested = 0
    programs_timed_out = 0
    programs_out_of_range = 0
    programs_errored = 0
    
    # Generate and filter all programs (filtering done in parallel)
    use_optimization = not args.no_optimization
    all_programs = generate_all_programs(
        args.num_instructions, 
        args.max_register, 
        use_optimization,
        workers=args.workers
    )
    total_programs = len(all_programs)
    print(f"Testing {total_programs:,} programs with {args.workers} workers...\n")
    
    # Run programs in parallel with worker reuse
    try:
        with Pool(
            processes=args.workers,
            initializer=init_worker_pool,
            initargs=(counter_machine_path, temp_dir_base, args.max_steps)
        ) as pool:
            # Use larger chunksize to reduce inter-process communication overhead
            # For massive runs (100M+ programs), use much larger chunks
            if total_programs > 100_000_000:
                optimal_chunk = max(10000, min(50000, total_programs // (args.workers * 3)))
            elif total_programs > 1_000_000:
                optimal_chunk = max(2000, min(10000, total_programs // (args.workers * 5)))
            else:
                optimal_chunk = max(200, min(1000, total_programs // (args.workers * 10)))
            
            # Use imap_unordered for better performance (results come back as they complete)
            for program, (steps, exceeded_limit, error) in pool.imap_unordered(
                run_program_server, all_programs, chunksize=optimal_chunk
            ):
                programs_tested += 1
                
                if exceeded_limit:
                    programs_timed_out += 1
                elif error == "Out of range termination":
                    programs_out_of_range += 1
                elif error:
                    programs_errored += 1
                
                # Update best if this program ran longer
                if steps > best_steps:
                    best_steps = steps
                    best_program = program
                    print(f"\n{'*'*70}")
                    print(f"NEW BEST FOUND! Steps: {best_steps:,}")
                    print(f"{'*'*70}")
                    print(format_program(best_program))
                    print(f"{'*'*70}\n")
                
                # Progress update every 100 programs
                if programs_tested % 100 == 0:
                    percent = (programs_tested / total_programs) * 100
                    try:
                        print(f"Progress: {programs_tested:,}/{total_programs:,} ({percent:.1f}%) - "
                              f"best: {best_steps:,} steps "
                              f"(exceeded: {programs_timed_out}, out-of-range: {programs_out_of_range}, errors: {programs_errored})")
                    except BrokenPipeError:
                        # Silently exit if output pipe is closed (e.g., when piped to head)
                        sys.exit(0)
    
    except KeyboardInterrupt:
        print("\n\nSearch interrupted by user.")
    except BrokenPipeError:
        # Silently exit if output pipe is closed
        sys.exit(0)
    finally:
        # Cleanup temp directory
        shutil.rmtree(temp_dir_base, ignore_errors=True)
    
    # Final results
    print(f"\n{'='*70}")
    print(f"SEARCH COMPLETE")
    print(f"{'='*70}")
    print(f"Programs tested: {programs_tested:,}")
    print(f"Programs exceeded step limit: {programs_timed_out:,}")
    print(f"Programs went out of range: {programs_out_of_range:,}")
    print(f"Programs with errors: {programs_errored:,}")
    print(f"\nLONGEST RUNNING PROGRAM ({best_steps:,} steps):")
    print(f"{'='*70}")
    if best_program:
        print(format_program(best_program))
    else:
        print("No successful programs found.")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()
