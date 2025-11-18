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

# Available opcodes (excluding HALT which is always last)
OPCODES = ['IF', 'DEC', 'INC', 'COPY', 'GOTO', 'CLR']

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
    if len(program) < 2:  # Need at least one instruction before HALT
        return True
    
    # Get the last instruction before HALT
    last_instruction = program[-2] if program[-1] == 'HALT' else program[-1]
    
    # Parse the instruction
    parts = last_instruction.split()
    opcode = parts[0]
    
    # GOTO always jumps, so check if it jumps away from HALT
    if opcode == 'GOTO':
        jump_target = int(parts[1])
        halt_line = len(program)  # HALT is at position len(program)
        # If GOTO jumps to anywhere except HALT line, HALT is unreachable
        if jump_target != halt_line:
            return False
    
    return True

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

def generate_all_programs(num_instructions, max_register):
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
    print(f"Applying optimizations...")
    
    # Track seen normalized programs to avoid duplicates
    seen_programs = set()
    programs_generated = 0
    programs_skipped_unreachable = 0
    programs_skipped_duplicate = 0
    
    for program_tuple in itertools.product(*all_position_instructions):
        program = list(program_tuple)
        program.append('HALT')  # Always add HALT as the final instruction
        
        # Skip if HALT is unreachable
        if not is_halt_reachable(program):
            programs_skipped_unreachable += 1
            continue
        
        # Skip if this is a duplicate (normalized form already seen)
        normalized = normalize_program(program, max_register)
        if normalized in seen_programs:
            programs_skipped_duplicate += 1
            continue
        
        seen_programs.add(normalized)
        programs_generated += 1
        yield program
    
    print(f"Programs after optimization: {programs_generated:,}")
    print(f"  - Skipped {programs_skipped_unreachable:,} with unreachable HALT")
    print(f"  - Skipped {programs_skipped_duplicate:,} duplicate/symmetric programs")
    print()

def create_empty_registers(filepath):
    with open(filepath, 'w') as f:
        f.write('0\n')

def run_program(program, timeout_seconds, counter_machine_path, work_dir):
    program_file = work_dir / 'program.txt'
    registers_file = work_dir / 'registers.txt'
    
    # Write program to file
    with open(program_file, 'w') as f:
        f.write('\n'.join(program) + '\n')
    
    # Create empty registers file
    create_empty_registers(registers_file)
    
    try:
        # Run the counter machine with timeout
        result = subprocess.run(
            [counter_machine_path],
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=timeout_seconds
        )
    
        # Parse output for step count
        output = result.stdout
        
        # Look for "Program halted after X steps"
        if "Program halted after" in output:
            parts = output.split("Program halted after")[1].split("steps")[0].strip()
            steps = int(parts)
            return steps, False, None
        elif "Program terminated because PC went out of range" in output:
            # Count this as termination, try to extract step count
            # The program doesn't print steps for out-of-range termination
            # We'll treat this as 0 or minimal steps
            return 0, False, "Out of range termination"
        else:
            return 0, False, "Unknown output format"
    
    except subprocess.TimeoutExpired:
        return 0, True, "Timeout"
    
    except Exception as e:
        return 0, False, str(e)

def run_program_wrapper(args):
    """
    Wrapper function for multiprocessing.
    Creates a unique temporary directory for each worker process.
    """
    program, timeout_seconds, counter_machine_path, temp_dir_base = args
    
    # Create a unique temporary directory for this process
    work_dir = Path(tempfile.mkdtemp(dir=temp_dir_base, prefix='worker_'))
    
    try:
        result = run_program(program, timeout_seconds, counter_machine_path, work_dir)
        return (program, result)
    finally:
        # Cleanup temporary directory
        shutil.rmtree(work_dir, ignore_errors=True)

def format_program(program):
    return '\n'.join(f"{i+1}: {instr}" for i, instr in enumerate(program))

def main():
    parser = argparse.ArgumentParser(
        description='Brute force search for busy beaver programs in counter machine'
    )
    parser.add_argument('num_instructions', type=int,
                        help='Number of instructions (excluding HALT)')
    parser.add_argument('--max-register', type=int, default=3,
                        help='Maximum register number to use (default: 3)')
    parser.add_argument('--timeout', type=float, default=0.5,
                        help='Timeout in seconds for each program (default: 0.5)')
    parser.add_argument('--counter-machine', type=str, default='../countermachine',
                        help='Path to counter machine executable (default: ../countermachine)')
    parser.add_argument('--workers', type=int, default=cpu_count(),
                        help=f'Number of parallel workers (default: {cpu_count()} - all CPU cores)')
    
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
    print(f"BUSY BEAVER BRUTE FORCE SEARCH (PARALLEL)")
    print(f"{'='*70}")
    print(f"Instructions (excluding HALT): {args.num_instructions}")
    print(f"Max register: {args.max_register}")
    print(f"Timeout per program: {args.timeout} seconds")
    print(f"Parallel workers: {args.workers}")
    print(f"Counter machine: {counter_machine_path}")
    print(f"{'='*70}\n")
    
    # Track the best program
    best_program = None
    best_steps = 0
    programs_tested = 0
    programs_timed_out = 0
    programs_errored = 0
    
    # Generate all programs first (needed for parallel processing)
    print("Generating programs...")
    all_programs = list(generate_all_programs(args.num_instructions, args.max_register))
    total_programs = len(all_programs)
    print(f"Testing {total_programs:,} programs with {args.workers} workers...\n")
    
    # Prepare arguments for each program
    program_args = [
        (program, args.timeout, counter_machine_path, temp_dir_base)
        for program in all_programs
    ]
    
    # Run programs in parallel
    try:
        with Pool(processes=args.workers) as pool:
            # Use imap_unordered for better performance (results come back as they complete)
            for program, (steps, timed_out, error) in pool.imap_unordered(
                run_program_wrapper, program_args, chunksize=10
            ):
                programs_tested += 1
                
                if timed_out:
                    programs_timed_out += 1
                elif error and error != "Out of range termination":
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
                    print(f"Progress: {programs_tested:,}/{total_programs:,} ({percent:.1f}%) - "
                          f"best: {best_steps:,} steps "
                          f"(timeouts: {programs_timed_out}, errors: {programs_errored})")
    
    except KeyboardInterrupt:
        print("\n\nSearch interrupted by user.")
    finally:
        # Cleanup temp directory
        shutil.rmtree(temp_dir_base, ignore_errors=True)
    
    # Final results
    print(f"\n{'='*70}")
    print(f"SEARCH COMPLETE")
    print(f"{'='*70}")
    print(f"Programs tested: {programs_tested:,}")
    print(f"Programs timed out: {programs_timed_out:,}")
    print(f"Programs errored: {programs_errored:,}")
    print(f"\nLONGEST RUNNING PROGRAM ({best_steps:,} steps):")
    print(f"{'='*70}")
    if best_program:
        print(format_program(best_program))
    else:
        print("No successful programs found.")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()
