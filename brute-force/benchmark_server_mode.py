#!/usr/bin/env python3
"""
Quick benchmark to demonstrate server mode performance improvement.
Compares old method (spawn process per program) vs new server mode.
"""
import subprocess
import time
from pathlib import Path

# Test programs
test_programs = [
    ["INC 1", "DEC 1", "HALT"],
    ["IF 1 3", "INC 1", "HALT"],
    ["DEC 1", "INC 2", "HALT"],
    ["COPY 1 2", "CLR 1", "HALT"],
    ["GOTO 1", "HALT"],
] * 20  # 100 programs total

counter_machine = Path(__file__).parent.parent / "countermachine"

print("=" * 70)
print("SERVER MODE PERFORMANCE BENCHMARK")
print("=" * 70)
print(f"Testing {len(test_programs)} programs\n")

# Method 1: Old way - spawn process per program
print("Method 1: Spawn new process per program...")
start = time.time()
for program in test_programs:
    # Create temp files
    with open('/tmp/test_program.txt', 'w') as f:
        f.write('\n'.join(program) + '\n')
    with open('/tmp/test_registers.txt', 'w') as f:
        f.write('0\n')
    
    # Run counter machine
    subprocess.run(
        [str(counter_machine), '-s', '1000'],
        cwd='/tmp',
        capture_output=True,
        text=True
    )
old_time = time.time() - start
print(f"Time: {old_time:.3f} seconds\n")

# Method 2: New way - server mode
print("Method 2: Server mode (persistent process)...")
start = time.time()

# Start server
proc = subprocess.Popen(
    [str(counter_machine), '--server', '-s', '1000'],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    bufsize=1
)

for program in test_programs:
    # Send program to server
    proc.stdin.write(f"{len(program)}\n")
    for line in program:
        proc.stdin.write(f"{line}\n")
    proc.stdin.write("END\n")
    proc.stdin.flush()
    
    # Read result
    proc.stdout.readline()

proc.terminate()
proc.wait()

new_time = time.time() - start
print(f"Time: {new_time:.3f} seconds\n")

# Results
print("=" * 70)
print(f"SPEEDUP: {old_time / new_time:.1f}x faster with server mode")
print(f"Time saved: {old_time - new_time:.3f} seconds ({(1 - new_time/old_time)*100:.1f}% reduction)")
print("=" * 70)
