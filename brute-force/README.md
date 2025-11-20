# Busy Beaver Brute Force Search

This directory contains both Python and C implementations of a brute-force search for busy beaver programs in the counter machine.

## Files

- `busy_beaver.py` - Python implementation with multiprocessing support
- `busy_beaver.c` - C implementation (rewrite of the Python version)
- `Makefile` - Build file for the C version
- `benchmark_server_mode.py` - Performance benchmarking script

## Building the C Version

```bash
make
```

This will compile `busy_beaver.c` into an executable called `busy_beaver`.

## Usage

### Python Version

```bash
python3 busy_beaver.py <num_instructions> [OPTIONS]

Options:
  --max-register N      Maximum register number (default: 3)
  --max-steps N         Maximum steps before timeout (default: 100000)
  --counter-machine P   Path to counter machine executable (default: ../countermachine)
  --workers N           Number of parallel workers (default: CPU count)
  --no-optimization     Disable all optimizations
```

### C Version

```bash
./busy_beaver <num_instructions> [OPTIONS]

Options:
  --max-register N      Maximum register number (default: 3)
  --max-steps N         Maximum steps before timeout (default: 100000)
  --counter-machine P   Path to counter machine executable (default: ../countermachine)
  --workers N           Number of parallel workers (default: CPU count)
```

## Examples

Search with 2 instructions:
```bash
./busy_beaver 2
```

Search with 3 instructions, using 4 workers:
```bash
./busy_beaver 3 --workers 4
```

Search with custom settings:
```bash
./busy_beaver 3 --max-register 4 --max-steps 1000000
```

## How It Works

Both implementations:
1. Generate all possible counter machine programs of a given length
2. Apply static analysis filters to skip obvious infinite loops
3. Run each program using the counter machine executable in server mode
4. Track the program that runs for the most steps before halting
5. Periodically restart server processes to prevent hangs

The C version is currently single-threaded but maintains the same core logic as the Python version.

## Performance Notes

- The C version should be faster for program generation and filtering
- Server mode communication allows running many programs without process overhead
- Both versions periodically restart the server to prevent hangs
- Memory-efficient streaming generation (not fully implemented in C version yet)

## Requirements

- Python version: Python 3.6+
- C version: GCC with pthread support
- Both require the `countermachine` executable to be built first