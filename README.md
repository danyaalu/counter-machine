# Counter Machine Simulator

A simple counter machine simulator written in C that executes programs using register-based operations.

## What is a Counter Machine?

A counter machine is a theoretical model of computation that uses a set of registers (counters) and basic operations to perform computations. This implementation supports dynamic register allocation and executes programs written in a simple assembly-like language.

## Building

Compile the program using gcc:

```bash
gcc -o countermachine main.c
```

## Running

To run a counter machine program:

```bash
./countermachine
```

The simulator will:
1. Load the program from `program.txt`
2. Load initial register values from `registers.txt`
3. Execute the program
4. Display the final register values

## Program Format

Programs are written in `program.txt` with one instruction per line. Register numbers are 1-based in the program file.

### Supported Instructions

- `IF <reg> <line>` - If register is 0, jump to line number; otherwise continue
- `DEC <reg>` - Decrement register (minimum value is 0)
- `INC <reg>` - Increment register
- `COPY <src> <dest>` - Copy value from source register to destination register
- `GOTO <line>` - Unconditional jump to line number
- `CLR <reg>` - Clear register (set to 0)
- `HALT` - Stop program execution

### Example Program

```
# Move value from R1 to R2
IF 1 5
DEC 1
INC 2
GOTO 1
HALT
```

This program transfers the value in register 1 to register 2.

## Register File Format

Create a `registers.txt` file with initial register values (one value per line):

```
5
0
10
```

This initializes R1=5, R2=0, R3=10. Registers not specified default to 0.

## Notes

- Line numbers in jump instructions (IF, GOTO) are 1-based
- Comments start with `#`
- Blank lines are ignored
- Registers are dynamically allocated as needed
- The program terminates when HALT is executed or the program counter goes out of range