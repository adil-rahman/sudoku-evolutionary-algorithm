# Sudoku Evolutionary Algorithm

The program `sudoky_solver.py` is an implementation of an evolutionary algorithm to solve sudoky puzzles.

## Input Sudoku Puzzle

The puzzle should be in a format where:
* Empty cells are full stops '.'
* The character '!' is a semantic column separator
*	The string '---!---!---' is a semantic row separator. <br>

See `example.txt` for an example of a Sudoky puzzle.

## Running the Program

The program takes two command line arguments. <br>
The first is the name of the file containing the pizzle, which should be in the same directory as the file. <br>
The second argument is the population size.

Ensure that you are running python 3 before running the program.

Example command to run this program is:<br>
`python sudoku_solver.py exampl.txt 100` <br>
Where the filename is `example.txt` and the population size is 100.

## Authors

* **Adil Rahman** - *Initial work* - His project.
