READ ME

The program sudoku_solver.py takes two command line parameters.
The first is the name of the file containing the puzzle which should be in the same directory as the file.
	The puzzle should be in the format where:
        	Empty cells are full stops '.',
        	The character '!'is a semantic column separator,
       		The string '---!---!---' is a semantic row separator. 
The second argument is the population size.

This python file is written in python 3.

An example command to run this program is:

> python sudoku_solver.py "example.txt" 100

Where the filename is "example.txt" and the population size is 100.