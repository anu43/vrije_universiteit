# Import libraries
import os
import sys
import time
import mxklabs.dimacs


def dpll(clauses: list, init_clauses: dict,
         unprocessed: set):
    pass


if __name__ == '__main__':
    # Declare file name
    file_name = sys.argv[2]
    # Read file and get the clauses
    clauses: list = mxklabs.dimacs.read(file_name).clauses
    # Convert DIMACS format to set
    clauses: list = list(map(lambda clause: {*clause}, clauses))
    # Initiliaze literals with None
    init_clauses: set = dict.fromkeys(sorted({
        abs(literal) for clause in clauses for literal in clause
    }), None)
    # Declare solver
    if sys.argv[1] == '-S1':
        solution = dpll()
