# Import libraries
import os
import sys
import time
import mxklabs.dimacs
from collections import Counter


def modify_unit_clause(clauses: list, unit: int):
    return clauses


def pure_literal(clauses: list):
    # Declare the counter of literals
    counter: Counter = Counter([literal for clause in clauses for literal in clause])
    # Declare lists
    assignment: list = list()  # assignment
    pure_literals: list = list()  # pure literals
    # Iterate through counter to check whether there is a pure literal
    for literal, occur in counter.items():
        # If polar literal does not occur in counter
        if -literal not in counter:
            # Append the literal as pure literal
            pure_literals.append(literal)
    # Iterate through pure literals to unit check
    for pure in pure_literals:
        # Modify unit clauses
        clauses = modify_unit_clause(clauses, pure)
    # Assign pure literals
    assignment += pure_literals
    # Return clauses and assignment
    print(clauses)
    print(assignment)
    return clauses, assignment


def dpll(clauses: list, init_clauses: dict):
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
        solution = dpll(clauses, init_clauses)
