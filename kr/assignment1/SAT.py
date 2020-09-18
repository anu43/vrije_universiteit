# Import libraries
import os
import sys
import time
import random
import mxklabs.dimacs
from collections import Counter


def modify_unit_clause(clauses: list, unit: int):
    # Declare an empty list for modified version of clauses
    modified: list = list()
    # Iterate through clauses to check the unit literals
    for clause in clauses:
        # If unit is in clause
        if unit in clause:
            continue
        # If polar unit is in clause
        elif -unit in clause:
            # Take literals except -unit
            cla = [cla for cla in clause if cla != -unit]
            # If the length of the literals is 0
            if len(cla) == 0:
                # Then, return -1
                return -1
            # Append cla to modified list
            modified.append(cla)
        # If something else
        else:
            modified.append(clause)
    # Return modified version of clauses
    return modified


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


def unit_prop(clauses: list):
    # Declare an empty list for assignments
    assignments: list = list()
    # Search for unit clauses
    unit_clauses: list = [c for c in clauses if len(c) == 1]
    # Iterate through unit clauses
    while len(unit_clauses) > 0:
        # Assign the first one
        unit = unit_clauses[0]
        # Modify clauses
        clauses = modify_unit_clause(clauses, unit[0])
        # Append to the assignments list
        assignments.append([unit[0]])
        # If the length of the literals is -1
        if clauses == -1:
            # Return -1 and an empty assignment list
            return -1, list()
        # If there are clauses
        if not clauses:
            # Return clauses and the assignments
            return clauses, assignments
        # Assign new unit clause list
        unit_clauses = [c for c in clauses if len(c) == 1]
    # Return clauses and the assignments
    return clauses, assignments


def rand_var_selection(clauses):
    # Declare the counter of literals
    counter: Counter = Counter([literal for clause in clauses for literal in clause])
    # Return a random choice
    return random.choice(counter.keys())


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
