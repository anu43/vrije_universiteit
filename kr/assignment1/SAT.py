# Import libraries
import os
import sys
import math
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


def dpll(clauses: list, assignments: list):
    # Check for the pure literals
    clauses, pure_assignments = pure_literal(clauses)
    # Check for the unit clauses
    clauses, unit_assignments = unit_prop(clauses)
    # Concat the assignments
    assignments = assignments + pure_assignments + unit_assignments
    # If the length of the literals is -1
    if clauses == -1:
        # Return an empty list
        return list()
    # Otherwise return the assignments
    if not clauses:
        return assignments

    # Select literal to move on
    lit = rand_var_selection(clauses)
    # Backtrack
    solution = dpll(modify_unit_clause(clauses, lit), assignments + [lit])
    # If there is not a solution
    if not solution:
        # Try with assignment polar form of the literal
        solution = dpll(modify_unit_clause(clauses, -lit), assignments + [-lit])
    # Return solution
    return solution


def print_sudoku_solution(solutions: list):
    # Declare an empty dict
    solution: dict = dict()
    # Convert solution list to a dict
    for sol in solutions:
        # If literal has NOT sign
        if sol[0] < 0:
            solution[sol[0]] = False
        # Otherwise
        else:
            solution[sol[0]] = True
    # Sort the solution
    solution: list = [
        *sorted(filter(lambda _variable: solution[_variable], solution))
    ]
    # Declare the square root of solution
    sqrt: float = math.sqrt(len(solution))
    # If it is square
    if sqrt.is_integer():
        # Print the solution by iterating by row
        for row in range(int(sqrt)):
            print(solution[:int(sqrt)])
            solution = solution[int(sqrt):]


if __name__ == '__main__':
    # Declare file name
    file_name = sys.argv[2]
    # Read file and get the clauses
    clauses: list = mxklabs.dimacs.read(file_name).clauses
    # Track time
    start = time.perf_counter()
    # Declare solver
    if sys.argv[1] == '-S1':
        solutions = dpll(clauses, list())
    # Track time
    print(f'Solved in {time.perf_counter() - start} seconds')
    # Print the sudoku solution
    print_sudoku_solution(solutions)
