import re
import os
import sys
import shutil
import random
import subprocess
import numpy as np
import pandas as pd
from typing import Union
from time import perf_counter


def insertSigOneByOneHeuristic(justification: str,
                               signatures: list) -> Union[str, bool]:
    # Iterate through signatures
    for signature in signatures:
        # If signature is not equal to the justification
        if justification[2] != signature[2]:
            # Return the classes from signature
            return signature[2]  # Classes
    # Trace
    print("Couldn't find any signature to search. Ending the program\n")
    # Return None
    return None


def insertSigTwoByTwoHeuristic(justification: str,
                               signatures: list) -> Union[str, bool]:
    # Initialize an empty signature str
    signs: str = ''
    # Array length checker
    arr: list = list()
    # If the length of the signatures is bigger than 2
    if len(signatures) > 2:
        # Take two signatures from the main list
        for signature in signatures:
            # If signature is not equal to the justification
            if justification[2] != signature[2]:
                # Track the process by the array checker
                arr.append(signature[2])
                # Add the signature to the signatures
                signs += signature[2] + '\n'
            # If the length is equal to 2
            if len(arr) == 2:
                # Return the signatures
                return signs
    # If the length is less than 3
    else:
        # Run as insertSigTwoByTwoHeuristic
        return insertSigTwoByTwoHeuristic(justification,
                                          signatures)


def insertSigRandomlyHeuristic(justification: str,
                               signatures: list) -> Union[str, bool]:
    # Infinite loop
    while True:
        # Pick a signature randomly
        signature = random.sample(signatures, 1)
        # If signature is not equal to the justification
        if justification[2] != signature[2]:
            # Return the classes from signature
            return signature[2]  # Classes
    # Trace
    print("Couldn't find any signature to search. Ending the program\n")
    # Return None
    return None


# Set a seed
random.seed(43)

# Declare results path
result_path = 'datasets/results/'
# If results folder exists
if os.path.isdir(result_path):
    # Delete it
    shutil.rmtree(result_path)
    # Create it again
    os.mkdir(result_path)
# Otherwise
else:
    # Create it
    os.mkdir(result_path)


# Declare the justification to search
justification = [
    'http://example.com/myOntology/Tutor',
    'http://www.w3.org/2000/01/rdf-schema#subClassOf',
    'http://example.com/myOntology/Professor'
    ]

# Decide on a method for the forgetter (check the papers of LETHE to understand the different options).
# The default is 1, I believe.
# 1 - ALCHTBoxForgetter
# 2 - SHQTBoxForgetter
# 3 - ALCOntologyForgetter
methopMap: dict = {
    '1': 'ALCHTBoxForgetter',
    '2': 'SHQTBoxForgetter',
    '3': 'ALCOntologyForgetter'
}

# Declare empty lists
runs: list = list()  # the number of run records
methods: list = list()  # method records
heuristics: list = list()  # heuristics records
subClassRuntimes: list = list()  # subClass runtime records
letheRuntimes: list = list()  # lethe runtime records

# Run for each method
for method in ['1', '2', '3']:
    # Run for each heuristic
    for heuristic in ['insertSigOneByOneHeuristic',
                      'insertSigTwoByTwoHeuristic',
                      'insertSigRandomlyHeuristic']:
        # Choose the ontology (in the OWL format) for which you want to explain the entailed subsumption relations.
        inputOntology = "datasets/university_v3.owl"

        # Choose the set of subclass for which you want to find an explanation.
        # this file can be generated using the second command (saveAllSubClasses)
        inputSubclassStatements = "./datasets/subClasses.nt"

        # If subClasses.nt exists
        if os.path.isfile(inputSubclassStatements):
            # Delete it
            os.remove(inputSubclassStatements)

        # Choose the ontology to which you want to apply forgetting. This can be the inputOntology, but in practise
        # should be a smaller ontology, e.g. created as a justification for a subsumption
        forgetOntology = "datasets/university_V2.owl"

        # Declare the runMore to end while loop
        runMore = True
        # Declare run counter
        run = 0

        # Run the application iteratively
        while runMore:
            # Increment counter by one
            run += 1
            # 1. PRINT ALL SUBCLASSES (inputOntology):
            # print all subClass statements (explicit and inferred) in the inputOntology
            # --> uncomment the following line to run this function
            # os.system('java -jar kr_functions.jar ' + 'printAllSubClasses' + " " + inputOntology)

            # 2. SAVE ALL SUBCLASSES (inputOntology):
            # save all subClass statements (explicit and inferred) in the inputOntology to file datasets/subClasses.nt
            # --> uncomment the following line to run this function
            # Track time
            start = perf_counter()
            os.system('java -jar kr_functions.jar ' + 'saveAllSubClasses' + " " + inputOntology)
            # Save the time passed
            subClassRuntime = perf_counter() - start
            # Print the running time
            print(f'EXPORTING subClasses took {round(subClassRuntime, 2)} sec in the {run}th run')
            # Declare empty lists for couple purposes
            signatures: list = list()  # Signatures

            # Read subClasses file
            with open(inputSubclassStatements, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    # Split variables of a line
                    line = line.split()
                    line = [re.sub(r'[<>]', '', l) for l in line]
                    # If first element contain owl:Thing
                    if ('owl:Thing)' in line[1]) or ('script' in line[2]):
                        # Continue
                        continue
                    # Save variables to the signature list
                    signatures.append((
                        line[0],  # subClass
                        line[1],  # relation
                        line[2]  # class
                    ))

            # Choose the symbols which you want to forget.
            signature_path = "datasets/signature.txt"

            # If signature.txt exists
            if os.path.isfile(signature_path):
                # Delete it
                os.remove(signature_path)

            # Try the insertSigOneByOneHeuristic Heuristic for signature creation
            signature = insertSigOneByOneHeuristic(justification, signatures)
            # If signature is None
            if signature is None:
                # Iterate through solutions
                for signature in signatures:
                    # Get the only element
                    signature = signatures[0]
                    # If the remaining signature matches with the justification
                    if signature[2] == justification[2]:
                        # Print the statement
                        print(f"KB |= {signature[0].split('/')[-1]} is subSumption of {signature[2].split('/')[-1]}")
                    # Otherwise
                    else:
                        print("Couldn't find a solution\n")
                        print(f'SIGNATURE: {signature}')
                        print(f'JUSTIFICATION: {justification}')

                # Close the program
                runMore = False
                # Save how many runs made
                runMade = run

                # 3. PRINT ALL EXPLANATIONS (inputOntology, inputSubclassStatements):
                # print explanations for each subClass statement in the inputSubclassStatements
                # --> uncomment the following line to run this function
                # os.system('java -jar kr_functions.jar ' + 'printAllExplanations' +
                #           " " + inputOntology + " " + inputSubclassStatements)

                # 4. SAVE ALL EXPLANATIONS (inputOntology, inputSubclassStatements):
                # save explanations for each subClass statement in the inputSubclassStatements to file datasets/exp-#.owl
                # --> uncomment the following line to run this function
                # os.system('java -jar kr_functions.jar ' + 'saveAllExplanations' +
                #           " " + inputOntology + " " + inputSubclassStatements)

            # Otherwise continue running
            else:
                with open(signature_path, 'w') as f:
                    f.write(f'{signature}')

                # For running LETHE forget command:
                # --> uncomment the following line to run this function
                # os.system('java -cp lethe-standalone.jar uk.ac.man.cs.lethe.internal.application.ForgettingConsoleApplication --owlFile ' +
                #           forgetOntology + ' --method ' + method + ' --signature ' + signature_path)
                # Track time
                start = perf_counter()
                subprocess.Popen('java -cp lethe-standalone.jar uk.ac.man.cs.lethe.internal.application.ForgettingConsoleApplication --owlFile ' +
                                 forgetOntology + ' --method ' + method + ' --signature ' + signature_path, shell=True, stdout=subprocess.DEVNULL).wait()
                # Save the time passed
                letheRuntime = perf_counter() - start

                # Append letheRuntime to the general recorder
                letheRuntimes.append(letheRuntime)
                # Append subClassRuntime to the general recorder
                subClassRuntimes.append(subClassRuntime)
                # Append method to the methods recorder
                methods.append(methopMap[method])
                # Append heuristic to the heuristic recorder
                heuristics.append(heuristic)
                # Append run to the number of run recorder
                runs.append(run)

                # Copy the result.owl file into
                # Declare destionation path
                dst = f'{result_path}run_{run}.owl'
                shutil.copyfile('result.owl', dst)
                # Delete the original result file
                os.remove('result.owl')

                # Assign the input and forget ontology for the following run
                inputOntology = dst
                forgetOntology = dst
                # And subClasses
                inputSubclassStatements = "./datasets/results/subClasses.nt"


# Create a dictionary containing the variables for a pandas frame
frame: dict = {
    'numberOfRuns': runs,
    'method': methods,
    'heuristic': heuristics,
    'subClassRunTimes': subClassRuntimes,
    'letheRunTimes': letheRuntimes
}

# Create a csv file
pd.DataFrame(data=frame).to_csv('datasets/results/runtimeTable.csv',
                                index=False)
