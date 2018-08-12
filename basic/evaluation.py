##
## COMP90042 Web Search and Text Analysis
## Project
##
## File: evaluation.py
## Description: Some evaluation metrices functions.
##
## Team: Mainframe
## Members:
## Name         | Student ID
## Kuan QIAN    | 686464
## Zequn MA     | 696586
## Yueni CHANG  | 884622
##

def reciprocal_rank(answer, results):
    if answer not in results:
        return 0.0
    else:
        return 1.0 / (results.index(answer) + 1)

def is_partial_match(answer, target):
    if answer == target: return False
    return target.find(answer) >= 0 or answer.find(target) >= 0
