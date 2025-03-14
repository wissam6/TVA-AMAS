import numpy as np

def generate_random_preferences_matrix(num_alternatives, num_voters):
    return np.array([np.random.permutation(num_alternatives) for _ in range(num_voters)]).T
    
def print_side_by_side(A, B):
    # Convert A and B into their string representations line by line
    A_str = str(A).splitlines()
    B_str = str(B).splitlines()
    
    # Pad the shorter list so they have the same number of lines
    n_lines = max(len(A_str), len(B_str))
    A_str += [""] * (n_lines - len(A_str))
    B_str += [""] * (n_lines - len(B_str))
    
    # Compute the width of the left block for alignment
    left_width = max(len(line) for line in A_str)
    
    # Calculate the "middle" line
    mid_line_idx = n_lines // 2
    
    # Print line by line
    for i, (left_line, right_line) in enumerate(zip(A_str, B_str)):
        if i == mid_line_idx:
            # Only print " -> " in the middle line
            print(left_line.ljust(left_width) + " -> " + right_line)
        else:
            # Print spacing on non-middle lines
            print(left_line.ljust(left_width) + "    " + right_line)