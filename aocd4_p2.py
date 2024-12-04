import numpy as np
from scipy.signal import convolve2d

def read_and_convert_to_matrix(file_path):
    """
    Reads a file with characters x, m, a, s and converts it to a matrix 
    with corresponding numerical values: x -> 1, m -> 2, a -> 3, s -> 4.
    """
    char_to_num = {'X': 1, 'M': 2, 'A': 3, 'S': 4}
    with open(file_path, 'r') as file:
        lines = file.readlines()
    matrix = np.array([[char_to_num[char] for char in line.strip()] for line in lines])
    return matrix.astype(np.float32)

if __name__ == "__main__":
    # File path to the text file
    file_path = 'aocd4-input.txt'  # Replace with your text file path
    weights = [ 0.35320646, 1.6758898, -2.6483374 ]

    # Read and convert to matrix
    matrix = read_and_convert_to_matrix(file_path)
    matrix = np.where(matrix == 1, 0, matrix)  # Remove 'x' characters (set to 0)

    # Define kernels for all "mas" patterns (forward and reverse)
    kernels = [
        np.diag(weights),                          # Forward diagonal
        np.diag(weights[::-1]),                    # Reverse diagonal
        np.fliplr(np.diag(weights)),               # Forward anti-diagonal
        np.fliplr(np.diag(weights[::-1]))          # Reverse anti-diagonal
    ]

    # Compute convolution for each kernel
    results = []
    for i, kernel in enumerate(kernels):
        convolved = convolve2d(matrix, kernel, mode="valid")
        results.append(convolved)

    # Detect matches (overlap "mas" patterns)
    threshold = -4.859267234802246
    masks = [np.isclose(result, threshold,0.001) for result in results]

    # Find overlapping matches (simultaneous `True` in two or more masks)
    combined_mask = np.zeros_like(masks[0], dtype=bool)
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            combined_mask |= np.logical_and(masks[i], masks[j])
    print(np.sum(combined_mask))
