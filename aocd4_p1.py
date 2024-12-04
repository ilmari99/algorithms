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
    file_path = 'aocd4-input.txt'
    weights = [0.7390609, 1.9268732, -1.8851099, -4.169684]

    # Read and convert to matrix
    matrix = read_and_convert_to_matrix(file_path)

    # Define convolution kernels
    kernel_1x4 = np.array([weights]).reshape(1, 4)
    kernel_4x4 = np.diag(weights)
    kernel_4x4_flipped = np.fliplr(kernel_4x4)
    kernel_4x1 = np.array([weights]).reshape(4,1)

    convolved_matrix_1x4 = convolve2d(matrix, kernel_1x4,"valid")
    convolved_matrix_4x4 = convolve2d(matrix, kernel_4x4,"valid")
    convolved_matrix_4x4_flipped = convolve2d(matrix, kernel_4x4_flipped,"valid")
    convolved_matrix_4x1 = convolve2d(matrix, kernel_4x1,"valid")

    # Count how many close to -17.7412584 and 0.7969594 values
    target_1 = -17.7412584
    target_2 = 0.7969594
    tolerance = 0.001

    num_correct = (
        np.sum(np.isclose(convolved_matrix_1x4, target_1, atol=tolerance)) +
        np.sum(np.isclose(convolved_matrix_4x4, target_1, atol=tolerance)) +
        np.sum(np.isclose(convolved_matrix_4x1, target_1, atol=tolerance)) +
        np.sum(np.isclose(convolved_matrix_4x4_flipped, target_1, atol=tolerance))
    )
    print(f"Count of values close to {target_1}: {num_correct}")

    num_reversed = (
        np.sum(np.isclose(convolved_matrix_1x4, target_2, atol=tolerance)) +
        np.sum(np.isclose(convolved_matrix_4x4, target_2, atol=tolerance)) +
        np.sum(np.isclose(convolved_matrix_4x1, target_2, atol=tolerance)) +
        np.sum(np.isclose(convolved_matrix_4x4_flipped, target_2, atol=tolerance))
    )
    print(f"Count of values close to {target_2}: {num_reversed}")

    total_num = num_correct + num_reversed
    print(f"Total count: {total_num}")
