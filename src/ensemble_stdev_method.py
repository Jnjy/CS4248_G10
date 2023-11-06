import numpy as np

def ensemble_stdev_2d(*models_output):
    # Assuming you have three arrays: arr1, arr2, and arr3
    # Replace these with your actual data

    # Combine the three arrays into one 2D array
    matrix = np.array(list(models_output))
        
    mean_values = np.mean(matrix, axis=0)
    std_values = np.std(matrix, axis=0)
    
    # Initialize a matrix to store the results
    averaged_matrix = np.zeros(matrix[0].shape, dtype=float)

    # Iterate through the cells and find values within 2 standard deviations
    for i in range(matrix[0].shape[0]):
        for j in range(matrix[0].shape[1]):
            # TODO: Look into 1 SD
            lower_limit = mean_values[i, j] - 2 * std_values[i, j]
            upper_limit = mean_values[i, j] + 2 * std_values[i, j]

            index_values = [m[i][j] for m in matrix]
            valid_values = list(filter(lambda x: x >= lower_limit and x <= upper_limit, index_values))
            
            if len(valid_values) > 0:
                averaged_matrix[i, j] = np.mean(valid_values)
            else:
                averaged_matrix[i, j] = 0

    return averaged_matrix

# # Usage
# model_1 = [[-8.517149925231934, -9.662267684936523, -9.75690746307373, -9.64969253540039]]
# model_2 = [[-8.517149925231934, -9.662267684936523, -9.75690746307373, -9.64969253540039]]
# model_3 = [[800.517149925231934, -9.662267684936523, -9.75690746307373, -9.64969253540039]]

# ensemble_stdev_2d(model_1, model_2, model_3)