import numpy as np

def pca(data: np.ndarray, k: int) -> np.ndarray:
    """
    Perform PCA and return the top k principal components.
    
    Args:
        data: Input array of shape (n_samples, n_features)
        k: Number of principal components to return
    
    Returns:
        Principal components of shape (n_features, k), rounded to 4 decimals.
        Each eigenvector's sign is fixed so its first non-zero element is positive.
    """
    # Step 1: Standardize the dataset
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    # Avoid division by zero: if std is 0, set to 1
    std = np.where(std == 0, 1, std)
    data_standardized = (data - mean) / std
    
    # Step 2: Compute covariance matrix
    n_samples = data_standardized.shape[0]
    cov_matrix = (data_standardized.T @ data_standardized) / (n_samples - 1)
    
    # Step 3: Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Step 4: Sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Step 5: Apply sign convention
    for i in range(eigenvectors.shape[1]):
        first_nonzero_idx = np.where(np.abs(eigenvectors[:, i]) > 1e-10)[0]
        if len(first_nonzero_idx) > 0:
            first_idx = first_nonzero_idx[0]
            if eigenvectors[first_idx, i] < 0:
                eigenvectors[:, i] *= -1
    
    # Step 6: Select top k principal components
    principal_components = eigenvectors[:, :k]
    
    # Step 7: Round to 4 decimal places
    principal_components = np.round(principal_components, 4)
    
    return principal_components
