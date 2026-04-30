import numpy as np

def compute_pmi(joint_counts, total_counts_x, total_counts_y, total_samples):
    # Calculate probabilities
    P_xy = joint_counts / total_samples
    P_x = total_counts_x / total_samples
    P_y = total_counts_y / total_samples
    
    # Calculate PMI: log(P(x,y) / (P(x) * P(y)))
    epsilon = 1e-10
    pmi = np.log2(P_xy / (P_x * P_y + epsilon) + epsilon)
    
    return float(pmi)