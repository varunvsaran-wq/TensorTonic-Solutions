import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """

    _, counts = np.unique(y, return_counts=True)

    entropy = 0.0
    
    for count in counts:
        p = count / counts.sum()
        entropy += p * np.log2(p)
        
    return -entropy