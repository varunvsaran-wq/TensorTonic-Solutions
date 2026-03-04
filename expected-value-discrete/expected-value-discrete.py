import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    
    
    # Write code here
    x = np.asarray(x)
    p = np.asarray(p)

    if not np.isclose(p.sum(), 1.0, atol=10e-6):
        raise ValueError("probabilities don't sum to 1")
    
    expected = x @ p
    
    return expected
