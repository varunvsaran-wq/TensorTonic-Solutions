import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    
    # Write code here
    x = np.asarray(x)
    p = np.asarray(p)

    expected = x @ p
    if not np.isclose(p.sum(), 1.0, atol=10e-6):
        raise ValueError("got error")
    return expected
