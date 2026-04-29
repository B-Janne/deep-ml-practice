import numpy as np

def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
    if not a:
        return []
    try:
        return np.array(a).reshape(new_shape).tolist()
    except ValueError:  # Invalid reshape (size mismatch)
        return []
    except MemoryError:
        return []