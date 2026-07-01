import numpy as np

def find_index_sign_revert(data_list, epsilon=5e-3):
    """
    Finds the first index where the sign changes and persists for
    more than one element.

    Values with absolute magnitude smaller than epsilon are treated as having the same sign as the previous element, effectively ignoring small fluctuations around zero.

    Args:
        data_list (list): A list of numbers.
        epsilon (float): The threshold below which a value is considered to have "no sign change" relative to the previous value.

    Returns:
        int or None: The first index of a persistent sign change,
                     or None if no such change is found.
    """
    if len(data_list) < 3:
        return None

    effective_signs = [np.sign(data_list[0])]

    for i in range(1, len(data_list)):
        val = data_list[i]
        prev_sign = effective_signs[-1]

        if abs(val) < epsilon:
            effective_signs.append(prev_sign)
        else:
            effective_signs.append(np.sign(val))
    # print(effective_signs)

    # Now search for the persistent sign change using these cleaned signs
    for i in range(1, len(effective_signs) - 1):
        if effective_signs[i] != effective_signs[i-1]:
            if effective_signs[i+1] == effective_signs[i]:
                return i

    return None

def find_indices_sign_revert(data_list, epsilon=5e-3):
    """
    Finds ALL indices where the sign changes and persists for
    more than one element.

    Values with absolute magnitude smaller than epsilon are treated as having
    the same sign as the previous element, effectively ignoring small
    fluctuations around zero.

    Args:
        data_list (list): A list of numbers.
        epsilon (float): The threshold below which a value is considered to
                         have "no sign change" relative to the previous value.

    Returns:
        list of int: A list of indices where persistent sign changes occur.
                     Returns an empty list if no changes are found.
    """
    if len(data_list) < 3:
        return []

    # 1. Pre-process to create 'effective signs' based on epsilon
    effective_signs = [np.sign(data_list[0])]

    for i in range(1, len(data_list)):
        val = data_list[i]
        prev_sign = effective_signs[-1]

        if abs(val) < epsilon:
            effective_signs.append(prev_sign)
        else:
            effective_signs.append(np.sign(val))

    # 2. Search for persistent sign changes
    found_indices = []

    # Iterate up to the second to last element (requires i+1 to exist)
    for i in range(1, len(effective_signs) - 1):
        # Check if current sign is different from previous
        if effective_signs[i] != effective_signs[i - 1]:
            # Check if current sign persists in the next element
            if effective_signs[i + 1] == effective_signs[i]:
                found_indices.append(i)

    return found_indices


# --- Example Usage ---
# data = [1, 1, -0.001]  #, -1, -1, 1, 1, 0.002, -1, -1]
# indices = find_indices_sign_revert(data, epsilon=0.01)
# print(f"Found indices: {indices}")
# Output: Found indices: [3, 5, 8]