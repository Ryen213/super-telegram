import numpy as np

def generate_ballots(votes=100, candidates=6, target_results=[30, 27, 21, 13, 6, 3]):
    '''
    Generate random ballot data given a total number of ballots (votes),
    a total number of candidates, and a target probability distribution
    of preferences.
    The probabilities are randomly attributed to each candidate for each
    level of preference.
    
    Returns a NumPy array with shape (votes, candidates), where each row
    is one ranked-choice ballot for one voter, and each column corresponds
    to one candidate.
    '''
    # Initialise a random number generator
    rng = np.random.default_rng()
    
    # Set target probabilities for each stage (normalised)
    prob = np.array(target_results, dtype=float)
    prob = np.tile(prob, (candidates, 1))
    # shuffle probabilities so they're applied differently for each rank; add some noise too
    prob = np.abs(rng.permuted(prob, axis=1) + rng.normal(scale=2, size=prob.shape))
    
    # Create an empty array to store the ballots
    ballots = np.zeros((votes, candidates))
    
    # Create each ballot one after the other
    for v in range(votes):
        # Voter ranks at most "candidates" candidates; introduce "stages" for clarity
        stages = candidates
        for r in range(stages):
            
            # Generates rth preference for an arbitrary candidate (use normalised probabilities)
            chosen_candidate = rng.choice(candidates, p=prob[r, :]/prob[r, :].sum())
            
            if ballots[v, chosen_candidate-1] > 0:
                # Arbitrarily decide that voter is done if they choose the same candidate twice
                break
            else:
                # If they hadn't previously chosen that candidate, choose it as rth preference (r indexes from 0)
                ballots[v, chosen_candidate-1] = r + 1
    
    return ballots


def select_ballots(ballots, rank, candidate):
    '''
    Returns a selector for all ballots which have allocated a given rank to a given candidate.
    '''
    # Create a bool mask: look at one candidate's column,
    # and find all the rows where that candidate's rank is rank
    return ballots[:, candidate] == rank

def tally_preferences(ballots, rank):
    """
    Tally the number of preferences at the given rank for each candidate.

    Parameters:
    - ballots (np.ndarray): A 2D NumPy array with one row per ballot and one column per candidate.
    - rank (int): The rank of preference to tally.

    Returns:
    - np.ndarray: A 1D array with the count of preferences at the given rank for each candidate.
    """
    # Check if the provided rank is within the valid range
    if not (1 <= rank <= ballots.shape[1]):
        raise ValueError("Rank must be between 1 and the number of candidates.")
    
    # Tally the preferences
    preferences = np.sum(ballots == rank, axis=0)
    return preferences


