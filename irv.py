import numpy as np
import ballot_data as bd


def find_least_popular(ballots, eliminated):
    """
    Finds the next candidate to eliminate in an IRV system with tie-breaking.

    Parameters:
    - ballots: A NumPy array containing ballot data.
    - eliminated: A list of indices of already eliminated candidates.

    Returns:
    The index of the candidate to eliminate next.
    """
    num_candidates = ballots.shape[1]
    remaining_candidates = set(range(num_candidates)) - set(eliminated)
    
    # Loop over each rank until we find a candidate to eliminate
    for rank in range(1, num_candidates + 1):
        # Tally preferences for the current rank only among remaining candidates
        preferences = bd.tally_preferences(ballots, rank)[list(remaining_candidates)]
        
        # Find the least popular candidate(s) at the current rank
        min_votes = np.min(preferences)
        candidates_with_min_votes = np.where(preferences == min_votes)[0]
        
        # Adjust the indices of candidates_with_min_votes based on the remaining candidates
        candidates_with_min_votes = [list(remaining_candidates)[i] for i in candidates_with_min_votes]
        
        # If there's only one candidate with the minimum votes, return them
        if len(candidates_with_min_votes) == 1:
            return candidates_with_min_votes[0]
        else:
            # If there's a tie, only consider these candidates in the next rank
            remaining_candidates = set(candidates_with_min_votes)
    
    # If all preferences are tied, return the candidate with the smallest index
    return min(remaining_candidates)



def update_ballots(ballots, to_eliminate):
    """
    Modifies ballots to eliminate a candidate and adjust remaining rankings.

    Parameters:
    - ballots (numpy.ndarray): 2D array with rows as ballots and columns as candidate rankings.
    - to_eliminate (int): Index of the candidate to be eliminated.

    Returns:
    - numpy.ndarray: The updated ballots array.
    """
    # Iterate over each ballot
    for ballot in ballots:
        # Check if the eliminated candidate is ranked
        if ballot[to_eliminate] != 0:
            # Find the rank of the eliminated candidate
            rank_of_eliminated = ballot[to_eliminate]
            # Set the rank of the eliminated candidate to 0
            ballot[to_eliminate] = 0
            # Upgrade the rank of candidates who were ranked lower than the eliminated candidate
            ballot[ballot > rank_of_eliminated] -= 1
    return ballots

def calculate_results(ballots):
    """
    Calculates the winner in an IRV election and the order of elimination of candidates.

    Iterates through the election process by repeatedly finding and eliminating the least popular
    candidate and updating the ballots until one candidate remains.

    Parameters:
    - ballots (numpy.ndarray): 2D array with rows as ballots and columns as candidate rankings.

    Returns:
    - winner (int): The index of the winning candidate.
    - eliminated (list of int): Indices of eliminated candidates in the order they were eliminated.
    """
    num_candidates = ballots.shape[1]
    eliminated = []

    # Continue until we have only one candidate left
    while len(eliminated) < num_candidates - 1:
        # Find the least popular candidate not yet eliminated
        least_popular = find_least_popular(ballots, eliminated)
        # Add this candidate to the list of eliminated candidates
        eliminated.append(least_popular)
        # Update ballots to remove the eliminated candidate
        ballots = update_ballots(ballots, least_popular)
    
    # The winner is the last candidate remaining
    winner = set(range(num_candidates)) - set(eliminated)
    winner = winner.pop()  # There will only be one element left, so pop it out

    return winner, eliminated



