import numpy as np
import matplotlib.pyplot as plt
import ballot_data as bd

def positional_voting(ballots, weights):
    """
    Calculate the total number of points obtained by each candidate using the
    positional voting system with the given weights.

    Parameters:
    - ballots: A NumPy array containing ballot data, one row per ballot.
    - weights: A list of weights for each preference rank.

    Returns:
    A list or NumPy array with the total points for each candidate.
    """
    # Check if the weights list has the same length as the number of candidates
    if len(weights) != ballots.shape[1]:
        raise ValueError("The length of weights must match the number of candidates.")

    # Initialize the results list with zeros
    results = np.zeros(ballots.shape[1])

    # For each rank, tally the preferences and add the weighted sum to results
    for rank, weight in enumerate(weights, start=1):
        preferences = np.sum(ballots == rank, axis=0)
        results += weight * preferences

    return results.tolist()



def display_results(ballots, weight_sets):
    """
    Generates a figure containing N subplots, each displaying a bar chart with
    the overall score for each candidate calculated with the positional voting
    method using a given set of weights.

    Parameters:
    - ballots: A NumPy array containing ballot data.
    - weight_sets: A list of N lists, each containing a different set of weights.

    The function displays the figure with the bar charts.
    """
    num_subplots = len(weight_sets)
    fig, axes = plt.subplots(num_subplots, 1, figsize=(10, num_subplots * 5))

    # If there's only one set of weights, axes will not be an array, so we wrap it
    if num_subplots == 1:
        axes = [axes]
    
    # Loop through each set of weights and plot the results
    for index, weights in enumerate(weight_sets):
        results = positional_voting(ballots, weights)
        axes[index].bar(range(len(results)), results, color='gray')
        # Highlight the winner with a different color
        winner_index = np.argmax(results)
        axes[index].bar(winner_index, results[winner_index], color='green')
        # Annotate the plot with the weight set used, formatting weights to 1 decimal place
        formatted_weights = [f'{w:.2f}' for w in weights]
        axes[index].set_title(f"Results using weights: {formatted_weights}")
        axes[index].set_xlabel('Candidates')
        axes[index].set_ylabel('Total Points')

    plt.tight_layout()
    plt.show()