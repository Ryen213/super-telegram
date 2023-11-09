# 2 simple tests for STV -- you can copy/paste this code into the last cell in the notebook.
importlib.reload(stv);

# This election should elect candidates 3 and 2, and eliminate candidates 0 and 1.
ballots = np.loadtxt('testing/task7_1.txt')
elected, eliminated = stv.stv_results(ballots, seats=2)
assert elected == [3, 2]
assert eliminated == [0, 1]
print('Passed test 1.')

# This election should elect candidates 2 and 1, and eliminate candidate 3.
# Candidate 0 should also be eliminated by default after candidates 2 and 1 are elected,
# as there are only 2 seats available.
ballots = np.loadtxt('testing/task7_2.txt')
elected, eliminated = stv.stv_results(ballots, seats=2)
assert elected == [1, 2]
# Pass the test whether algorithm proceeded with the final elimination,
# or stopped early as soon as 2 candidates were elected
assert eliminated in [[0], [0, 3]]
print('Passed test 2.')


