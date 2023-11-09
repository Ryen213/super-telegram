# Another test for STV -- you can copy/paste this code into the last cell in the notebook.
importlib.reload(stv);

# This tests your algorithm on the same data as in the ballotbox.scot explainer.

ballots = np.loadtxt('testing/task7_glasgow.txt')
elected, eliminated = stv.stv_results(ballots, seats=4)
assert elected == [2, 3, 5, 4]
assert eliminated == [0, 1, 7, 6]
print('Passed test with Glasgow data.')

# To make sure you transfer surplus votes correctly, you can also make your function
# print out the first-preference vote tally (appropriately weighted) at every round,
# and check that you get the same numbers in each stage as on the webpage.
