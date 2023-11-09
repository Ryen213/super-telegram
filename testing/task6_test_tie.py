# A test which involves a tie for Task 6.
importlib.reload(irv);

# This election should elect candidate 3, and eliminate candidates 0, 4, 1, 2 in this order.
# If ties are not resolved in Task 4, the result will likely be 0, 1, 4, 2 instead, but this test should still pass and say so.
ballots = np.loadtxt('testing/task6_tie.txt')
elected, eliminated = irv.calculate_results(ballots)
assert elected == 3
try:
    assert eliminated == [0, 4, 1, 2]
    print('Passed test.')
except AssertionError:
    assert eliminated == [0, 1, 4, 2], 'Incorrect result.'
    print('Passed test with no-tie version of Task 4.')
