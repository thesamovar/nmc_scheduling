import masked_nmf
import matrix_completion
import numpy as np
import pickle
import scipy
import scipy.sparse

def create_true_vote_matrix(data):
    talks = []
    people = []
    for i, d in enumerate(data['prefs']):
        talks += list(d)
        people += [i] * len(d)

    V = scipy.sparse.coo_matrix(([1] * len(talks), (talks, people))).tocsr()
    return V

def simulate_voting(true_votes, data, votes_per=10, voting_population=10000):
    # Use a brain-dead voting mechanism: present things at random such that each 
    # participant has to vote on 10 things, and those 10 things are uniformily 
    # distributed.

    vote_allocations = (np.arange(voting_population * votes_per) % true_votes.shape[0]).astype(np.int)
    np.random.shuffle(vote_allocations)

    people = []
    for i in range(voting_population):
        people += [i] * votes_per

    mask = scipy.sparse.coo_matrix(([1] * len(vote_allocations), (vote_allocations, people)))
    vals = []
    rows = []
    cols = []
    for i, j in zip(mask.row, mask.col):
        if true_votes[i, j]:
            vals.append(1)
            rows.append(i)
            cols.append(j)
    votes = scipy.sparse.coo_matrix((vals, (rows, cols)), shape=(true_votes.shape[0], voting_population)).tocsr()
    return mask, votes

def complete_matrix_naive(mask, votes):
    # Use a naive matrix completion mechanism, where we are assuming 
    # missing-at-random-ness
    score = np.asarray(votes.sum(1)) / np.asarray(mask.sum(1))
    inferred_votes = score.reshape((-1, 1)) * np.ones(mask.shape[1]).reshape((1, -1))
    for i, j in zip(mask.row, mask.col):
        inferred_votes[i, j] = votes[i, j]
    return inferred_votes

def complete_matrix_mf(mask, votes):
    # Use a matrix factorization scheme.
    inferred_votes = matrix_completion.pmf_solve(
        np.asarray(votes.todense()).astype(np.float), 
        np.asarray(mask.todense()).astype(np.float),
        20,
        1.0)
    return inferred_votes

def complete_matrix_nmf(mask, votes):
    # Use a nonnegative matrix factorization scheme
    votes_all = -np.ones(votes.shape)
    for i, j in zip(mask.row, mask.col):
        votes_all[i, j] = votes[i, j]
    
    A, F = masked_nmf.nmf(votes_all, 10)
    inferred_votes = A.dot(F)
    return inferred_votes


def evaluate_matrix(true_votes, inferred):
    return np.corrcoef(
        np.asarray(true_votes.sum(1))[:, 0], 
        inferred.sum(1))[0, 1]


if __name__ == '__main__':
    # Load data from the 10k dataset
    with open('times_and_prefs_10k.pickle', 'rb') as f:
        data = pickle.load(f)

    true_votes = create_true_vote_matrix(data)

    mask, votes = simulate_voting(true_votes, data, voting_population=2000, votes_per=25)
    assert votes.shape == (1000, 2000)

    inferred_votes_naive = complete_matrix_naive(mask, votes)
    inferred_votes_mf = complete_matrix_mf(mask, votes)
    inferred_votes_nmf = complete_matrix_nmf(mask, votes)

    print(evaluate_matrix(true_votes, inferred_votes_naive))
    print(evaluate_matrix(true_votes, inferred_votes_nmf))
    print(evaluate_matrix(true_votes, inferred_votes_mf))

    import matplotlib.pyplot as plt
    plt.subplot(121)
    plt.plot(np.asarray(true_votes.sum(1))[:, 0], 
        inferred_votes_naive.sum(1), '.'
    )
    plt.subplot(122)
    plt.plot(np.asarray(true_votes.sum(1))[:, 0], 
        inferred_votes_nmf.sum(1), '.'
    )
    plt.show()