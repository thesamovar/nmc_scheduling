#import masked_nmf
import collections
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

def simulate_voting_enhanced(true_votes, 
                             data, 
                             votes_per=10, 
                             voting_population=10000,
                             p_enhanced=.5):
    # Use a voting mechanism where we present with higher probability talks from 
    # the matching cluster.
    vote_allocations = (np.arange(voting_population * votes_per) % true_votes.shape[0]).astype(np.int)
    np.random.shuffle(vote_allocations)

    enhanced_votes = int(p_enhanced * votes_per)
    unenhanced_votes = votes_per - enhanced_votes

    people = []
    true_vote_allocations = []
    k = 0
    for i in range(voting_population):
        people += [i] * votes_per
        cluster = data['participant_clusters'][i]
        enhanced = np.where(data['talk_clusters'] == cluster)[0]
        np.random.shuffle(enhanced)

        true_vote_allocations += list(enhanced[:enhanced_votes])
        true_vote_allocations += list(vote_allocations[k:k+unenhanced_votes])
        k += unenhanced_votes

    vote_allocations = true_vote_allocations
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

def complete_matrix_mf(mask, votes, ncomponents=10):
    # Use a matrix factorization scheme.
    inferred_votes = matrix_completion.biased_mf_solve(
        np.asarray(votes.todense()).astype(np.float), 
        np.asarray(mask.todense()).astype(np.float),
        ncomponents,
        .1)

    return inferred_votes

def complete_matrix_nmf(mask, votes):
    # Use a nonnegative matrix factorization scheme
    votes_all = -np.ones(votes.shape)
    for i, j in zip(mask.row, mask.col):
        votes_all[i, j] = votes[i, j]
    
    A, F = masked_nmf.nmf(votes_all, ncomponents)
    inferred_votes = A.dot(F)
    return inferred_votes

def infer_total_votes(mask, votes, data):
    cluster_counts = dict(collections.Counter(list(data['talk_clusters'])))
    
    precision = 11

    # estimate the intrinsic quality of a paper based off of interest.
    upvote = np.arange(precision)
    upvote = upvote / upvote.sum()
    upvote = np.log(upvote)
    upvote[0] = -1000

    downvote_in_cluster = 1 - np.arange(precision) / (precision - 1) * .3
    downvote_in_cluster = downvote_in_cluster / downvote_in_cluster.sum()
    downvote_in_cluster = np.log(downvote_in_cluster)

    downvote_out_cluster = 1 - np.arange(precision) / (precision - 1) * .05
    downvote_out_cluster = downvote_out_cluster / downvote_out_cluster.sum()
    downvote_out_cluster = np.log(downvote_out_cluster)

    W = np.zeros((3, precision))
    W[0, :] = upvote
    W[1, :] = downvote_in_cluster
    W[2, :] = downvote_out_cluster

    quality_dists = np.zeros((mask.shape[0], 3))

    adjusted_votes = votes.copy()
    for i, j in zip(mask.row, mask.col):
        same_cluster = (data['talk_clusters'][i] == data['participant_clusters'][j])
        if votes[i, j]:
            quality_dists[i, 0] += 1
        elif same_cluster:
            quality_dists[i, 1] += 1
        else:
            quality_dists[i, 2] += 1

    true_q = quality_dists @ W
    true_q = true_q - true_q.max(axis=1, keepdims=True)
    true_q = np.exp(true_q)
    true_q = true_q / true_q.sum(axis=1, keepdims=True)
    adjusted_q = true_q @ (np.arange(precision) / (precision - 1))

    adjusted_counts = np.zeros(mask.shape[0])
    for i in range(mask.shape[0]):
        adjusted_counts[i] = cluster_counts[data['talk_clusters'][i]] * adjusted_q[i]
    
    return adjusted_counts

def evaluate_matrix(true_votes, inferred):
    return np.corrcoef(
        np.asarray(true_votes.sum(1)).squeeze(), 
        np.asarray(inferred.sum(1)).squeeze())[0, 1]


if __name__ == '__main__':
    # Load data from the 10k dataset
    with open('times_and_prefs_10k.pickle', 'rb') as f:
        data = pickle.load(f)

    true_votes = create_true_vote_matrix(data)

    mask, votes = simulate_voting(true_votes, data, voting_population=2000, votes_per=25)
    print(votes.getnnz())

    assert votes.shape == (1000, 2000)

    print(evaluate_matrix(true_votes, votes))

    inferred_votes_naive = complete_matrix_naive(mask, votes)
    print('Correlation, before enhancement')
    print(evaluate_matrix(true_votes, inferred_votes_naive))

    # Do label smoothing
    mask, votes = simulate_voting_enhanced(true_votes, data, voting_population=2000, votes_per=25, p_enhanced=1.0)
    print(votes.getnnz())
    inferred_votes_enhanced = complete_matrix_naive(mask, votes)
    print('Correlation, after enhancement')
    print(evaluate_matrix(true_votes, inferred_votes_enhanced))
    
    inferred_votes_corrected = infer_total_votes(mask, votes, data)
    print('Correlation, after Bayesian correction')
    print(evaluate_matrix(true_votes, inferred_votes_corrected.reshape((-1, 1))))

    #print(evaluate_matrix(true_votes, inferred_votes_nmf))

    import matplotlib.pyplot as plt
    plt.subplot(121)
    plt.plot(np.asarray(true_votes.sum(1))[:, 0], 
        inferred_votes_naive.sum(1), '.'
    )
    plt.subplot(122)
    plt.plot(np.asarray(true_votes.sum(1))[:, 0], 
        inferred_votes_enhanced.sum(1), '.'
    )
    plt.show()