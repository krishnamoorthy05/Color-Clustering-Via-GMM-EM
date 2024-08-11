import numpy as np
import math

def generate_data():
    # Generating synthetic data from three Gaussian distributions
    data_1 = np.random.normal(0, 2, (50, 1))
    data_2 = np.random.normal(3, 0.5, (50, 1))
    data_3 = np.random.normal(6, 3, (50, 1))
    data = np.concatenate((data_1, data_2, data_3), axis=0)
    return data

def gaussian(x, mean, cov):
    # Compute the Gaussian probability density function
    n_feat = x.shape[0]
    cov_inv = np.linalg.inv(cov)  # Compute the inverse of covariance matrix
    diff = x - mean
    N = (2.0 * np.pi) ** (-n_feat / 2.0) * (np.linalg.det(cov) ** (-0.5)) * \
        np.exp(-0.5 * np.sum(diff @ cov_inv * diff, axis=1))
    return N.reshape(-1, 1)

def gmm(data, K):
    n_feat = data.shape[1]
    n_obs = data.shape[0]

    def initialize():
        mean = data[np.random.choice(n_obs)]
        cov = np.random.rand(n_feat, n_feat) * 0.5 + 0.5  # Random covariance matrix
        cov = np.dot(cov, cov.T)  # Ensure positive semi-definite
        return {'mean': mean, 'cov': cov}
    
    bound = 0.0001
    max_itr = 1000

    parameters = [initialize() for _ in range(K)]
    mix_c = [1. / K] * K
    log_likelihoods = []

    for itr in range(max_itr):
        print(f"Iteration: {itr}")
        cluster_prob = np.zeros((n_obs, K))

        for cluster in range(K):
            cluster_prob[:, cluster] = gaussian(data, parameters[cluster]['mean'], parameters[cluster]['cov']).flatten() * mix_c[cluster]

        cluster_sum = np.sum(cluster_prob, axis=1)
        log_likelihood = np.sum(np.log(cluster_sum))
        log_likelihoods.append(log_likelihood)
        
        # Normalize probabilities
        cluster_prob /= cluster_sum[:, np.newaxis]

        Nk = np.sum(cluster_prob, axis=0)

        # Update means and covariances
        for cluster in range(K):
            weighted_data = cluster_prob[:, cluster, np.newaxis] * data
            new_mean = np.sum(weighted_data, axis=0) / Nk[cluster]
            diff = data - new_mean
            new_cov = (cluster_prob[:, cluster, np.newaxis] * diff).T @ diff / Nk[cluster]
            
            parameters[cluster]['mean'] = new_mean
            parameters[cluster]['cov'] = new_cov
            mix_c[cluster] = Nk[cluster] / n_obs

        if len(log_likelihoods) >= 2 and abs(log_likelihood - log_likelihoods[-2]) < bound:
            break
    
    return mix_c, parameters

# Generate synthetic data
data = generate_data()
K = 3
mix_c, parameters = gmm(data, K)

# Print the results
for i in range(K):
    print(f"Mean {i+1}:", parameters[i]['mean'])
    print(f"Covariance {i+1}:", parameters[i]['cov'])
    print(f"Mixing Coefficient {i+1}:", mix_c[i])
