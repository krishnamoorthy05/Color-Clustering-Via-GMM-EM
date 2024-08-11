import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from imutils import contours

# Function to generate training data from images in a specified directory
def generate_data():
    stack = []
    for filename in os.listdir("green_train"):
        image = cv2.imread(os.path.join("green_train", filename))
        resized = cv2.resize(image, (40, 40), interpolation=cv2.INTER_LINEAR)
        cropped = resized[13:27, 13:27]
        image = cropped[:, :, 1]  # Extract green channel
        ch = 1
        nx, ny = image.shape
        image = np.reshape(image, (nx * ny, ch))
        
        stack.extend(image)
        
    return np.array(stack)

# Function to compute the Gaussian probability density function
def gaussian(x, mean, cov):
    cov_inv = 1 / cov
    diff = x - mean
    n_features = x.shape[1]
    
    N = (2.0 * np.pi) ** (-n_features / 2.0) * (1.0 / cov ** 0.5) * \
        np.exp(-0.5 * np.sum(diff @ np.diag(cov_inv) * diff, axis=1))
    
    return N.reshape(-1, 1)

# Function to perform Gaussian Mixture Model clustering using Expectation-Maximization
def gmm(data, K):
    def initialize():
        mean = np.array([data[np.random.choice(data.shape[0])]])
        cov = [np.random.randint(1, 255)]
        return {'mean': mean, 'cov': cov}
    
    bound = 0.0001
    max_itr = 1000
    
    parameters = [initialize() for _ in range(K)]
    n_features = data.shape[1]
    cluster_prob = np.zeros((data.shape[0], K))
    mix_c = [1.0 / K] * K
    log_likelihoods = []

    for itr in range(max_itr):
        # E-step: Compute the probability of each data point belonging to each cluster
        for cluster in range(K):
            cluster_prob[:, cluster] = gaussian(data, parameters[cluster]['mean'], parameters[cluster]['cov']).flatten() * mix_c[cluster]
        
        # Normalize the probabilities and compute log-likelihood
        cluster_sum = np.sum(cluster_prob, axis=1)
        log_likelihood = np.sum(np.log(cluster_sum))
        log_likelihoods.append(log_likelihood)
        
        cluster_prob /= cluster_sum[:, np.newaxis]
        Nk = np.sum(cluster_prob, axis=0)
        
        # M-step: Update the parameters of the Gaussian Mixture Model
        for cluster in range(K):
            new_mean = np.sum(cluster_prob[:, cluster, np.newaxis] * data, axis=0) / Nk[cluster]
            diff = data - new_mean
            new_cov = np.sum(cluster_prob[:, cluster, np.newaxis] * (diff ** 2), axis=0) / Nk[cluster]
            parameters[cluster]['mean'] = new_mean
            parameters[cluster]['cov'] = new_cov
            mix_c[cluster] = Nk[cluster] / data.shape[0]
        
        # Check convergence
        if len(log_likelihoods) >= 2 and np.abs(log_likelihood - log_likelihoods[-2]) < bound:
            break

    return mix_c, parameters

# Generate training data for GMM
data = generate_data()
K = 3
mix_c, parameters = gmm(data, K)

# Save the GMM weights and parameters
np.save('weights_1d_g.npy', mix_c)
np.save('parameters_1d_g.npy', parameters)

# Open video file and process frames
name = "detectbuoy.avi"
cap = cv2.VideoCapture(name)
images = []

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Process each frame
    test_image = frame
    nx, ny = test_image.shape[:2]
    img = test_image[:, :, 1]
    img = np.reshape(img, (nx * ny, 1))
    
    # Load the trained GMM parameters
    weights = np.load('weights_1d_g.npy')
    parameters = np.load('parameters_1d_g.npy', allow_pickle=True)
    
    # Compute probabilities and likelihoods
    prob = np.zeros((nx * ny, K))
    likelihood = np.zeros((nx * ny, K))
    for cluster in range(K):
        prob[:, cluster] = weights[cluster] * gaussian(img, parameters[cluster]['mean'], parameters[cluster]['cov']).flatten()
    
    likelihood = prob.sum(axis=1)
    probabilities = np.reshape(likelihood, (nx, ny))
    probabilities[probabilities > np.max(probabilities) / 2.0] = 255
    
    # Create output image with detected colors
    output = np.zeros_like(frame)
    output[:, :, 1] = probabilities

    cv2.imshow("output", output)
    
    # Apply morphological operations to detect contours
    blur = cv2.medianBlur(output, 3)
    edged = cv2.Canny(blur, 20, 255)
    cnts, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts_sorted, _ = contours.sort_contours(cnts, method="right-to-left")
    
    # Draw contours and circles around detected objects
    if cnts_sorted:
        hull = cv2.convexHull(cnts_sorted[0])
        (x, y), radius = cv2.minEnclosingCircle(hull)
        
        if radius > 5:
            cv2.circle(test_image, (int(x), int(y) - 1), int(radius + 1), 255, 4)
    
    cv2.imshow("Final output", test_image)
    images.append(test_image)
    cv2.waitKey(9)

# Release resources and save the processed video
cap.release()

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('1D_gauss_green.avi', fourcc, 5.0, (640, 480))
for image in images:
    out.write(image)
    cv2.waitKey(10)

out.release()
