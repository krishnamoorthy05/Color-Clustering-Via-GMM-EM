import numpy as np
import cv2
import math
import os
import imutils
from imutils import contours

# Function to generate training data from images in a specified directory
def generate_data():
    stack = []
    for filename in os.listdir("yellow_train"):
        image = cv2.imread(os.path.join("yellow_train", filename))
        resized = cv2.resize(image, (40, 40), interpolation=cv2.INTER_LINEAR)
        cropped = resized[13:27, 13:27]
        
        green_channel = cropped[:, :, 1].ravel()
        red_channel = cropped[:, :, 2].ravel()
        combined_image = np.concatenate((green_channel, red_channel), axis=0)
        combined_image = combined_image.reshape(-1, 1)
        
        stack.extend(combined_image)
    
    return np.array(stack)

# Function to compute the Gaussian probability density function
def gaussian(x, mean, cov):
    n_feat = x.shape[0]
    cov_inv = 1 / cov[0]
    diff = x - mean
    N = (2.0 * np.pi) ** (-len(x[1]) / 2.0) * (1.0 / cov[0] ** 0.5) * \
        np.exp(-0.5 * np.sum(np.multiply(diff * cov_inv, diff), axis=1))
    return np.reshape(N, (n_feat, 1))

# Function to perform Gaussian Mixture Model clustering using Expectation-Maximization
def gmm(data, K):
    n_feat = data.shape[0]
    
    def initialize():
        mean = np.array([data[np.random.choice(n_feat)]], np.float64)
        cov = [np.random.randint(1, 255)]
        return {'mean': mean, 'cov': cov}
    
    bound = 0.0001
    max_itr = 1000
    parameters = [initialize() for _ in range(K)]
    cluster_prob = np.ndarray([n_feat, K], np.float64)
    mix_c = [1. / K] * K
    log_likelihoods = []
    
    # EM Algorithm
    for itr in range(max_itr):
        print(itr)
        for cluster in range(K):
            cluster_prob[:, cluster] = gaussian(data, parameters[cluster]['mean'], parameters[cluster]['cov']).flatten() * mix_c[cluster]
        
        cluster_sum = np.sum(cluster_prob, axis=1)
        log_likelihood = np.sum(np.log(cluster_sum))
        log_likelihoods.append(log_likelihood)
        
        cluster_prob /= np.tile(cluster_sum, (K, 1)).T
        Nk = np.sum(cluster_prob, axis=0)
        
        for cluster in range(K):
            new_mean = np.sum(cluster_prob[:, cluster] * data.T, axis=1) / Nk[cluster]
            diff = data - new_mean
            new_cov = np.dot(np.multiply(diff.T, cluster_prob[:, cluster]), diff) / Nk[cluster]
            parameters[cluster]['mean'] = new_mean
            parameters[cluster]['cov'] = new_cov
            mix_c[cluster] = Nk[cluster] / n_feat
        
        if len(log_likelihoods) >= 2 and np.abs(log_likelihood - log_likelihoods[-2]) < bound:
            break

    return mix_c, parameters

# Generate training data for GMM
data = generate_data()
K = 2
mix_c, parameters = gmm(data, K)

# Save the GMM weights and parameters
np.save('weights_1d_y.npy', mix_c)
np.save('parameters_1d_y.npy', parameters)

# Open video file and process frames
name = "detectbuoy.avi"
cap = cv2.VideoCapture(name)
images = []

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    test_image = frame
    nx, ny = test_image.shape[:2]
    
    green_channel = test_image[:, :, 1].ravel()
    red_channel = test_image[:, :, 2].ravel()
    combined_image = np.concatenate((green_channel, red_channel), axis=0)
    combined_image = combined_image.reshape(-1, 1)
    
    weights = np.load('weights_1d_y.npy')
    parameters = np.load('parameters_1d_y.npy', allow_pickle=True)
    prob = np.zeros((combined_image.shape[0], K))
    likelihood = np.zeros((combined_image.shape[0], K))
    
    for cluster in range(K):
        prob[:, cluster] = weights[cluster] * gaussian(combined_image, parameters[cluster]['mean'], parameters[cluster]['cov']).flatten()
    
    likelihood = prob.sum(axis=1)
    
    green_prob = likelihood[:nx*ny]
    red_prob = likelihood[nx*ny:]
    
    prob = green_prob + red_prob
    prob[red_prob > np.max(red_prob) / 3.0] = 255
    
    output = np.zeros_like(frame)
    output[:, :, 2] = np.reshape(prob, (nx, ny))
    output[:, :, 1] = np.reshape(prob, (nx, ny))
    
    cv2.imshow("Output", output)
    blur = cv2.medianBlur(output, 3)
    edged = cv2.Canny(blur, 50, 255)
    cnts, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts_sorted = sorted(cnts, key=cv2.contourArea, reverse=True)
    
    if cnts_sorted:
        hull = cv2.convexHull(cnts_sorted[0])
        (x, y), radius = cv2.minEnclosingCircle(hull)
        
        if radius > 7:
            cv2.circle(test_image, (int(x), int(y) - 1), int(radius + 1), 255, 4)

    cv2.imshow("Final Output", test_image)
    images.append(test_image)
    cv2.waitKey(5)

# Save the processed video
cap.release()

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('1D_gauss_yellow.avi', fourcc, 5.0, (640, 480))
for image in images:
    out.write(image)
    cv2.waitKey(10)

out.release()
