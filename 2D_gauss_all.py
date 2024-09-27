import numpy as np
import cv2
import os
import math
from imutils import contours

def get_data(folder_name):
    data = []
    for filename in os.listdir(folder_name):
        img = cv2.imread(os.path.join(folder_name, filename))
        resized = cv2.resize(img, (40, 40), interpolation=cv2.INTER_LINEAR)
        img = resized[13:27, 13:27]

        nx, ny, ch = img.shape
        img = img.reshape(nx * ny, ch)
        data.extend(img)
    return np.array(data)

def gaussian(x, mean, cov):
    det_cov = np.linalg.det(cov)
    cov_inv = np.linalg.inv(cov)
    diff = x - mean
    norm_factor = (2.0 * np.pi) ** (-x.shape[1] / 2.0) * (1.0 / np.sqrt(det_cov))
    exponent = -0.5 * np.sum(diff @ cov_inv * diff, axis=1)
    return norm_factor * np.exp(exponent)

def initialize(data, n_obs):
    mean = data[np.random.choice(data.shape[0], 1)]
    cov = np.eye(n_obs) * np.random.uniform(1, 255)
    return {'mean': mean, 'cov': cov}

def gmm(data, K):
    n_feat, n_obs = data.shape
    bound = 0.0001
    max_itr = 500
    parameters = [initialize(data, n_obs) for _ in range(K)]
    mix_c = np.ones(K) / K
    log_likelihoods = []
    
    for itr in range(max_itr):
        print(f"Iteration {itr}")
        
        # E-Step
        cluster_prob = np.zeros((n_feat, K))
        for cluster in range(K):
            cluster_prob[:, cluster] = gaussian(data, parameters[cluster]['mean'], parameters[cluster]['cov']) * mix_c[cluster]
        
        cluster_sum = np.sum(cluster_prob, axis=1)
        log_likelihood = np.sum(np.log(cluster_sum))
        log_likelihoods.append(log_likelihood)
        
        # Normalize probabilities
        cluster_prob /= cluster_sum[:, np.newaxis]
        
        Nk = np.sum(cluster_prob, axis=0)
        
        # M-Step
        for cluster in range(K):
            weighted_sum = np.sum(cluster_prob[:, cluster, np.newaxis] * data, axis=0)
            parameters[cluster]['mean'] = weighted_sum / Nk[cluster]
            diff = data - parameters[cluster]['mean']
            parameters[cluster]['cov'] = (cluster_prob[:, cluster] * diff.T @ diff) / Nk[cluster]
            mix_c[cluster] = Nk[cluster] / n_feat
        
        if len(log_likelihoods) > 1 and abs(log_likelihood - log_likelihoods[-2]) < bound:
            break
    
    return mix_c, parameters

def test(frame, K, weights, parameters, div, color, r):
    def compute_probabilities(img, K, weights, parameters):
        prob = np.zeros((img.shape[0], K))
        for cluster in range(K):
            prob[:, cluster] = weights[cluster] * gaussian(img, parameters[cluster]['mean'], parameters[cluster]['cov'])
        return prob.sum(axis=1)
    
    nx, ny, ch = frame.shape
    img = frame.reshape(nx * ny, ch)
    
    likelihood = compute_probabilities(img, K, weights, parameters)
    probabilities = likelihood.reshape(nx, ny)
    probabilities[probabilities > np.max(probabilities) / div] = 255
    
    output = np.zeros_like(frame)
    output[:, :, :] = probabilities[:, :, np.newaxis]
    blur = cv2.GaussianBlur(output, (3, 3), 5)
    edged = cv2.Canny(blur, 50, 255)
    
    cnts, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    (cnts_sorted, _) = contours.sort_contours(cnts, method="left-to-right")
    
    if cnts_sorted:
        hull = cv2.convexHull(cnts_sorted[0])
        (x, y), radius = cv2.minEnclosingCircle(hull)
        if radius > r:
            cv2.circle(frame, (int(x), int(y) - 1), int(radius + 1), color, 4)
    
    return frame

def main():
    green_train_data = get_data("green_train")
    orange_train_data = get_data("orange_train")
    yellow_train_data = get_data("yellow_train")

    mix_c, parameters = gmm(green_train_data, 4)
    np.save('weights_g.npy', mix_c)
    np.save('parameters_g.npy', parameters)
    mix_c, parameters = gmm(orange_train_data, 6)
    np.save('weights_o.npy', mix_c)
    np.save('parameters_o.npy', parameters)
    mix_c, parameters = gmm(yellow_train_data, 7)
    np.save('weights_y.npy', mix_c)
    np.save('parameters_y.npy', parameters)

    images = []
    video = "detectbuoy.avi"
    cap = cv2.VideoCapture(video)
    weights_o = np.load('weights_o.npy')
    parameters_o = np.load('parameters_o.npy')
    weights_g = np.load('weights_g.npy')
    parameters_g = np.load('parameters_g.npy')
    weights_y = np.load('weights_y.npy')
    parameters_y = np.load('parameters_y.npy')

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = test(frame, 4, weights_g, parameters_g, div=8.5, color=(0, 255, 0), r=9)
        frame = test(frame, 7, weights_y, parameters_y, div=9.5, color=(0, 255, 255), r=7)
        frame = test(frame, 6, weights_o, parameters_o, div=3.0, color=(0, 128, 255), r=7)

        images.append(frame)
        cv2.imshow("output", frame)
        cv2.waitKey(5)

    # Writing to a video
    source = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('2D_gauss_all.avi', source, 5.0, (640, 480))
    for image in images:
        out.write(image)
    out.release()
    cap.release()

if __name__ == "__main__":
    main()
