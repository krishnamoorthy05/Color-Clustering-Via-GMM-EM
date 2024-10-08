import numpy as np
import cv2
import os
from imutils import contours

def get_data(folder_name):
    stack = []
    for filename in os.listdir(folder_name):
        image = cv2.imread(os.path.join(folder_name, filename))
        resized = cv2.resize(image, (40, 40), interpolation=cv2.INTER_LINEAR)
        cropped = resized[13:27, 13:27]

        ch = cropped.shape[2]
        reshaped = cropped.reshape(-1, ch)
        stack.extend(reshaped)
    
    return np.array(stack)

def gaussian(data, mean, cov):
    det_cov = np.linalg.det(cov)
    cov_inv = np.linalg.inv(cov)
    diff = data - mean
    norm_factor = (2.0 * np.pi) ** (-data.shape[1] / 2.0) * (1.0 / np.sqrt(det_cov))
    exponent = -0.5 * np.sum(diff @ cov_inv * diff, axis=1)
    return norm_factor * np.exp(exponent)

def initialize_parameters(data, K):
    n_feat, n_obs = data.shape
    parameters = []
    for _ in range(K):
        mean = data[np.random.choice(n_feat, 1)].flatten()
        cov = np.eye(n_obs) * np.random.uniform(1, 255)
        parameters.append({'mean': mean, 'cov': cov})
    return parameters

def gmm(data, K):
    n_feat, n_obs = data.shape
    bound = 0.0001
    max_itr = 500
    parameters = initialize_parameters(data, K)
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
            parameters[cluster]['cov'] = (cluster_prob[:, cluster] * (diff.T @ diff)) / Nk[cluster]
            mix_c[cluster] = Nk[cluster] / n_feat

        # Check convergence
        if len(log_likelihoods) > 1 and abs(log_likelihood - log_likelihoods[-2]) < bound:
            break

    return mix_c, parameters

def process_frame(frame, K, weights, parameters):
    nx, ny, ch = frame.shape
    img = frame.reshape(nx * ny, ch)
    
    prob = np.zeros((nx * ny, K))
    for cluster in range(K):
        prob[:, cluster] = weights[cluster] * gaussian(img, parameters[cluster]['mean'], parameters[cluster]['cov'])
    
    likelihood = prob.sum(axis=1)
    probabilities = likelihood.reshape(nx, ny)
    probabilities[probabilities > np.max(probabilities) / 9.5] = 255

    output = np.zeros_like(frame)
    output[:, :, 0] = probabilities
    output[:, :, 1] = probabilities
    output[:, :, 2] = probabilities
    blur = cv2.GaussianBlur(output, (3, 3), 5)
    
    edged = cv2.Canny(blur, 50, 255)
    cnts, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        (cnts_sorted, _) = contours.sort_contours(cnts, method="left-to-right")
        hull = cv2.convexHull(cnts_sorted[0])
        (x, y), radius = cv2.minEnclosingCircle(hull)
        if radius > 7:
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 4)

    return frame

def main():
    train_data = get_data("yellow_train")
    K = 7
    mix_c, parameters = gmm(train_data, K)
    np.save('weights_y.npy', mix_c)
    np.save('parameters_y.npy', parameters)

    cap = cv2.VideoCapture("detectbuoy.avi")
    images = []

    weights = np.load('weights_y.npy')
    parameters = np.load('parameters_y.npy', allow_pickle=True)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break    
        processed_frame = process_frame(frame, K, weights, parameters)
        images.append(processed_frame)
        cv2.imshow("Final output", processed_frame)
        cv2.waitKey(5)
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('2D_gauss_yellow.avi', fourcc, 5.0, (640, 480))
    for image in images:
        out.write(image)
    out.release()
    cap.release()

if __name__ == "__main__":
    main()
