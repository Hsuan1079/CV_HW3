import os
import random
import numpy as np
import cv2

def sift(img):
    # Get the key points and descriptors
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)
    kp = np.float32([i.pt for i in kp])
    return kp, des

def matching_features(des1, des2):
    # find the first and second best matches
    ratio_dis = 0.75
    matches = []
    for i, d1 in enumerate(des1):
        dis = []
        for j, d2 in enumerate(des2):
            distance = np.linalg.norm(d1 - d2) # Euclidean distance
            dis.append((distance, j))

        # Sort the distances by the first element of the tuple    
        dis.sort(key=lambda x: x[0])

        # Check if the first distance is less than the second distance
        if (dis[0][0] / dis[1][0]) < ratio_dis:
            matches.append((i, dis[0][1]))
    
    return matches

def Draw_matches(matches, img1, img2, kp1, kp2):
    # Draw the matches linking the two images
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    img_matches = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
    img_matches[:h1, :w1] = img1
    img_matches[:h2, w1:] = img2
    for i, j in matches:
        pt1 = (int(kp1[i][0]), int(kp1[i][1]))
        pt2 = (int(kp2[j][0]) + w1, int(kp2[j][1]))

        # draw a circle in the keypoints
        cv2.circle(img_matches, pt1, 5, (0, 0, 255), 1)
        cv2.circle(img_matches, pt2, 5, (0, 255, 0), 1)

        # draw a line between the keypoints
        cv2.line(img_matches, pt1, pt2, (255, 0, 0), 1)

    # Save the image
    cv2.imwrite(os.path.join('output', 'matches.jpg'), img_matches)
    
def find_homography(p1, p2):
    # compute the homography matrix
    A = []
    for (x, y), (xp, yp) in zip(p1, p2):
        A.append([-x, -y, -1, 0, 0, 0, x * xp, y * xp, xp])
        A.append([0, 0, 0, -x, -y, -1, x * yp, y * yp, yp])
    A = np.array(A)

    # Apply SVD
    U, S, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    # Normalize H
    H = H / H[-1, -1]

    return H
    
def homomat(kp1, kp2, matches):
    # RANSAC to find homography matrix H
    pts1 = np.float32([kp1[i] for (i,_)in matches])
    pts2 = np.float32([kp2[i] for (_,i) in matches])

    num = np.shape(pts1)[0]
    coord_1 = np.zeros((num, 3), dtype=float)
    coord_2 = np.zeros((num, 3), dtype=float)
    for i in range(num):
        coord_1[i] = np.array([pts1[i][0], pts1[i][1], 1])
        coord_2[i] = np.array([pts2[i][0], pts2[i][1], 1])
    
    S = 4 # at least 4 samples
    N = 2000 # iterate for N times
    best_inliers = 0
    best_H = np.zeros((3,3), dtype=float)
    threshold = 5

    # get the best homography matrix with smallest number of outliers
    for _ in range(N):
        # 1. sample S correspondences from the feature matching results
        inliners = 0
        sampleidx = np.random.choice(num, S, replace=False) # sample S index of points
        p1_s = pts1[sampleidx]
        p2_s = pts2[sampleidx]

        # 2. compute the homography matrix based on these sampled correspondences
        H = find_homography(p1_s, p2_s)

        # 3. check the number of inliers/outliers by a threshold
        for j in range(num):
            mapped_point = np.dot(H, coord_1[j])
            mapped_point /= mapped_point[2]

            dis = np.linalg.norm(coord_2[j,:2] - mapped_point[:2])
            if dis < threshold:
                inliners += 1
        
        if inliners > best_inliers:
            best_inliers = inliners
            best_H = H
    
    return best_H

def warp(image1, image2, H):
    # Combine 2 image
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    inv_H = np.linalg.inv(H)
    stitch_img = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)

    # Applying inv(H) on image 1
    for i in range(stitch_img.shape[0]):
        for j in range(stitch_img.shape[1]):
            # i for height, j for width
            coord2 = np.array([j, i, 1])
            coord1 = inv_H @ coord2
            coord1 /= coord1[2]
            coord1 = np.around(coord1[:2])
            new_j, new_i = int(coord1[0]), int(coord1[1]) 
            if 0 <= new_i < h1 and 0 <= new_j < w1:
                stitch_img[i][j] = image1[new_i][new_j]
    
    stitch_img[0:h2, 0:w2] = image2
    
    return stitch_img

def image_stitching(img1, img2):
    # print("1. Interest points detection & feature description by SIFT")
    kp1, des1 = sift(img2)
    kp2, des2 = sift(img1)
    # print("2. Feature matching by SIFT features")
    matches = matching_features(des1, des2)
    Draw_matches(matches, img2, img1, kp1, kp2)
    # print("3. RANSAC to find homography matrix H")
    H = homomat(kp1, kp2, matches)
    # print("4. Warp image to create panoramic image")
    result_image = warp(img2, img1,H)

    return result_image

if __name__ == '__main__':
    # Read the images from data folder
    folder = 'my_data'
    prefix = 'Daikakuji'
    img1 = cv2.imread(os.path.join(folder, f'{prefix}1.jpg'))
    img2 = cv2.imread(os.path.join(folder, f'{prefix}2.jpg'))
    result_image = image_stitching(img1, img2)
    cv2.imshow('Result Image', result_image)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()
    cv2.imwrite(os.path.join('output', f'{prefix}_result.jpg'), result_image)
    
