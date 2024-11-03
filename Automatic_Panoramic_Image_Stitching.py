import os
import cv2
import numpy as np

def sift(img):
    # Get the key points and descriptors
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)
    kp = np.float32([i.pt for i in kp])
    return kp, des

def mser(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(img_gray)

    # Convert MSER regions to keypoints for SIFT descriptor extraction
    keypoints = []
    for region in regions:
        x, y = np.mean(region, axis=0)
        keypoints.append(cv2.KeyPoint(x=x, y=y, size=1.0))  # Set a fixed size of 1.0

    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.compute(img_gray, keypoints)
    keypoints = np.float32([kp.pt for kp in keypoints])

    return keypoints, descriptors

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
        pt1 = (int(kp1[i][0]) + w1, int(kp1[i][1]))
        pt2 = (int(kp2[j][0]), int(kp2[j][1]))

        # draw a circle in the keypoints
        cv2.circle(img_matches, pt1, 5, (0, 0, 255), 1)
        cv2.circle(img_matches, pt2, 5, (0, 255, 0), 1)

        # draw a line between the keypoints
        cv2.line(img_matches, pt1, pt2, (255, 0, 0), 1)

    # Save the image
    cv2.imwrite(os.path.join('output', 'matches.jpg'), img_matches)

def Draw_matches_mser(matches, img1, img2, kp1, kp2):
    # Draw the matches linking the two images
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    img_matches = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
    img_matches[:h1, :w1] = img1
    img_matches[:h2, w1:] = img2
    for i, j in matches:
        pt1 = (int(kp1[i][0]) + w1, int(kp1[i][1]))
        pt2 = (int(kp2[j][0]), int(kp2[j][1]))

        # draw a circle in the keypoints
        cv2.circle(img_matches, pt1, 5, (0, 0, 255), 1)
        cv2.circle(img_matches, pt2, 5, (0, 255, 0), 1)

        # draw a line between the keypoints
        cv2.line(img_matches, pt1, pt2, (255, 0, 0), 1)

    # Save the image
    cv2.imwrite(os.path.join('output', 'matches_mser.jpg'), img_matches)


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

def create_mask(img1, img2,version):
    # Creates the mask using query and train images for blending the images,
    # using a gaussian smoothing window/kernel
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    lowest_width = min(w1, w2)
    smoothing_window_percent = 0.10 # consider increasing or decreasing[0.00, 1.00] 
    smoothing_window_size = max(100, min(smoothing_window_percent * lowest_width, 1000))
    offset = int(smoothing_window_size / 2)
    barrier = w1 - offset

    mask = np.zeros((max(h1, h2), w1 + w2))
    if version == "left_image":
        mask[:, barrier - offset : barrier + offset] = np.tile(
            np.linspace(1, 0, 2 * offset).T, (max(h1, h2), 1)
        )
        mask[:, : barrier - offset] = 1
    elif version == "right_image":
        mask[:, barrier - offset : barrier + offset] = np.tile(
            np.linspace(0, 1, 2 * offset).T, (max(h1, h2), 1)
        )
        mask[:, barrier + offset :] = 1
    
    mask_rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=mask.dtype)
    mask_rgb[:, :, 0] = mask
    mask_rgb[:, :, 1] = mask
    mask_rgb[:, :, 2] = mask
    
    return mask_rgb

def warp(img, H, output_size):
    height, width = output_size
    warped_image = np.zeros((height, width, 3), dtype=img.dtype)

    inv_H = np.linalg.inv(H)

    for i in range(height):
        for j in range(width):
            original_coord = inv_H @ np.array([j, i, 1]) 
            original_coord /= original_coord[2] 
            new_x, new_y = int(original_coord[0]), int(original_coord[1])
            if 0 <= new_x < img.shape[1] and 0 <= new_y < img.shape[0]:
                warped_image[i, j] = img[new_y, new_x]

    return warped_image

def blending_smoothing(img1, img2, H):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    height = max(h1, h2)
    width = w1 + w2

    panorama1 = np.zeros((height, width, 3))
    mask1 = create_mask(img1, img2, version="left_image")
    panorama1[0 : h1, 0 : w1, :] = img1
    panorama1 *= mask1
    mask2 = create_mask(img1, img2, version="right_image")
    panorama2 = warp(img2, H, (height, width)) * mask2
    result = panorama1 + panorama2

    # remove extra blackspace
    rows, cols = np.where(result[:, :, 0] != 0)
    min_row, max_row = min(rows), max(rows) + 1
    min_col, max_col = min(cols), max(cols) + 1

    result = result[min_row:max_row, min_col:max_col, :]
    return result

def image_stitching(img1, img2):
    print("1. Interest points detection & feature description by SIFT")
    kp1, des1 = sift(img2)
    kp2, des2 = sift(img1)
    print("2. Feature matching by SIFT features")
    matches = matching_features(des1, des2)
    Draw_matches(matches, img1, img2, kp1, kp2)
    print("3. RANSAC to find homography matrix H")
    H = homomat(kp1, kp2, matches)
    print("4. Warp image to create panoramic image")
    result_image = blending_smoothing(img1, img2, H)

    return result_image

def image_stitching_mser(img1, img2):
    print("1. Interest points detection & feature description by SIFT")
    kp1, des1 = mser(img2)
    kp2, des2 = mser(img1)
    print("2. Feature matching by SIFT features")
    matches = matching_features(des1, des2)
    Draw_matches_mser(matches, img1, img2, kp1, kp2)
    print("3. RANSAC to find homography matrix H")
    H = homomat(kp1, kp2, matches)
    print("4. Warp image to create panoramic image")
    result_image = blending_smoothing(img1, img2, H)

    return result_image

if __name__ == '__main__':
    # Read the images from data folder
    folder = 'my_data'
    prefix = 'Daikakuji'
    img1 = cv2.imread(os.path.join(folder, f'{prefix}1.jpg'))
    img2 = cv2.imread(os.path.join(folder, f'{prefix}2.jpg'))
    result_image = image_stitching(img1, img2)
    result_image_mser = image_stitching_mser(img1, img2)
    cv2.imwrite(os.path.join('output', f'{prefix}_result.jpg'), result_image)
    cv2.imwrite(os.path.join('output', f'{prefix}_mser_result.jpg'), result_image_mser)