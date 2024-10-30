import numpy as np
import cv2

def sift(img):
    # chage img to gray scale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)
    return kp, des

def matching_features(des1, des2):
    # find the first and second best matches
    ratio_dis = 0.6
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

def Draw_matches(matches,img1,img2, kp1, kp2):
    # Draw the matches linking the two images
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    img_matches = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
    img_matches[:h1, :w1] = img1
    img_matches[:h2, w1:] = img2
    for i, j in matches:
        pt1 = (int(kp1[i].pt[0]), int(kp1[i].pt[1]))
        pt2 = (int(kp2[j].pt[0]) + w1, int(kp2[j].pt[1]))

        color = np.random.randint(0, high=255, size=(3,)) # make visualization more colorful
        color = tuple([int(x) for x in color])

        cv2.line(img_matches, pt1, pt2, color, 1)

    cv2.imshow('Matches', img_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    

def image_stitching(img1, img2):
    # Get the key points and descriptors
    kp1, des1 = sift(img1)
    kp2, des2 = sift(img2)

    # Match the SIFT descriptors
    Draw_matches(matching_features(des1, des2),img1,img2,kp1,kp2)


if __name__ == '__main__':
    # Read the images from data folder
    img1 = cv2.imread('data/hill1.JPG')
    img2 = cv2.imread('data/hill2.JPG')
    image_stitching(img1, img2)