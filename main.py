import os
import cv2

sample = cv2.imread("SOCOFing/Altered/Altered-Hard/1__M_Left_ring_finger_Zcut.BMP")

best_score = 0
filename = None
image = None
kp1, kp2, mp = None, None, None

counter = 0
for file in [file for file in os.listdir("SOCOFing/Real")][:1000]:
    counter += 1  # Increment the counter
    if counter % 10 == 0:
        print("Processed files:", counter)

    fingerprint_image = cv2.imread("SOCOFing/Real/" + file)
    sift = cv2.SIFT_create()

    keypoints_1, descriptors_1 = sift.detectAndCompute(sample, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(fingerprint_image, None)

    # Check if descriptors are found
    if descriptors_1 is not None and descriptors_2 is not None:
        matches = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 10}, {}).knnMatch(descriptors_1, descriptors_2, k=2)

        match_points = []

        for p, q in matches:
            if p.distance < 0.34 * q.distance:
                match_points.append(p)

        keypoints = min(len(keypoints_1), len(keypoints_2))

        if len(match_points) / keypoints * 100 > best_score:
            best_score = len(match_points) / keypoints * 100
            filename = file
            image = fingerprint_image
            kp1, kp2, mp = keypoints_1, keypoints_2, match_points

            print("Best MATCH: " + filename)
            print("SCORE: " + str(best_score))

# Display the best match outside the loop
if image is not None:
    result = cv2.drawMatches(sample, kp1, image, kp2, mp, None)
    result = cv2.resize(result, None, fx=4, fy=4)
    cv2.imshow("Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No match found")