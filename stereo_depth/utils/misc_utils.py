"""Miscellaneous helpers - ported from MiscUtils.py."""
import numpy as np


def sift_features_to_array(sift_matches, kp1, kp2):
    """Convert OpenCV DMatch list + keypoints into Nx4 array [x1,y1,x2,y2]."""
    matched_pairs = []
    for m1 in sift_matches:
        pt1 = kp1[m1.queryIdx].pt
        pt2 = kp2[m1.trainIdx].pt
        matched_pairs.append([pt1[0], pt1[1], pt2[0], pt2[1]])
    return np.array(matched_pairs).reshape(-1, 4)
