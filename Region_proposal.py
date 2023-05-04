import cv2
import numpy as np


class Region_proposal():
    def __init__(self, THRESH):
        self.thresh = THRESH

    # input RGB img
    def img_process(self, img):
        proposal = []
        height, width, _ = img.shape
        # Convert RBG img to GRAY img
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # for normal img threshold.  200 is ok
        _, binary_img = cv2.threshold(img_gray, self.thresh, 255, cv2.THRESH_BINARY)
        binary_img = cv2.bitwise_not(binary_img)
        contours, _ = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        noise_binary_white_area = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 400:
                noise_binary_white_area.append(contour)
            else:
                continue
        cv2.fillPoly(binary_img, noise_binary_white_area, 0)
        binary_img = ~binary_img
        dla_kernel = np.ones((20, 20), np.uint8)
        thresh_ots = cv2.dilate(binary_img, dla_kernel)
        contours, _ = cv2.findContours(thresh_ots, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Region proposal.
        for c in range(len(contours)):
            area_temp = cv2.contourArea(contours[c])
            # Select area. (under 50 cm is ok)
            if 2000 < area_temp < 4000:
                rect = cv2.minAreaRect(contours[c])
                v = rect[1][0] / rect[1][1]
                # Select length-width ratio.
                if 0.5 < v < 2:
                    proposal.append(cv2.boundingRect(contours[c]))
        # print(len(proposal))

        # Region proposal expand.
        for i in range(len(proposal)):
            x, y, w, h = proposal[i]
            x = (x - w //4) if (x -  w //4) > 0 else 0
            y = (y - h // 4) if (y - h // 4) > 0 else 0
            w = int(w * 1.25) if int(x + w * 1.25) < width else w
            h = int(h * 1.25) if int(y + h * 1.25) < height else h
            proposal[i] = [x, y, w, h]
        return proposal
