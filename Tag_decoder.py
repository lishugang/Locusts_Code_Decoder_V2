import os
import cv2
import numpy as np
import collections
import json


class Tag_decoder():
    def __init__(self, total_frames, THRESH):
        self.grid_num = 6
        self.length = 60
        self.grid_length = self.length // self.grid_num
        self.img_grid_4_npy = np.load('npy_set/tags_data.npy')
        self.result_dict = collections.defaultdict(list)
        self.total_frames = total_frames
        self.result_dict['total_frame'] = self.total_frames
        self.thresh = THRESH


    def apply_PerspectiveTransform(self, img_gray, img_bgr):
        h, w = img_gray.shape
        area = h * w
        # (200 is ok)
        _, img_gray = cv2.threshold(img_gray, self.thresh, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(img_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        max_area = 0
        max_contours_ids = -1
        max_rect = None
        for i in range(len(contours)):
            rect = cv2.minAreaRect(contours[i])
            contour_area = cv2.contourArea(contours[i])
            ratio_area = contour_area / area
            if 0.1 < ratio_area < 0.90:
                if contour_area >= max_area:
                    max_area = contour_area
                    max_contours_ids = i
                    max_rect = rect
        box = cv2.boxPoints(max_rect)
        box = list(np.int0(box))
        min_rect = []
        if max_contours_ids == -1:
            return max_contours_ids
        # print(max_contours_ids)
        max_contours = contours[max_contours_ids]
        approx = cv2.approxPolyDP(max_contours, 1, True)
        for rect_p in box:
            rect_p = np.array(rect_p)
            min_dis = 100000
            min_p = []
            for approx_p in approx:
                approx_p_cp = approx_p[0]
                dis = np.sqrt(np.sum(np.square(rect_p-approx_p_cp)))
                if dis < min_dis:
                    min_p = approx_p_cp
                    min_dis = dis
            min_rect.append(list(min_p))
        p_gen = np.float32([[0, 0], [0, self.length], [self.length, self.length], [self.length, 0]])
        p_rect = np.float32(min_rect)
        M = cv2.getPerspectiveTransform(p_rect, p_gen)
        pes_bgr = cv2.warpPerspective(img_bgr, M, (self.length, self.length))
        return pes_bgr


    def decoder(self, code):
        code_gray = cv2.cvtColor(code, cv2.COLOR_BGR2GRAY)
        # flip img
        code_gray = cv2.flip(code_gray,1,dst=None)
        gap = self.length//self.grid_num
        begin_h = begin_w = self.length // (self.grid_num*2) + gap
        res = []
        img_mean = np.mean(code_gray)
        for i in range(4):
            res_temp = []
            w = begin_w + gap * i
            for j in range(4):
                h = begin_h + gap * j
                value = 1 if code_gray[w][h] > img_mean else 0
                res_temp.append(value)
            res.append(res_temp)
        res = np.array(res)
        for r in range(4):
            res = np.rot90(res, k=r)
            res_cp = res.flatten()
            for t in range(len(self.img_grid_4_npy)):
                if all(res_cp == self.img_grid_4_npy[t]):
                    return t
        return -1

    def run(self, regions, img, frame, return_img=False):
        for i in range(len(regions)):
            x, y, w, h = regions[i]
            # print(regions[i])
            cut_img = img[y: y + h, x: x + w]
            # print(cut_img.shape)
            cut_img_gray = cv2.cvtColor(cut_img, cv2.COLOR_BGR2GRAY)
            pes_bgr = self.apply_PerspectiveTransform(cut_img_gray, cut_img)
            if type(pes_bgr) == np.ndarray:
                code = self.decoder(pes_bgr)
                if code != -1:
                    self.result_dict[str(code)].append([frame, x + w//2, y + h//2])
                    # draw
                    if return_img:
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        img = cv2.putText(img, str(code), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        if return_img:
            # print(str(frame))
            # cv2.imwrite("result_img/region_proposal_" + str(frame) + ".png", img)
            return img

    def save_result_dict(self):
        np.save(os.path.join('npy_set','result_dict.npy'), self.result_dict)
