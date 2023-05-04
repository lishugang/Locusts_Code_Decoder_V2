import os
import cv2
import numpy as np
import collections
import time


class Gen_videos():
    def __init__(self, image_set):
        self.image_set = image_set
        self.localtime = time.localtime(time.time())
        self.date_info = str(self.localtime.tm_year) + str(self.localtime.tm_mon) + str(self.localtime.tm_mday) + str(self.localtime.tm_hour)
        self.dict_file = os.path.join('npy_set', "result_dict.npy")
        if os.path.exists(self.dict_file):
            self.dict_data = np.load(self.dict_file, allow_pickle=True).item()
            self.total_frame = self.dict_data['total_frame']
            del self.dict_data['total_frame']
            self.temp_5_nums = collections.defaultdict(list)
        self.ori_width = 5120
        self.ori_height = 5120
        self.width = 512
        self.height = 512
        self.r_width = self.ori_width//self.width
        self.r_herght = self.ori_height//self.height
        self.img = np.ones([self.width, self.height, 3], np.uint8)
        self.img[:, :, 0] = 255
        self.img[:, :, 1] = 255
        self.img[:, :, 2] = 255
        cv2.circle(self.img, (self.width//2, self.height//2), 100, (0,0,0), 2)
        cv2.circle(self.img, (self.width // 2, self.height // 2), 270, (0, 0, 0), 2)
        #               黄色           印度红        青色         绿色          橘黄        深红          粉红              蓝色       胡紫色         草绿色
        self.color = [(0,255,255), (31,23,176),(255,255,0),  (0,255,0) , (0,128,255), (255,0,255), (203,192,255), (255,0,0), (250,51,140), (35,142,107)]
        self.color_dict = {}
        self.fps = 5
        self.item_per_video = 10

    def get_pre_name(self):
        img_list = os.listdir(self.image_set)
        sample_img = img_list[0]
        return sample_img[0:8]

    def gen_video(self, video_idx):
        video = cv2.VideoWriter("video_set/" + self.date_info + str(video_idx) + ".avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                self.fps, (self.width, self.height))
        pre_name = self.get_pre_name()
        for i in range(self.total_frame):
            print(i)
            # img_copy = self.img.copy()
            img_copy = cv2.imread(os.path.join(self.image_set, str(pre_name) + str(i+1) + '.bmp'))
            img_copy = cv2.resize(img_copy,(self.height, self.width))

            if str(i) in self.temp_5_nums.keys():
                for locusts_info in self.temp_5_nums[str(i)]:
                    code, x, y, = locusts_info
                    cv2.circle(img_copy, (x//self.r_width,y//self.r_herght), 2, self.color_dict[str(code)], 4)
                    cv2.putText(img_copy, str(code), (x//self.r_width,y//self.r_herght), 2, 0.5, self.color_dict[str(code)], 1)
                # cv2.imshow("1", img_copy)
                # cv2.waitKey(500)
                video.write(img_copy)
        video.release()

    def run(self):
        video_idx = 0
        count = 0
        for code in self.dict_data.keys():
            if count >= self.item_per_video:
                self.gen_video(video_idx)
                self.temp_5_nums.clear()
                count = 0
                video_idx += 1
            self.color_dict[code] = self.color[count]
            for frame_data in self.dict_data[code]:
                self.temp_5_nums[str(frame_data[0])].append([int(code), frame_data[1], frame_data[2]])
            count += 1
        if self.temp_5_nums:
            self.gen_video(video_idx)
            video_idx += 1


if __name__ == "__main__":
    image_set = 'image_set'
    gen_videos = Gen_videos(image_set)
    gen_videos.run()
