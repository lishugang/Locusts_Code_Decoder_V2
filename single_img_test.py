from Region_proposal import *
from Tag_decoder import *


if __name__ == "__main__":
    # 待处理文件夹中的一张图片
    img_name = 'image_set/202304272.bmp'
    # 在 100-250 的范围内观察调节
    THRESH = 100
    while THRESH < 250:
        tag_decoder = Tag_decoder(1, THRESH)
        region_proposal = Region_proposal(THRESH)
        img = cv2.imread(img_name)
        # The input image is 3 channel.
        regions = region_proposal.img_process(img)
        total_regions = len(regions)
        print("阈值在 {} 检测到了 {} 个疑似编码区域".format(THRESH, total_regions))
        # tag_decoder.run(regions, img, i, True)
        draw_img = tag_decoder.run(regions, img, 0, True)
        cv2.imwrite('result_img/' + str(THRESH)+'.png', draw_img)
        THRESH += 10