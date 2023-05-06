from Region_proposal import *
from Tag_decoder import *
from Gen_videos import *
from tqdm import tqdm

def find_num(img_name):
    length = len(img_name)
    num = img_name[7:length-4]
    return int(num)


if __name__ == "__main__":
    # 图片文件夹的路径
    img_set = 'image_set'
    # 在 100-250 的范围内观察调节
    THRESH = 170
    # 跳过前 BEGIN_IMG_ID 张图片
    BEGIN_IMG_ID = 0
    img_list = os.listdir(img_set)
    img_list.sort(key=find_num)
    total_frames = len(img_list) - BEGIN_IMG_ID
    region_proposal = Region_proposal(THRESH)
    tag_decoder = Tag_decoder(total_frames, THRESH)
    for i in tqdm(range(total_frames)):
        id = BEGIN_IMG_ID + i
        img_name = img_list[id].split('.')[0]
        img_file = img_set + "/" + img_name + ".bmp"
        img = cv2.imread(img_file)
        # The input image is 3 channel.
        regions = region_proposal.img_process(img)
        tag_decoder.run(regions, img, id)
    tag_decoder.save_result_dict()
    gen_videos = Gen_videos(img_set, BEGIN_IMG_ID)
    gen_videos.run()
