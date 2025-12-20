import pickle
import os
import numpy as np
import cv2

# ============================================================================================================
# ============================================================================================================
result_img_dir = './284/' #模型推理图文件夹
merged_img_dir = './284_result' #这部分merge后的大图保存文件夹,也是算法最终的结果
pad_info_save_path =  './pad_infos.pkl'
# ============================================================================================================
# ============================================================================================================

if os.path.exists(merged_img_dir):
    os.system(f'rm -rf {merged_img_dir}')
os.makedirs(merged_img_dir)


pad_infos = pickle.load(open(pad_info_save_path,'rb'))


imgs = os.listdir(result_img_dir) #读取所有小图
# 根据前缀分组图像,前缀相同就是从同一个大图切的，这里我是根据"_"划分前缀和列序号和行序号的。images_dict是字典，key是大图文件名，value是对应的多个小图文件名
images_dict = {}
for img_file in imgs:
    # 根据实际情况调整分隔符和前缀部分,例如'IMG_2205_0_0.png'根据'_'划分后，['IMG','2205','0','0.png']
    pwideix = "_".join(img_file.split("_")[:-2])
    if pwideix in images_dict:
        images_dict[pwideix].append(img_file)
    else:
        images_dict[pwideix] = [img_file]

# 对每个分组的图像进行合并
for pwideix, imgs in images_dict.items():
    # 对应一张完整的大图
    # key:row id
    
    file_base = pwideix
    pad_info = pad_infos[file_base]
    W_starts = pad_info['W_starts']
    H_starts = pad_info['H_starts']
    overlap = pad_info['overlap']
    small_size = pad_info['small_size']
    padded_W, padded_H = pad_info['padded_W'], pad_info['padded_H']
    left_pad, right_pad, top_pad, bottom_pad = pad_info['left_pad'], pad_info['right_pad'], pad_info['top_pad'], pad_info['bottom_pad']

    inner_images_dict = {}
    for img in imgs:
        
        inner_pwideix = img.split('_')[-2]
        if inner_pwideix in inner_images_dict:
            inner_images_dict[inner_pwideix].append(img)
        else:
            inner_images_dict[inner_pwideix] = [img]
    
    row_imgs_list = {}
    print(inner_images_dict.keys())
    for inner_pwideix, img_crops in inner_images_dict.items():
        # 对应完整的大图的一行,小图使用alpha-blending拼成1列,对同一行的图片进行排序
        img_crops = sorted(img_crops, key=lambda x:int(x.split('_')[-1].split('.')[0]))
        new_row_img = np.zeros(shape=(small_size, padded_W, 3))
        overlap_mask = (np.array(list(range(overlap)))/overlap)[np.newaxis,:].repeat(small_size,0)[:,:,np.newaxis].repeat(3, -1)
        
        new_row_img[:small_size,:small_size] = cv2.imread(os.path.join(result_img_dir, img_crops[0]))
        for num in range(1,len(img_crops)):
            mask_a = np.ones(shape=(small_size,(small_size-overlap)*(num-1)+small_size,3))
            mask_b = np.ones(shape=(small_size, small_size,3))
            mask_a[:,-overlap:] = 1-overlap_mask
            mask_b[:,:overlap] = overlap_mask
            
            img_crop_mat = cv2.imread(os.path.join(result_img_dir, img_crops[num]))
            new_row_img[:,:(small_size-overlap)*(num-1)+small_size] = mask_a*new_row_img[:,:(small_size-overlap)*(num-1)+small_size] 
            new_row_img[:,(small_size-overlap)*num:(small_size-overlap)*num+small_size] += mask_b* img_crop_mat
        row_imgs_list[inner_pwideix] = new_row_img
    # print(row_imgs_list.keys())
    row_imgs_list = dict(sorted(row_imgs_list.items(), key=lambda x:x[0]))

    # 将每行的子图使用alpha-blending合成大图
    new_img = np.zeros(shape=(padded_H, padded_W, 3))
    
    new_img[:small_size] = row_imgs_list['0']
    
    overlap_mask = (np.array(list(range(overlap)))/overlap)[:,np.newaxis].repeat(padded_W,1)[:,:,np.newaxis].repeat(3, -1)
    for row in range(1, len(row_imgs_list)):
        mask_a = np.ones(shape=((row-1)*(small_size-overlap)+small_size,padded_W,3))
        mask_b = np.ones(shape=(small_size, padded_W,3))
        mask_a[-overlap:] = 1-overlap_mask
        mask_b[:overlap] = overlap_mask 
        
        new_img[:(row-1)*(small_size-overlap)+small_size] = mask_a*new_img[:(row-1)*(small_size-overlap)+small_size] 
        new_img[row*(small_size-overlap):(small_size-overlap)*row+small_size] += mask_b*row_imgs_list[str(row)] 

    new_img = new_img[top_pad:-bottom_pad,left_pad:-right_pad]

    cv2.imwrite(os.path.join(merged_img_dir, f'{pwideix}.png'), new_img)

print("所有图片合并完成！")