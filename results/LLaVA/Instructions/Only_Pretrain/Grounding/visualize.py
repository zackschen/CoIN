import os
import cv2


if __name__ == '__main__':
    fname = os.path.join('/home/chencheng/Code/LLaVA/cl_dataset/COCO2014/train2014/COCO_train2014_000000580957.jpg')
    output_path = './results/Instructions/Only_Pretrain/Grounding/COCO_train2014_000000580957.png'
    img = cv2.imread(fname)

    box = [0.0, 70.8875, 0.22999999999999998, 71.0413125]
    pt1 = (int(box[0]), int(box[1])) 
    pt2 = (int(box[2]), int(box[3]))
    cv2.rectangle(img, pt1, pt2, (0, 255, 0), 2)

    box = [467.2, 72.76, 640.0, 149.79999999999998]
    pt1 = (int(box[0]), int(box[1])) 
    pt2 = (int(box[2]), int(box[3]))
    cv2.rectangle(img, pt1, pt2, (0, 0, 255), 2)

    cv2.imwrite(output_path, img)