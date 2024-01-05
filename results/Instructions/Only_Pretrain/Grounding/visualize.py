import os
import cv2


if __name__ == '__main__':
    fname = os.path.join('/home/chencheng/Code/LLaVA/cl_dataset/VisualGenome/images/4.jpg')
    output_path = './results/Instructions/Only_Pretrain/Grounding/4.png'
    box = [0.34, 0.52, 0.42, 0.63]
    img = cv2.imread(fname)

    size = [img.shape[1],img.shape[0]]

    max_height_width = max(size[0],size[1])
    dif = abs(size[0]-size[1])
    if size[0] > size[1]:
        box[0] = box[0] * size[0]
        box[1] = box[1] * size[1]
        box[2] = box[2] * size[0]
        box[3] = box[3] * size[1]
    else:
        box[0] = box[0] * max_height_width - dif/2.0
        box[1] = box[1] * max_height_width
        box[2] = box[2] * max_height_width - dif/2.0
        box[3] = box[3] * max_height_width

    pt1 = (int(box[0]), int(box[1])) 
    pt2 = (int(box[2]), int(box[3]))

    cv2.rectangle(img, pt1, pt2, (0, 255, 0), 2)
    cv2.imwrite(output_path, img)