import logging
import mmengine

det_prompt = 'The following sentences describe each object in the image. ' \
             'Each object is represented in the format c: [x1, y1, x2, y2], ' \
             'where c is the class name, and [x1, y1, x2, y2] is ' \
             'the relative box coordinates of the object, where x1 is the horizontal coordinate ' \
             'of the left top corner, y1 is the vertical coordinate of the left top corner, x2 ' \
             'is the horizontal coordinate of the right bottom corner, y2 is the vertical coordinate ' \
             'of the right bottom corner. The following are the descriptions: '


def load_det_res(res_path):
    det_res = mmengine.load(res_path)
    return {k.split('/')[-1]: v for k, v in det_res.items()}


def get_det_res_str(image_path, img_path_to_ann_dict):
    image_path = image_path.split('/')[-1]
    if img_path_to_ann_dict is None:
        return None
    if image_path not in img_path_to_ann_dict:
        return None
    det_res = img_path_to_ann_dict[image_path]
    if len(det_res) == 0:
        return None
    det_res_str = ''
    for res in det_res:
        cat_name = res['category_name']
        bbox = res['bbox']
        img_height, img_width = res['image_height'], res['image_width']
        x1, y1, w, h = bbox
        x2, y2 = x1 + w, y1 + h
        x1 /= img_width
        x2 /= img_width
        y1 /= img_height
        y2 /= img_height
        det_res_str += f'{cat_name}: [{x1:.3f}, {y1:.3f}, {x2:.3f}, {y2:.3f}], '
    det_res_str = det_res_str.strip(', ') + '.'
    det_res_str = det_prompt + det_res_str
    return det_res_str


def compare_imgs(src_imgs, det_results_dict, dataset_name):
    src_imgs = [img.split('/')[-1] for img in src_imgs]
    unique_imgs = set(src_imgs)
    contain = 0
    not_contain = 0
    for img in unique_imgs:
        if img in det_results_dict:
            contain += 1
        else:
            not_contain += 1
    logging.info(
        f'{dataset_name}: [{contain}/{len(unique_imgs)}] imgs have det res'
    )
