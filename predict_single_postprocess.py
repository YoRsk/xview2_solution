import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

import os
import cv2
import numpy as np
import skimage.io
from skimage import measure
from skimage.segmentation import watershed
import argparse

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

def label_mask(loc, labels, intensity, mask, seed_threshold=0.8):
    """
    Identify and label individual building regions based on damage predictions.
    For each building region:
    - Examine the distribution of damage predictions within the region.
    - If a certain type of damage is dominant (>60%) and the building shape is reasonable 
      (not too elongated, no holes), update the labels of low-confidence areas with this 
      dominant damage type.
    """
    av_pred = 1 * (loc > seed_threshold)
    av_pred = av_pred.astype(np.uint8)
    y_pred = measure.label(av_pred, background=0)

    nucl_msk = (1 - loc)
    nucl_msk = nucl_msk.astype('uint8')
    y_pred = watershed(nucl_msk, y_pred, mask=mask, watershed_line=False)
    props = measure.regionprops(y_pred)
    for i in range(1, np.max(y_pred)):
        reg_labels = labels[y_pred == i]
        unique, counts = np.unique(reg_labels, return_counts=True)
        max_idx = np.argmax(counts)
        out_label = unique[max_idx]
        if out_label > 0:
            prop = props[i - 1]
            if counts[max_idx] > 0.6 * sum(counts) \
                    and prop.eccentricity < 1.5 \
                    and prop.euler_number == 1:
                labels[(y_pred == i) & (intensity < 0.6)] = out_label

    return y_pred

def post_process(loc, damage, out_loc, out_damage):
    localization = loc/255.
    damage = damage/255.
    first = np.zeros((1024, 1024, 1))
    first[:, :, 0] = 1 - np.sum(damage, axis=2)
    first[:, :, :] *= 0.8

    damage_pred = np.concatenate([first, damage], axis=-1)

    damage_pred[:, :, 2] *= 2.
    damage_pred[:, :, 3] *= 2.
    damage_pred[:, :, 4] *= 2.

    argmax = np.argmax(damage_pred, axis=-1)
    loc = 1 * ((localization > 0.25) | (argmax > 0))
    # argmax = np.argmax(damage, axis=-1) + 1
    max = np.max(damage, axis=-1)
    label_mask(localization, argmax, max, loc)
    
    print("Saving final results to:", out_loc, out_damage)
    print("loc shape:", loc.shape)
    print("argmax shape:", argmax.shape)
    
    os.makedirs(os.path.dirname(out_loc), exist_ok=True)
    os.makedirs(os.path.dirname(out_damage), exist_ok=True)
    
    try:
        cv2.imwrite(out_loc, (loc * 255).astype(np.uint8))
        cv2.imwrite(out_damage, argmax.astype(np.uint8))
        print("Successfully saved final results")
    except Exception as e:
        print(f"Error saving final results: {e}")

def main():
    parser = argparse.ArgumentParser("Post-processing Script")
    parser.add_argument('--out-loc', type=str, required=True, help='Path to output localization image')
    parser.add_argument('--out-damage', type=str, required=True, help='Path to output damage image')
    args = parser.parse_args()

    # 读取已生成的中间文件
    damage = skimage.io.imread("_damage.png")
    localization = cv2.imread("_localization.png", cv2.IMREAD_GRAYSCALE)
    post_process(localization, damage, args.out_loc, args.out_damage)

if __name__ == "__main__":
    main()