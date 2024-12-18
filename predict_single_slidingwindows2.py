import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
import argparse
import os
import cv2
import numpy as np
import skimage.io
import torch
from albumentations.pytorch.transforms import img_to_tensor
from skimage import measure
from skimage.segmentation import watershed
import models
from tools.config import load_config
from collections import namedtuple

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

ModelConfig = namedtuple("ModelConfig", "config_path weight_path type weight")
weight_path = "weights"
configs = [
    ModelConfig("configs/d92_loc.json", "localization_dpn_unet_dpn92_0_best_dice", "localization", 1),
    ModelConfig("configs/d92_loc.json", "localization_dpn_unet_dpn92_1_best_dice", "localization", 1),
    ModelConfig("configs/d92_loc.json", "localization_dpn_unet_dpn92_2_best_dice", "localization", 1),
    ModelConfig("configs/d92_loc.json", "localization_dpn_unet_dpn92_3_best_dice", "localization", 1),
    ModelConfig("configs/d161_loc.json", "localization_densenet_unet_densenet161_3_0_best_dice", "localization", 1),
    ModelConfig("configs/d161_loc.json", "localization_densenet_unet_densenet161_3_1_best_dice", "localization", 1),

    ModelConfig("configs/d92_softmax.json", "softmax_dpn_seamese_unet_shared_dpn92_0_best_xview", "damage", 1),
    ModelConfig("configs/d92_softmax.json", "softmax_dpn_seamese_unet_shared_dpn92_2_best_xview", "damage", 1),
    ModelConfig("configs/d92_softmax.json", "pseudo_dpn_seamese_unet_shared_dpn92_0_best_xview", "damage", 1),
    ModelConfig("configs/d92_softmax.json", "pseudo_dpn_seamese_unet_shared_dpn92_2_best_xview", "damage", 1),

    ModelConfig("configs/d161_softmax.json", "softmax_densenet_seamese_unet_shared_densenet161_0_best_xview", "damage", 1),
    ModelConfig("configs/d161_softmax.json", "softmax_densenet_seamese_unet_shared_densenet161_2_best_xview", "damage", 1),
    ModelConfig("configs/d161_softmax.json", "pseudo_densenet_seamese_unet_shared_densenet161_0_best_xview", "damage", 1),
    ModelConfig("configs/d161_softmax.json", "pseudo_densenet_seamese_unet_shared_densenet161_2_best_xview", "damage", 1),

    ModelConfig("configs/se50_softmax.json", "pseudo_scseresnext_seamese_unet_shared_seresnext50_0_best_xview", "damage", 1),
    ModelConfig("configs/se50_softmax.json", "pseudo_scseresnext_seamese_unet_shared_seresnext50_1_best_xview", "damage", 1),
    ModelConfig("configs/se50_softmax.json", "pseudo_scseresnext_seamese_unet_shared_seresnext50_2_best_xview", "damage", 1),
    ModelConfig("configs/se50_softmax.json", "pseudo_scseresnext_seamese_unet_shared_seresnext50_3_best_xview", "damage", 1),

    ModelConfig("configs/b2_softmax.json", "softmax_sampling_efficient_seamese_unet_shared_efficientnet-b2_0_best_xview", "damage", 1),
    ModelConfig("configs/b2_softmax.json", "softmax_sampling_efficient_seamese_unet_shared_efficientnet-b2_1_best_xview", "damage", 1),

    ModelConfig("configs/r101_softmax_sgd.json", "sgd_resnext_seamese_unet_shared_resnext101_0_best_xview", "damage", 2)
]

def predict_localization(image, config: ModelConfig):
    conf = load_config(config.config_path)
    model = models.__dict__[conf['network']](seg_classes=1, backbone_arch=conf['encoder'])
    checkpoint_path = os.path.join(weight_path, config.weight_path)
    print(f"=> loading checkpoint '{checkpoint_path}'")
    checkpoint = torch.load(checkpoint_path, map_location="cuda")
    model.load_state_dict({k.replace("module.", ""): v for k, v in checkpoint['state_dict'].items()})
    model.eval()
    model = model.cuda()

    h, w = image.shape[:2]
    window_size = 1024
    stride = 768
    
    output = np.zeros((h, w), dtype=np.float32)
    counts = np.zeros((h, w), dtype=np.float32)
    
    normalize = {
        'mean': conf["input"]["normalize"]["mean"][:3],
        'std': conf["input"]["normalize"]["std"][:3]
    }
    
    with torch.no_grad():
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                end_y = min(y + window_size, h)
                end_x = min(x + window_size, w)
                y1, x1 = max(0, y), max(0, x)
                
                window = image[y1:end_y, x1:end_x]
                
                if window.shape[0] != window_size or window.shape[1] != window_size:
                    pad_h = window_size - window.shape[0]
                    pad_w = window_size - window.shape[1]
                    window = np.pad(window, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
                
                window = img_to_tensor(window, normalize).cuda()
                
                if "dpn" in config.weight_path:
                    window_cpu = window.cpu().numpy()
                    window_cpu = np.pad(window_cpu, [(0, 0), (16, 16), (16, 16)], mode='reflect')
                    window = torch.from_numpy(window_cpu).cuda()

                aug_windows = torch.stack([
                    window,
                    window.flip(1),
                    window.flip(2),
                    window.flip(1).flip(2)
                ])
                
                aug_predictions = []
                for i in range(4):
                    logits = model(aug_windows[i:i+1])
                    pred = torch.sigmoid(logits).cpu().numpy()[0]
                    
                    if i == 1:
                        pred = pred[:, ::-1, :]
                    elif i == 2:
                        pred = pred[:, :, ::-1]
                    elif i == 3:
                        pred = pred[:, ::-1, ::-1]
                    
                    aug_predictions.append(pred)
                
                pred = np.mean(aug_predictions, axis=0)
                
                if "dpn" in config.weight_path:
                    pred = pred[:, 16:-16, 16:-16]
                
                pred = pred[0]
                pred = pred[:end_y-y1, :end_x-x1]
                
                output[y1:end_y, x1:end_x] += pred
                counts[y1:end_y, x1:end_x] += 1

                torch.cuda.empty_cache()

    valid_mask = counts > 0
    output[valid_mask] /= counts[valid_mask]
    output = np.expand_dims(output, axis=0)
    
    return output

def predict_damage(image, config: ModelConfig):
    conf = load_config(config.config_path)
    model = models.__dict__[conf['network']](seg_classes=5, backbone_arch=conf['encoder'])
    checkpoint_path = os.path.join(weight_path, config.weight_path)
    print(f"=> loading checkpoint '{checkpoint_path}'")
    checkpoint = torch.load(checkpoint_path, map_location="cuda")
    model.load_state_dict({k[7:]: v for k, v in checkpoint['state_dict'].items()})
    model.eval()
    model = model.cuda()

    h, w = image.shape[:2]
    window_size = 1024
    stride = 768
    
    output = np.zeros((5, h, w), dtype=np.float32)
    counts = np.zeros((h, w), dtype=np.float32)
    
    with torch.no_grad():
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                end_y = min(y + window_size, h)
                end_x = min(x + window_size, w)
                y1, x1 = max(0, y), max(0, x)
                
                window = image[y1:end_y, x1:end_x]
                
                if window.shape[0] != window_size or window.shape[1] != window_size:
                    pad_h = window_size - window.shape[0]
                    pad_w = window_size - window.shape[1]
                    window = np.pad(window, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
                
                window = img_to_tensor(window, conf["input"]["normalize"]).cuda()
                
                aug_windows = torch.stack([
                    window,
                    window.flip(1),
                    window.flip(2),
                    window.flip(1).flip(2)
                ])
                aug_predictions = []
                for i in range(4):
                    logits = model(aug_windows[i:i+1])
                    pred = torch.softmax(logits, dim=1).cpu().numpy()[0]  # (5, h, w)
                    # 不需要转置维度
                    
                    if i == 1:
                        pred = pred[:, ::-1, :]
                    elif i == 2:
                        pred = pred[:, :, ::-1]
                    elif i == 3:
                        pred = pred[:, ::-1, ::-1]
                    
                    aug_predictions.append(pred)
                
                pred = np.mean(aug_predictions, axis=0)  # (5, h, w)
                pred = pred[:, :end_y-y1, :end_x-x1]  # 保持通道在前
                
                output[:, y1:end_y, x1:end_x] += pred
                counts[y1:end_y, x1:end_x] += 1

                torch.cuda.empty_cache()

    valid_mask = counts > 0
    for i in range(5):
        output[i][valid_mask] /= counts[valid_mask]
    
    return output

def predict_localization_ensemble(pre_path):
    image = cv2.imread(pre_path, cv2.IMREAD_COLOR)[:, :, ::-1]
    preds = []
    for model_config in configs:
        if model_config.type == "localization":
            preds.append((predict_localization(image, model_config) * 255).astype(np.uint8))
    return np.average(preds, axis=0)

def predict_damage_ensemble(pre_path, post_path):
    image_pre = cv2.imread(pre_path, cv2.IMREAD_COLOR)[:, :, ::-1]
    image_post = cv2.imread(post_path, cv2.IMREAD_COLOR)[:, :, ::-1]
    image = np.concatenate([image_pre, image_post], axis=-1)
    preds = []
    for model_config in configs:
        if model_config.type == "damage":
            damage = (predict_damage(image, model_config) * 255).astype(np.uint8)
            preds.append(damage)
            if model_config.weight == 2:
                preds.append(damage)
    return np.average(preds, axis=0)

def label_mask(loc, labels, intensity, mask, seed_threshold=0.8):
    av_pred = (loc > seed_threshold).astype(np.uint8)
    y_pred = measure.label(av_pred, background=0)
    
    nucl_msk = (1 - loc).astype('uint8')
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
    
    first = np.zeros((damage.shape[0], damage.shape[1], 1))
    first[:, :, 0] = 1 - np.sum(damage, axis=2)
    first[:, :, :] *= 0.8

    damage_pred = np.concatenate([first, damage], axis=-1)

    damage_pred[:, :, 2] *= 2.
    damage_pred[:, :, 3] *= 2.
    damage_pred[:, :, 4] *= 2.

    argmax = np.argmax(damage_pred, axis=-1)
    loc = 1 * ((localization > 0.25) | (argmax > 0))
    max_val = np.max(damage, axis=-1)
    
    label_mask(localization, argmax, max_val, loc)
    print(f"Saving final results to: {out_loc}, {out_damage}")
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
    parser = argparse.ArgumentParser("Xview Predictor")
    arg = parser.add_argument
    arg('--pre', type=str, help='Path to pre test image')
    arg('--post', type=str, help='Path to post test image')
    arg('--out-loc', type=str, help='Path to output localization image')
    arg('--out-damage', type=str, help='Path to output damage image')
    args = parser.parse_args()

    print("Predicting localization...")
    localization = predict_localization_ensemble(args.pre)
    skimage.io.imsave("_localization2.png", localization[0, :, :].astype(np.uint8))
    
    print("Predicting damage...")
    damage = predict_damage_ensemble(args.pre, args.post)
    preds = (np.moveaxis(damage, 0, -1)).astype(np.uint8)
    cv2.imwrite("_damage2.png", cv2.cvtColor(preds[:, :, 1:], cv2.COLOR_RGBA2BGRA))
    
    print("Post-processing results...")
    damage = skimage.io.imread("_damage2.png")
    localization = cv2.imread("_localization2.png", cv2.IMREAD_GRAYSCALE)
    post_process(localization, damage, args.out_loc, args.out_damage)


if __name__ == '__main__':
    main()