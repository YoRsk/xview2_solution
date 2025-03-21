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
import torchmetrics
from torchmetrics import JaccardIndex
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
    """Predict localization using sliding window"""
    conf = load_config(config.config_path)
    model = models.__dict__[conf['network']](seg_classes=1, backbone_arch=conf['encoder'])
    checkpoint_path = os.path.join(weight_path, config.weight_path)
    print(f"=> loading checkpoint '{checkpoint_path}'")
    
    # 直接加载到GPU
    checkpoint = torch.load(checkpoint_path, map_location="cuda")
    model.load_state_dict({k.replace("module.", ""): v for k, v in checkpoint['state_dict'].items()})
    model.eval()
    model = model.cuda()

    h, w = image.shape[:2]
    window_size = 1024
    stride = 768
    
    # 在CPU上初始化输出数组(大型numpy数组在CPU上处理更好)
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
                
                # CPU上处理图像预处理
                window = image[y1:end_y, x1:end_x]
                
                if window.shape[0] != window_size or window.shape[1] != window_size:
                    pad_h = window_size - window.shape[0]
                    pad_w = window_size - window.shape[1]
                    window = np.pad(window, 
                                  ((0, pad_h), (0, pad_w), (0, 0)), 
                                  mode='reflect')
                
                # 转换成tensor并直接送到GPU
                window = img_to_tensor(window, normalize).cuda()
                
                if "dpn" in config.weight_path:
                    # CPU上做padding操作
                    window_cpu = window.cpu().numpy()
                    window_cpu = np.pad(window_cpu, 
                                      [(0, 0), (16, 16), (16, 16)], 
                                      mode='reflect')
                    window = torch.from_numpy(window_cpu).cuda()
                
                # GPU上做数据增强
                aug_windows = torch.stack([
                    window,
                    window.flip(1),  # 水平翻转
                    window.flip(2),  # 垂直翻转
                    window.flip(1).flip(2)  # 同时翻转
                ]).float()
                
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
    """Predict damage using sliding window"""
    conf = load_config(config.config_path)
    model = models.__dict__[conf['network']](seg_classes=5, backbone_arch=conf['encoder'])
    checkpoint_path = os.path.join(weight_path, config.weight_path)
    print(f"=> loading checkpoint '{checkpoint_path}'")
    checkpoint = torch.load(checkpoint_path, map_location="cuda")
    model.load_state_dict({k[7:]: v for k, v in checkpoint['state_dict'].items()})
    model.eval()
    model = model.cuda()
    h, w = image.shape[:2]
    window_size = 512
    stride = 256
    # window_size = 1024
    # stride = 768
    
    # 输出维度保持 (5,h,w)
    output = np.zeros((5, h, w), dtype=np.float32)
    counts = np.zeros((h, w), dtype=np.float32)
    
    with torch.no_grad():
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                # 处理窗口
                end_y = min(y + window_size, h)
                end_x = min(x + window_size, w)
                y1, x1 = max(0, y), max(0, x)
                
                window = image[y1:end_y, x1:end_x]
                if window.shape[0] != window_size or window.shape[1] != window_size:
                    pad_h = window_size - window.shape[0]
                    pad_w = window_size - window.shape[1]
                    window = np.pad(window, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
                
                # 转换和增强
                window = img_to_tensor(window, conf["input"]["normalize"]).cuda()
                aug_windows = torch.stack([
                    window,
                    window.flip(1),
                    window.flip(2),
                    window.flip(1).flip(2)
                ])
                
                # 预测并累积结果
                aug_predictions = []
                for i in range(4):
                    logits = model(aug_windows[i:i+1])
                    pred = torch.softmax(logits, dim=1).cpu().numpy()[0]  # 已经是(5,h,w)格式
                    if i == 1:
                        pred = pred[:, ::-1, :]
                    elif i == 2:
                        pred = pred[:, :, ::-1]
                    elif i == 3:
                        pred = pred[:, ::-1, ::-1]
                    aug_predictions.append(pred)
                
                pred = np.mean(aug_predictions, axis=0)  # (5,h,w)
                pred = pred[:, :end_y-y1, :end_x-x1]
                
                output[:, y1:end_y, x1:end_x] += pred
                counts[y1:end_y, x1:end_x] += 1
                
                torch.cuda.empty_cache()
    
    # 处理平均值
    valid_mask = counts > 0
    for i in range(5):
        output[i][valid_mask] /= counts[valid_mask]
    
    return output  # 返回(5,h,w)
def predict_localization_ensemble(pre_path):
    image = cv2.imread(pre_path, cv2.IMREAD_COLOR)[:, :, ::-1]
    preds = []
    for model_config in configs:
        if model_config.type == "localization":
            preds.append((predict_localization(image, model_config) * 255).astype(np.uint8))
            torch.cuda.empty_cache()  # 每次预测后清理
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
    """Label and process mask regions"""
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
    h, w = loc.shape[:2]  # 获取实际图像大小
    localization = loc/255.
    damage = damage/255.  # damage已经是(h,w,4)格式
    
    # 创建"无损伤"通道，形状与输入图像一致
    first = np.zeros((h, w, 1))
    first[:, :, 0] = 1 - np.sum(damage, axis=2)  # 计算无损伤概率
    first *= 0.8  # 缩放因子

    # 将"无损伤"通道与其他损伤类型拼接，得到完整的预测
    damage_pred = np.concatenate([first, damage], axis=2)  # (h,w,5)

    # 增强后面三种损伤类型的权重
    damage_pred[:, :, 2] *= 2.
    damage_pred[:, :, 3] *= 2.
    damage_pred[:, :, 4] *= 2.

    argmax = np.argmax(damage_pred, axis=-1)
    loc = 1 * ((localization > 0.25) | (argmax > 0))
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

def calculate_metrics(prediction, ground_truth):
    """
    计算评估指标:
    - 5分类指标：准确率、F1分数、IoU、精确率、召回率
    - 二分类指标：准确率、F1分数、IoU、精确率、召回率
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 计算5分类指标
    prediction = prediction.astype(np.int64)
    ground_truth = ground_truth.astype(np.int64)
    prediction_tensor = torch.from_numpy(prediction).long().unsqueeze(0).to(device)
    ground_truth_tensor = torch.from_numpy(ground_truth).long().unsqueeze(0).to(device)
    
    # 初始化5分类metrics（排除背景类）
    accuracy_5 = torchmetrics.Accuracy(task='multiclass', num_classes=5, 
                                     ignore_index=0, validate_args=False).to(device)
    precision_5 = torchmetrics.Precision(task='multiclass', num_classes=5, 
                                       ignore_index=0, average='macro', validate_args=False).to(device)
    recall_5 = torchmetrics.Recall(task='multiclass', num_classes=5, 
                                  ignore_index=0, average='macro', validate_args=False).to(device)
    f1_score_5 = torchmetrics.F1Score(task='multiclass', num_classes=5, 
                                     ignore_index=0, average='macro').to(device)
    f1_score_5_per_class = torchmetrics.F1Score(task='multiclass', num_classes=5, 
                                               ignore_index=0, average=None).to(device)
    iou_5 = torchmetrics.JaccardIndex(task="multiclass", num_classes=5, 
                                     ignore_index=0).to(device)
    
    # 更新metrics
    accuracy_5.update(prediction_tensor, ground_truth_tensor)
    precision_5.update(prediction_tensor, ground_truth_tensor)
    recall_5.update(prediction_tensor, ground_truth_tensor)
    f1_score_5.update(prediction_tensor, ground_truth_tensor)
    f1_score_5_per_class.update(prediction_tensor, ground_truth_tensor)
    iou_5.update(prediction_tensor, ground_truth_tensor)
    
    # 计算结果
    acc_5 = accuracy_5.compute()
    prec_5 = precision_5.compute()
    rec_5 = recall_5.compute()
    f1_5 = f1_score_5.compute()
    f1_5_per_class = f1_score_5_per_class.compute()
    iou_score_5 = iou_5.compute()
    
    # 创建二分类版本 (保持0为背景)
    binary_prediction = prediction.copy()
    binary_ground_truth = ground_truth.copy()
    
    # 将2,3,4类转换为2（损坏类）
    binary_prediction[(binary_prediction == 2) | (binary_prediction == 3) | (binary_prediction == 4)] = 2
    binary_ground_truth[(binary_ground_truth == 2) | (binary_ground_truth == 3) | (binary_ground_truth == 4)] = 2
    
    # 转换为tensor
    binary_prediction_tensor = torch.from_numpy(binary_prediction).long().unsqueeze(0).to(device)
    binary_ground_truth_tensor = torch.from_numpy(binary_ground_truth).long().unsqueeze(0).to(device)
    
    # 初始化二分类metrics（排除背景类）
    accuracy_2 = torchmetrics.Accuracy(task='multiclass', num_classes=3, 
                                     ignore_index=0, validate_args=False).to(device)
    precision_2 = torchmetrics.Precision(task='multiclass', num_classes=3, 
                                       ignore_index=0, average='macro', validate_args=False).to(device)
    recall_2 = torchmetrics.Recall(task='multiclass', num_classes=3, 
                                  ignore_index=0, average='macro', validate_args=False).to(device)
    f1_score_2 = torchmetrics.F1Score(task='multiclass', num_classes=3, 
                                     ignore_index=0, average='macro').to(device)
    f1_score_2_per_class = torchmetrics.F1Score(task='multiclass', num_classes=3, 
                                               ignore_index=0, average=None).to(device)
    iou_2 = torchmetrics.JaccardIndex(task="multiclass", num_classes=3, 
                                     ignore_index=0).to(device)
    
    # 更新二分类metrics
    accuracy_2.update(binary_prediction_tensor, binary_ground_truth_tensor)
    precision_2.update(binary_prediction_tensor, binary_ground_truth_tensor)
    recall_2.update(binary_prediction_tensor, binary_ground_truth_tensor)
    f1_score_2.update(binary_prediction_tensor, binary_ground_truth_tensor)
    f1_score_2_per_class.update(binary_prediction_tensor, binary_ground_truth_tensor)
    iou_2.update(binary_prediction_tensor, binary_ground_truth_tensor)
    
    # 计算二分类结果
    acc_2 = accuracy_2.compute()
    prec_2 = precision_2.compute()
    rec_2 = recall_2.compute()
    f1_2 = f1_score_2.compute()
    f1_2_per_class = f1_score_2_per_class.compute()
    iou_score_2 = iou_2.compute()
    
    # 清除缓存
    for metric in [accuracy_5, precision_5, recall_5, f1_score_5, iou_5, f1_score_5_per_class,
                  accuracy_2, precision_2, recall_2, f1_score_2, iou_2, f1_score_2_per_class]:
        metric.reset()
    
    return (acc_5.item(), f1_5.item(), iou_score_5.item(), prec_5.item(), rec_5.item(), f1_5_per_class.tolist(),
            acc_2.item(), f1_2.item(), iou_score_2.item(), prec_2.item(), rec_2.item(), f1_2_per_class.tolist())

def main():
    parser = argparse.ArgumentParser("Xview Predictor")
    arg = parser.add_argument
    arg('--pre', type=str, help='Path to pre test image')
    arg('--post', type=str, help='Path to post test image')
    arg('--out-loc', type=str, help='Path to output localization image')
    arg('--out-damage', type=str, help='Path to output damage image')
    # 只添加这一行参数
    arg('--ground-truth', type=str, default='', help='Path to ground truth mask file (for evaluation)')
    args = parser.parse_args()
    
    # [保持原有代码完全不变，包括所有注释]
    localization = predict_localization_ensemble(args.pre)
    skimage.io.imsave("_localization.png", localization[0, :, :].astype(np.uint8))
    damage = predict_damage_ensemble(args.pre, args.post)
    preds = (np.moveaxis(damage, 0, -1)).astype(np.uint8)
    cv2.imwrite("_damage.png", cv2.cvtColor(preds[:, :, 1:], cv2.COLOR_RGBA2BGRA))
    damage = skimage.io.imread("_damage.png")
    localization = cv2.imread("_localization.png", cv2.IMREAD_GRAYSCALE)
    post_process(localization, damage, args.out_loc, args.out_damage)

    # 在最后添加评估部分
    if args.ground_truth:
        try:
            # 使用 skimage.io 替代 cv2.imread
            prediction = skimage.io.imread(args.out_damage)
            ground_truth = skimage.io.imread(args.ground_truth)
            
            # 确保是单通道灰度图
            if len(prediction.shape) > 2:
                prediction = prediction[:,:,0]
            if len(ground_truth.shape) > 2:
                ground_truth = ground_truth[:,:,0]
                
            if ground_truth.shape != prediction.shape:
                ground_truth = cv2.resize(ground_truth, (prediction.shape[1], prediction.shape[0]), 
                                    interpolation=cv2.INTER_NEAREST)
            
            metrics = calculate_metrics(prediction, ground_truth)
            
            print("\nMulti-classes evaluation result:")
            print(f'Accuracy: {metrics[0]:.4f}')
            print(f'F1 Score: {metrics[1]:.4f}')
            print(f'IoU: {metrics[2]:.4f}')
            print(f'Precision: {metrics[3]:.4f}')
            print(f'Recall: {metrics[4]:.4f}')
            print("\nEach class f1 score:")
            class_names_5 = ['Background', 'No Damage', 'Minor Damage', 'Major Damage', 'Destroyed']
            for i, f1 in enumerate(metrics[5]):
                print(f'{class_names_5[i]}: {f1:.4f}')
            
            print("\nBinary-classes (undamaged vs damaged):")
            print(f'Accuracy: {metrics[6]:.4f}')
            print(f'F1 Score: {metrics[7]:.4f}')
            print(f'IoU: {metrics[8]:.4f}')
            print(f'Precision: {metrics[9]:.4f}')
            print(f'Recall: {metrics[10]:.4f}')
            print("\nEach class f1 score:")
            class_names_2 = ['Background', 'Undamaged', 'Damaged']
            for i, f1 in enumerate(metrics[11]):
                print(f'{class_names_2[i]}: {f1:.4f}')
                
        except Exception as e:
            print(f"Error during evaluation: {e}")
if __name__ == '__main__':
    main()