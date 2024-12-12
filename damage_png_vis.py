import cv2
import numpy as np
import argparse

def visualize_damage(damage_path, output_path):
    """
    将损伤预测结果可视化为彩色图像
    0: 黑色 - 无损伤/背景
    1: 蓝色 - 轻微损伤
    2: 绿色 - 中等损伤
    3: 黄色 - 严重损伤
    4: 红色 - 完全损毁
    """
    # 读取损伤预测图像
    damage = cv2.imread(damage_path, cv2.IMREAD_GRAYSCALE)
    
    # 创建彩色输出图像
    colored_damage = np.zeros((damage.shape[0], damage.shape[1], 3), dtype=np.uint8)
    
    # 定义颜色映射
    # BGR 格式
    colors = {
        0: (0, 0, 0),      # 黑色
        1: (255, 0, 0),    # 蓝色
        2: (0, 255, 0),    # 绿色
        3: (0, 255, 255),  # 黄色
        4: (0, 0, 255)     # 红色
    }
    
    # 应用颜色映射
    for damage_class, color in colors.items():
        mask = (damage == damage_class)
        colored_damage[mask] = color
    
    # 保存结果
    cv2.imwrite(output_path, colored_damage)
    print(f"Saved colored visualization to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Visualize damage prediction with colors")
    parser.add_argument("--damage", type=str, required=True, help="Path to damage prediction image")
    parser.add_argument("--output", type=str, required=True, help="Path to save colored visualization")
    args = parser.parse_args()
    
    visualize_damage(args.damage, args.output)

if __name__ == "__main__":
    main()
    # Usage: python damage_png_vis.py --damage ./result/damage1.png --output colored_damage.png