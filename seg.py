import os
import cv2
import numpy as np
import pandas as pd



def get_vid_frame(video_path):
    """
    从视频中读取第一帧返回，关闭硬件加速规避buffer报错
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_HW_ACCELERATION, 0)
    
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")

    ret, first_frame = cap.read()
    cap.release()

    if not ret or first_frame is None:
        raise ValueError(f"无法读取第一帧: {video_path}")

    return first_frame


def final_optimized_vessel_segmentation(frame, output_path):
    """
    单张眼底OCTA图像血管分割，保存mask
    """
    # 1. 图像已读入，直接处理
    img = frame
    h, w = img.shape[:2]

    # 2. 裁剪眼底圆形区域（去除背景噪声）
    mask_circle = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask_circle, (w//2, h//2), min(w, h)//2 - 15, 255, -1)
    img_masked = cv2.bitwise_and(img, img, mask=mask_circle)

    # 3. 预处理：绿色通道+多步去噪增强
    green_channel = img_masked[:, :, 1]
    
    # 3.1 双边滤波（去噪同时保留边缘）
    denoised = cv2.bilateralFilter(green_channel, d=5, sigmaColor=75, sigmaSpace=75)
    
    # 3.2 CLAHE增强
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    # 4. 血管分割：双阈值+形态学组合
    # 4.1 自适应阈值分割
    thresh = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 17, 4
    )
    
    # 4.2 Laplacian边缘检测强化血管
    laplacian = cv2.Laplacian(enhanced, cv2.CV_64F)
    laplacian = cv2.convertScaleAbs(laplacian)
    _, laplacian_thresh = cv2.threshold(laplacian, 10, 255, cv2.THRESH_BINARY)
    
    # 4.3 合并两种分割结果
    combined = cv2.bitwise_or(thresh, laplacian_thresh)

    # 5. 形态学处理：轻量去噪+血管连接
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_open, iterations=1)
    
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    # 6. 连通域过滤：去掉白点噪声 + 保留血管
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 5  
    filtered = np.zeros_like(cleaned)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            cv2.drawContours(filtered, [cnt], -1, 255, thickness=cv2.FILLED)

    # 7. 去掉圆形边缘
    inner_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(inner_mask, (w//2, h//2), min(w, h)//2 - 35, 255, -1)
    final_mask = cv2.bitwise_and(filtered, filtered, mask=inner_mask)

    # 8. 保存结果
    cv2.imwrite(output_path, final_mask)
    print(f"✅ 已保存mask: {output_path}")
    return final_mask


def main():
    # ---------------------- 配置路径 ----------------------
    # 1. 数据集根目录
    root_dir = "./data/OCTAvideo"
    # 2. Excel标签文件路径（请替换为你的实际路径）
    excel_path = "./data/视频对应标签(脑卒)_扩增版.xlsx"
    # 3. mask保存根目录
    save_root_1 = "./data/output_seg/1"
    save_root_0 = "./data/output_seg/0"
    # ------------------------------------------------------

    # 确保保存目录存在
    os.makedirs(save_root_1, exist_ok=True)
    os.makedirs(save_root_0, exist_ok=True)

    # 读取Excel标签
    df = pd.read_excel(excel_path)
    # 构建标签字典：key=视频文件名，value=1/0（是=1，否=0）
    label_dict = {}
    for _, row in df.iterrows():
        video_name = str(row["AVI文件名"]).strip()
        label = 1 if str(row["是否脑卒"]).strip() == "是" else 0
        label_dict[video_name] = label
    print(f"📋 共加载 {len(label_dict)} 个视频标签")

    # 遍历所有子文件夹（1028, 1101, 1103...）
    for sub_dir in os.listdir(root_dir):
        sub_path = os.path.join(root_dir, sub_dir)
        if not os.path.isdir(sub_path):
            continue
        print(f"\n📂 正在处理文件夹: {sub_path}")

        # 【修改点】同时匹配 avi / mp4 两种格式，大小写兼容
        video_files = []
        for f in os.listdir(sub_path):
            lower_f = f.lower()
            if lower_f.endswith(".avi") or lower_f.endswith(".mp4"):
                video_files.append(f)
        print(f"📹 找到 {len(video_files)} 个视频文件(avi/mp4)")

        for video_name in video_files:
            if video_name not in label_dict:
                print(f"⚠️  标签中未找到视频 {video_name}，跳过")
                continue

            label = label_dict[video_name]
            video_path = os.path.join(sub_path, video_name)
            
            # 确定mask保存路径（文件名与视频同名，后缀改为.png）
            mask_save_name = os.path.splitext(video_name)[0] + ".png"
            if label == 1:
                mask_save_path = os.path.join(save_root_1, mask_save_name)
            else:
                mask_save_path = os.path.join(save_root_0, mask_save_name)

            try:
                # 步骤1：从视频中提取最清晰的一帧
                print(f"🔍 正在处理视频: {video_name}")
                best_frame = get_vid_frame(video_path)
                
                # 步骤2：血管分割并保存mask
                final_optimized_vessel_segmentation(best_frame, mask_save_path)
            except Exception as e:
                print(f"❌ 处理失败: {video_name}, 错误: {e}")


if __name__ == "__main__":
    main()