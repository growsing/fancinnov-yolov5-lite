#!/usr/bin/env python3
"""
detect_control.py — 单字母命令遥控 + YOLO 视觉跟踪 + MiDaS 实时深度估计（双窗口显示）
命令映射：
  a -> arm (解锁)      d -> disarm (上锁)
  t -> takeoff (起飞)  l -> land (降落)
  k -> track (启动视觉跟踪 + 深度估计，双窗口)
  h -> halt (悬停/停止跟踪)
  q -> 在跟踪过程中输入 → 停止跟踪，返回命令菜单

特性：
- 左侧窗口：YOLO 目标检测与跟踪（含控制量标绘）
- 右侧窗口：MiDaS_small 实时彩色深度图（Inferno 映射）
- 同时保存原始跟踪视频和深度视频
- 状态日志与深度日志独立保存
"""

import argparse
import time
from pathlib import Path
import threading
import select
import sys
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from datetime import datetime

# ===== 原有 YOLO 导入 =====
from models.experimental import attempt_load
from utils.datasets import LoadPicamera2, LoadImages
from utils.general import (check_img_size, check_requirements, check_imshow,
                           non_max_suppression, apply_classifier,
                           scale_coords, xyxy2xywh, strip_optimizer, set_logging,
                           increment_path)
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

# ===== 树莓派摄像头与数据链 =====
from picamera2 import Picamera2
from datalink_serial import datalink

# ===== MiDaS 深度估计依赖 =====
try:
    import timm
except ImportError:
    raise ImportError("请先安装 timm: pip install timm")
# 可选：matplotlib 用于生成彩色深度图（若不安装可用 OpenCV 的 `applyColorMap` 代替）
try:
    import matplotlib.pyplot as plt
except ImportError:
    print("未安装 matplotlib，将使用 OpenCV color map 替代")

# MiDaS 全局模型（只加载一次）
midas_model = None
midas_transform = None
midas_device = None


def load_midas_once():
    """加载 MiDaS_small，缓存为全局变量"""
    global midas_model, midas_transform, midas_device
    if midas_model is not None:
        return
    midas_device = torch.device('cpu')   # 树莓派5 使用 CPU
    print(f"正在加载 MiDaS_small 模型到 {midas_device} ...")
    midas_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    midas_model.to(midas_device)
    midas_model.eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    midas_transform = midas_transforms.small_transform
    print("MiDaS_small 加载完成。")


def run_midas_full(rgb_image):
    """
    输入 RGB 图像 (H,W,3)，返回全分辨率深度图 (H,W) numpy 数组
    """
    input_batch = midas_transform(rgb_image).to(midas_device)
    with torch.no_grad():
        prediction = midas_model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=rgb_image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    return prediction.cpu().numpy()


def depth_to_colormap(depth_map, min_depth=None, max_depth=None):
    """
    将深度图转换为彩色图像 (BGR) 用于显示/保存
    """
    if min_depth is None:
        min_depth = np.min(depth_map)
    if max_depth is None:
        max_depth = np.max(depth_map)
    # 归一化到 0-1
    depth_norm = (depth_map - min_depth) / (max_depth - min_depth + 1e-8)
    depth_norm = np.clip(depth_norm, 0, 1)
    depth_uint8 = (depth_norm * 255).astype(np.uint8)
    # 使用 OpenCV 的 Inferno 颜色映射（需要 OpenCV >= 4.2，否则使用 JET 等）
    try:
        colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)
    except:
        colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
    return colored


def get_roi_depth_median(depth_map, xyxy):
    """
    从全图深度图中提取目标框内的中值深度（扩大10%框）
    """
    x1, y1, x2, y2 = map(int, xyxy)
    h, w = depth_map.shape
    # 扩大框：向内偏移一点，避免边缘噪声
    pad_w = max(1, int(0.1 * (x2 - x1)))
    pad_h = max(1, int(0.1 * (y2 - y1)))
    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = min(w, x2 + pad_w)
    y2 = min(h, y2 + pad_h)
    roi = depth_map[y1:y2, x1:x2]
    if roi.size == 0:
        return -1.0
    return float(np.median(roi))


# ================= 原有常量（保持不变） =================
W_img, H_img = 1280, 720
FOV_x, FOV_y = 77, 44
W_real = 0.1
H_real = 1.7
safe_distance = 0.8
k = 1.0
Kp_dx, Kp_dy, Kp_dalt, Kp_dyaw = 0.6, 0.6, 0.7, 0.3
YAW_SEARCH_SPEED = 0.2

dl = None
stop_track_event = threading.Event()


# ================= 状态线程（保留原样） =================
def status_loop(dl_obj, log_path):
    """每秒打印无人机状态并写入日志文件，低压时红色警告"""
    with open(log_path, 'a', buffering=1) as f:
        while True:
            alt = getattr(dl_obj, 'relative_alt', 0.0)
            batt_v = getattr(dl_obj, 'batt_voltage', 0.0)
            batt_i = getattr(dl_obj, 'batt_current', 0.0)
            pos_x = getattr(dl_obj, 'pos_x', 0.0)
            pos_y = getattr(dl_obj, 'pos_y', 0.0)
            pos_z = getattr(dl_obj, 'pos_z', 0.0)
            yaw = getattr(dl_obj, 'att_yaw', 0.0)

            ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            line = (f"[{ts}] Alt:{alt:.1f}m Batt:{batt_v:.1f}V "
                    f"Pos:({pos_x:.1f},{pos_y:.1f},{pos_z:.1f}) Yaw:{yaw:.2f}rad")
            print(line)
            f.write(line + "\n")

            if batt_v < 6.7 and batt_v > 0:
                warning = "\033[91m\033[1m!!! BATTERY LOW !!! PLEASE LAND !!!\033[0m"
                print(warning)
                f.write(warning + "\n")
            time.sleep(1)


# ================= 跟踪循环（双窗口显示 + 深度视频保存） =================
def tracking_loop(opt, dl_obj, base_dir):
    stop_track_event.clear()
    source, weights, save_txt, imgsz = opt.source, opt.weights, opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    view_img = check_imshow()
    save_dir = base_dir
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    # 深度日志文件
    depth_log_path = save_dir / f'depth_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
    depth_log_f = open(depth_log_path, 'w', encoding='utf-8')
    depth_log_f.write(f"Depth Log started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    depth_log_f.write("=" * 60 + "\n")
    depth_log_f.flush()

    # 设备与模型（YOLO）
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'
    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())
    imgsz = check_img_size(imgsz, s=stride)
    if half:
        model.half()

    # 数据加载
    vid_path, vid_writer = None, None
    depth_vid_path, depth_vid_writer = None, None   # 深度视频
    if webcam:
        cudnn.benchmark = True
        dataset = LoadPicamera2(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

    print("视觉跟踪 + 深度估计双窗口已启动。")
    if view_img:
        print("按 OpenCV 窗口 'q' 键停止跟踪。")
    print("在本终端输入 'q' + 回车 也能停止跟踪。")
    t0 = time.time()

    lost_start_time = None
    last_no_target_print = 0

    # 加载 MiDaS 模型（仅一次）
    load_midas_once()

    try:
        for path, img, im0s, vid_cap in dataset:
            # 非阻塞检测终端输入 'q'
            while sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                line = sys.stdin.readline().strip().lower()
                if line == 'q':
                    print("终端输入 'q'，停止跟踪。")
                    stop_track_event.set()
                    break
            if stop_track_event.is_set():
                break

            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()
            img /= 255.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # YOLO 推理
            t1 = time_synchronized()
            pred = model(img, augment=opt.augment)[0]
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,
                                       classes=[0], agnostic=opt.agnostic_nms)
            t2 = time_synchronized()
            inference_time = t2 - t1

            det = pred[0] if len(pred) > 0 else []
            im0 = im0s.copy() if isinstance(im0s, np.ndarray) else im0s[0].copy()

            # ===== ⭐ 每帧进行全图深度估计（无论有无目标） =====
            rgb_for_depth = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
            t_depth_start = time.time()
            depth_map = run_midas_full(rgb_for_depth)          # 全图深度
            depth_time = (time.time() - t_depth_start) * 1000  # ms

            # 生成彩色深度图用于显示和保存
            colored_depth = depth_to_colormap(depth_map)

            # 根据有无目标获取深度值（用于日志/控制显示）
            target_depth = -1.0
            if len(det) > 0:
                # 后续处理时会更新最可信目标的框，这里先占位，后面会重新计算
                pass

            # ========= 控制决策 =========
            if len(det) == 0:
                now = time.time()
                if lost_start_time is None:
                    lost_start_time = now
                    dl_obj.set_pose(0, 0, 0, 0)
                    print(f"[丢失目标] 悬停，开始计时 (5秒后自动旋转搜寻)")
                    depth_log_f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 丢失目标\n")
                else:
                    elapsed = now - lost_start_time
                    if elapsed >= 5.0:
                        dl_obj.set_pose(0, 0, 0, YAW_SEARCH_SPEED)
                        if elapsed - 5.0 < 0.5:
                            print(f"[自动旋转] 已丢失 {elapsed:.1f} 秒，偏航搜寻")
                    else:
                        dl_obj.set_pose(0, 0, 0, 0)

                if time.time() - last_no_target_print >= 1.0:
                    # 计算画面中心深度（简单均值）
                    h, w = depth_map.shape
                    center_depth = np.mean(depth_map[h//4:3*h//4, w//4:3*w//4])
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [YOLO] no target | "
                          f"[MiDaS] center_depth≈{center_depth:.2f}m")
                    last_no_target_print = time.time()
                    # 无目标时记录中心区域深度
                    h, w = depth_map.shape
                    center_depth = np.mean(depth_map[h//4:3*h//4, w//4:3*w//4])
                    depth_log_f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 无目标 center_depth={center_depth:.2f}m\n")
                    depth_log_f.flush()
            else:
                # 恢复跟踪
                if lost_start_time is not None:
                    print("[恢复跟踪] 重新检测到目标，停止旋转")
                    depth_log_f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 恢复跟踪\n")
                    lost_start_time = None
                    dl_obj.set_pose(0, 0, 0, 0)

                # 处理检测结果：只保留置信度最高的目标
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                max_conf_idx = torch.argmax(det[:, 4])

                for idx, (*xyxy, conf, cls) in enumerate(det):
                    if save_txt:
                        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)
                        frame_id = getattr(dataset, 'frame', 0)
                        txt_path = save_dir / 'labels' / f"{Path(path).stem}_{frame_id}" if dataset.mode != 'image' else save_dir / 'labels' / Path(path).stem
                        txt_path = str(txt_path) + '.txt'
                        with open(txt_path, 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if idx == max_conf_idx:
                        # 绘制检测框
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                        # 原有控制量计算
                        f_x = f_y = 2120
                        x1, y1, x2, y2 = xyxy
                        W_qr = x2 - x1
                        H_qr = y2 - y1
                        cx_qr = (x1 + x2) / 2
                        cy_qr = (y1 + y2) / 2
                        cx_img = W_img / 2
                        cy_img = H_img / 2
                        dx = cx_qr - cx_img
                        dy = cy_qr - cy_img
                        angle_x_rad = k * np.arctan(dx / f_x)
                        dz_m = k * (W_real * f_x) / W_qr
                        dy_m = k * (dy / f_y) * dz_m
                        dx_m = k * (dx / f_x) * dz_m
                        dx_1 = dz_m - safe_distance
                        dy_1 = dx_m
                        d_alt_1 = -dy_m
                        d_yaw = angle_x_rad

                        # 从全图深度图中提取目标区域深度
                        target_depth = get_roi_depth_median(depth_map, xyxy)

                        # 发送控制指令
                        try:
                            dl_obj.set_pose(Kp_dx * dx_1, Kp_dy * dy_1,
                                            Kp_dalt * d_alt_1, Kp_dyaw * d_yaw)
                        except Exception as e:
                            print(f"控制发送失败: {e}")

                        # 终端输出（含深度信息）
                        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        print(f"[{ts}] [YOLO] conf={conf:.2f} dist={dz_m:.2f}m infer={inference_time:.2f}s | "
                              f"[MiDaS] depth={target_depth:.2f}m cost={depth_time:.0f}ms")
                        depth_log_f.write(f"[{ts}] DETECT drone conf={conf:.2f} "
                                          f"model_dist={dz_m:.2f}m midas_depth={target_depth:.2f}m "
                                          f"infer={inference_time:.3f}s depth_ms={depth_time:.0f}\n")
                        depth_log_f.flush()

            # ===== 双窗口显示 =====
            if view_img:
                # 左窗口：YOLO 跟踪结果
                cv2.namedWindow("YOLOv5 Tracking", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("YOLOv5 Tracking", 640, 360)
                cv2.imshow("YOLOv5 Tracking", im0)

                # 右窗口：MiDaS 深度图（彩色）
                cv2.namedWindow("MiDaS Depth", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("MiDaS Depth", 640, 360)
                cv2.imshow("MiDaS Depth", colored_depth)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("窗口 'q' 键按下，停止跟踪。")
                    stop_track_event.set()
                    break
            else:
                # 即使不显示窗口，也要执行 waitKey 使 OpenCV 事件循环运行（避免卡死）
                cv2.waitKey(1)

            # ===== 保存视频（原图 + 深度图） =====
            if save_img:
                # 原始跟踪视频
                if dataset.mode == 'image':
                    cv2.imwrite(str(save_dir / Path(path).name), im0)
                else:
                    if vid_path != str(save_dir / 'output.mp4'):
                        vid_path = str(save_dir / 'output.mp4')
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()
                        if vid_cap:
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:
                            fps, w, h = 15, im0.shape[1], im0.shape[0]   # 未获取到摄像头信息时默认15fps
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        vid_writer = cv2.VideoWriter(vid_path, fourcc, fps, (w, h))

                    vid_writer.write(im0)

                # 深度视频
                if depth_vid_path != str(save_dir / 'depth_output.mp4'):
                    depth_vid_path = str(save_dir / 'depth_output.mp4')
                    if isinstance(depth_vid_writer, cv2.VideoWriter):
                        depth_vid_writer.release()
                    # 深度图尺寸可能与原图相同
                    dh, dw = colored_depth.shape[:2]
                    if vid_cap:
                        depth_fps = fps
                    else:
                        depth_fps = 15
                    depth_vid_writer = cv2.VideoWriter(depth_vid_path, fourcc, depth_fps, (dw, dh))

                depth_vid_writer.write(colored_depth)

            if stop_track_event.is_set():
                break

    finally:
        depth_log_f.close()
        if vid_writer is not None:
            vid_writer.release()
        if depth_vid_writer is not None:
            depth_vid_writer.release()
        cv2.destroyAllWindows()
        if save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
            print(f"结果保存至 {save_dir}{s}")
        print(f"跟踪循环结束 (总耗时 {time.time() - t0:.3f}s)")


# ================= 命令处理（保留所有功能） =================
def execute_command(cmd_char, opt, base_dir):
    global dl, stop_track_event
    cmd_char = cmd_char.lower()

    if cmd_char == 'a':
        dl.set_arm()
        print("解锁(arm)指令已发送")
    elif cmd_char == 'd':
        dl.set_disarm()
        print("上锁(disarm)指令已发送")
    elif cmd_char == 't':
        dl.set_takeoff()
        print("起飞(takeoff)指令已发送")
    elif cmd_char == 'l':
        dl.set_land()
        print("降落(land)指令已发送")
    elif cmd_char == 'k':
        stop_track_event.set()
        time.sleep(0.5)
        stop_track_event.clear()
        tracking_loop(opt, dl, base_dir)
    elif cmd_char == 'h':
        dl.set_pose(0, 0, 0, 0)
        stop_track_event.set()
        print("悬停(halt)，跟踪已停止")
    else:
        print("? 支持命令: a(解锁) d(上锁) t(起飞) l(降落) k(跟踪+深度) h(悬停)")


# ================= 主入口（保持不变） =================
def main():
    global dl

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,
                        default='/home/pi/YOLOv5-Lite/weights/best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='0', help='file/folder, 0 for webcam')
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.06, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.2, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print("参数:", opt)
    check_requirements(exclude=('pycocotools', 'thop'))

    base_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    base_dir.mkdir(parents=True, exist_ok=True)

    status_log_path = base_dir / f"status_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    dl = datalink()
    threading.Thread(target=dl.drone, daemon=True).start()
    threading.Thread(target=dl.heartbeat, daemon=True).start()
    threading.Thread(target=status_loop, args=(dl, str(status_log_path)), daemon=True).start()

    print("\n===== 无人机单键遥控 + 深度跟踪双窗口 =====")
    print("命令: a = 解锁 d = 上锁 t = 起飞 l = 降落 k = 跟踪+深度双窗 h = 悬停")
    print(f"输出目录: {base_dir}")
    print(f"状态日志: {status_log_path.name}")
    print("按 Ctrl+C 退出\n")

    try:
        while True:
            cmd_input = input(">> ").strip().lower()
            if cmd_input:
                if len(cmd_input) == 1:
                    execute_command(cmd_input, opt, base_dir)
                else:
                    print("只接受单字母命令: a/d/t/l/k/h")
    except KeyboardInterrupt:
        print("\n用户退出。")
    finally:
        stop_track_event.set()
        print("程序已结束。")


if __name__ == '__main__':
    main()