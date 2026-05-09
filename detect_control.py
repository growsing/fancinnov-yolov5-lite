#!/usr/bin/env python3
"""
detect_control.py — 单字母命令遥控脚本（适合远程 SSH）
命令映射：
  a -> arm (解锁)
  d -> disarm (上锁)
  t -> takeoff (起飞)
  l -> land (降落)
  k -> track (启动视觉跟踪)
  h -> halt (悬停/停止跟踪)
  q -> (在跟踪过程中输入) 停止跟踪，返回命令菜单

状态信息每秒打印到终端，同时写入 exp 目录下的状态日志文件。
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

from models.experimental import attempt_load
from utils.datasets import LoadPicamera2, LoadImages
from utils.general import (check_img_size, check_requirements, check_imshow,
                           non_max_suppression, apply_classifier,
                           scale_coords, xyxy2xywh, strip_optimizer, set_logging,
                           increment_path)
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

import libcamera
from picamera2 import Picamera2
from datalink_serial import datalink

# ================= 常量 =================
W_img, H_img = 1280, 720
FOV_x, FOV_y = 77, 44
W_real = 0.1
H_real = 1.7
safe_distance = 1.0
k = 1.0
Kp_dx, Kp_dy, Kp_dalt, Kp_dyaw = 0.6, 0.6, 0.7, 0.3

YAW_SEARCH_SPEED = 0.2      # 自动旋转搜寻速度 (rad/s)

dl = None
stop_track_event = threading.Event()

# ================= 状态线程 =================
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

# ================= 跟踪循环（修复显示与输出） =================
def tracking_loop(opt, dl_obj, base_dir):
    stop_track_event.clear()
    source, weights, save_txt, imgsz = opt.source, opt.weights, opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    view_img = check_imshow()  # 如果支持X11转发就显示窗口
    save_dir = base_dir
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    det_log_path = save_dir / f'detection_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
    log_f = open(det_log_path, 'w', encoding='utf-8')
    log_f.write(f"Log started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_f.write("=" * 60 + "\n")
    log_f.flush()

    # 设备与模型
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'
    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())
    imgsz = check_img_size(imgsz, s=stride)
    if half:
        model.half()
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # 数据加载
    vid_path, vid_writer = None, None
    if webcam:
        cudnn.benchmark = True
        dataset = LoadPicamera2(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

    print("视觉跟踪已启动。")
    if view_img:
        print("按 OpenCV 窗口 'q' 键停止跟踪。")
    print("在本终端输入 'q' + 回车 也能停止跟踪。")
    t0 = time.time()

    # 目标丢失计时
    lost_start_time = None          # None 表示当前有目标
    last_no_target_print = 0        # 上次输出“未检测到目标”的时间戳

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

            t1 = time_synchronized()
            pred = model(img, augment=opt.augment)[0]
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,
                                       classes=[0], agnostic=opt.agnostic_nms)
            t2 = time_synchronized()
            inference_time = t2 - t1

            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            # 注意：每个 batch 可能有多张图，这里按照原逻辑只取第一张（webcam 时只有一个）
            # 简化处理：只处理第一个 detection 结果
            det = pred[0] if len(pred) > 0 else []
            im0 = im0s.copy() if isinstance(im0s, np.ndarray) else im0s[0].copy()

            # ========= 控制决策（有/无目标） =========
            if len(det) == 0:
                now = time.time()
                if lost_start_time is None:
                    lost_start_time = now
                    dl_obj.set_pose(0, 0, 0, 0)
                    print(f"[丢失目标] 悬停，开始计时 (5秒后自动旋转搜寻)")
                    log_f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 丢失目标，悬停\n")
                    log_f.flush()
                else:
                    elapsed = now - lost_start_time
                    if elapsed >= 5.0:
                        # 自动旋转搜寻
                        dl_obj.set_pose(0, 0, 0, YAW_SEARCH_SPEED)
                        if elapsed - 5.0 < 0.5:  # 仅首次进入旋转时打印一次
                            print(f"[自动旋转] 已丢失 {elapsed:.1f} 秒，开始缓慢偏航搜寻 (速度 {YAW_SEARCH_SPEED} rad/s)")
                            log_f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 自动旋转搜寻启动\n")
                            log_f.flush()
                    else:
                        dl_obj.set_pose(0, 0, 0, 0)
                # 终端输出“未检测到目标”（每秒最多一次）
                if time.time() - last_no_target_print >= 1.0:
                    print("[状态] 未检测到目标")
                    last_no_target_print = time.time()
                    log_f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 未检测到目标\n")
                    log_f.flush()
            else:
                # 有目标：重置丢失计时
                if lost_start_time is not None:
                    print("[恢复跟踪] 重新检测到目标，停止旋转")
                    log_f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 恢复跟踪\n")
                    log_f.flush()
                    lost_start_time = None
                    dl_obj.set_pose(0, 0, 0, 0)   # 先停止旋转，后面会覆盖

                # 处理检测结果，计算控制量
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                max_conf_idx = torch.argmax(det[:, 4])
                best_target_info = None

                for idx, (*xyxy, conf, cls) in enumerate(det):
                    if save_txt:
                        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)
                        # 构建标签保存路径（保持与原代码一致）
                        frame_id = getattr(dataset, 'frame', 0)
                        txt_path = save_dir / 'labels' / f"{Path(path).stem}_{frame_id}" if dataset.mode != 'image' else save_dir / 'labels' / Path(path).stem
                        txt_path = str(txt_path) + '.txt'
                        with open(txt_path, 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    # 只在置信度最高的目标上绘制框和控制
                    if idx == max_conf_idx:
                        # 在图像上绘制检测框
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                        # 计算控制量
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

                        try:
                            dl_obj.set_pose(Kp_dx * dx_1, Kp_dy * dy_1,
                                            Kp_dalt * d_alt_1, Kp_dyaw * d_yaw)
                        except Exception as e:
                            print(f"控制发送失败: {e}")

                        best_target_info = {
                            'class': "drone",
                            'conf': conf,
                            'depth': dz_m,
                            'dx': dy_1,
                            'dy': d_alt_1,
                            'yaw': d_yaw
                        }

                        # 终端输出识别信息
                        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        print(f"[{ts}] 检测到目标 | 置信度:{conf:.2f} 距离:{dz_m:.2f}m "
                              f"dx:{dy_1:.3f}m dy:{d_alt_1:.3f}m yaw:{d_yaw:.3f}rad 推理时间:{inference_time:.3f}s")
                        log_f.write(f"[{ts}] DETECT drone conf={conf:.2f} depth={dz_m:.2f}m "
                                    f"dx={dy_1:.3f}m dy={d_alt_1:.3f}m yaw={d_yaw:.3f}rad infer={inference_time:.3f}s\n")
                        log_f.flush()

            # ========= 图像显示（无论有无目标都显示） =========
            if view_img:
                win_name = "YOLOv5 Tracking"
                cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(win_name, 640, 360)
                cv2.imshow(win_name, im0)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("窗口 'q' 键按下，停止跟踪。")
                    stop_track_event.set()
                    break

            # ========= 保存图像/视频 =========
            if save_img:
                if dataset.mode == 'image':
                    save_path = str(save_dir / Path(path).name)
                    cv2.imwrite(save_path, im0)
                else:
                    # 视频写入
                    if vid_path != str(save_dir / 'output.mp4'):
                        vid_path = str(save_dir / 'output.mp4')
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()
                        if vid_cap:
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        vid_writer = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

            if stop_track_event.is_set():
                break

    finally:
        log_f.close()
        if vid_writer is not None:
            vid_writer.release()
        cv2.destroyAllWindows()
        if save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
            print(f"结果保存至 {save_dir}{s}")
        print(f"跟踪循环结束 (总耗时 {time.time() - t0:.3f}s)")

# ================= 单字母命令处理 =================
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
        print("? 支持命令: a(解锁) d(上锁) t(起飞) l(降落) k(跟踪) h(悬停)")

# ================= 主入口 =================
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

    print("\n===== 无人机单键遥控 =====")
    print("命令: a = 解锁 d = 上锁 t = 起飞 l = 降落 k = 跟踪 h = 悬停")
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