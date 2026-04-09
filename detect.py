# 核心脚本
# 实时运行 YOLOv5 目标检测（默认使用摄像头输入），并根据检测到的目标（如二维码）计算其空间位置偏移，通过 datalink_serial 控制无人机跟踪或悬停。

# 工作流程：
# 启动两个后台线程：dl.drone() 接收飞控数据，dl.heartbeat() 发送心跳维持连接。
# 对每帧图像进行推理，若检测到目标则利用相机内参估算目标在三维空间中的相对位置（dx_m, dy_m, dz_m）。
# 调用 dl.set_pose() 调整无人机位置，实现视觉引导的自主飞行。
# 依赖：datalink_serial、YOLOv5 模型文件（如 v5lite-s.pt）、PyTorch 及相关工具库。


import argparse
import time
from pathlib import Path
import threading
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadPicamera2, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

import libcamera
from picamera2 import Picamera2
from datalink_serial import datalink

W_img = 1280  #图像宽度（像素）
H_img = 720
FOV_x = 77  #水平视场角（度）
FOV_y = 44 
W_real = 0.1  #目标的实际宽度（米）
H_real = 1.7  #目标实际高度
safe_distance = 1.0  #期望距离目标保持的距离
k = 1.0  #经验调节系数（目前为 1.0）
Kp_dx = 0.5
Kp_dy = 0.5
Kp_dalt = 0.5
Kp_dyaw = 0.3

def status_loop(dl):
    """每隔1秒打印一次无人机状态（高度、电池、位置等）"""
    while True:
        # 读取当前状态，若属性不存在则使用默认值
        alt = getattr(dl, 'relative_alt', 0.0)
        batt_v = getattr(dl, 'batt_voltage', 0.0)
        batt_i = getattr(dl, 'batt_current', 0.0)
        pos_x = getattr(dl, 'pos_x', 0.0)
        pos_y = getattr(dl, 'pos_y', 0.0)
        pos_z = getattr(dl, 'pos_z', 0.0)
        yaw = getattr(dl, 'att_yaw', 0.0)
        
        # 打印状态（使用 \r 可以在一行内刷新，但考虑到可能有其他输出，这里直接换行打印）
        print(f"[Status] Alt: {alt:.2f}m | Batt: {batt_v:.1f}V {batt_i:.1f}A | "
              f"Pos: ({pos_x:.1f}, {pos_y:.1f}, {pos_z:.1f}) | Yaw: {yaw:.2f}rad")
        time.sleep(1)


def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadPicamera2(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=[0], agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    # if int(c) == 0:
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    
                # 找到最高置信度索引
                max_conf_idx = torch.argmax(det[:, 4])

                # Write results
                for idx, (*xyxy, conf, cls) in enumerate(det):
                    # if int(cls) == 0:
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                    
                    if idx == max_conf_idx:    
                      # f_x = (W_img / 2) / np.tan(np.radians(FOV_x / 2))
                      f_x = 2120
                      # f_y = (H_img / 2) / np.tan(np.radians(FOV_y / 2))
                      f_y = 2120
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
      
                      dl.set_pose(Kp_dx * dx_1, Kp_dy * dy_1, Kp_dalt * d_alt_1, Kp_dyaw * d_yaw)
                    
                      print(f"[{frame}] {names[int(cls)]} conf={conf:.2f}, "f"depth={dz_m:.2f} m, "f"dx={dy_1:.3f} m, "f"dy={d_alt_1:.3f} m, "f"yaw={d_yaw:.3f} rad")

    
                    

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results 摄像头实时展示
            view_img = 0
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections) 保存视频
            save_img = 1
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')



if __name__ == '__main__':
    dl = datalink()
    data_thread = threading.Thread(target=dl.drone)
    heartbeat_thread = threading.Thread(target=dl.heartbeat)
    data_thread.start()
    heartbeat_thread.start()
    
    # 启动状态打印线程
    status_thread = threading.Thread(target=status_loop, args=(dl,), daemon=True)
    status_thread.start()
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='/home/pi/YOLOv5-Lite/weights/best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
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
    print(opt)
    check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
