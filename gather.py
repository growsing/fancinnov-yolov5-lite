#!/usr/bin/env python3
"""
================================================================================
无人机红外聚集运动模块 - IR Swarm Aggregation Module
================================================================================
"""

import time
import threading
import sys
from pathlib import Path
from datetime import datetime
import math

# 红外硬件依赖
import lgpio
import spidev
from rpi_hardware_pwm import HardwarePWM

# 无人机数据链
from datalink_serial import datalink

# 串口解析依赖（UWB）
try:
    import serial
except ImportError:
    serial = None

# ═══════════════════════════════════════════════════════════════════════════════
# 配置区
# ═══════════════════════════════════════════════════════════════════════════════

# ---- UWB 配置 ----
UWB_PORT = '/dev/ttyACM0'
UWB_BAUD = 921600

# ---- 红外硬件配置 ----
PWM_CHANNEL = 0
PWM_FREQ = 38000
PWM_DUTY = 50
CHIP = 4
IR_GPIO = 22
SPI_BUS = 0
SPI_DEVICE = 0
SPI_SPEED = 1000000

# ---- NEC 时序参数 (单位: 微秒) ----
NEC_HEADER_MARK = 9000
NEC_HEADER_SPACE = 4500
NEC_BIT_MARK = 562
NEC_ZERO_SPACE = 562
NEC_ONE_SPACE = 1687
NEC_TOLERANCE = 400
NEC_REPEAT_SPACE = 2250
IDLE_TIMEOUT_US = 120000
SAMPLE_INTERVAL_NS = 200000

# ---- 聚集模式参数 ----
SWARM_ADDR = 0xAA
SWARM_CMD = 0xBB
TX_INTERVAL = 0.25

# ---- 方向映射 (MCP3008 通道 → 方向) ----
CHANNEL_TO_DIR = {0: "右", 1: "后", 2: "左", 3: "前"}
CHANNEL_TO_ANGLE = {0: 90, 1: 180, 2: 270, 3: 0}

# ---- 四向运动控制 (dx, dy, d_alt, d_yaw) ----
DIR_TO_CONTROL = {
    "前": (0.3, 0.0, 0.0, 0.0),
    "右": (0.0, 0.3, 0.0, 0.0),
    "后": (-0.3, 0.0, 0.0, 0.0),
    "左": (0.0, -0.3, 0.0, 0.0),
}

# ---- 底噪配置 ----
NOISE_FLOOR = 590

# ---- 距离控制阈值 ----
MOD_RELIABLE = 4
MOD_APPROACH = 50
MOD_HOLD = 80
MOD_RETREAT = 180

# ---- 运动速度参数 ----
SPEED_APPROACH = 0.1
SPEED_RETREAT = -0.1
YAW_SEARCH_SPEED = 0.2

# ---- 控制增益 ----
Kp_dx, Kp_dy, Kp_dalt, Kp_dyaw = 0.6, 0.6, 0.7, 0.3

# ---- 场地边界 (m) ----
FIELD_X_MIN, FIELD_X_MAX = 0.0, 5.6
FIELD_Y_MIN, FIELD_Y_MAX = 0.0, 4.6
BOUNDARY_MARGIN = 0.5

# ═══════════════════════════════════════════════════════════════════════════════
# 全局变量
# ═══════════════════════════════════════════════════════════════════════════════

dl = None
gather_running = False
gather_thread = None
stop_gather_event = threading.Event()

# 红外硬件句柄
pwm = None
handle = None
adc = None

# UWB 全局状态
uwb_pos = {'x': None, 'y': None, 'z': None, 'valid': False}
uwb_lock = threading.Lock()
uwb_thread = None
uwb_running = False

# 日志句柄
log_file = None
log_lock = threading.Lock()

# 统计信息
# raw_adc: 四通道原始 ADC 最大值缓存，仅在接收到红外信号时更新，
# 未更新时保留上一次有效值，避免为日志单独读取硬件
stats = {
    'last_peer_direction': "无",
    'last_predicted_angle': None,
    'last_max_reading': 0,
    'last_modulus': 0,
    'last_action': "待机",
    'last_uwb_pos': "无信号",
    'last_batt': 0.0,
    'raw_adc': [0.0, 0.0, 0.0, 0.0],
}
stats_lock = threading.Lock()

# ═══════════════════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════════════════

def now_str():
    return time.strftime("%H:%M:%S")


def now_ms_str():
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def delay_us(us):
    start = time.perf_counter_ns()
    target = start + us * 1000
    while time.perf_counter_ns() < target:
        pass


def adc_read_all():
    vals = []
    for ch in range(4):
        resp = adc.xfer2([1, (8 + ch) << 4, 0])
        vals.append(((resp[1] & 3) << 8) + resp[2])
    return vals


def in_range(value, target):
    return abs(value - target) < NEC_TOLERANCE


def log_write(line: str):
    """
    写入日志行。
    注意：调用方需自行提供完整格式化的行内容（含时间戳），
    本函数仅负责线程安全地写入文件并 flush，不再额外添加前缀。
    """
    global log_file
    if log_file is None:
        return
    with log_lock:
        try:
            log_file.write(line + "\n")
            log_file.flush()
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# UWB 位置读取 (Node_Frame2 实时解析)
# ═══════════════════════════════════════════════════════════════════════════════

def uwb_loop():
    global uwb_pos, uwb_running
    if serial is None:
        return
    try:
        ser = serial.Serial(UWB_PORT, UWB_BAUD, timeout=0.05)
    except Exception:
        return

    buf = bytearray()
    while uwb_running:
        try:
            chunk = ser.read(ser.in_waiting or 1)
            if not chunk:
                continue
            buf.extend(chunk)

            while len(buf) >= 4:
                idx = buf.find(b'\x55\x04')
                if idx == -1:
                    buf = buf[-1:] if buf[-1] == 0x55 else bytearray()
                    break
                if len(buf) < idx + 4:
                    buf = buf[idx:]
                    break

                frame_len = int.from_bytes(buf[idx+2:idx+4], 'little')
                if not (120 <= frame_len <= 512):
                    buf = buf[idx+1:]
                    continue
                if len(buf) < idx + frame_len:
                    buf = buf[idx:]
                    break

                frame = buf[idx:idx+frame_len]
                if (sum(frame[:-1]) & 0xFF) == frame[-1]:
                    try:
                        def i24(b):
                            return int.from_bytes(b, 'little', signed=True)
                        px = i24(frame[13:16]) / 1000.0
                        py = i24(frame[16:19]) / 1000.0
                        pz = i24(frame[19:22]) / 1000.0
                        with uwb_lock:
                            uwb_pos = {'x': px, 'y': py, 'z': pz, 'valid': True}
                    except Exception:
                        pass
                buf = buf[idx+frame_len:]
        except Exception:
            time.sleep(0.1)

    try:
        ser.close()
    except Exception:
        pass


def get_uwb_pos():
    with uwb_lock:
        return uwb_pos.copy()


# ═══════════════════════════════════════════════════════════════════════════════
# 信号分析
# ═══════════════════════════════════════════════════════════════════════════════

def compute_signals(all_max_vals, noise_floor=NOISE_FLOOR):
    return [max(0, v - noise_floor) for v in all_max_vals]


def compute_modulus(signal_vals):
    return math.sqrt(sum(v ** 2 for v in signal_vals))


def analyze_signal(samples, noise_floor=NOISE_FLOOR):
    if not samples or len(samples) == 0:
        return "未知", 0, 0.0, [0, 0, 0, 0]

    all_max_vals = [max(s[i] for s in samples) for i in range(4)]
    best_idx = all_max_vals.index(max(all_max_vals))
    max_reading = all_max_vals[best_idx]
    direction = CHANNEL_TO_DIR.get(best_idx, "未知")
    signal_vals = compute_signals(all_max_vals, noise_floor)
    modulus = compute_modulus(signal_vals)

    return direction, max_reading, modulus, all_max_vals


def predict_direction(all_max_vals, noise_floor=NOISE_FLOOR):
    signal_vals = compute_signals(all_max_vals, noise_floor)
    total_signal = sum(signal_vals)

    if total_signal == 0:
        return None

    x_sum = 0.0
    y_sum = 0.0

    for i, val in enumerate(signal_vals):
        angle_rad = math.radians(CHANNEL_TO_ANGLE[i])
        x_sum += val * math.cos(angle_rad)
        y_sum += val * math.sin(angle_rad)

    predicted_angle = math.degrees(math.atan2(y_sum, x_sum))
    if predicted_angle < 0:
        predicted_angle += 360

    return predicted_angle


# ═══════════════════════════════════════════════════════════════════════════════
# 红外发射模块
# ═══════════════════════════════════════════════════════════════════════════════

def send_nec_frame(address, command):
    pwm.start(PWM_DUTY)
    delay_us(NEC_HEADER_MARK)
    pwm.stop()
    delay_us(NEC_HEADER_SPACE)

    data_bytes = [
        address & 0xFF,
        (~address) & 0xFF,
        command & 0xFF,
        (~command) & 0xFF
    ]

    for byte in data_bytes:
        for i in range(8):
            bit = (byte >> i) & 1
            pwm.start(PWM_DUTY)
            delay_us(NEC_BIT_MARK)
            pwm.stop()
            delay_us(NEC_ONE_SPACE if bit else NEC_ZERO_SPACE)

    pwm.start(PWM_DUTY)
    delay_us(NEC_BIT_MARK)
    pwm.stop()


# ═══════════════════════════════════════════════════════════════════════════════
# 红外接收模块
# ═══════════════════════════════════════════════════════════════════════════════

def read_frame():
    timeout_start = time.time_ns() // 1000
    while lgpio.gpio_read(handle, IR_GPIO) == 1:
        if (time.time_ns() // 1000 - timeout_start) > 500000:
            return None, None

    pulses = []
    samples = []
    last_state = 0
    last_time = time.time_ns() // 1000
    last_sample = time.time_ns()

    while True:
        current_state = lgpio.gpio_read(handle, IR_GPIO)
        current_time = time.time_ns() // 1000

        if current_state != last_state:
            duration = current_time - last_time
            pulses.append((last_state, duration))
            last_state = current_state
            last_time = current_time

        if len(pulses) == 0 and current_state == 0:
            now_ns = time.time_ns()
            if (now_ns - last_sample) >= SAMPLE_INTERVAL_NS:
                samples.append(adc_read_all())
                last_sample = now_ns

        if (current_time - last_time) > IDLE_TIMEOUT_US and len(pulses) > 0:
            pulses.append((last_state, current_time - last_time))
            break

    return pulses, samples


def decode_pulses(pulses):
    if not pulses or len(pulses) < 4:
        return None

    for i in range(len(pulses) - 1):
        mark_dur = pulses[i][1]
        space_dur = pulses[i+1][1]

        if in_range(mark_dur, NEC_HEADER_MARK):
            if in_range(space_dur, NEC_HEADER_SPACE):
                return decode_data_bits(pulses, i + 2)
            elif in_range(space_dur, NEC_REPEAT_SPACE):
                return {'repeat': True, 'valid': True, 'address': None, 'command': None}

    return None


def decode_data_bits(pulses, start_idx):
    bits = []
    idx = start_idx

    while len(bits) < 32 and idx < len(pulses) - 1:
        mark_dur = pulses[idx][1]
        space_dur = pulses[idx+1][1]

        if mark_dur < 200:
            break
        if space_dur > 1000:
            bits.append(1)
        elif space_dur > 200:
            bits.append(0)
        else:
            break
        idx += 2

    if len(bits) != 32:
        return {'valid': False, 'error': f'仅收到 {len(bits)} 位'}

    address = sum(bits[i] << i for i in range(8))
    address_inv = sum(bits[i+8] << i for i in range(8))
    command = sum(bits[i+16] << i for i in range(8))
    command_inv = sum(bits[i+24] << i for i in range(8))

    valid = ((address ^ address_inv) == 0xFF) and ((command ^ command_inv) == 0xFF)

    return {
        'address': address,
        'command': command,
        'valid': valid,
        'repeat': False
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 运动决策 (基于向量模长)
# ═══════════════════════════════════════════════════════════════════════════════

def decide_movement(direction, modulus):
    if modulus < MOD_RELIABLE:
        return 0.0, 0.0, 0.0, YAW_SEARCH_SPEED, f"搜寻 模={modulus:.2f}"

    if direction == "未知":
        return 0.0, 0.0, 0.0, YAW_SEARCH_SPEED, f"搜寻 模={modulus:.2f}"

    if modulus > MOD_RETREAT:
        base = DIR_TO_CONTROL.get(direction, (0.0, 0.0, 0.0, 0.0))
        dx = -base[0] if base[0] != 0 else 0.0
        dy = -base[1] if base[1] != 0 else 0.0
        return dx, dy, 0.0, 0.0, f"远离{direction}"

    elif modulus > MOD_HOLD:
        return 0.0, 0.0, 0.0, 0.0, f"保持{direction}"

    else:
        base = DIR_TO_CONTROL.get(direction, (0.0, 0.0, 0.0, 0.0))
        return base[0], base[1], 0.0, 0.0, f"靠近{direction}"


# ═══════════════════════════════════════════════════════════════════════════════
# 聚集主循环
# ═══════════════════════════════════════════════════════════════════════════════

def gather_loop():
    global gather_running
    # 记录聚集模式启动时刻，用于计算运行时长
    start_time = time.time()
    last_tx_time = 0
    last_peer_time = 0
    # 日志与终端输出计时器，与 TX_INTERVAL（0.25s）对齐，
    # 保证记录频率与发射/决策周期一致
    last_log_time = time.time()
    lost_peer_time = None
    last_boundary_print = 0
    boundary_alert_active = False

    print(f"[{now_str()}] 🟢 聚集模式已启动")
    print(f"   发射地址: 0x{SWARM_ADDR:02X}  命令: 0x{SWARM_CMD:02X}")
    print(f"   底噪读数: {NOISE_FLOOR}")
    print(f"   模长可靠阈值: {MOD_RELIABLE}")
    print(f"   靠近阈值: 模长 < {MOD_APPROACH}")
    print(f"   保持阈值: {MOD_HOLD}")
    print(f"   远离阈值: 模长 > {MOD_RETREAT}")
    print(f"   场地边界: X[{FIELD_X_MIN},{FIELD_X_MAX}] Y[{FIELD_Y_MIN},{FIELD_Y_MAX}] 安全边距:{BOUNDARY_MARGIN}m")

    try:
        while gather_running and not stop_gather_event.is_set():
            # ── 初始化本轮循环的边界与 UWB 状态变量 ──
            # 确保在后续日志输出块中始终可访问，避免未定义
            in_boundary = False
            uwb_x = uwb_y = uwb_z = ""
            pos_str = "无信号"

            # ── 0. UWB 位置读取与边界检测 ──
            uwb = get_uwb_pos()

            if uwb['valid'] and uwb['x'] is not None:
                x, y, z = uwb['x'], uwb['y'], uwb['z']
                # UWB 坐标保留两位小数，用于日志与终端显示
                uwb_x = f"{x:.2f}"
                uwb_y = f"{y:.2f}"
                uwb_z = f"{z:.2f}"
                pos_str = f"({uwb_x},{uwb_y},{uwb_z})"
                if (x < FIELD_X_MIN + BOUNDARY_MARGIN or
                    x > FIELD_X_MAX - BOUNDARY_MARGIN or
                    y < FIELD_Y_MIN + BOUNDARY_MARGIN or
                    y > FIELD_Y_MAX - BOUNDARY_MARGIN):
                    in_boundary = True
                    if time.time() - last_boundary_print >= 1.0:
                        print(f"[{now_str()}] 🚨 边界警报！位置({x:.2f},{y:.2f})靠近边界，强制悬停！")
                        last_boundary_print = time.time()
                    dl.set_pose(0, 0, 0, 0)
                    with stats_lock:
                        stats['last_action'] = "边界悬停"
                        stats['last_uwb_pos'] = pos_str
                    boundary_alert_active = True

            if not in_boundary:
                boundary_alert_active = False
                with stats_lock:
                    stats['last_uwb_pos'] = pos_str

            # ── 1. 发射聚集信号 (定时) ──
            now = time.time()
            if now - last_tx_time >= TX_INTERVAL:
                send_nec_frame(SWARM_ADDR, SWARM_CMD)
                last_tx_time = now

            # ── 2. 接收红外信号 (非阻塞轮询) ──
            poll_start = time.time_ns() // 1000
            pulses = None
            while (time.time_ns() // 1000 - poll_start) < 50000:
                if lgpio.gpio_read(handle, IR_GPIO) == 0:
                    pulses, samples = read_frame()
                    break

            # ── 3. 处理接收到的信号 ──
            peer_detected = False

            if pulses:
                result = decode_pulses(pulses)
                if result and result.get('valid'):
                    addr = result.get('address')
                    cmd = result.get('command')

                    is_peer = (addr == SWARM_ADDR and cmd == SWARM_CMD)
                    is_repeat = result.get('repeat', False)

                    if is_peer or is_repeat:
                        peer_detected = True

                        direction, max_reading, modulus, all_vals = analyze_signal(samples)
                        predicted_angle = predict_direction(all_vals)

                        # 更新统计：将四通道原始 ADC 最大值缓存到 stats，
                        # 供后续日志统一输出；未检测到信号时保留上一次有效值
                        with stats_lock:
                            stats['last_peer_direction'] = direction
                            stats['last_max_reading'] = max_reading
                            stats['last_modulus'] = modulus
                            stats['last_predicted_angle'] = predicted_angle
                            stats['raw_adc'] = [float(v) for v in all_vals]

                        # 仅在未触发边界保护时执行运动决策
                        if not in_boundary:
                            dx, dy, d_alt, d_yaw, action = decide_movement(direction, modulus)
                            try:
                                dl.set_pose(
                                    Kp_dx * dx,
                                    Kp_dy * dy,
                                    Kp_dalt * d_alt,
                                    Kp_dyaw * d_yaw
                                )
                            except Exception as e:
                                print(f"[{now_str()}] ⚠️  控制指令发送失败: {e}")

                            with stats_lock:
                                stats['last_action'] = action
                        else:
                            with stats_lock:
                                stats['last_action'] = "边界悬停"

                        # 恢复跟踪状态（终端提示，但不写入日志）
                        if lost_peer_time is not None:
                            angle_str = f" 预测角={predicted_angle:.2f}°" if predicted_angle is not None else ""
                            print(f"[{now_str()}] 📡 发现同类 {direction}{angle_str} 模={modulus:.2f}")
                            lost_peer_time = None

                        last_peer_time = time.time()

            # ── 4. 丢失目标处理 ──
            if not peer_detected and (time.time() - last_peer_time > 2.0):
                if lost_peer_time is None:
                    lost_peer_time = time.time()
                    if not in_boundary:
                        dl.set_pose(0, 0, 0, 0)
                    with stats_lock:
                        stats['last_action'] = "丢失-悬停"
                        stats['last_peer_direction'] = "无"
                        stats['last_max_reading'] = 0
                        stats['last_modulus'] = 0
                        stats['last_predicted_angle'] = None
                        # 丢失目标时不重置 raw_adc，保留最后一次有效读数，
                        # 避免日志中出现无意义的零值跳变
                    print(f"[{now_str()}] 🔍 丢失目标")

                elapsed = time.time() - lost_peer_time
                if elapsed >= 5.0 and not in_boundary:
                    try:
                        dl.set_pose(0, 0, 0, YAW_SEARCH_SPEED)
                    except Exception:
                        pass
                    with stats_lock:
                        stats['last_action'] = f"丢失-自旋({elapsed:.0f}s)"

            # ── 5. 统一日志与终端输出（每 TX_INTERVAL 秒，即 0.25s） ──
            # 日志频率与发射/决策周期严格对齐，确保每完成一轮读取-决策就记录一次
            if time.time() - last_log_time >= TX_INTERVAL:
                # 计算进入聚集模式后的运行时间（单位：秒，保留两位小数）
                elapsed = time.time() - start_time

                # 获取电池电压（兜底 None，保留两位小数）
                batt_v = getattr(dl, 'batt_voltage', 0.0) or 0.0

                with stats_lock:
                    stats['last_batt'] = batt_v

                    # 格式化预测角度：保留两位小数，无目标时以 0.00 占位避免 CSV 列错位
                    pred_angle_str = f"{stats['last_predicted_angle']:.2f}" if stats['last_predicted_angle'] is not None else "0.00"

                    # 格式化四通道原始 ADC 读数：保留两位小数
                    raw_adc_strs = [f"{v:.2f}" for v in stats['raw_adc']]

                    # 构建 CSV 数据行，逗号分隔，便于直接导入 Excel
                    # 列顺序：时间戳, 运行时间, CH0, CH1, CH2, CH3, 方向, 预测角, 最大读数, 模长, 动作, UWB_X, UWB_Y, UWB_Z, 边界, 电量
                    csv_line = (
                        f"{now_ms_str()},{elapsed:.2f},"
                        f"{raw_adc_strs[0]},{raw_adc_strs[1]},{raw_adc_strs[2]},{raw_adc_strs[3]},"
                        f"{stats['last_peer_direction']},{pred_angle_str},"
                        f"{stats['last_max_reading']:.2f},{stats['last_modulus']:.2f},"
                        f"{stats['last_action']},{uwb_x},{uwb_y},{uwb_z},"
                        f"{'YES' if in_boundary else 'NO'},{batt_v:.2f}"
                    )

                    # 写入日志文件（仅周期性数据，不含提示性事件）
                    log_write(csv_line)

                    # 终端输出：中文可读，运行时间/预测角/模/ADC 显示整数，其余保留两位小数
                    angle_disp = f" 预测角={int(stats['last_predicted_angle'])}°" if stats['last_predicted_angle'] is not None else ""
                    print(
                        f"[{now_str()}] ⏱ {elapsed:.0f}s | "
                        f"{stats['last_peer_direction']}{angle_disp} | "
                        f"模={stats['last_modulus']:.0f} | "
                        f"{stats['last_action']} | "
                        f"UWB={pos_str} | "
                        f"电量={batt_v:.2f}V | "
                        f"ADC=[{','.join(str(int(v)) for v in stats['raw_adc'])}]"
                    )

                last_log_time = time.time()

            time.sleep(0.01)

    except Exception as e:
        print(f"[{now_str()}] ❌ 聚集循环异常: {e}")
    finally:
        try:
            dl.set_pose(0, 0, 0, 0)
        except:
            pass
        print(f"[{now_str()}] 🔴 聚集模式已停止")


# ═══════════════════════════════════════════════════════════════════════════════
# 命令执行
# ═══════════════════════════════════════════════════════════════════════════════

def execute_command(cmd_char):
    global gather_running, gather_thread, uwb_running, uwb_thread
    cmd_char = cmd_char.lower()

    if cmd_char == 'a':
        dl.set_arm()
        print(f"[{now_str()}] 🔓 解锁(arm)指令已发送")

    elif cmd_char == 'd':
        dl.set_disarm()
        print(f"[{now_str()}] 🔒 上锁(disarm)指令已发送")

    elif cmd_char == 't':
        dl.set_takeoff()
        print(f"[{now_str()}] 🚀 起飞(takeoff)指令已发送")

    elif cmd_char == 'l':
        dl.set_land()
        print(f"[{now_str()}] 🛬 降落(land)指令已发送")

    elif cmd_char == 'g':
        if gather_running:
            print(f"[{now_str()}] ⚠️  聚集模式已在运行中")
            return
        if not uwb_running:
            uwb_running = True
            uwb_thread = threading.Thread(target=uwb_loop, daemon=True)
            uwb_thread.start()
        stop_gather_event.clear()
        gather_running = True
        gather_thread = threading.Thread(target=gather_loop, daemon=True)
        gather_thread.start()

    elif cmd_char == 'h':
        if gather_running:
            gather_running = False
            stop_gather_event.set()
            if gather_thread and gather_thread.is_alive():
                gather_thread.join(timeout=2.0)
        dl.set_pose(0, 0, 0, 0)
        print(f"[{now_str()}] ⏹  悬停(halt)，聚集模式已停止")

    else:
        print(f"[{now_str()}] ❓ 未知命令。支持: a(解锁) d(上锁) t(起飞) l(降落) g(聚集) h(悬停)")


# ═══════════════════════════════════════════════════════════════════════════════
# 状态监控
# ═══════════════════════════════════════════════════════════════════════════════

def status_loop(log_path):
    global log_file
    while True:
        # 防止 batt_voltage 为 None，终端低电量提示保留，但不写入日志文件
        batt_v = getattr(dl, 'batt_voltage', 0.0) or 0.0
        if batt_v < 6.7 and batt_v > 0:
            warning = f"[{now_str()}] !!! BATTERY LOW ({batt_v:.2f}V) !!! PLEASE LAND !!!"
            print(f"\033[91m\033[1m{warning}\033[0m")
        time.sleep(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 主程序
# ═══════════════════════════════════════════════════════════════════════════════

def init_hardware():
    global pwm, handle, adc

    print("═" * 70)
    print("   无人机红外聚集运动模块  IR Swarm Aggregation")
    print("   模长=sqrt(Σsignal²) | 四向移动 | 圆周阵列 | 预测方向(展示)")
    print("   UWB定位: /dev/ttyACM0 | 边界保护: 靠近<0.5m悬停")
    print("   日志输出: 每 0.25s CSV 数据记录（含四通道原始ADC、UWB、电量、运行时间）")
    print("═" * 70)

    pwm = HardwarePWM(pwm_channel=PWM_CHANNEL, hz=PWM_FREQ, chip=0)
    pwm.stop()
    print("   [TX] HardwarePWM (GPIO12)  38kHz  已就绪")

    handle = lgpio.gpiochip_open(CHIP)
    try:
        lgpio.gpio_claim_input(handle, IR_GPIO, lgpio.SET_PULL_UP)
    except AttributeError:
        try:
            lgpio.gpio_claim_input(handle, IR_GPIO, lgpio.SET_BIAS_PULL_UP)
        except AttributeError:
            lgpio.gpio_claim_input(handle, IR_GPIO)
    print(f"   [RX] TSOP34438 (GPIO{IR_GPIO})       已就绪")

    adc = spidev.SpiDev()
    adc.open(SPI_BUS, SPI_DEVICE)
    adc.max_speed_hz = SPI_SPEED
    adc.mode = 0
    print("   [AD] MCP3008 (4通道圆周阵列)         已就绪")
    print("        CH0=前  CH1=右  CH2=后  CH3=左")
    print("═" * 70)


def cleanup_hardware():
    global pwm, handle, adc

    if pwm:
        pwm.stop()
    if handle:
        try:
            lgpio.gpio_free(handle, IR_GPIO)
        except:
            pass
        lgpio.gpiochip_close(handle)
    if adc:
        adc.close()

    print(f"[{now_str()}] 🔧 硬件资源已释放")


def main():
    global dl, uwb_running, uwb_thread, log_file

    init_hardware()

    dl = datalink()
    threading.Thread(target=dl.drone, daemon=True).start()
    threading.Thread(target=dl.heartbeat, daemon=True).start()

    base_dir = Path("runs/swarm") / f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    base_dir.mkdir(parents=True, exist_ok=True)
    status_log_path = base_dir / "status.log"

    log_file = open(str(status_log_path), 'a', buffering=1)
    # 写入 CSV 表头，便于 Excel 直接识别列并导入
    log_file.write(
        "timestamp,elapsed_time,ch0_raw,ch1_raw,ch2_raw,ch3_raw,"
        "peer_dir,pred_angle,max_reading,modulus,action,"
        "uwb_x,uwb_y,uwb_z,in_boundary,batt_v\n"
    )
    log_file.flush()

    threading.Thread(target=status_loop, args=(str(status_log_path),), daemon=True).start()

    print(f"\n===== 无人机红外聚集运动模块 =====")
    print(f"命令: a(解锁) d(上锁) t(起飞) l(降落) g(聚集) h(悬停)")
    print(f"聚集地址: 0x{SWARM_ADDR:02X}  命令: 0x{SWARM_CMD:02X}")
    print(f"底噪读数: {NOISE_FLOOR} (无信号时校准值)")
    print(f"输出目录: {base_dir}")
    print(f"日志文件: {status_log_path}\n")

    try:
        while True:
            cmd = input(">> ").strip().lower()
            if cmd:
                if len(cmd) == 1:
                    execute_command(cmd)
                else:
                    print(f"[{now_str()}] ⚠️  只接受单字母命令: a/d/t/l/g/h")
    except KeyboardInterrupt:
        print(f"\n[{now_str()}] 用户中断")
    finally:
        gather_running = False
        stop_gather_event.set()
        if gather_thread and gather_thread.is_alive():
            gather_thread.join(timeout=2.0)
        uwb_running = False
        if uwb_thread and uwb_thread.is_alive():
            uwb_thread.join(timeout=1.0)
        try:
            dl.set_pose(0, 0, 0, 0)
        except:
            pass
        if log_file:
            log_file.close()
        cleanup_hardware()
        print(f"[{now_str()}] 👋 程序已结束")


if __name__ == '__main__':
    main()