#!/usr/bin/env python3
"""
================================================================================
无人机红外聚集运动模块 - IR Swarm Aggregation Module
================================================================================
功能:
  • 聚集模式：持续发射红外 NEC 信号，同时监听同类信号
  • 同类识别：通过预设的聚集地址和命令码识别同类无人机
  • 方向感知：通过 MCP3008 + 4路圆周阵列光敏传感器判断信号来源方向
  • 自主靠近：根据信号强度和方向自主调整飞行姿态
  • 安全距离：基于向量模长的阈值控制，防止碰撞

硬件接线:
  发射端:
    GPIO12  →  HardwarePWM 38kHz → 栅极驱动 → 4×TSAL6200 红外LED
  接收端:
    GPIO22  →  TSOP34438 红外接收头
    SPI0    →  MCP3008 ADC (4路模拟光敏传感器，圆周阵列)
      CH0 → 前方光敏传感器
      CH1 → 右方光敏传感器
      CH2 → 后方光敏传感器
      CH3 → 左方光敏传感器

NEC 协议:
  引导码: 9ms 载波 + 4.5ms 空闲
  数据位: 0 = 0.56ms载波 + 0.56ms空闲
          1 = 0.56ms载波 + 1.69ms空闲
  帧格式: [地址8位][地址反码8位][命令8位][命令反码8位] (LSB first)

信号强度定义:
  对每个通道: signal = max(0, 读数 - 底噪)
  向量模长 = sqrt(signal_0² + signal_1² + signal_2² + signal_3²)
  模长反映整体信号强度，用于阈值判断

聚集策略:
  模长 < 可靠阈值: 信号不可靠，悬停/自旋搜寻
  模长 ≥ 可靠阈值: 根据最强通道判断方向并移动

  距离控制 (基于向量模长):
    模长 < 靠近阈值: 距离较远，向信号源方向移动
    靠近阈值 ≤ 模长 ≤ 保持阈值: 距离适中，悬停
    模长 > 远离阈值: 距离过近，反向移动

运动方向 (四向):
  前/后/左/右 —— 圆周阵列直接对应

预测方向 (展示用):
  基于四个通道的信号增量做向量合成，计算相对角度
  仅用于展示，不参与运动控制

命令映射:
  a -> arm (解锁)      d -> disarm (上锁)
  t -> takeoff (起飞)  l -> land (降落)
  g -> gather (启动聚集模式)
  h -> halt (悬停/停止聚集)
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

# ═══════════════════════════════════════════════════════════════════════════════
# 配置区
# ═══════════════════════════════════════════════════════════════════════════════

# ---- 红外硬件配置 ----
PWM_CHANNEL = 0          # HardwarePWM 通道 0 (对应 GPIO12)
PWM_FREQ = 38000         # 载波频率 38kHz
PWM_DUTY = 50            # 占空比 50%
CHIP = 4                 # lgpio 芯片号 (树莓派5 = gpiochip4)

IR_GPIO = 22             # TSOP34438 信号输出接 GPIO22
SPI_BUS = 0              # SPI 总线
SPI_DEVICE = 0           # SPI 设备 (CE0)
SPI_SPEED = 1000000      # SPI 时钟 1MHz

# ---- NEC 时序参数 (单位: 微秒) ----
NEC_HEADER_MARK = 9000
NEC_HEADER_SPACE = 4500
NEC_BIT_MARK = 562
NEC_ZERO_SPACE = 562
NEC_ONE_SPACE = 1687
NEC_TOLERANCE = 400      # 容差 ±400μs
NEC_REPEAT_SPACE = 2250
IDLE_TIMEOUT_US = 120000
SAMPLE_INTERVAL_NS = 200000  # ADC 采样间隔 200μs

# ---- 聚集模式参数 ----
SWARM_ADDR = 0xAA        # 聚集模式专用地址 (同类识别码)
SWARM_CMD = 0xBB         # 聚集模式专用命令
TX_INTERVAL = 0.25        # 发射间隔（秒）

# ---- 方向映射 (MCP3008 通道 → 方向) ----
# 圆周阵列: 0=前, 1=右, 2=后, 3=左
CHANNEL_TO_DIR = {0: "右", 1: "后", 2: "左", 3: "前"}
# 通道对应的角度 (度)
CHANNEL_TO_ANGLE = {0: 90, 1: 180, 2: 270, 3: 0}

# ---- 四向运动控制 (dx, dy, d_alt, d_yaw) ----
DIR_TO_CONTROL = {
    "前": (0.3, 0.0, 0.0, 0.0),
    "右": (0.0, 0.3, 0.0, 0.0),
    "后": (-0.3, 0.0, 0.0, 0.0),
    "左": (0.0, -0.3, 0.0, 0.0),
}

# ---- 底噪配置 ----
NOISE_FLOOR = 608        # 无信号时的 ADC 底噪读数

# ---- 距离控制阈值 (基于向量模长) ----
# 模长 = sqrt(Σ(signal_i²)), signal_i = max(0, 读数_i - 底噪)
MOD_RELIABLE = 5        # 模长可靠阈值, 超过此值认为信号有效
MOD_APPROACH = 100       # 低于此值: 距离较远，主动靠近
MOD_HOLD = 150           # 此值附近: 距离适中，悬停保持
MOD_RETREAT = 280        # 高于此值: 距离过近，反向远离

# ---- 运动速度参数 ----
SPEED_APPROACH = 0.1     # 靠近速度 (m/s)
SPEED_RETREAT = -0.1     # 远离速度 (m/s)
YAW_SEARCH_SPEED = 0.2   # 搜寻自旋速度 (rad/s)

# ---- 控制增益 ----
Kp_dx, Kp_dy, Kp_dalt, Kp_dyaw = 0.6, 0.6, 0.7, 0.3

# ═══════════════════════════════════════════════════════════════════════════════
# 全局变量
# ═══════════════════════════════════════════════════════════════════════════════

dl = None                          # 无人机数据链对象
gather_running = False             # 聚集模式运行标志
gather_thread = None               # 聚集线程
stop_gather_event = threading.Event()

# 红外硬件句柄
pwm = None
handle = None
adc = None

# 统计信息
stats = {
    'last_peer_direction': "无",
    'last_predicted_angle': None,
    'last_max_reading': 0,
    'last_modulus': 0,
    'last_action': "待机",
}
stats_lock = threading.Lock()

# ═══════════════════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════════════════

def now_str():
    """返回当前时间字符串 HH:MM:SS"""
    return time.strftime("%H:%M:%S")


def delay_us(us):
    """微秒级忙等待延时"""
    start = time.perf_counter_ns()
    target = start + us * 1000
    while time.perf_counter_ns() < target:
        pass


def adc_read_all():
    """读取 MCP3008 4个通道的 ADC 值"""
    vals = []
    for ch in range(4):
        resp = adc.xfer2([1, (8 + ch) << 4, 0])
        vals.append(((resp[1] & 3) << 8) + resp[2])
    return vals


def in_range(value, target):
    """判断数值是否在目标容差范围内"""
    return abs(value - target) < NEC_TOLERANCE


# ═══════════════════════════════════════════════════════════════════════════════
# 信号分析
# ═══════════════════════════════════════════════════════════════════════════════

def compute_signals(all_max_vals, noise_floor=NOISE_FLOOR):
    """
    计算各通道的信号增量 (读数 - 底噪)

    参数:
        all_max_vals: 4个通道的最大读数列表
        noise_floor: 底噪读数

    返回:
        4个通道的信号增量列表 (负值归零)
    """
    return [max(0, v - noise_floor) for v in all_max_vals]


def compute_modulus(signal_vals):
    """
    计算向量模长 = sqrt(Σ(signal_i²))

    参数:
        signal_vals: 4个通道的信号增量列表

    返回:
        向量模长 (float)
    """
    return math.sqrt(sum(v ** 2 for v in signal_vals))


def analyze_signal(samples, noise_floor=NOISE_FLOOR):
    """
    分析红外信号的方向和强度

    参数:
        samples: ADC 采样列表 [[ch0, ch1, ch2, ch3], ...]
        noise_floor: 底噪读数

    返回:
        (direction, max_reading, modulus, all_max_vals)
        direction: 信号来源方向 ("前"/"右"/"后"/"左"/"未知")
        max_reading: 最强通道的读数
        modulus: 向量模长 (sqrt(Σ(signal_i²)))
        all_max_vals: 4个通道各自的最大读数列表
    """
    if not samples or len(samples) == 0:
        return "未知", 0, 0.0, [0, 0, 0, 0]

    # 取每个通道在采样期间的最大值
    all_max_vals = [max(s[i] for s in samples) for i in range(4)]

    # 找出最强通道 (运动方向依据)
    best_idx = all_max_vals.index(max(all_max_vals))
    max_reading = all_max_vals[best_idx]
    direction = CHANNEL_TO_DIR.get(best_idx, "未知")

    # 计算向量模长 (阈值判断依据)
    signal_vals = compute_signals(all_max_vals, noise_floor)
    modulus = compute_modulus(signal_vals)

    return direction, max_reading, modulus, all_max_vals


def predict_direction(all_max_vals, noise_floor=NOISE_FLOOR):
    """
    基于四个通道的读数，预测信号源的相对角度 (展示用，不参与运动控制)

    使用向量合成法：以各通道信号增量为权重，对通道角度进行加权平均

    参数:
        all_max_vals: 4个通道的读数列表 [ch0, ch1, ch2, ch3]
        noise_floor: 底噪读数

    返回:
        预测角度 (度, 0°=前, 顺时针增加), 或 None 如果信号太弱
    """
    signal_vals = compute_signals(all_max_vals, noise_floor)
    total_signal = sum(signal_vals)

    if total_signal == 0:
        return None

    # 向量合成
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
    """
    发送标准 NEC 帧
    参数:
        address: 8位地址 (0-255)
        command: 8位命令 (0-255)
    """
    # 引导码: 9ms 载波 + 4.5ms 空闲
    pwm.start(PWM_DUTY)
    delay_us(NEC_HEADER_MARK)
    pwm.stop()
    delay_us(NEC_HEADER_SPACE)

    # 32位数据: [地址][地址反码][命令][命令反码] (LSB first)
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

    # 停止位
    pwm.start(PWM_DUTY)
    delay_us(NEC_BIT_MARK)
    pwm.stop()


# ═══════════════════════════════════════════════════════════════════════════════
# 红外接收模块
# ═══════════════════════════════════════════════════════════════════════════════

def read_frame():
    """
    读取一帧红外信号
    返回: (pulses, samples)
        pulses:  [(电平, 时长μs), ...]
        samples: [[ch0, ch1, ch2, ch3], ...]  引导码期间的 ADC 采样
    """
    # 等待信号开始 (TSOP 输出低电平)
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

        # 边沿检测
        if current_state != last_state:
            duration = current_time - last_time
            pulses.append((last_state, duration))
            last_state = current_state
            last_time = current_time

        # 引导码 mark 期间 (前 9ms) 采样 ADC
        if len(pulses) == 0 and current_state == 0:
            now_ns = time.time_ns()
            if (now_ns - last_sample) >= SAMPLE_INTERVAL_NS:
                samples.append(adc_read_all())
                last_sample = now_ns

        # 超时结束
        if (current_time - last_time) > IDLE_TIMEOUT_US and len(pulses) > 0:
            pulses.append((last_state, current_time - last_time))
            break

    return pulses, samples


def decode_pulses(pulses):
    """
    从脉冲序列解码 NEC 帧
    返回: dict 或 None
    """
    if not pulses or len(pulses) < 4:
        return None

    # 查找引导码
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
    """解码 32 位数据 (LSB first)"""
    bits = []
    idx = start_idx

    while len(bits) < 32 and idx < len(pulses) - 1:
        mark_dur = pulses[idx][1]
        space_dur = pulses[idx+1][1]

        if mark_dur < 200:   # 太短，视为噪声
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

    # LSB first: bits[i] 就是第 i 位的值
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
    """
    根据方向和向量模长决定运动策略

    参数:
        direction: 信号来源方向
        modulus: 向量模长 sqrt(Σ(signal_i²))

    返回:
        (dx, dy, d_alt, d_yaw, action_desc)
    """
    # 1. 首先检查模长是否可靠
    if modulus < MOD_RELIABLE:
        # 信号不可靠
        return 0.0, 0.0, 0.0, YAW_SEARCH_SPEED, f"搜寻 模={modulus:.0f}"

    # 2. 模长可靠，根据距离判断
    if direction == "未知":
        return 0.0, 0.0, 0.0, YAW_SEARCH_SPEED, f"搜寻 模={modulus:.0f}"

    if modulus > MOD_RETREAT:
        # 模长过大 → 距离过近 → 反向远离
        base = DIR_TO_CONTROL.get(direction, (0.0, 0.0, 0.0, 0.0))
        dx = -base[0] if base[0] != 0 else 0.0
        dy = -base[1] if base[1] != 0 else 0.0
        return dx, dy, 0.0, 0.0, f"远离{direction}"

    elif modulus > MOD_HOLD:
        # 模长适中 → 距离合适 → 悬停
        return 0.0, 0.0, 0.0, 0.0, f"保持{direction}"

    else:
        # 模长较小 → 距离较远 → 向信号源靠近
        base = DIR_TO_CONTROL.get(direction, (0.0, 0.0, 0.0, 0.0))
        return base[0], base[1], 0.0, 0.0, f"靠近{direction}"


# ═══════════════════════════════════════════════════════════════════════════════
# 聚集主循环
# ═══════════════════════════════════════════════════════════════════════════════

def gather_loop():
    """
    聚集模式主循环：
    1. 持续发射聚集信号
    2. 持续监听红外信号
    3. 识别同类并分析方向/模长
    4. 根据模长和距离控制运动
    """
    global gather_running
    last_tx_time = 0
    last_peer_time = 0
    last_status_print = 0
    lost_peer_time = None

    print(f"[{now_str()}] 🟢 聚集模式已启动")
    print(f"   发射地址: 0x{SWARM_ADDR:02X}  命令: 0x{SWARM_CMD:02X}")
    print(f"   底噪读数: {NOISE_FLOOR}")
    print(f"   模长可靠阈值: {MOD_RELIABLE}")
    print(f"   靠近阈值: 模长 < {MOD_APPROACH}")
    print(f"   保持阈值: {MOD_HOLD}")
    print(f"   远离阈值: 模长 > {MOD_RETREAT}")

    try:
        while gather_running and not stop_gather_event.is_set():
            # ── 1. 发射聚集信号 (定时) ──
            now = time.time()
            if now - last_tx_time >= TX_INTERVAL:
                send_nec_frame(SWARM_ADDR, SWARM_CMD)
                last_tx_time = now

            # ── 2. 接收红外信号 (非阻塞轮询) ──
            poll_start = time.time_ns() // 1000
            pulses = None
            while (time.time_ns() // 1000 - poll_start) < 50000:  # 50μs 轮询
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

                    # 检查是否是同类（聚集信号）
                    is_peer = (addr == SWARM_ADDR and cmd == SWARM_CMD)
                    is_repeat = result.get('repeat', False)

                    if is_peer or is_repeat:
                        peer_detected = True

                        # 分析方向和模长
                        direction, max_reading, modulus, all_vals = analyze_signal(samples)

                        # 预测方向 (展示用)
                        predicted_angle = predict_direction(all_vals)

                        # 更新统计
                        with stats_lock:
                            stats['last_peer_direction'] = direction
                            stats['last_max_reading'] = max_reading
                            stats['last_modulus'] = modulus
                            stats['last_predicted_angle'] = predicted_angle

                        # 判断运动策略
                        dx, dy, d_alt, d_yaw, action = decide_movement(direction, modulus)

                        # 发送控制指令
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

                        # 恢复跟踪状态
                        if lost_peer_time is not None:
                            angle_str = f" 预测角={predicted_angle:.1f}°" if predicted_angle is not None else ""
                            print(f"[{now_str()}] 📡 发现同类 {direction}{angle_str} 模={modulus:.0f}")
                            lost_peer_time = None

                        last_peer_time = time.time()

            # ── 4. 丢失目标处理 ──
            if not peer_detected and (time.time() - last_peer_time > 2.0):
                if lost_peer_time is None:
                    lost_peer_time = time.time()
                    dl.set_pose(0, 0, 0, 0)
                    with stats_lock:
                        stats['last_action'] = "丢失目标"
                        stats['last_peer_direction'] = "无"
                        stats['last_max_reading'] = 0
                        stats['last_modulus'] = 0
                        stats['last_predicted_angle'] = None
                    print(f"[{now_str()}] 🔍 丢失目标")

                # 丢失超过 5 秒，开始自旋搜寻
                elapsed = time.time() - lost_peer_time
                if elapsed >= 5.0:
                    try:
                        dl.set_pose(0, 0, 0, YAW_SEARCH_SPEED)
                    except Exception as e:
                        pass
                    with stats_lock:
                        stats['last_action'] = f"自旋({elapsed:.0f}s)"

            # ── 5. 定时状态打印 (每 2 秒) ──
            if time.time() - last_status_print >= 2.0:
                with stats_lock:
                    angle_str = ""
                    if stats['last_predicted_angle'] is not None:
                        angle_str = f" 预测角={stats['last_predicted_angle']:.1f}°"
                    print(f"[{now_str()}] 📊 {stats['last_peer_direction']}{angle_str} | "
                          f"模={stats['last_modulus']:.0f} | "
                          f"{stats['last_action']}")
                last_status_print = time.time()

            # 短暂休眠避免 CPU 占满
            time.sleep(0.01)

    except Exception as e:
        print(f"[{now_str()}] ❌ 聚集循环异常: {e}")
    finally:
        # 停止运动
        try:
            dl.set_pose(0, 0, 0, 0)
        except:
            pass
        print(f"[{now_str()}] 🔴 聚集模式已停止")


# ═══════════════════════════════════════════════════════════════════════════════
# 命令执行
# ═══════════════════════════════════════════════════════════════════════════════

def execute_command(cmd_char):
    """执行单字母命令"""
    global gather_running, gather_thread
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
        # 启动聚集模式
        if gather_running:
            print(f"[{now_str()}] ⚠️  聚集模式已在运行中")
            return
        stop_gather_event.clear()
        gather_running = True
        gather_thread = threading.Thread(target=gather_loop, daemon=True)
        gather_thread.start()

    elif cmd_char == 'h':
        # 停止聚集模式
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
    """低电压警报监控"""
    with open(log_path, 'a', buffering=1) as f:
        while True:
            batt_v = getattr(dl, 'batt_voltage', 0.0)
            if batt_v < 6.7 and batt_v > 0:
                warning = f"[{now_str()}] !!! BATTERY LOW ({batt_v:.2f}V) !!! PLEASE LAND !!!"
                print(f"\033[91m\033[1m{warning}\033[0m")
                f.write(warning + "\n")
            time.sleep(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 主程序
# ═══════════════════════════════════════════════════════════════════════════════

def init_hardware():
    """初始化红外硬件"""
    global pwm, handle, adc

    print("═" * 70)
    print("   无人机红外聚集运动模块  IR Swarm Aggregation")
    print("   模长=sqrt(Σsignal²) | 四向移动 | 圆周阵列 | 预测方向(展示)")
    print("═" * 70)

    # 发射: HardwarePWM 独占 GPIO12
    pwm = HardwarePWM(pwm_channel=PWM_CHANNEL, hz=PWM_FREQ, chip=0)
    pwm.stop()
    print("   [TX] HardwarePWM (GPIO12)  38kHz  已就绪")

    # 接收: GPIO22 输入 + 上拉
    handle = lgpio.gpiochip_open(CHIP)
    try:
        lgpio.gpio_claim_input(handle, IR_GPIO, lgpio.SET_PULL_UP)
    except AttributeError:
        try:
            lgpio.gpio_claim_input(handle, IR_GPIO, lgpio.SET_BIAS_PULL_UP)
        except AttributeError:
            lgpio.gpio_claim_input(handle, IR_GPIO)
    print(f"   [RX] TSOP34438 (GPIO{IR_GPIO})       已就绪")

    # ADC: MCP3008 4通道光敏传感器 (圆周阵列)
    adc = spidev.SpiDev()
    adc.open(SPI_BUS, SPI_DEVICE)
    adc.max_speed_hz = SPI_SPEED
    adc.mode = 0
    print("   [AD] MCP3008 (4通道圆周阵列)         已就绪")
    print("        CH0=前  CH1=右  CH2=后  CH3=左")
    print("═" * 70)


def cleanup_hardware():
    """清理红外硬件资源"""
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
    global dl

    # 初始化硬件
    init_hardware()

    # 初始化无人机数据链
    dl = datalink()
    threading.Thread(target=dl.drone, daemon=True).start()
    threading.Thread(target=dl.heartbeat, daemon=True).start()

    # 创建日志目录
    base_dir = Path("runs/swarm") / f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    base_dir.mkdir(parents=True, exist_ok=True)
    status_log_path = base_dir / "status.log"

    # 启动状态监控
    threading.Thread(target=status_loop, args=(str(status_log_path),), daemon=True).start()

    print(f"\n===== 无人机红外聚集运动模块 =====")
    print(f"命令: a(解锁) d(上锁) t(起飞) l(降落) g(聚集) h(悬停)")
    print(f"聚集地址: 0x{SWARM_ADDR:02X}  命令: 0x{SWARM_CMD:02X}")
    print(f"底噪读数: {NOISE_FLOOR} (无信号时校准值)")
    print(f"输出目录: {base_dir}\n")

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
        # 清理
        gather_running = False
        stop_gather_event.set()
        if gather_thread and gather_thread.is_alive():
            gather_thread.join(timeout=2.0)
        try:
            dl.set_pose(0, 0, 0, 0)
        except:
            pass
        cleanup_hardware()
        print(f"[{now_str()}] 👋 程序已结束")


if __name__ == '__main__':
    main()