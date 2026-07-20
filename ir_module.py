#!/usr/bin/env python3
"""
================================================================================
红外收发一体终端 - IR Communication Terminal
================================================================================
功能:
  • 发射: 通过 HardwarePWM (GPIO12) 发送 NEC 编码红外信号
  • 接收: 通过 TSOP34438 (GPIO22) 解码 NEC 信号
  • 测向: 通过 MCP3008 + 4路光敏传感器检测信号强度与方向

硬件接线:
  发射端:
    GPIO12  →  HardwarePWM 38kHz → 栅极驱动 → 4×TSAL6200 红外LED
  接收端:
    GPIO22  →  TSOP34438 红外接收头
    SPI0    →  MCP3008 ADC (4路模拟光敏传感器)

NEC 协议:
  引导码: 9ms 载波 + 4.5ms 空闲
  数据位: 0 = 0.56ms载波 + 0.56ms空闲
          1 = 0.56ms载波 + 1.69ms空闲
  帧格式: [地址8位][地址反码8位][命令8位][命令反码8位] (LSB first)

使用说明:
  tx              发送默认帧 (地址0x00, 命令0x46)
  tx 0x00 0x46    发送指定 NEC 帧
  q / quit        退出程序
================================================================================
"""

import time
import lgpio
import spidev
import sys
import select
from rpi_hardware_pwm import HardwarePWM

# ═══════════════════════════════════════════════════════════════════════════════
# 配置区
# ═══════════════════════════════════════════════════════════════════════════════
PWM_CHANNEL = 0          # HardwarePWM 通道 0 (对应 GPIO12)
PWM_FREQ = 38000         # 载波频率 38kHz
PWM_DUTY = 50            # 占空比 50%
CHIP = 4                 # lgpio 芯片号 (树莓派5 = gpiochip4)

IR_GPIO = 22             # TSOP34438 信号输出接 GPIO22
SPI_BUS = 0              # SPI 总线
SPI_DEVICE = 0           # SPI 设备 (CE0)
SPI_SPEED = 1000000      # SPI 时钟 1MHz

# NEC 时序参数 (单位: 微秒)
NEC_HEADER_MARK = 9000
NEC_HEADER_SPACE = 4500
NEC_BIT_MARK = 562
NEC_ZERO_SPACE = 562
NEC_ONE_SPACE = 1687
NEC_TOLERANCE = 400      # 容差 ±400μs
NEC_REPEAT_SPACE = 2250
IDLE_TIMEOUT_US = 120000
SAMPLE_INTERVAL_NS = 200000  # ADC 采样间隔 200μs

# 方向映射 (MCP3008 通道 → 方向)
CHANNEL_TO_DIR = {0: "前", 1: "右", 2: "后", 3: "左"}

# 默认发射参数
DEFAULT_ADDR = 0x00
DEFAULT_CMD = 0x46

# ═══════════════════════════════════════════════════════════════════════════════
# 初始化
# ═══════════════════════════════════════════════════════════════════════════════
print("═" * 70)
print("   红外收发一体终端  IR Communication Terminal")
print("═" * 70)

# 发射: HardwarePWM 独占 GPIO12，绝不通过 lgpio 操作该引脚
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

# ADC: MCP3008 4通道光敏传感器
adc = spidev.SpiDev()
adc.open(SPI_BUS, SPI_DEVICE)
adc.max_speed_hz = SPI_SPEED
adc.mode = 0
print("   [AD] MCP3008 (4通道强度)             已就绪")
print("═" * 70)

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
# 发射模块
# ═══════════════════════════════════════════════════════════════════════════════
def send_nec_frame(address, command):
    """
    发送标准 NEC 帧
    参数:
        address: 8位地址 (0-255)
        command: 8位命令 (0-255)
    """
    ts = now_str()
    print(f"\n[{ts}] 📤 发射 → 地址=0x{address:02X}  命令=0x{command:02X}")

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

    ts = now_str()
    print(f"[{ts}] ✅ 发射完成")

# ═══════════════════════════════════════════════════════════════════════════════
# 接收模块
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
                return {'repeat': True, 'valid': True}

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
# 非阻塞键盘输入
# ═══════════════════════════════════════════════════════════════════════════════
def check_input():
    """检查是否有键盘输入 (非阻塞)"""
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.readline().strip()
    return None

# ═══════════════════════════════════════════════════════════════════════════════
# 主程序
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print(f"\n[{now_str()}] 🟢 系统启动，开始监听...")
    print("""
┌──────────────────────────────────────────────────────────────────────┐
│  可用命令:                                                           │
│    tx              发送默认帧 (地址 0x00, 命令 0x46)                  │
│    tx 0x00 0x46    发送自定义 NEC 帧                                  │
│    q / quit        退出程序                                          │
└──────────────────────────────────────────────────────────────────────┘
""")

    last_command = None
    rx_count = 0
    tx_count = 0

    try:
        while True:
            # ── 检查键盘输入 ──
            cmd = check_input()
            if cmd in ("q", "quit", "exit"):
                break
            elif cmd == "tx":
                send_nec_frame(DEFAULT_ADDR, DEFAULT_CMD)
                tx_count += 1
            elif cmd and cmd.startswith("tx "):
                parts = cmd.split()
                if len(parts) == 3:
                    try:
                        addr = int(parts[1], 0)
                        cmd_val = int(parts[2], 0)
                        send_nec_frame(addr, cmd_val)
                        tx_count += 1
                    except ValueError:
                        print(f"[{now_str()}] ⚠️  格式错误，示例: tx 0x00 0x46")
                else:
                    print(f"[{now_str()}] ⚠️  格式错误，示例: tx 0x00 0x46")
            elif cmd:
                print(f"[{now_str()}] ⚠️  未知命令: '{cmd}'")

            # ── 检查红外信号 ──
            poll_start = time.time_ns() // 1000
            pulses = None
            while (time.time_ns() // 1000 - poll_start) < 100000:  # 100μs 轮询
                if lgpio.gpio_read(handle, IR_GPIO) == 0:
                    pulses, samples = read_frame()
                    break

            if not pulses:
                continue

            result = decode_pulses(pulses)
            if not result:
                continue

            ts = now_str()
            rx_count += 1

            # 计算方向/强度
            if samples and len(samples) > 0:
                max_vals = [max(s[i] for s in samples) for i in range(4)]
                best_idx = max_vals.index(max(max_vals))
                direction = CHANNEL_TO_DIR.get(best_idx, "未知")
                sv = sorted(max_vals, reverse=True)
                snr = sv[0] - sv[1] if len(sv) > 1 else 0
                sample_count = len(samples)
            else:
                max_vals, direction, snr, sample_count = [0,0,0,0], "未知", 0, 0

            # 输出结果
            if result.get('repeat'):
                if last_command is not None:
                    print(f"[{ts}] 🔄 重复码  命令=0x{last_command:02X}  "
                          f"方向={direction}  强度={max_vals}  SNR={snr}  "
                          f"采样={sample_count}")
            elif result.get('valid'):
                addr = result['address']
                cmd = result['command']
                last_command = cmd
                print(f"[{ts}] 📡 接收    地址=0x{addr:02X}  命令=0x{cmd:02X}  "
                      f"方向={direction}  强度={max_vals}  SNR={snr}  "
                      f"采样={sample_count}")
            else:
                err = result.get('error', '未知错误')
                print(f"[{ts}] ⚠️  解码失败  {err}  强度={max_vals}")

    except KeyboardInterrupt:
        pass
    finally:
        # 清理资源
        pwm.stop()
        try:
            lgpio.gpio_free(handle, IR_GPIO)
        except:
            pass
        lgpio.gpiochip_close(handle)
        adc.close()

        ts = now_str()
        print(f"\n[{ts}] 🔴 系统关闭")
        print(f"   统计: 接收 {rx_count} 帧  |  发射 {tx_count} 帧")
        print("═" * 70)

if __name__ == "__main__":
    main()