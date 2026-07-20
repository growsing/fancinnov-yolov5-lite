#!/usr/bin/env python3
"""
红外发射测试 - 最终版
- 精确微秒延时（忙等待）
- 标准NEC帧发送（引导码 + 32位数据 + 停止位）
- 支持循环选择模式
- 彻底关断载波（PWM停止 + 控制IO拉低）
"""

import time
import lgpio
from rpi_hardware_pwm import HardwarePWM

# ==================== 配置 ====================
PWM_PIN = 12          # 硬件PWM引脚 (GPIO12)
CTRL_PIN = 17         # 发射控制IO
PWM_CHANNEL = 0       # 对应 GPIO12
PWM_FREQ = 38000      # 38kHz
PWM_DUTY = 50         # 占空比 50%

# ==================== 初始化 ====================
# 1. 硬件PWM
pwm = HardwarePWM(pwm_channel=PWM_CHANNEL, hz=PWM_FREQ, chip=0)
pwm.stop()  # 默认停止

# 2. GPIO控制 (树莓派5使用chip4)
handle = lgpio.gpiochip_open(4)
lgpio.gpio_claim_output(handle, CTRL_PIN)
lgpio.gpio_write(handle, CTRL_PIN, 0)  # 初始关闭

# ==================== 基础函数 ====================
def delay_us(us):
    """精确微秒延时（忙等待）"""
    start = time.perf_counter_ns()
    target = start + us * 1000
    while time.perf_counter_ns() < target:
        pass

def led_on():
    """开启载波：启动PWM + 控制IO高"""
    pwm.start(PWM_DUTY)
    lgpio.gpio_write(handle, CTRL_PIN, 1)

def led_off():
    """彻底关断：停止PWM + 控制IO低"""
    pwm.stop()
    lgpio.gpio_write(handle, CTRL_PIN, 0)
    delay_us(10)  # 确保电平稳定

def send_carrier(duration_ms):
    """发送连续载波（强度测试）"""
    led_on()
    time.sleep(duration_ms / 1000.0)
    led_off()

def send_nec_pulse(high_us, low_us):
    """发送一个NEC脉冲对（载波高 + 低电平）"""
    led_on()
    delay_us(high_us)
    led_off()
    delay_us(low_us)

# ==================== NEC帧发送 ====================
def send_nec_frame(address, command):
    """
    发送标准NEC帧
    address: 8位地址 (0-255)
    command: 8位命令 (0-255)
    """
    print(f"发送NEC帧: 地址=0x{address:02X}, 命令=0x{command:02X}")

    # 1. 引导码: 9ms 高 + 4.5ms 低
    send_nec_pulse(9000, 4500)

    # 2. 发送 32 位数据 (LSB first)
    # 数据顺序: 地址(8) | 地址反码(8) | 命令(8) | 命令反码(8)
    data_bytes = [
        address & 0xFF,
        (~address) & 0xFF,
        command & 0xFF,
        (~command) & 0xFF
    ]

    for byte in data_bytes:
        for i in range(8):
            bit = (byte >> i) & 1
            if bit == 1:
                send_nec_pulse(560, 1687)   # 逻辑1
            else:
                send_nec_pulse(560, 560)    # 逻辑0

    # 3. 停止位: 至少 560µs 低电平 (NEC协议要求)
    led_off()
    delay_us(560)

    print("NEC帧发送完成")

# ==================== 主循环 ====================
def main():
    try:
        print("\n" + "="*55)
        print("  红外发射测试 - 最终版")
        print("  模式1: 发射5秒连续载波 (强度测试)")
        print("  模式2: 发送NEC帧 (地址0x00 命令0x46)")
        print("  模式3: 发送自定义NEC帧 (输入地址和命令)")
        print("  模式0: 退出程序")
        print("="*55)

        while True:
            print("\n请选择模式 (1/2/3/0): ", end="")
            choice = input().strip()

            if choice == "1":
                print("发射5秒连续载波...")
                send_carrier(5000)
                print("✅ 完成")

            elif choice == "2":
                send_nec_frame(0x00, 0x46)  # 常用音量+码值
                print("✅ 完成")

            elif choice == "3":
                try:
                    addr = int(input("输入地址 (0-255, 十六进制如0x00): "), 0)
                    cmd = int(input("输入命令 (0-255, 十六进制如0x46): "), 0)
                    if 0 <= addr <= 255 and 0 <= cmd <= 255:
                        send_nec_frame(addr, cmd)
                        print("✅ 完成")
                    else:
                        print("❌ 数值超出范围 (0-255)")
                except ValueError:
                    print("❌ 输入格式错误，请输入整数或十六进制数")

            elif choice == "0":
                print("退出程序")
                break
            else:
                print("❌ 无效输入，请输入 1、2、3 或 0")

    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        led_off()
        pwm.stop()
        lgpio.gpio_free(handle, CTRL_PIN)
        lgpio.gpiochip_close(handle)
        print("资源已释放")

if __name__ == "__main__":
    main()