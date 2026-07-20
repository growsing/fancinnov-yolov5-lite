#!/usr/bin/env python3
import spidev
import time

# 初始化 SPI
spi = spidev.SpiDev()
spi.open(0, 0)          # 打开 /dev/spidev0.0 (CE0)
spi.max_speed_hz = 500000  # 500kHz 足够

def read_channel(channel):
    """读取 MCP3008 指定通道 (0-7) 的 ADC 值 (0-1023)"""
    if channel < 0 or channel > 7:
        return -1
    # 构建命令：起始位1，单端模式1，通道号3位
    # 具体格式：发送3字节，第一个字节 0x01 表示起始，第二个字节 (8+ch)<<4，第三个字节 0x00
    cmd = [1, (8 + channel) << 4, 0]
    response = spi.xfer2(cmd)
    # 合并后10位数据
    adc_value = ((response[1] & 3) << 8) + response[2]
    return adc_value

def voltage_from_adc(adc_value, vref=3.3):
    """将ADC值转换为电压（VREF默认3.3V）"""
    return (adc_value * vref) / 1023.0

try:
    print("红外强度测试 - 按 Ctrl+C 退出")
    print("CH0\tCH1\tCH2\tCH3\t(ADC原始值)")
    while True:
        values = [read_channel(i) for i in range(4)]
        # 打印原始值
        print(f"{values[0]:4d}\t{values[1]:4d}\t{values[2]:4d}\t{values[3]:4d}")
        time.sleep(0.2)
except KeyboardInterrupt:
    print("\n退出")
finally:
    spi.close()