
# 修复接收端解码：LSB first的数据，解码时bits[0]是最低位
#!/usr/bin/env python3
"""
红外接收与方向判断 - 解码修复版
关键修复：LSB first的数据，bits[0]是最低位，解码时bit[0]应放在bit0位置
"""

import lgpio
import spidev
import time

# ==================== 配置 ====================
IR_GPIO = 22
CHIP = 4

SPI_BUS = 0
SPI_DEVICE = 0
SPI_SPEED = 1000000

NEC_HEADER_MARK = 9000
NEC_HEADER_SPACE = 4500
NEC_BIT_MARK = 562
NEC_ZERO_SPACE = 562
NEC_ONE_SPACE = 1687
NEC_TOLERANCE = 400
NEC_REPEAT_SPACE = 2250
IDLE_TIMEOUT_US = 120000
SAMPLE_INTERVAL_NS = 200000

CHANNEL_TO_DIR = {0: "前", 1: "右", 2: "后", 3: "左"}

def _get_pull_up_flag():
    try:
        return lgpio.SET_PULL_UP
    except AttributeError:
        try:
            return lgpio.SET_BIAS_PULL_UP
        except AttributeError:
            return 0

class MCP3008:
    def __init__(self, bus=SPI_BUS, device=SPI_DEVICE, speed=SPI_SPEED):
        self.spi = spidev.SpiDev()
        self.spi.open(bus, device)
        self.spi.max_speed_hz = speed
        self.spi.mode = 0

    def read(self, channel):
        if channel < 0 or channel > 7:
            return -1
        cmd = [1, (8 + channel) << 4, 0]
        resp = self.spi.xfer2(cmd)
        return ((resp[1] & 3) << 8) + resp[2]

    def read_all(self):
        return [self.read(i) for i in range(4)]

    def close(self):
        self.spi.close()

class IRDirectionReceiver:
    def __init__(self, gpio_pin=IR_GPIO, chip=CHIP):
        self.gpio = gpio_pin
        self.chip = chip
        self.handle = lgpio.gpiochip_open(chip)
        if self.handle < 0:
            raise RuntimeError(f"无法打开GPIO芯片 {chip}")
        pull_flag = _get_pull_up_flag()
        try:
            lgpio.gpio_claim_input(self.handle, gpio_pin, pull_flag)
        except:
            lgpio.gpio_claim_input(self.handle, gpio_pin)
        self.adc = MCP3008()
        print(f"✅ 接收器已启动 (GPIO{self.gpio})")

    def _in_range(self, value, target):
        return abs(value - target) < NEC_TOLERANCE

    def _read_pulses_and_sample(self):
        timeout_start = time.time_ns() // 1000
        while lgpio.gpio_read(self.handle, self.gpio) == 1:
            if (time.time_ns() // 1000 - timeout_start) > 500000:
                return None, None
        pulses = []
        samples = []
        last_state = 0
        last_time = time.time_ns() // 1000
        last_sample = time.time_ns()
        while True:
            current_state = lgpio.gpio_read(self.handle, self.gpio)
            current_time = time.time_ns() // 1000
            if current_state != last_state:
                duration = current_time - last_time
                pulses.append((last_state, duration))
                last_state = current_state
                last_time = current_time
            if len(pulses) == 0 and current_state == 0:
                now_ns = time.time_ns()
                if (now_ns - last_sample) >= SAMPLE_INTERVAL_NS:
                    samples.append(self.adc.read_all())
                    last_sample = now_ns
            if (current_time - last_time) > IDLE_TIMEOUT_US and len(pulses) > 0:
                pulses.append((last_state, current_time - last_time))
                break
        return pulses, samples

    def _decode_pulses(self, pulses):
        if not pulses or len(pulses) < 4:
            return None
        for i in range(len(pulses) - 1):
            mark_dur = pulses[i][1]
            space_dur = pulses[i+1][1]
            if (self._in_range(mark_dur, NEC_HEADER_MARK) and 
                self._in_range(space_dur, NEC_HEADER_SPACE)):
                return self._decode_data_bits(pulses, i+2)
            elif (self._in_range(mark_dur, NEC_HEADER_MARK) and 
                  self._in_range(space_dur, NEC_REPEAT_SPACE)):
                return {'repeat': True, 'valid': True}
        return None

    def _decode_data_bits(self, pulses, start_idx):
        bits = []
        idx = start_idx
        while len(bits) < 32 and idx < len(pulses)-1:
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
            return {'valid': False, 'error': f'仅收到{len(bits)}位'}
        
        # ==================== 关键修复 ====================
        # 发射端是LSB first: bits[0]=bit0, bits[1]=bit1, ...
        # 所以 bits[i] 就是第i位的值，不需要移位！
        address = sum(bits[i] << i for i in range(8))
        address_inv = sum(bits[i+8] << i for i in range(8))
        command = sum(bits[i+16] << i for i in range(8))
        command_inv = sum(bits[i+24] << i for i in range(8))
        # ==================================================
        
        valid = ((address ^ address_inv) == 0xFF) and ((command ^ command_inv) == 0xFF)
        return {
            'address': address,
            'command': command,
            'valid': valid,
            'raw_bits': ''.join(map(str, bits)),
            'repeat': False
        }

    def close(self):
        try:
            lgpio.gpio_free(self.handle, self.gpio)
        except:
            pass
        lgpio.gpiochip_close(self.handle)
        self.adc.close()
        print("👋 已关闭")

def main():
    print("="*60)
    print("   红外接收 + 方向判断 (解码修复版)")
    print("   修复：LSB first解码，bits[i]直接放在第i位")
    print("="*60)
    receiver = None
    try:
        receiver = IRDirectionReceiver()
        print("✅ 准备就绪，等待红外信号...\\n")
        last_command = None
        while True:
            pulses, samples = receiver._read_pulses_and_sample()
            if not pulses:
                continue
            result = receiver._decode_pulses(pulses)
            if not result:
                continue
            timestamp = time.strftime("%H:%M:%S")
            if samples and len(samples) > 0:
                max_vals = [max(s[i] for s in samples) for i in range(4)]
                sample_count = len(samples)
            else:
                max_vals = [0, 0, 0, 0]
                sample_count = 0
            best_val = max(max_vals)
            best_idx = max_vals.index(best_val)
            direction = CHANNEL_TO_DIR.get(best_idx, "未知")
            sorted_vals = sorted(max_vals, reverse=True)
            snr = sorted_vals[0] - sorted_vals[1] if len(sorted_vals) > 1 else 0
            if result.get('repeat'):
                if last_command is not None:
                    print(f"[{timestamp}] 🔄 重复码 | 命令: 0x{last_command:02X} | "
                          f"方向: {direction} | 强度: {max_vals} | 采样: {sample_count}")
            elif result.get('valid'):
                addr = result['address']
                cmd = result['command']
                last_command = cmd
                print(f"[{timestamp}] 📡 有效信号 | 地址: 0x{addr:02X} | "
                      f"命令: 0x{cmd:02X} | 方向: {direction} | "
                      f"强度: {max_vals} | 采样: {sample_count} | 信噪比: {snr}")
            else:
                print(f"[{timestamp}] ⚠️ 解码失败 | 强度: {max_vals}")
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\\n\\n用户中断")
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if receiver:
            receiver.close()

if __name__ == "__main__":
    main()
