#!/usr/bin/env python3
"""
红外接收模块 - TSOP34438 + 树莓派5 (GPIO22)
NEC编码协议解码器 - 修复版（放宽容差）
"""

import lgpio
import time
from collections import deque

# ==================== 配置 ====================
IR_GPIO = 22
CHIP = 4

# NEC协议时序 (微秒) - 关键修复：放宽到400us容差
NEC_HEADER_MARK = 9000
NEC_HEADER_SPACE = 4500
NEC_BIT_MARK = 562
NEC_ZERO_SPACE = 562
NEC_ONE_SPACE = 1687
NEC_TOLERANCE = 400       # 从250放宽到400
NEC_REPEAT_SPACE = 2250

IDLE_TIMEOUT_US = 120000  # 120ms超时


def _get_pull_up_flag():
    try:
        return lgpio.SET_PULL_UP
    except AttributeError:
        try:
            return lgpio.SET_BIAS_PULL_UP
        except AttributeError:
            return 0


class NECReceiver:
    def __init__(self, gpio_pin=IR_GPIO, chip=CHIP):
        self.gpio = gpio_pin
        self.chip = chip
        self.handle = lgpio.gpiochip_open(chip)
        if self.handle < 0:
            raise RuntimeError(f"无法打开GPIO芯片 {chip}")
        
        pull_flag = _get_pull_up_flag()
        try:
            lgpio.gpio_claim_input(self.handle, gpio_pin, pull_flag)
        except Exception:
            lgpio.gpio_claim_input(self.handle, gpio_pin)
        
        self.last_tick = 0
        self.receiving = False
        self.bits = []
        self.command_buffer = deque(maxlen=10)
        
        print(f"✅ NEC红外接收器已启动 (GPIO{self.gpio}, lgpio芯片{self.chip})")
        print("等待红外信号... (按 Ctrl+C 退出)\\n")
    
    def _in_range(self, value, target):
        """检查值是否在目标容差范围内"""
        return abs(value - target) < NEC_TOLERANCE
    
    def _read_pulses(self):
        """读取一帧红外信号的原始脉冲序列"""
        timeout_start = time.time_ns() // 1000
        while lgpio.gpio_read(self.handle, self.gpio) == 1:
            if (time.time_ns() // 1000 - timeout_start) > 500000:
                return None
        
        pulses = []
        last_state = 0
        last_time = time.time_ns() // 1000
        
        while True:
            current_state = lgpio.gpio_read(self.handle, self.gpio)
            current_time = time.time_ns() // 1000
            
            if current_state != last_state:
                duration = current_time - last_time
                pulses.append((last_state, duration))
                last_state = current_state
                last_time = current_time
            
            if (current_time - last_time) > IDLE_TIMEOUT_US and len(pulses) > 0:
                pulses.append((last_state, current_time - last_time))
                break
        
        return pulses
    
    def _decode_pulses(self, pulses):
        """从原始脉冲解码NEC协议"""
        if not pulses or len(pulses) < 4:
            return None
        
        # 查找引导码 - 放宽容差
        for i in range(len(pulses) - 1):
            mark_dur = pulses[i][1]
            space_dur = pulses[i+1][1]
            
            # 引导码: 高电平~9000us + 低电平~4500us
            if (self._in_range(mark_dur, NEC_HEADER_MARK) and 
                self._in_range(space_dur, NEC_HEADER_SPACE)):
                return self._decode_data_bits(pulses, i + 2)
            
            # 重复码
            elif (self._in_range(mark_dur, NEC_HEADER_MARK) and 
                  self._in_range(space_dur, NEC_REPEAT_SPACE)):
                return {'repeat': True, 'valid': True}
        
        return None
    
    def _decode_data_bits(self, pulses, start_idx):
        """解码32位数据 - 关键修复：更宽松的位判断"""
        bits = []
        idx = start_idx
        
        while len(bits) < 32 and idx < len(pulses) - 1:
            mark_level, mark_dur = pulses[idx]
            if idx + 1 < len(pulses):
                space_level, space_dur = pulses[idx + 1]
            else:
                break
            
            # 关键修复：mark容差放宽，且允许小幅超限
            if mark_dur < 200:  # 太短，可能是噪声
                break
            
            # 关键修复：用阈值法判断0/1，不严格要求space在容差内
            if space_dur > 1000:  # 大于1ms认为是1
                bits.append(1)
            elif space_dur > 200:  # 大于200us认为是0
                bits.append(0)
            else:
                break
            
            idx += 2
        
        if len(bits) != 32:
            return {'valid': False, 'error': f'仅收到{len(bits)}位'}
        
        # 解析32位数据
        address = 0
        address_inv = 0
        command = 0
        command_inv = 0
        
        for i in range(8):
            address = (address << 1) | bits[i]
            address_inv = (address_inv << 1) | bits[i + 8]
            command = (command << 1) | bits[i + 16]
            command_inv = (command_inv << 1) | bits[i + 24]
        
        # 验证校验
        valid = ((address ^ address_inv) == 0xFF) and ((command ^ command_inv) == 0xFF)
        
        return {
            'address': address,
            'command': command,
            'valid': valid,
            'raw_bits': ''.join(map(str, bits)),
            'repeat': False
        }
    
    def run(self):
        """主循环：持续接收和解码"""
        last_command = None
        last_time = 0
        stats = {'ok': 0, 'fail': 0}
        
        try:
            while True:
                pulses = self._read_pulses()
                
                if not pulses:
                    continue
                
                result = self._decode_pulses(pulses)
                timestamp = time.strftime("%H:%M:%S")
                
                if not result:
                    stats['fail'] += 1
                    # 静默处理无法识别的格式，避免刷屏
                elif result.get('repeat'):
                    if last_command is not None:
                        print(f"[{timestamp}] 🔄 重复码 | 命令: 0x{last_command:02X}")
                elif result.get('valid'):
                    stats['ok'] += 1
                    addr = result['address']
                    cmd = result['command']
                    last_command = cmd
                    last_time = time.time()
                    print(f"[{timestamp}] 📡 有效信号 | 地址: 0x{addr:02X} | "
                          f"命令: 0x{cmd:02X} | 二进制: {cmd:08b}")
                else:
                    stats['fail'] += 1
                    err = result.get('error', '未知错误')
                    # 只显示前几次失败，避免刷屏
                    if stats['fail'] <= 3:
                        print(f"[{timestamp}] ⚠️ 解码失败: {err} | 脉冲数: {len(pulses)}")
                
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print(f"\\n\\n统计: 成功{stats['ok']} 失败{stats['fail']}")
    
    def close(self):
        """清理资源"""
        try:
            lgpio.gpio_free(self.handle, self.gpio)
        except:
            pass
        lgpio.gpiochip_close(self.handle)
        print("👋 接收器已关闭")


if __name__ == "__main__":
    import sys
    
    print("=" * 55)
    print("   TSOP34438 NEC红外接收器 - 修复版")
    print("   树莓派5 - GPIO22")
    print("=" * 55)
    
    receiver = None
    try:
        receiver = NECReceiver(IR_GPIO, CHIP)
        receiver.run()
    except ImportError as e:
        print(f"\\n❌ 缺少依赖: {e}")
        print("sudo apt-get install python3-lgpio")
        sys.exit(1)
    except PermissionError:
        print("\\n❌ 权限不足，请使用sudo运行")
        sys.exit(1)
    except RuntimeError as e:
        print(f"\\n❌ 运行时错误: {e}")
        print("尝试修改 CHIP = 0")
        sys.exit(1)
    except Exception as e:
        print(f"\\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if receiver:
            receiver.close()
