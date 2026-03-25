'''
MAVLink X25 CRC code

Copyright Andrew Tridgell
Released under GNU LGPL version 3 or later
'''
from builtins import object

# 作用：实现 MAVLink 协议所需的 X.25 CRC 校验算法，供 datalink_serial.py 计算消息校验和。

class x25crc(object):
    '''x25 CRC - based on checksum.h from mavlink library'''
    def __init__(self, buf=None):
        self.crc = 0xffff
        if buf is not None:
            if isinstance(buf, str):
                self.accumulate_str(buf)
            else:
                self.accumulate(buf)

    def accumulate(self, buf):
        '''add in some more bytes'''
        accum = self.crc
        for b in buf:
            tmp = b ^ (accum & 0xff)
            tmp = (tmp ^ (tmp<<4)) & 0xFF
            accum = (accum>>8) ^ (tmp<<8) ^ (tmp<<3) ^ (tmp>>4)
        self.crc = accum

    def accumulate_str(self, buf):
        if isinstance(buf, str):
            buf = buf.encode('utf-8')
        self.accumulate(buf)
