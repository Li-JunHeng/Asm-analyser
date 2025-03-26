class Stack:
    def __init__(self, start_address=0x0FFFFFFF, min_address=0x1000):
        self.int_memory = {}  # 整数值映射
        self.float_memory = {}  # 浮点值映射
        self.rsp = start_address  # 栈顶指针
        self.min_address = min_address  # 栈的最小地址

    def push(self, value, size=8, is_float=False):
        if self.rsp - size < self.min_address:
            raise OverflowError("栈溢出，地址超出最小限制")
        if is_float:
            self.float_memory[self.rsp] = float(value)
        else:
            self.int_memory[self.rsp] = int(value)
        self.rsp -= size
        return self.rsp

    def pop(self, size=8, expect_float=False):
        if self.rsp + size not in self.int_memory and self.rsp + size not in self.float_memory:
            raise ValueError("栈中无数据可弹出")
        if expect_float and self.rsp + size in self.float_memory:
            value = self.float_memory.pop(self.rsp + size)
        elif self.rsp + size in self.int_memory:
            value = self.int_memory.pop(self.rsp + size)
        else:
            raise ValueError("栈顶数据类型不匹配")
        self.rsp += size
        return value, self.rsp

    def get_value(self, address, as_float=False):
        if as_float and address in self.float_memory:
            return self.float_memory[address]
        return self.int_memory.get(address, 0)

    def set_value(self, address, value, is_float=False):
        if address < self.min_address or address > 0xFFFFFFFF:
            raise ValueError(f"地址 {hex(address)} 超出有效范围")
        if is_float:
            self.float_memory[address] = float(value)
            if address in self.int_memory:
                del self.int_memory[address]  # 避免类型混淆
        else:
            self.int_memory[address] = int(value)
            if address in self.float_memory:
                del self.float_memory[address]

    def display(self):
        if not self.int_memory and not self.float_memory:
            print("栈当前为空")
            return
        print("栈状态：")
        print(f"{'地址':<16} {'类型':<8} {'值':<16}")
        all_addresses = set(self.int_memory.keys()) | set(self.float_memory.keys())
        for addr in sorted(all_addresses, reverse=True):
            if addr in self.int_memory:
                value = self.int_memory[addr]
                print(f"{hex(addr):<16} {'整数':<8} {hex(value):<16} ({value})")
            elif addr in self.float_memory:
                value = self.float_memory[addr]
                print(f"{hex(addr):<16} {'浮点':<8} {value:<16}")