class Registers:
    def __init__(self, start_rsp=None):
        self.registers = {}  # 整数寄存器值
        self.float_registers = {}  # 浮点寄存器值（XMM 等）
        # 扩展寄存器映射，包含更多 x86_64 寄存器
        self.register_map = {
            # 原有整数寄存器
            "%rax": 64, "%rbx": 64, "%rcx": 64, "%rdx": 64,
            "%rsi": 64, "%rdi": 64, "%rbp": 64, "%rsp": 64,
            "%eax": 32, "%ebx": 32, "%ecx": 32, "%edx": 32,
            "%esi": 32, "%edi": 32, "%ebp": 32, "%esp": 32,
            "%ax": 16, "%bx": 16, "%cx": 16, "%dx": 16,
            "%r8": 64, "%r9": 64, "%r10": 64, "%r11": 64,
            "%r12": 64, "%r13": 64, "%r14": 64, "%r15": 64,
            "%r8d": 32, "%r9d": 32, "%r10d": 32, "%r11d": 32,
            "%r12d": 32, "%r13d": 32, "%r14d": 32, "%r15d": 32,
            "%r8w": 16, "%r9w": 16, "%r10w": 16, "%r11w": 16,
            "%r12w": 16, "%r13w": 16, "%r14w": 16, "%r15w": 16,
            # 浮点寄存器 (128 位 XMM，支持单精度和双精度)
            "%xmm0": 128, "%xmm1": 128, "%xmm2": 128, "%xmm3": 128,
            "%xmm4": 128, "%xmm5": 128, "%xmm6": 128, "%xmm7": 128,
            "%xmm8": 128, "%xmm9": 128, "%xmm10": 128, "%xmm11": 128,
            "%xmm12": 128, "%xmm13": 128, "%xmm14": 128, "%xmm15": 128,
        }
        self.custom_registers = {}
        
        if start_rsp is not None:
            self.registers["%rsp"] = start_rsp
            self.registers["%rbp"] = start_rsp

    def _normalize_register(self, reg):
        reg = reg.lower().strip()  # 确保去除多余空格
        # 检查是否是自定义寄存器
        if reg in self.custom_registers:
            return reg
        # 检查是否是内置寄存器
        if reg not in self.register_map:
            raise ValueError(f"未知寄存器: {reg}")
        base_regs = {
            "%eax": "%rax", "%ax": "%rax", "%ebx": "%rbx", "%bx": "%rbx",
            "%ecx": "%rcx", "%cx": "%rcx", "%edx": "%rdx", "%dx": "%rdx",
            "%esi": "%rsi", "%edi": "%rdi", "%ebp": "%rbp", "%esp": "%rsp",
            "%r8d": "%r8", "%r8w": "%r8", "%r9d": "%r9", "%r9w": "%r9",
            "%r10d": "%r10", "%r10w": "%r10", "%r11d": "%r11", "%r11w": "%r11",
            "%r12d": "%r12", "%r12w": "%r12", "%r13d": "%r13", "%r13w": "%r13",
            "%r14d": "%r14", "%r14w": "%r14", "%r15d": "%r15", "%r15w": "%r15",
        }
        return base_regs.get(reg, reg)

    def add_custom_register(self, reg_name, bit_width):
        if not reg_name.startswith("%"):
            reg_name = "%" + reg_name
        if reg_name in self.register_map:
            raise ValueError(f"寄存器 {reg_name} 已存在于内置寄存器中，无法自定义")
        if bit_width not in [8, 16, 32, 64, 128]:
            raise ValueError(f"位宽 {bit_width} 不支持，仅支持 8、16、32、64、128")
        self.custom_registers[reg_name] = bit_width
        print(f"已添加自定义寄存器: {reg_name} ({bit_width} 位)")

    def set_register(self, reg, value, suffix=None, is_float=False):
        reg = self._normalize_register(reg)
        # 检查寄存器来源并获取位宽
        if reg in self.custom_registers:
            bit_width = self.custom_registers[reg]
            print(f"从 custom_registers 获取位宽: {bit_width}")
        elif reg in self.register_map:
            bit_width = self.register_map[reg]
            print(f"从 register_map 获取位宽: {bit_width}")
        else:
            raise ValueError(f"寄存器 {reg} 未定义在 custom_registers 或 register_map 中")
        
        if is_float or reg.startswith("%xmm"):
            self.float_registers[reg] = float(value)
        else:
            if suffix == "q" or bit_width == 64:
                value = value & 0xFFFFFFFFFFFFFFFF
            elif suffix == "l" or bit_width == 32:
                value = value & 0xFFFFFFFF
            elif suffix == "w" or bit_width == 16:
                value = value & 0xFFFF
            elif bit_width == 8:
                value = value & 0xFF
            else:
                print(f"警告：未知位宽 {bit_width}，默认按 64 位处理")
                value = value & 0xFFFFFFFFFFFFFFFF
            self.registers[reg] = value
        print(f"已设置 {reg} = {value}")

    def get_register(self, reg, as_float=False):
        reg = self._normalize_register(reg)
        if as_float or reg.startswith("%xmm"):
            return self.float_registers.get(reg, 0.0)
        return self.registers.get(reg, 0)

    def display(self):
        if not self.registers and not self.float_registers and not self.custom_registers:
            print("寄存器未被使用")
            return
        
        # 整数寄存器
        if self.registers:
            int_regs = [(reg, self.registers[reg]) for reg in sorted(self.registers.keys())
                        if reg in self.register_map and not reg.startswith("%xmm")]
            if int_regs:
                name_width = max(len(reg) for reg, _ in int_regs) + 2
                hex_width = max(len(hex(val)) for _, val in int_regs) + 2
                dec_width = max(len(str(val)) for _, val in int_regs) + 2
                print("整数寄存器状态：")
                print(f"{'名称':<{name_width}} {'十六进制值':<{hex_width}} {'十进制值':<{dec_width}}")
                for reg, value in int_regs:
                    print(f"{reg:<{name_width}} {hex(value):<{hex_width}} {value:<{dec_width}}")
        
        # 浮点寄存器
        if self.float_registers:
            float_regs = [(reg, self.float_registers[reg]) for reg in sorted(self.float_registers.keys())]
            if float_regs:
                name_width = max(len(reg) for reg, _ in float_regs) + 2
                val_width = max(len(str(val)) for _, val in float_regs) + 2
                print("\n浮点寄存器状态：")
                print(f"{'名称':<{name_width}} {'值':<{val_width}}")
                for reg, value in float_regs:
                    print(f"{reg:<{name_width}} {value:<{val_width}}")
        
        # 自定义寄存器
            if self.custom_registers:
                custom_regs = [(reg, self.float_registers.get(reg) if reg in self.float_registers else self.registers.get(reg, 0),
                            reg in self.float_registers) for reg in sorted(self.custom_registers.keys())]
                if custom_regs:
                    name_width = max(len(reg) for reg, _, _ in custom_regs) + 2
                    # 计算值列最大宽度
                    val_lengths = [len(str(val) if is_float else f"{hex(val)} ({val})") 
                                for _, val, is_float in custom_regs]
                    val_width = max(val_lengths) + 2
                    print("\n自定义寄存器状态：")
                    print(f"{'名称':<{name_width}} {'值':<{val_width}}")
                    for reg, value, is_float in custom_regs:
                        val_str = str(value) if is_float else f"{hex(value)} ({value})"
                        suffix = " (浮点)" if is_float else ""
                        print(f"{reg:<{name_width}} {val_str + suffix:<{val_width}}")