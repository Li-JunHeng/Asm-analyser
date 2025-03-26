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
            print("整数寄存器状态：")
            print(f"{'名称':<8} {'十六进制值':<16} {'十进制值':<16}")
            for reg in sorted(self.registers.keys()):
                if reg in self.register_map and not reg.startswith("%xmm"):
                    value = self.registers[reg]
                    print(f"{reg:<8} {hex(value):<16} {value:<16}")
        # 浮点寄存器
        if self.float_registers:
            print("\n浮点寄存器状态：")
            print(f"{'名称':<8} {'值':<16}")
            for reg in sorted(self.float_registers.keys()):
                value = self.float_registers[reg]
                print(f"{reg:<8} {value:<16}")
        # 自定义寄存器
        if self.custom_registers:
            print("\n自定义寄存器状态：")
            print(f"{'名称':<8} {'值':<16}")
            for reg in sorted(self.custom_registers.keys()):
                if reg in self.float_registers:
                    value = self.float_registers[reg]
                    print(f"{reg:<8} {value:<16} (浮点)")
                else:
                    value = self.registers.get(reg, 0)
                    print(f"{reg:<8} {hex(value):<16} {value:<16}")

class Flags:
    def __init__(self):
        # 整数标志
        self.ZF = False  # 零标志
        self.SF = False  # 符号标志
        self.OF = False  # 溢出标志
        self.CF = False  # 进位标志
        # 浮点标志（模拟 FPU/IEEE 754）
        self.FZF = False  # 浮点零标志
        self.FSF = False  # 浮点符号标志
        self.FOF = False  # 浮点溢出标志
        self.FUF = False  # 浮点下溢标志
        self.FPE = False  # 浮点精度异常

    def update(self, result, op1, op2, operation):
        """更新整数标志"""
        self.ZF = (result == 0)
        self.SF = (result < 0)
        if operation == "add":
            self.OF = ((op1 > 0 and op2 > 0 and result < 0) or 
                       (op1 < 0 and op2 < 0 and result > 0))
            self.CF = (result < op1 or result < op2)
        elif operation == "sub" or operation == "cmp":
            self.OF = ((op1 > 0 and op2 < 0 and result < 0) or 
                       (op1 < 0 and op2 > 0 and result > 0))
            self.CF = (op1 < op2)
        elif operation == "test" or operation == "and" or operation == "or":
            self.OF = False
            self.CF = False
        elif operation == "mul":
            self.OF = (result > 0xFFFFFFFF)  # 假设 32 位溢出检测
            self.CF = self.OF

    def update_float(self, result, op1, op2, operation):
        """更新浮点标志"""
        import math
        self.FZF = (result == 0.0)
        self.FSF = (result < 0.0)
        # 溢出检测
        self.FOF = math.isinf(result)
        # 下溢检测
        self.FUF = (result != 0.0 and abs(result) < 2.2e-308)  # 接近双精度最小值
        # 精度异常（简单模拟）
        self.FPE = (result != op1 + op2 and operation == "add" and not self.FOF) or \
                   (result != op1 - op2 and operation == "sub" and not self.FOF) or \
                   (result != op1 * op2 and operation == "mul" and not self.FOF) or \
                   (result != op1 / op2 and operation == "div" and not self.FOF)

    def display(self):
        print("标志位状态：")
        print("整数标志：")
        print(f"ZF: {int(self.ZF)}  SF: {int(self.SF)}  OF: {int(self.OF)}  CF: {int(self.CF)}")
        print("浮点标志：")
        print(f"FZF: {int(self.FZF)}  FSF: {int(self.FSF)}  FOF: {int(self.FOF)}  FUF: {int(self.FUF)}  FPE: {int(self.FPE)}")

class Assembler:
    def __init__(self, start_address=0x0FFFFFFF):
        self.stack = Stack(start_address)
        self.registers = Registers(start_address)
        self.flags = Flags()  # 后续会替换为支持浮点的版本
        self.pc = 0
        self.instructions = []
        self.labels = {}
        self.handlers = {
            "mov": self.mov, "push": self.push, "pop": self.pop,
            "add": self.add, "sub": self.sub, "cmp": self.comp,
            "test": self.test, "jmp": self.jmp, "je": self.je,
            "jne": self.jne, "jg": self.jg, "jl": self.jl,
            "call": self.call, "ret": self.ret, "inc": self.inc, 
            "dec": self.dec, "mul": self.mul, 
            "lea": self.lea, "and": self.and_, "or": self.or_,
            "shl": self.shl, "sal": self.shl, "shr": self.shr, "sar": self.sar,
            # 单精度浮点
            "movss": self.movss, "addss": self.addss, "subss": self.subss,
            "mulss": self.mulss, "divss": self.divss,
            # 双精度浮点
            "movsd": self.movsd, "addsd": self.addsd, "subsd": self.subsd,
            "mulsd": self.mulsd, "divsd": self.divsd,
        }
        self.history = []
        self.breakpoints = set()

    def parse_instruction(self, instruction):
        instruction = instruction.strip()
        # 移除行内注释
        if "#" in instruction:
            instruction = instruction.split("#", 1)[0].strip()
        if instruction.endswith(":"):  # 标签定义
            label = instruction[:-1]
            self.labels[label] = len(self.instructions)
            return None, None, []
        parts = instruction.split(maxsplit=1)
        if not parts:
            return None, None, []
        opcode = parts[0].lower()
        operands = []
        if len(parts) > 1:
            raw_operands = parts[1].strip()
            current = ""
            paren_count = 0
            for char in raw_operands:
                if char == "(":
                    paren_count += 1
                    current += char
                elif char == ")":
                    paren_count -= 1
                    current += char
                elif char == "," and paren_count == 0:
                    operands.append(current.strip())
                    current = ""
                else:
                    current += char
            if current:
                operands.append(current.strip())
        if opcode[-1] in ["q", "l", "w"]:
            base_op = opcode[:-1]
            suffix = opcode[-1]
        else:
            base_op = opcode
            suffix = None
        return base_op, suffix, operands

    def parse_operand(self, operand, is_target=False):
        operand = operand.strip()
        if operand.startswith("$"):
            imm = operand[1:]
            try:
                return float(imm) if "." in imm else int(imm, 16) if imm.startswith("0x") else int(imm)
            except ValueError:
                raise ValueError(f"无法解析的立即数: {imm}")
        elif operand.startswith("%"):
            if operand.startswith("%xmm"):
                return self.registers.get_register(operand, as_float=True)
            return self.registers.get_register(operand)
        elif "(" in operand and ")" in operand:
            if operand.startswith("("):
                displacement = 0
                inner = operand[1:-1].strip()
            else:
                displacement, inner = operand.split("(", 1)
                inner = inner[:-1].strip()
                displacement = int(displacement, 16) if displacement.startswith("0x") else int(displacement)
            parts = [p.strip() for p in inner.split(",")]
            base = parts[0] if parts else None
            index = parts[1] if len(parts) > 1 else None
            scale = int(parts[2]) if len(parts) > 2 else 1
            address = 0
            if base and base.startswith("%"):
                address += self.registers.get_register(base)
            if index and index.startswith("%"):
                address += self.registers.get_register(index) * scale
            address += displacement
            return self.stack.get_value(address)
        elif is_target and operand in self.labels:
            return self.labels[operand]
        else:
            try:
                return float(operand) if "." in operand else int(operand, 16) if operand.startswith("0x") else int(operand)
            except ValueError:
                raise ValueError(f"无法解析的操作数: {operand}")

    def parse_address(self, operand):
        operand = operand.strip()
        if "(" not in operand or ")" not in operand:
            raise ValueError(f"不是有效的内存引用: {operand}")
        if operand.startswith("("):
            displacement = 0
            inner = operand[1:-1].strip()
        else:
            displacement, inner = operand.split("(", 1)
            inner = inner[:-1].strip()
            displacement = int(displacement, 16) if displacement.startswith("0x") else int(displacement)
        parts = [p.strip() for p in inner.split(",")]
        base = parts[0] if parts else None
        index = parts[1] if len(parts) > 1 else None
        scale = int(parts[2]) if len(parts) > 2 else 1
        address = 0
        if base and base.startswith("%"):
            address += self.registers.get_register(base)
        if index and index.startswith("%"):
            address += self.registers.get_register(index) * scale
        address += displacement
        return address

    def add_instruction(self, instruction):
        base_op, _, _ = self.parse_instruction(instruction)
        if instruction.startswith("#reg "):
            parts = instruction.split()
            if len(parts) == 3:
                reg_name, bit_width = parts[1], int(parts[2])
                self.registers.add_custom_register(reg_name, bit_width)
                print(f"从指令中定义寄存器: {reg_name} ({bit_width} 位)")
            return
        if base_op is not None or instruction.endswith(":"):  # 有效指令或标签
            self.instructions.append(instruction)
            if instruction.endswith(":"):
                print(f"已添加标签: {instruction[:-1]} -> 指令索引 {len(self.instructions) - 1}")
            else:
                print(f"已添加指令: {instruction}")

    # 执行单条指令（核心逻辑不变）
    def execute(self):
        if 0 <= self.pc < len(self.instructions):
            instruction = self.instructions[self.pc]
            print(f"执行指令 (PC={self.pc}): {instruction}")
            base_op, suffix, operands = self.parse_instruction(instruction)
            if base_op in self.handlers:
                self.handlers[base_op](suffix, operands)
            else:
                print(f"未知指令: {instruction}")
                self.pc += 1
            return True
        else:
            print("程序计数器超出指令范围，执行结束")
            return False

    def run(self):
        self.pc = 0
        while self.execute():
            self.refresh()
        print("程序执行完成")

    def mov(self, suffix, operands):
        if len(operands) != 2:
            raise ValueError(f"mov 需要 2 个操作数，收到 {len(operands)} 个: {operands}")
        src, dst = operands
        value = self.parse_operand(src)
        if dst.startswith("%"):
            self.registers.set_register(dst, value, suffix)
        elif "(" in dst and ")" in dst:
            address = self.parse_address(dst)
            self.stack.set_value(address, value)
        else:
            raise ValueError(f"目标操作数无效: {dst}")
        self.pc += 1

    def add(self, suffix, operands):
        if len(operands) != 2:
            raise ValueError(f"add 需要 2 个操作数，收到 {len(operands)} 个: {operands}")
        src, dst = operands
        src_value = self.parse_operand(src)
        if dst.startswith("%"):
            current = self.registers.get_register(dst)
            result = current + src_value
            self.registers.set_register(dst, result, suffix)
            self.flags.update(result, current, src_value, "add")
        elif "(" in dst and ")" in dst:
            address = self.parse_address(dst)
            current = self.stack.get_value(address)
            result = current + src_value
            self.stack.set_value(address, result)
            self.flags.update(result, current, src_value, "add")
        else:
            raise ValueError(f"目标操作数无效: {dst}")
        self.pc += 1

    def sub(self, suffix, operands):
        if len(operands) != 2:
            raise ValueError(f"sub 需要 2 个操作数，收到 {len(operands)} 个: {operands}")
        src, dst = operands
        src_value = self.parse_operand(src)
        if dst.startswith("%"):
            current = self.registers.get_register(dst)
            result = current - src_value
            self.registers.set_register(dst, result, suffix)
            self.flags.update(result, current, src_value, "sub")
        elif "(" in dst and ")" in dst:
            address = self.parse_address(dst)
            current = self.stack.get_value(address)
            result = current - src_value
            self.stack.set_value(address, result)
            self.flags.update(result, current, src_value, "sub")
        else:
            raise ValueError(f"目标操作数无效: {dst}")
        self.pc += 1

    def push(self, suffix, operands):
        if len(operands) != 1:
            raise ValueError(f"push 需要 1 个操作数，收到 {len(operands)} 个: {operands}")
        src = operands[0]
        value = self.parse_operand(src)
        size = {"q": 8, "l": 4, "w": 2}.get(suffix, 8)
        is_float = src.startswith("%xmm") or isinstance(value, float)
        new_rsp = self.stack.push(value, size, is_float=is_float)
        self.registers.set_register("%rsp", new_rsp, suffix)
        self.pc += 1

    def pop(self, suffix, operands):
        if len(operands) != 1:
            raise ValueError(f"pop 需要 1 个操作数，收到 {len(operands)} 个: {operands}")
        dst = operands[0]
        size = {"q": 8, "l": 4, "w": 2}.get(suffix, 8)
        expect_float = dst.startswith("%xmm")
        value, new_rsp = self.stack.pop(size, expect_float=expect_float)
        if dst.startswith("%"):
            self.registers.set_register(dst, value, suffix, is_float=expect_float)
            self.registers.set_register("%rsp", new_rsp, suffix)
        else:
            raise ValueError(f"目标操作数无效: {dst}")
        self.pc += 1

    def comp(self, suffix, operands):
        if len(operands) != 2:
            raise ValueError(f"cmp 需要 2 个操作数，收到 {len(operands)} 个: {operands}")
        src, dst = operands
        src_value = self.parse_operand(src)
        dst_value = self.parse_operand(dst)
        result = dst_value - src_value
        self.flags.update(result, dst_value, src_value, "cmp")
        self.pc += 1

    def test(self, suffix, operands):
        if len(operands) != 2:
            raise ValueError(f"test 需要 2 个操作数，收到 {len(operands)} 个: {operands}")
        src, dst = operands
        src_value = self.parse_operand(src)
        dst_value = self.parse_operand(dst)
        result = src_value & dst_value
        self.flags.update(result, src_value, dst_value, "test")
        self.pc += 1

    def jmp(self, suffix, operands):
        if len(operands) != 1:
            raise ValueError(f"jmp 需要 1 个操作数，收到 {len(operands)} 个: {operands}")
        target = self.parse_operand(operands[0], is_target=True)
        if not (0 <= target < len(self.instructions)):
            raise ValueError(f"跳转目标 {target} 超出指令范围")
        self.pc = target

    def je(self, suffix, operands):
        if len(operands) != 1:
            raise ValueError(f"je 需要 1 个操作数，收到 {len(operands)} 个: {operands}")
        target = self.parse_operand(operands[0], is_target=True)
        if not (0 <= target < len(self.instructions)):
            raise ValueError(f"跳转目标 {target} 超出指令范围")
        if self.flags.ZF:
            self.pc = target
        else:
            self.pc += 1

    def jne(self, suffix, operands):
        if len(operands) != 1:
            raise ValueError(f"jne 需要 1 个操作数，收到 {len(operands)} 个: {operands}")
        target = self.parse_operand(operands[0], is_target=True)
        if not (0 <= target < len(self.instructions)):
            raise ValueError(f"跳转目标 {target} 超出指令范围")
        if not self.flags.ZF:
            self.pc = target
        else:
            self.pc += 1

    def jg(self, suffix, operands):
        if len(operands) != 1:
            raise ValueError(f"jg 需要 1 个操作数，收到 {len(operands)} 个: {operands}")
        target = self.parse_operand(operands[0], is_target=True)
        if not (0 <= target < len(self.instructions)):
            raise ValueError(f"跳转目标 {target} 超出指令范围")
        if not self.flags.ZF and self.flags.SF == self.flags.OF:
            self.pc = target
        else:
            self.pc += 1

    def jl(self, suffix, operands):
        if len(operands) != 1:
            raise ValueError(f"jl 需要 1 个操作数，收到 {len(operands)} 个: {operands}")
        target = self.parse_operand(operands[0], is_target=True)
        if not (0 <= target < len(self.instructions)):
            raise ValueError(f"跳转目标 {target} 超出指令范围")
        if self.flags.SF != self.flags.OF:
            self.pc = target
        else:
            self.pc += 1

    def call(self, suffix, operands):
        if len(operands) != 1:
            raise ValueError(f"call 需要 1 个操作数，收到 {len(operands)} 个: {operands}")
        target = self.parse_operand(operands[0], is_target=True)
        if not (0 <= target < len(self.instructions)):
            raise ValueError(f"调用目标 {target} 超出指令范围")
        return_address = self.pc + 1
        size = {"q": 8, "l": 4, "w": 2}.get(suffix, 8)
        new_rsp = self.stack.push(return_address, size)  # 已更新，无需修改
        self.registers.set_register("%rsp", new_rsp, suffix)
        self.pc = target

    def ret(self, suffix, operands):
        if len(operands) != 0:
            raise ValueError(f"ret 不需要操作数，收到 {len(operands)} 个: {operands}")
        size = {"q": 8, "l": 4, "w": 2}.get(suffix, 8)
        return_address, new_rsp = self.stack.pop(size)  # 已更新，无需修改
        if not (0 <= return_address <= len(self.instructions)):
            raise ValueError(f"返回地址 {return_address} 无效")
        self.registers.set_register("%rsp", new_rsp, suffix)
        self.pc = return_address

    def inc(self, suffix, operands):
        if len(operands) != 1:
            raise ValueError(f"inc 需要 1 个操作数，收到 {len(operands)} 个: {operands}")
        dst = operands[0]
        if dst.startswith("%"):
            current = self.registers.get_register(dst)
            result = current + 1
            self.registers.set_register(dst, result, suffix)
            self.flags.update(result, current, 1, "add")
        elif "(" in dst and ")" in dst:
            address = self.parse_address(dst)
            current = self.stack.get_value(address)
            result = current + 1
            self.stack.set_value(address, result)
            self.flags.update(result, current, 1, "add")
        else:
            raise ValueError(f"目标操作数无效: {dst}")
        self.pc += 1

    def dec(self, suffix, operands):
        if len(operands) != 1:
            raise ValueError(f"dec 需要 1 个操作数，收到 {len(operands)} 个: {operands}")
        dst = operands[0]
        if dst.startswith("%"):
            current = self.registers.get_register(dst)
            result = current - 1
            self.registers.set_register(dst, result, suffix)
            self.flags.update(result, current, 1, "sub")
        elif "(" in dst and ")" in dst:
            address = self.parse_address(dst)
            current = self.stack.get_value(address)
            result = current - 1
            self.stack.set_value(address, result)
            self.flags.update(result, current, 1, "sub")
        else:
            raise ValueError(f"目标操作数无效: {dst}")
        self.pc += 1

    def mul(self, suffix, operands):
        if len(operands) != 1:
            raise ValueError(f"mul 需要 1 个操作数，收到 {len(operands)} 个: {operands}")
        src = operands[0]
        src_value = self.parse_operand(src)
        rax = self.registers.get_register("%rax")
        result = rax * src_value
        if suffix == "q":
            self.registers.set_register("%rax", result & 0xFFFFFFFFFFFFFFFF)
            self.registers.set_register("%rdx", (result >> 64) & 0xFFFFFFFFFFFFFFFF)
        elif suffix == "l":
            self.registers.set_register("%eax", result & 0xFFFFFFFF)
            self.registers.set_register("%edx", (result >> 32) & 0xFFFFFFFF)
        elif suffix == "w":
            self.registers.set_register("%ax", result & 0xFFFF)
            self.registers.set_register("%dx", (result >> 16) & 0xFFFF)
        else:
            self.registers.set_register("%rax", result)
        self.flags.update(result, rax, src_value, "mul")
        self.pc += 1

    def lea(self, suffix, operands):
        """LEA指令：加载有效地址到目标寄存器"""
        if len(operands) != 2:
            raise ValueError(f"lea 需要 2 个操作数，收到 {len(operands)} 个: {operands}")
        src, dst = operands
        if not dst.startswith("%"):
            raise ValueError(f"lea 目标必须是寄存器: {dst}")
        # 计算有效地址
        if "(" in src and ")" in src:
            address = self.parse_address(src)  # 直接复用 parse_address 计算地址
        else:
            raise ValueError(f"lea 源操作数必须是内存引用: {src}")
        # 将地址存入目标寄存器
        self.registers.set_register(dst, address, suffix)
        self.pc += 1

    def and_(self, suffix, operands):
        """AND指令：按位与运算"""
        if len(operands) != 2:
            raise ValueError(f"and 需要 2 个操作数，收到 {len(operands)} 个: {operands}")
        src, dst = operands
        src_value = self.parse_operand(src)
        # 目标是寄存器
        if dst.startswith("%"):
            current = self.registers.get_register(dst)
            result = current & src_value
            self.registers.set_register(dst, result, suffix)
            self.flags.update(result, current, src_value, "and")
        # 目标是内存
        elif "(" in dst and ")" in dst:
            address = self.parse_address(dst)
            current = self.stack.get_value(address)
            result = current & src_value
            self.stack.set_value(address, result)
            self.flags.update(result, current, src_value, "and")
        else:
            raise ValueError(f"and 目标操作数无效: {dst}")
        self.pc += 1

    def or_(self, suffix, operands):
        """OR指令：按位或运算"""
        if len(operands) != 2:
            raise ValueError(f"or 需要 2 个操作数，收到 {len(operands)} 个: {operands}")
        src, dst = operands
        src_value = self.parse_operand(src)
        # 目标是寄存器
        if dst.startswith("%"):
            current = self.registers.get_register(dst)
            result = current | src_value
            self.registers.set_register(dst, result, suffix)
            self.flags.update(result, current, src_value, "or")
        # 目标是内存
        elif "(" in dst and ")" in dst:
            address = self.parse_address(dst)
            current = self.stack.get_value(address)
            result = current | src_value
            self.stack.set_value(address, result)
            self.flags.update(result, current, src_value, "or")
        else:
            raise ValueError(f"or 目标操作数无效: {dst}")
        self.pc += 1

    def shl(self, suffix, operands):
        """SHL/SAL 指令：逻辑/算术左移"""
        if len(operands) != 2:
            raise ValueError(f"shl/sal 需要 2 个操作数，收到 {len(operands)} 个: {operands}")
        src, dst = operands
        shift_count = self.parse_operand(src) & 0x1F  # 移位计数限制在 0-31 位（x86 规范）
        # 目标是寄存器
        if dst.startswith("%"):
            current = self.registers.get_register(dst)
            if shift_count == 0:
                self.pc += 1
                return  # 移位 0 次，不改变状态
            result = current << shift_count
            # 根据后缀截断结果
            if suffix == "q":
                result &= 0xFFFFFFFFFFFFFFFF
            elif suffix == "l":
                result &= 0xFFFFFFFF
            elif suffix == "w":
                result &= 0xFFFF
            self.registers.set_register(dst, result, suffix)
            # 更新标志位，获取寄存器位宽
            bit_width = self.registers.register_map[dst]
            self.flags.ZF = (result == 0)
            self.flags.SF = (result < 0)
            self.flags.CF = ((current >> (bit_width - shift_count)) & 1) if shift_count <= bit_width else 0
            self.flags.OF = (self.flags.SF != ((current >> (bit_width - 1)) & 1)) if shift_count == 1 else False
        # 目标是内存
        elif "(" in dst and ")" in dst:
            address = self.parse_address(dst)
            current = self.stack.get_value(address)
            if shift_count == 0:
                self.pc += 1
                return
            result = current << shift_count
            if suffix == "q":
                result &= 0xFFFFFFFFFFFFFFFF
            elif suffix == "l":
                result &= 0xFFFFFFFF
            elif suffix == "w":
                result &= 0xFFFF
            self.stack.set_value(address, result)
            # 内存操作默认按 64 位处理
            self.flags.ZF = (result == 0)
            self.flags.SF = (result < 0)
            self.flags.CF = ((current >> (64 - shift_count)) & 1) if shift_count <= 64 else 0
            self.flags.OF = (self.flags.SF != ((current >> 63) & 1)) if shift_count == 1 else False
        else:
            raise ValueError(f"shl/sal 目标操作数无效: {dst}")
        self.pc += 1
    
    def shr(self, suffix, operands):
        """SHR 指令：逻辑右移"""
        if len(operands) != 2:
            raise ValueError(f"shr 需要 2 个操作数，收到 {len(operands)} 个: {operands}")
        src, dst = operands
        shift_count = self.parse_operand(src) & 0x1F  # 移位计数限制在 0-31 位
        # 目标是寄存器
        if dst.startswith("%"):
            current = self.registers.get_register(dst)
            if shift_count == 0:
                self.pc += 1
                return
            # 无符号右移
            result = current >> shift_count
            if suffix == "q":
                result &= 0xFFFFFFFFFFFFFFFF
            elif suffix == "l":
                result &= 0xFFFFFFFF
            elif suffix == "w":
                result &= 0xFFFF
            self.registers.set_register(dst, result, suffix)
            # 更新标志位
            self.flags.ZF = (result == 0)
            self.flags.SF = (result < 0)
            self.flags.CF = ((current >> (shift_count - 1)) & 1) if shift_count > 0 else 0
            self.flags.OF = ((current >> (self.register_map[dst] - 1)) & 1) if shift_count == 1 else False
        # 目标是内存
        elif "(" in dst and ")" in dst:
            address = self.parse_address(dst)
            current = self.stack.get_value(address)
            if shift_count == 0:
                self.pc += 1
                return
            result = current >> shift_count
            if suffix == "q":
                result &= 0xFFFFFFFFFFFFFFFF
            elif suffix == "l":
                result &= 0xFFFFFFFF
            elif suffix == "w":
                result &= 0xFFFF
            self.stack.set_value(address, result)
            # 更新标志位
            self.flags.ZF = (result == 0)
            self.flags.SF = (result < 0)
            self.flags.CF = ((current >> (shift_count - 1)) & 1) if shift_count > 0 else 0
            self.flags.OF = ((current >> 63) & 1) if shift_count == 1 else False
        else:
            raise ValueError(f"shr 目标操作数无效: {dst}")
        self.pc += 1
    
    def shr(self, suffix, operands):
        """SHR 指令：逻辑右移"""
        if len(operands) != 2:
            raise ValueError(f"shr 需要 2 个操作数，收到 {len(operands)} 个: {operands}")
        src, dst = operands
        shift_count = self.parse_operand(src) & 0x1F  # 移位计数限制在 0-31 位
        # 目标是寄存器
        if dst.startswith("%"):
            current = self.registers.get_register(dst)
            if shift_count == 0:
                self.pc += 1
                return
            # 无符号右移
            result = current >> shift_count
            if suffix == "q":
                result &= 0xFFFFFFFFFFFFFFFF
            elif suffix == "l":
                result &= 0xFFFFFFFF
            elif suffix == "w":
                result &= 0xFFFF
            self.registers.set_register(dst, result, suffix)
            # 更新标志位
            bit_width = self.registers.register_map[dst]
            self.flags.ZF = (result == 0)
            self.flags.SF = (result < 0)
            self.flags.CF = ((current >> (shift_count - 1)) & 1) if shift_count > 0 else 0
            self.flags.OF = ((current >> (bit_width - 1)) & 1) if shift_count == 1 else False
        # 目标是内存
        elif "(" in dst and ")" in dst:
            address = self.parse_address(dst)
            current = self.stack.get_value(address)
            if shift_count == 0:
                self.pc += 1
                return
            result = current >> shift_count
            if suffix == "q":
                result &= 0xFFFFFFFFFFFFFFFF
            elif suffix == "l":
                result &= 0xFFFFFFFF
            elif suffix == "w":
                result &= 0xFFFF
            self.stack.set_value(address, result)
            # 内存操作默认 64 位
            self.flags.ZF = (result == 0)
            self.flags.SF = (result < 0)
            self.flags.CF = ((current >> (shift_count - 1)) & 1) if shift_count > 0 else 0
            self.flags.OF = ((current >> 63) & 1) if shift_count == 1 else False
        else:
            raise ValueError(f"shr 目标操作数无效: {dst}")
        self.pc += 1
    
    def sar(self, suffix, operands):
        """SAR 指令：算术右移"""
        if len(operands) != 2:
            raise ValueError(f"sar 需要 2 个操作数，收到 {len(operands)} 个: {operands}")
        src, dst = operands
        shift_count = self.parse_operand(src) & 0x1F  # 移位计数限制在 0-31 位
        # 目标是寄存器
        if dst.startswith("%"):
            current = self.registers.get_register(dst)
            if shift_count == 0:
                self.pc += 1
                return
            # 有符号右移，Python 的 >> 默认是算术右移
            result = current >> shift_count
            if suffix == "q":
                result &= 0xFFFFFFFFFFFFFFFF
            elif suffix == "l":
                result &= 0xFFFFFFFF
            elif suffix == "w":
                result &= 0xFFFF
            self.registers.set_register(dst, result, suffix)
            # 更新标志位
            bit_width = self.registers.register_map[dst]
            self.flags.ZF = (result == 0)
            self.flags.SF = (result < 0)
            self.flags.CF = ((current >> (shift_count - 1)) & 1) if shift_count > 0 else 0
            self.flags.OF = False  # SAR 的 OF 在 shift_count == 1 时恒为 0
        # 目标是内存
        elif "(" in dst and ")" in dst:
            address = self.parse_address(dst)
            current = self.stack.get_value(address)
            if shift_count == 0:
                self.pc += 1
                return
            result = current >> shift_count
            if suffix == "q":
                result &= 0xFFFFFFFFFFFFFFFF
            elif suffix == "l":
                result &= 0xFFFFFFFF
            elif suffix == "w":
                result &= 0xFFFF
            self.stack.set_value(address, result)
            # 内存操作默认 64 位
            self.flags.ZF = (result == 0)
            self.flags.SF = (result < 0)
            self.flags.CF = ((current >> (shift_count - 1)) & 1) if shift_count > 0 else 0
            self.flags.OF = False
        else:
            raise ValueError(f"sar 目标操作数无效: {dst}")
        self.pc += 1

    # 单精度浮点指令（更新为支持浮点栈）
    def movss(self, suffix, operands):
        if len(operands) != 2:
            raise ValueError(f"movss 需要 2 个操作数，收到 {len(operands)} 个: {operands}")
        src, dst = operands
        value = self.parse_operand(src)
        if dst.startswith("%xmm"):
            self.registers.set_register(dst, value, is_float=True)
        elif "(" in dst and ")" in dst:
            address = self.parse_address(dst)
            self.stack.set_value(address, value, is_float=True)
        else:
            raise ValueError(f"movss 目标必须是 XMM 寄存器或内存: {dst}")
        self.pc += 1

    def addss(self, suffix, operands):
        if len(operands) != 2:
            raise ValueError(f"addss 需要 2 个操作数，收到 {len(operands)} 个: {operands}")
        src, dst = operands
        src_value = self.parse_operand(src)
        if dst.startswith("%xmm"):
            current = self.registers.get_register(dst, as_float=True)
            result = current + src_value
            self.registers.set_register(dst, result, is_float=True)
            self.flags.update_float(result, current, src_value, "add")
        elif "(" in dst and ")" in dst:
            address = self.parse_address(dst)
            current = self.stack.get_value(address, as_float=True)
            result = current + src_value
            self.stack.set_value(address, result, is_float=True)
            self.flags.update_float(result, current, src_value, "add")
        else:
            raise ValueError(f"addss 目标必须是 XMM 寄存器或内存: {dst}")
        self.pc += 1

    def subss(self, suffix, operands):
        if len(operands) != 2:
            raise ValueError(f"subss 需要 2 个操作数，收到 {len(operands)} 个: {operands}")
        src, dst = operands
        src_value = self.parse_operand(src)
        if dst.startswith("%xmm"):
            current = self.registers.get_register(dst, as_float=True)
            result = current - src_value
            self.registers.set_register(dst, result, is_float=True)
            self.flags.update_float(result, current, src_value, "sub")
        elif "(" in dst and ")" in dst:
            address = self.parse_address(dst)
            current = self.stack.get_value(address, as_float=True)
            result = current - src_value
            self.stack.set_value(address, result, is_float=True)
            self.flags.update_float(result, current, src_value, "sub")
        else:
            raise ValueError(f"subss 目标必须是 XMM 寄存器或内存: {dst}")
        self.pc += 1

    def mulss(self, suffix, operands):
        if len(operands) != 2:
            raise ValueError(f"mulss 需要 2 个操作数，收到 {len(operands)} 个: {operands}")
        src, dst = operands
        src_value = self.parse_operand(src)
        if dst.startswith("%xmm"):
            current = self.registers.get_register(dst, as_float=True)
            result = current * src_value
            self.registers.set_register(dst, result, is_float=True)
            self.flags.update_float(result, current, src_value, "mul")
        elif "(" in dst and ")" in dst:
            address = self.parse_address(dst)
            current = self.stack.get_value(address, as_float=True)
            result = current * src_value
            self.stack.set_value(address, result, is_float=True)
            self.flags.update_float(result, current, src_value, "mul")
        else:
            raise ValueError(f"mulss 目标必须是 XMM 寄存器或内存: {dst}")
        self.pc += 1

    def divss(self, suffix, operands):
        if len(operands) != 2:
            raise ValueError(f"divss 需要 2 个操作数，收到 {len(operands)} 个: {operands}")
        src, dst = operands
        src_value = self.parse_operand(src)
        if src_value == 0:
            raise ValueError("divss 除数不能为 0")
        if dst.startswith("%xmm"):
            current = self.registers.get_register(dst, as_float=True)
            result = current / src_value
            self.registers.set_register(dst, result, is_float=True)
            self.flags.update_float(result, current, src_value, "div")
        elif "(" in dst and ")" in dst:
            address = self.parse_address(dst)
            current = self.stack.get_value(address, as_float=True)
            result = current / src_value
            self.stack.set_value(address, result, is_float=True)
            self.flags.update_float(result, current, src_value, "div")
        else:
            raise ValueError(f"divss 目标必须是 XMM 寄存器或内存: {dst}")
        self.pc += 1

    # 双精度浮点指令
    def movsd(self, suffix, operands):
        if len(operands) != 2:
            raise ValueError(f"movsd 需要 2 个操作数，收到 {len(operands)} 个: {operands}")
        src, dst = operands
        value = self.parse_operand(src)
        if dst.startswith("%xmm"):
            self.registers.set_register(dst, value, is_float=True)
        elif "(" in dst and ")" in dst:
            address = self.parse_address(dst)
            self.stack.set_value(address, value, is_float=True)
        else:
            raise ValueError(f"movsd 目标必须是 XMM 寄存器或内存: {dst}")
        self.pc += 1

    def addsd(self, suffix, operands):
        if len(operands) != 2:
            raise ValueError(f"addsd 需要 2 个操作数，收到 {len(operands)} 个: {operands}")
        src, dst = operands
        src_value = self.parse_operand(src)
        if dst.startswith("%xmm"):
            current = self.registers.get_register(dst, as_float=True)
            result = current + src_value
            self.registers.set_register(dst, result, is_float=True)
            self.flags.update_float(result, current, src_value, "add")
        elif "(" in dst and ")" in dst:
            address = self.parse_address(dst)
            current = self.stack.get_value(address, as_float=True)
            result = current + src_value
            self.stack.set_value(address, result, is_float=True)
            self.flags.update_float(result, current, src_value, "add")
        else:
            raise ValueError(f"addsd 目标必须是 XMM 寄存器或内存: {dst}")
        self.pc += 1

    def subsd(self, suffix, operands):
        if len(operands) != 2:
            raise ValueError(f"subsd 需要 2 个操作数，收到 {len(operands)} 个: {operands}")
        src, dst = operands
        src_value = self.parse_operand(src)
        if dst.startswith("%xmm"):
            current = self.registers.get_register(dst, as_float=True)
            result = current - src_value
            self.registers.set_register(dst, result, is_float=True)
            self.flags.update_float(result, current, src_value, "sub")
        elif "(" in dst and ")" in dst:
            address = self.parse_address(dst)
            current = self.stack.get_value(address, as_float=True)
            result = current - src_value
            self.stack.set_value(address, result, is_float=True)
            self.flags.update_float(result, current, src_value, "sub")
        else:
            raise ValueError(f"subsd 目标必须是 XMM 寄存器或内存: {dst}")
        self.pc += 1

    def mulsd(self, suffix, operands):
        if len(operands) != 2:
            raise ValueError(f"mulsd 需要 2 个操作数，收到 {len(operands)} 个: {operands}")
        src, dst = operands
        src_value = self.parse_operand(src)
        if dst.startswith("%xmm"):
            current = self.registers.get_register(dst, as_float=True)
            result = current * src_value
            self.registers.set_register(dst, result, is_float=True)
            self.flags.update_float(result, current, src_value, "mul")
        elif "(" in dst and ")" in dst:
            address = self.parse_address(dst)
            current = self.stack.get_value(address, as_float=True)
            result = current * src_value
            self.stack.set_value(address, result, is_float=True)
            self.flags.update_float(result, current, src_value, "mul")
        else:
            raise ValueError(f"mulsd 目标必须是 XMM 寄存器或内存: {dst}")
        self.pc += 1

    def divsd(self, suffix, operands):
        if len(operands) != 2:
            raise ValueError(f"divsd 需要 2 个操作数，收到 {len(operands)} 个: {operands}")
        src, dst = operands
        src_value = self.parse_operand(src)
        if src_value == 0:
            raise ValueError("divsd 除数不能为 0")
        if dst.startswith("%xmm"):
            current = self.registers.get_register(dst, as_float=True)
            result = current / src_value
            self.registers.set_register(dst, result, is_float=True)
            self.flags.update_float(result, current, src_value, "div")
        elif "(" in dst and ")" in dst:
            address = self.parse_address(dst)
            current = self.stack.get_value(address, as_float=True)
            result = current / src_value
            self.stack.set_value(address, result, is_float=True)
            self.flags.update_float(result, current, src_value, "div")
        else:
            raise ValueError(f"divsd 目标必须是 XMM 寄存器或内存: {dst}")
        self.pc += 1

    def refresh(self):
        print("\n--- 当前系统状态 ---")
        print(f"程序计数器 (PC): {self.pc}")
        self.display_stack_frames()
        self.registers.display()
        self.flags.display()
        self.display_watched()
        print("--------------------\n")

    # 保存当前状态到历史
    def save_state(self):
        import copy
        state = {
            "int_memory": copy.deepcopy(self.stack.int_memory),  # 保存整数内存
            "float_memory": copy.deepcopy(self.stack.float_memory),  # 保存浮点内存
            "rsp": self.stack.rsp,
            "registers": copy.deepcopy(self.registers.registers),
            "float_registers": copy.deepcopy(self.registers.float_registers),
            "flags": {
                "ZF": self.flags.ZF, "SF": self.flags.SF, "OF": self.flags.OF, "CF": self.flags.CF,
                "FZF": self.flags.FZF, "FSF": self.flags.FSF, "FOF": self.flags.FOF, 
                "FUF": self.flags.FUF, "FPE": self.flags.FPE
            },
            "pc": self.pc
        }
        self.history.append(state)

    def restore_state(self):
        if not self.history:
            print("没有可回溯的状态")
            return False
        last_state = self.history.pop()
        self.stack.int_memory = last_state["int_memory"]
        self.stack.float_memory = last_state["float_memory"]
        self.stack.rsp = last_state["rsp"]
        self.registers.registers = last_state["registers"]
        self.registers.float_registers = last_state["float_registers"]
        self.flags.ZF = last_state["flags"]["ZF"]
        self.flags.SF = last_state["flags"]["SF"]
        self.flags.OF = last_state["flags"]["OF"]
        self.flags.CF = last_state["flags"]["CF"]
        self.flags.FZF = last_state["flags"]["FZF"]
        self.flags.FSF = last_state["flags"]["FSF"]
        self.flags.OF = last_state["flags"]["FOF"]
        self.flags.FUF = last_state["flags"]["FUF"]
        self.flags.FPE = last_state["flags"]["FPE"]
        self.pc = last_state["pc"]
        return True

    # 单步执行
    def step(self):
        self.save_state()
        if self.execute():
            self.refresh()
            if self.pc in self.breakpoints or self.check_conditional_breakpoints():
                print(f"到达断点或条件触发 PC={self.pc}，暂停执行")
                return False
            return True
        return False

    # 全速运行
    def run(self):
        self.pc = 0
        while True:
            self.save_state()
            if not self.execute():
                break
            self.refresh()
            if self.pc in self.breakpoints:
                print(f"到达断点 PC={self.pc}，暂停执行")
                break
        print("程序执行完成或暂停")

    # 回退一步
    def back(self):
        if self.restore_state():
            self.refresh()
            print(f"已回退到 PC={self.pc}")
        else:
            print("无法回退")

    # 设置断点
    def set_breakpoint(self, pc):
        if 0 <= pc < len(self.instructions):
            self.breakpoints.add(pc)
            print(f"已在 PC={pc} 设置断点")
        else:
            print(f"无效的断点位置: {pc}")

    # 删除断点
    def clear_breakpoint(self, pc):
        if pc in self.breakpoints:
            self.breakpoints.remove(pc)
            print(f"已删除 PC={pc} 的断点")
        else:
            print(f"PC={pc} 处没有断点")

    def set_conditional_breakpoint(self, condition):
        """设置条件断点，例如 'rax == 0x100'"""
        self.conditional_breakpoints = getattr(self, "conditional_breakpoints", [])
        self.conditional_breakpoints.append(condition)
        print(f"已设置条件断点: {condition}")

    def check_conditional_breakpoints(self):
        """检查条件断点是否触发"""
        if not hasattr(self, "conditional_breakpoints"):
            return False
        for cond in self.conditional_breakpoints:
            reg, op, val = cond.split()
            reg_val = self.registers.get_register(f"%{reg}")
            val = int(val, 16) if val.startswith("0x") else int(val)
            if op == "==" and reg_val == val:
                return True
            elif op == ">" and reg_val > val:
                return True
            elif op == "<" and reg_val < val:
                return True
        return False

    # 显示指令列表
    def show_instructions(self):
        if not self.instructions:
            print("指令列表为空")
            return
        print("当前指令列表：")
        for i, instr in enumerate(self.instructions):
            mark = " (断点)" if i in self.breakpoints else ""
            print(f"{i}: {instr}{mark}")
    
    def watch_variable(self, var):
        """添加要跟踪的变量，例如 %rax 或 0x0FFFFFFF"""
        self.watched_vars = getattr(self, "watched_vars", set())
        self.watched_vars.add(var)
        print(f"开始跟踪变量: {var}")

    def display_watched(self):
        if not hasattr(self, "watched_vars") or not self.watched_vars:
            return
        print("被跟踪的变量：")
        for var in self.watched_vars:
            if var.startswith("%"):
                if var.startswith("%xmm"):
                    val = self.registers.get_register(var, as_float=True)
                    print(f"{var}: {val} (浮点)")
                else:
                    val = self.registers.get_register(var)
                    print(f"{var}: {hex(val)} ({val})")
            elif var.startswith("0x"):
                addr = int(var, 16)
                if addr in self.stack.float_memory:
                    val = self.stack.get_value(addr, as_float=True)
                    print(f"内存[{var}]: {val} (浮点)")
                else:
                    val = self.stack.get_value(addr)
                    print(f"内存[{var}]: {hex(val)} ({val})")
            else:
                print(f"无效跟踪目标: {var}")
    
    def display_stack_frames(self):
        if not self.stack.int_memory and not self.stack.float_memory:
            print("栈当前为空，无栈帧")
            return
        print("栈帧信息：")
        all_addresses = set(self.stack.int_memory.keys()) | set(self.stack.float_memory.keys())
        if not all_addresses:
            return
        addresses = sorted(all_addresses, reverse=True)
        current_frame = []
        frames = []
        last_addr = None
        for addr in addresses:
            if last_addr is None or addr == last_addr - 8:
                if addr in self.stack.int_memory:
                    current_frame.append((addr, self.stack.int_memory[addr], "整数"))
                elif addr in self.stack.float_memory:
                    current_frame.append((addr, self.stack.float_memory[addr], "浮点"))
            else:
                if current_frame:
                    frames.append(current_frame)
                current_frame = [(addr, self.stack.int_memory.get(addr, self.stack.float_memory[addr]), 
                                "整数" if addr in self.stack.int_memory else "浮点")]
            last_addr = addr
        if current_frame:
            frames.append(current_frame)
        
        for i, frame in enumerate(frames):
            print(f"\n栈帧 {i}（从高地址到低地址）：")
            print(f"{'地址':<16} {'类型':<8} {'值':<16}")
            for addr, value, type_ in frame:
                if type_ == "整数":
                    print(f"{hex(addr):<16} {type_:<8} {hex(value):<16} ({value})")
                else:
                    print(f"{hex(addr):<16} {type_:<8} {value:<16}")
        print(f"当前 RSP: {hex(self.stack.rsp)}")

def interactive_mode(start_address=None):
    asm = Assembler(start_address)
    print("欢迎使用汇编模拟器！支持以下命令（含浮点指令如 movss、addss 等）：")
    print("- 输入汇编指令或标签直接添加（如 movss $3.14, %xmm0）")
    print("- '# reg add <name> [bit_width]': 添加自定义寄存器（如 # reg add myreg 32）")
    print("- 'load <filename>': 从文件加载汇编代码")
    print("- 'run': 全速运行程序")
    print("- 'step': 单步执行")
    print("- 'back': 回退一步")
    print("- 'bp <num>': 设置断点（如 bp 2）")
    print("- 'clear <num>': 清除断点")
    print("- 'list': 显示指令列表")
    print("- 'exit': 退出")
    while True:
        try:
            cmd = input("请输入指令> ").strip()
            if cmd.lower() == "exit":
                print("退出模拟器")
                break
            elif cmd.lower() == "run":
                if not asm.instructions:
                    print("指令列表为空，无法运行")
                else:
                    asm.run()
            elif cmd.lower() == "step":
                if not asm.instructions:
                    print("指令列表为空，无法单步")
                elif not asm.step():
                    print("单步执行结束")
            elif cmd.lower() == "back":
                asm.back()
            elif cmd.startswith("bp "):
                try:
                    pc = int(cmd.split()[1])
                    asm.set_breakpoint(pc)
                except (IndexError, ValueError):
                    print("断点格式错误，示例：bp 2")
            elif cmd.startswith("cbp "):
                try:
                    condition = cmd[4:].strip()
                    asm.set_conditional_breakpoint(condition)
                except Exception as e:
                    print(f"条件断点格式错误，示例：cbp rax == 0x100，错误: {e}")
            elif cmd.startswith("watch "):
                try:
                    var = cmd.split()[1]
                    asm.watch_variable(var)
                except IndexError:
                    print("watch 格式错误，示例：watch %rax 或 watch 0x0FFFFFFF")
            elif cmd.startswith("clear "):
                try:
                    pc = int(cmd.split()[1])
                    asm.clear_breakpoint(pc)
                except (IndexError, ValueError):
                    print("清除断点格式错误，示例：clear 2")
            elif cmd.lower() == "list":
                asm.show_instructions()
            elif cmd.startswith("load "):
                try:
                    filename = cmd.split(maxsplit=1)[1].strip()
                    with open(filename, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        print(f"正在从文件 {filename} 加载汇编代码...")
                        for i, line in enumerate(lines, 1):
                            line = line.strip()
                            if line.startswith("# reg add"):
                                parts = line.split()
                                if len(parts) == 5 and parts[0] == "#" and parts[1] == "reg" and parts[2] == "add":
                                    reg_name = parts[3]
                                    try:
                                        bit_width = int(parts[4])
                                        asm.registers.add_custom_register(reg_name, bit_width)
                                        print(f"行 {i}: 已添加自定义寄存器 {reg_name} ({bit_width} 位)")
                                    except ValueError:
                                        print(f"行 {i}: 位宽必须是整数: {parts[4]}")
                                else:
                                    reg_name = parts[3]
                                    print(f"检测到未定义的寄存器 {reg_name} 在行 {i}: {line}")
                                    define = input(f"是否定义 {reg_name} (输入位宽，如 32，或按回车跳过)？> ")
                                    if define:
                                        asm.registers.add_custom_register(reg_name, int(define))
                            elif line and not line.startswith('#'):
                                asm.add_instruction(line)
                        print(f"文件 {filename} 加载完成，共 {len(asm.instructions)} 条指令")
                except FileNotFoundError:
                    print(f"文件 {filename} 不存在，请检查路径或文件名")
                except Exception as e:
                    print(f"加载文件时发生错误: {e}")
            elif cmd.startswith("# reg add"):
                parts = cmd.split()
                if len(parts) == 4:
                    reg_name = parts[3]
                    try:
                        bit_width = int(input(f"请输入 {reg_name} 的位宽（例如 32）: "))
                        asm.registers.add_custom_register(reg_name, bit_width)
                    except ValueError:
                        print(f"位宽必须是整数，跳过定义 {reg_name}")
                elif len(parts) == 5:
                    reg_name = parts[3]
                    try:
                        bit_width = int(parts[4])
                        asm.registers.add_custom_register(reg_name, bit_width)
                    except ValueError:
                        print(f"位宽必须是整数: {parts[4]}")
                else:
                    print(f"无效的 # reg add 指令: {cmd}")
            elif cmd:
                asm.add_instruction(cmd)
            else:
                print("输入为空，请重新输入")
        except Exception as e:
            print(f"发生错误: {e}")

if __name__ == "__main__":
    interactive_mode(start_address=0x0FFFFFFF)