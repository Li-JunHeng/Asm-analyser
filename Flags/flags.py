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