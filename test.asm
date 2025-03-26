# test.asm - 用于测试汇编模拟器的所有功能
# reg add myreg 32

# 1. 基本整数操作
mov $0x1234, %rax       # 将立即数移动到 rax
add $0x5678, %rax       # 整数加法
sub $0x1000, %rax       # 整数减法
push %rax               # 将 rax 压栈
pop %rbx                # 从栈弹出到 rbx
cmp $0x1234, %rbx       # 比较，更新整数标志

# 2. 条件跳转测试
je equal_label          # 如果 ZF=1 跳转
mov $0xDEAD, %rcx       # 如果没跳转，rcx 会被赋值
jmp end_jump

equal_label:
mov $0xBEEF, %rcx       # 如果跳转，rcx 赋值为 0xBEEF

end_jump:

# 3. 单精度浮点操作
movss $3.14, %xmm0      # 单精度浮点数移动到 xmm0
addss $2.86, %xmm0      # 单精度加法
subss $1.0, %xmm0       # 单精度减法
mulss $2.0, %xmm0       # 单精度乘法
divss $4.0, %xmm0       # 单精度除法
push %xmm0              # 将浮点数压栈
pop %xmm1               # 从栈弹出到 xmm1

# 4. 双精度浮点操作
movsd $3.1415926535, %xmm2  # 双精度浮点数移动到 xmm2
addsd $2.7182818284, %xmm2  # 双精度加法
mulsd $1.4142135623, %xmm2  # 双精度乘法
push %xmm2                  # 将双精度结果压栈
pop %xmm3                   # 从栈弹出到 xmm3

# 5. 自定义寄存器测试（需要在交互模式中先定义）
# 假设已通过 "reg add myreg 32" 定义 %myreg
mov $0xABCD, %myreg     # 将值移动到自定义寄存器
add $0x1111, %myreg     # 对自定义寄存器加法

# 6. 位运算和移位
mov $0xFF00, %rdx       # 设置初始值
and $0x0F0F, %rdx       # 按位与
or $0xF0F0, %rdx        # 按位或
shl $2, %rdx            # 左移 2 位
shr $1, %rdx            # 逻辑右移 1 位
sar $1, %rdx            # 算术右移 1 位

# 7. 栈帧和内存操作
lea 8(%rsp), %rsi       # 计算栈地址并存入 rsi
mov $0xCAFE, (%rsi)     # 将值写入栈内存
add $0x10, (%rsi)       # 对栈内存加法

# 8. 函数调用模拟
call func_label         # 调用子程序
jmp end_program

func_label:
push %rbp               # 保存栈基指针
mov %rsp, %rbp          # 设置新的栈帧
mov $0x12345678, %rdi   # 子程序内操作
pop %rbp                # 恢复栈基指针
ret                     # 返回

end_program:
# 程序结束