from Assembler.assembler import Assembler

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