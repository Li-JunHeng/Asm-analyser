import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import ttkbootstrap as ttkb
from Assembler.assembler import Assembler  # 假设你的 Assembler 类在 Assembler/assembler.py 中

DEFAULT_ALPHA = 0.8

COLORS = {
    "bg_main": {"color": "#3a3a3a", "alpha": DEFAULT_ALPHA},  # 原#2b2b2b，提亮为更柔和的深灰
    "fg_text": {"color": "#d0d0d0", "alpha": DEFAULT_ALPHA},  # 原#ffffff，改为柔和浅灰
    "insert_bg": {"color": "#e0e0e0", "alpha": DEFAULT_ALPHA},  # 原#ffffff，改为柔和灰白
    "highlight_instruction": {"color": "#e6d98c", "alpha": DEFAULT_ALPHA},  # 原#ffff00，改为柔和淡黄
    "highlight_register": {"color": "#8cdfe6", "alpha": DEFAULT_ALPHA},  # 原#00ffff，改为柔和青色
    "highlight_immediate": {"color": "#a3d9a3", "alpha": DEFAULT_ALPHA},  # 原#00ff00，改为柔和浅绿
    "highlight_label": {"color": "#f2b589", "alpha": DEFAULT_ALPHA},  # 原#ffa500，改为柔和橙色
    "highlight_comment": {"color": "#a0a0a0", "alpha": DEFAULT_ALPHA},  # 原#808080，提亮为柔和中灰
    "highlight_current_line": {"color": "#5c7cfa", "alpha": DEFAULT_ALPHA},  # 原#0000ff，改为柔和蓝色
    "highlight_breakpoint": {"color": "#ff9999", "alpha": DEFAULT_ALPHA}  # 原#ff0000，改为柔和浅红
}

# 获取颜色值函数，方便在UI中使用
def get_color(key):
    """返回指定键的颜色值（不含透明度，Tkinter用）"""
    return COLORS[key]["color"]

# 示例：如果需要带透明度的颜色值（供未来扩展）
def get_color_with_alpha(key, alpha=None):
    """返回带透明度的颜色值（RGBA格式，供未来支持Alpha的库用）"""
    color = COLORS[key]["color"]
    alpha = COLORS[key]["alpha"] if alpha is None else alpha
    # 将#RRGGBB转换为RGBA
    r = int(color[1:3], 16)
    g = int(color[3:5], 16)
    b = int(color[5:7], 16)
    return (r, g, b, int(alpha * 255))  # RGBA元组

# 初始化主窗口
root = ttkb.Window(themename="darkly")
root.title("汇编模拟器")
root.geometry("1200x800")

# 定义字体
content_font = ("Cascadia Code", 14)

# 初始化汇编器实例
assembler = Assembler(start_address=0x0FFFFFFF)

# 行号到PC值的映射表
line_to_pc = {}  # 行号 -> PC
pc_to_line = {}  # PC -> 行号
breakpoints = set()  # 基于行号的断点集合

# --- 菜单栏 ---
menubar = tk.Menu(root)
root.config(menu=menubar)

file_menu = tk.Menu(menubar, tearoff=0)
menubar.add_cascade(label="文件", menu=file_menu)
file_menu.add_command(label="加载代码", command=lambda: load_file(), accelerator="Ctrl+O")
file_menu.add_command(label="保存代码", command=lambda: save_file(), accelerator="Ctrl+S")
file_menu.add_separator()
file_menu.add_command(label="退出", command=root.quit, accelerator="Ctrl+Q")

run_menu = tk.Menu(menubar, tearoff=0)
menubar.add_cascade(label="运行", menu=run_menu)
run_menu.add_command(label="全速运行", command=lambda: run_program(), accelerator="F5")
run_menu.add_command(label="单步执行", command=lambda: step_program(), accelerator="F10")
run_menu.add_command(label="回退一步", command=lambda: back_program(), accelerator="Ctrl+B")
run_menu.add_command(label="重置", command=lambda: reset_program(), accelerator="Ctrl+R")

debug_menu = tk.Menu(menubar, tearoff=0)
menubar.add_cascade(label="调试", menu=debug_menu)
debug_menu.add_command(label="设置断点", command=lambda: set_breakpoint(), accelerator="Ctrl+D")
debug_menu.add_command(label="清除断点", command=lambda: clear_breakpoint(), accelerator="Ctrl+Shift+D")
debug_menu.add_command(label="查看断点", command=lambda: status_notebook.select(breakpoints_frame), accelerator="Ctrl+B")
debug_menu.add_command(label="监视变量", command=lambda: watch_variable(), accelerator="Ctrl+W")

help_menu = tk.Menu(menubar, tearoff=0)
menubar.add_cascade(label="帮助", menu=help_menu)
help_menu.add_command(label="关于", command=lambda: show_about())

# --- 快捷键绑定 ---
root.bind("<Control-o>", lambda e: load_file())
root.bind("<Control-s>", lambda e: save_file())
root.bind("<Control-q>", lambda e: root.quit())
root.bind("<F5>", lambda e: run_program())
root.bind("<F10>", lambda e: step_program())
root.bind("<Control-b>", lambda e: back_program())
root.bind("<Control-r>", lambda e: reset_program())
root.bind("<Control-d>", lambda e: set_breakpoint())
root.bind("<Control-D>", lambda e: clear_breakpoint())
root.bind("<Control-B>", lambda e: status_notebook.select(breakpoints_frame))
root.bind("<Control-w>", lambda e: watch_variable())

# --- 主布局 ---
main_pane = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
main_pane.pack(fill=tk.BOTH, expand=True)

# 左侧代码编辑区 + 指令执行区
code_frame = ttk.Frame(main_pane)
main_pane.add(code_frame, weight=1)

code_label = ttk.Label(code_frame, text="汇编代码", font=content_font)
code_label.pack(pady=5)

code_container = ttk.Frame(code_frame)
code_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

line_numbers = tk.Text(code_container, width=4, bg=get_color("bg_main"), fg=get_color("fg_text"), font=content_font, state="disabled")
line_numbers.pack(side=tk.LEFT, fill=tk.Y)

code_text = tk.Text(code_container, height=25, bg=get_color("bg_main"), fg=get_color("fg_text"), insertbackground=get_color("insert_bg"), font=content_font)
code_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

scrollbar = ttk.Scrollbar(code_container, orient=tk.VERTICAL, command=lambda *args: (code_text.yview(*args), line_numbers.yview(*args)))
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
code_text.config(yscrollcommand=scrollbar.set)
line_numbers.config(yscrollcommand=scrollbar.set)

def update_line_numbers(event=None):
    line_numbers.config(state="normal")
    line_numbers.delete("1.0", tk.END)
    line_count = int(code_text.index("end-1c").split(".")[0])
    line_numbers.insert("1.0", "\n".join(str(i) for i in range(1, line_count + 1)))
    line_numbers.config(state="disabled")
    code_text.yview_moveto(line_numbers.yview()[0])

code_text.bind("<KeyRelease>", lambda e: (highlight_code(), update_line_numbers()))
code_text.bind("<MouseWheel>", update_line_numbers)

code_text.tag_configure("instruction", foreground=get_color("highlight_instruction"))
code_text.tag_configure("register", foreground=get_color("highlight_register"))
code_text.tag_configure("immediate", foreground=get_color("highlight_immediate"))
code_text.tag_configure("label", foreground=get_color("highlight_label"))
code_text.tag_configure("comment", foreground=get_color("highlight_comment"))
code_text.tag_configure("current_line", background=get_color("highlight_current_line"))
code_text.tag_configure("breakpoint", background=get_color("highlight_breakpoint"))

def highlight_code(event=None):
    code_text.tag_remove("instruction", "1.0", tk.END)
    code_text.tag_remove("register", "1.0", tk.END)
    code_text.tag_remove("immediate", "1.0", tk.END)
    code_text.tag_remove("label", "1.0", tk.END)
    code_text.tag_remove("comment", "1.0", tk.END)

    code = code_text.get("1.0", tk.END)
    lines = code.split("\n")
    for i, line in enumerate(lines):
        if "#" in line:
            comment_start = line.index("#")
            code_text.tag_add("comment", f"{i+1}.{comment_start}", f"{i+1}.end")
        if line.endswith(":"):
            code_text.tag_add("label", f"{i+1}.0", f"{i+1}.end")
        parts = line.split()
        if parts and parts[0] in assembler.handlers:
            code_text.tag_add("instruction", f"{i+1}.0", f"{i+1}.{len(parts[0])}")
        for part in parts[1:]:
            if part.startswith("%"):
                start = line.index(part)
                code_text.tag_add("register", f"{i+1}.{start}", f"{i+1}.{start + len(part)}")
            elif part.startswith("$"):
                start = line.index(part)
                code_text.tag_add("immediate", f"{i+1}.{start}", f"{i+1}.{start + len(part)}")

def highlight_current_line_and_breakpoints(current_line=None):
    code_text.tag_remove("current_line", "1.0", tk.END)
    code_text.tag_remove("breakpoint", "1.0", tk.END)
    
    if current_line is not None:
        code_text.tag_add("current_line", f"{current_line}.0", f"{current_line}.end")
    
    for bp in breakpoints:
        code_text.tag_add("breakpoint", f"{bp}.0", f"{bp}.end")

exec_frame = ttk.Frame(code_frame)
exec_frame.pack(fill=tk.X, padx=5, pady=5)
exec_label = ttk.Label(exec_frame, text="指令执行", font=content_font)
exec_label.pack(side=tk.LEFT)
exec_entry = tk.Entry(exec_frame, bg=get_color("bg_main"), fg=get_color("fg_text"), insertbackground=get_color("insert_bg"), font=content_font)
exec_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
exec_button = ttk.Button(exec_frame, text="执行", command=lambda: execute_single_instruction())
exec_button.pack(side=tk.RIGHT)
exec_entry.bind("<Return>", lambda e: execute_single_instruction())

# 右侧状态面板
status_frame = ttk.Frame(main_pane)
main_pane.add(status_frame, weight=1)
status_notebook = ttk.Notebook(status_frame)
status_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

registers_frame = ttk.Frame(status_notebook)
status_notebook.add(registers_frame, text="寄存器")
registers_tree = ttk.Treeview(registers_frame, columns=("名称", "值"), show="headings", height=15)
registers_tree.heading("名称", text="名称")
registers_tree.heading("值", text="值")
registers_tree.column("名称", width=100)
registers_tree.column("值", width=200)
style = ttk.Style()
style.configure("Treeview.Heading", font=content_font)
registers_tree.tag_configure("content", font=content_font)
registers_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

stack_frame = ttk.Frame(status_notebook)
status_notebook.add(stack_frame, text="堆栈")
stack_tree = ttk.Treeview(stack_frame, columns=("地址", "类型", "值"), show="headings", height=15)
stack_tree.heading("地址", text="地址")
stack_tree.heading("类型", text="类型")
stack_tree.heading("值", text="值")
stack_tree.column("地址", width=100)
stack_tree.column("类型", width=50)
stack_tree.column("值", width=150)
stack_tree.tag_configure("content", font=content_font)
stack_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

flags_frame = ttk.Frame(status_notebook)
status_notebook.add(flags_frame, text="标志")
flags_tree = ttk.Treeview(flags_frame, columns=("标志", "值"), show="headings", height=10)
flags_tree.heading("标志", text="标志")
flags_tree.heading("值", text="值")
flags_tree.column("标志", width=100)
flags_tree.column("值", width=50)
flags_tree.tag_configure("content", font=content_font)
flags_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

watch_frame = ttk.Frame(status_notebook)
status_notebook.add(watch_frame, text="监视变量")
watch_tree = ttk.Treeview(watch_frame, columns=("变量", "值"), show="headings", height=15)
watch_tree.heading("变量", text="变量")
watch_tree.heading("值", text="值")
watch_tree.column("变量", width=100)
watch_tree.column("值", width=200)
watch_tree.tag_configure("content", font=content_font)
watch_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

breakpoints_frame = ttk.Frame(status_notebook)
status_notebook.add(breakpoints_frame, text="断点")
breakpoints_tree = ttk.Treeview(breakpoints_frame, columns=("行号", "指令"), show="headings", height=15)
breakpoints_tree.heading("行号", text="行号")
breakpoints_tree.heading("指令", text="指令")
breakpoints_tree.column("行号", width=50)
breakpoints_tree.column("指令", width=250)
breakpoints_tree.tag_configure("content", font=content_font)
breakpoints_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

def breakpoint_context_menu(event):
    item = breakpoints_tree.identify_row(event.y)
    if item:
        breakpoints_tree.selection_set(item)
        line = int(breakpoints_tree.item(item, "values")[0])
        menu = tk.Menu(root, tearoff=0)
        menu.add_command(label="删除断点", command=lambda: [breakpoints.remove(line), update_ui(), log(f"已删除断点 行号={line}")])
        menu.post(event.x_root, event.y_root)

breakpoints_tree.bind("<Button-3>", breakpoint_context_menu)

console_frame = ttk.Frame(root)
console_frame.pack(fill=tk.X, padx=5, pady=5)
console_label = ttk.Label(console_frame, text="输出控制台", font=content_font)
console_label.pack()
console_text = tk.Text(console_frame, height=10, bg=get_color("bg_main"), fg=get_color("fg_text"), font=content_font)
console_text.pack(fill=tk.X, padx=5, pady=5)

# 自定义对话框
class CustomInputDialog(tk.Toplevel):
    def __init__(self, parent, title, prompt, input_type=str):
        super().__init__(parent)
        self.transient(parent)
        self.title(title)
        self.result = None
        self.input_type = input_type
        self.configure(bg=get_color("bg_main"))
        self.geometry("300x150")

        ttk.Label(self, text=prompt, font=content_font, foreground=get_color("fg_text")).pack(pady=10)
        self.entry = ttk.Entry(self, font=content_font)
        self.entry.pack(pady=5, padx=10, fill=tk.X)
        ttk.Button(self, text="确定", command=self.on_ok).pack(pady=10)

        self.entry.focus_set()
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self.on_cancel)

    def on_ok(self):
        value = self.entry.get()
        try:
            if self.input_type == int:
                self.result = int(value)
            else:
                self.result = value
            self.destroy()
        except ValueError:
            log("输入无效，请输入正确的值")

    def on_cancel(self):
        self.result = None
        self.destroy()

def custom_askinteger(title, prompt):
    dialog = CustomInputDialog(root, title, prompt, int)
    root.wait_window(dialog)
    return dialog.result

def custom_askstring(title, prompt):
    dialog = CustomInputDialog(root, title, prompt, str)
    root.wait_window(dialog)
    return dialog.result

# --- 函数定义 ---

def log(message):
    console_text.insert(tk.END, message + "\n")
    console_text.see(tk.END)

def load_file():
    file_path = filedialog.askopenfilename(filetypes=[("汇编文件", "*.asm"), ("所有文件", "*.*")])
    if file_path:
        with open(file_path, 'r', encoding='utf-8') as f:
            code_text.delete("1.0", tk.END)
            code_text.insert(tk.END, f.read())
        highlight_code()
        update_line_numbers()
        load_code_to_assembler()
        update_ui()
        log(f"已加载文件: {file_path}")

def save_file():
    file_path = filedialog.asksaveasfilename(defaultextension=".asm", filetypes=[("汇编文件", "*.asm")])
    if file_path:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(code_text.get("1.0", tk.END))
        log(f"已保存文件: {file_path}")

def run_program():
    load_code_to_assembler()
    code_lines = code_text.get("1.0", tk.END).strip().splitlines()
    assembler.pc = 0
    for i, line in enumerate(code_lines, 1):
        line = line.strip()
        if not line or line.startswith(".data") or line.startswith(".text") or line.startswith("#") or (assembler.in_data_section and line.split(maxsplit=1)[0].endswith(":")):
            continue
        if i in breakpoints:
            log(f"到达断点 行号={i}: {line}")
            highlight_current_line_and_breakpoints(i)
            update_ui()
            break
        assembler.add_instruction(line)
        if assembler.execute():
            log(f"执行行 {i}: {line}")
        else:
            log(f"行 {i} 执行失败: {line}")
            break
        update_ui()
    else:
        log("程序全速运行完成")
    highlight_current_line_and_breakpoints()

def step_program():
    if not assembler.instructions:
        load_code_to_assembler()
    if assembler.step():
        update_ui()
        log(f"单步执行: PC={assembler.pc}")
    else:
        log("单步执行结束")

def back_program():
    assembler.back()
    update_ui()
    log(f"回退到 PC={assembler.pc}")

def reset_program():
    global assembler
    assembler = Assembler(start_address=0x0FFFFFFF)
    breakpoints.clear()
    line_to_pc.clear()
    pc_to_line.clear()
    update_ui()
    log("模拟器已重置")

def set_breakpoint():
    bp = custom_askinteger("设置断点", "请输入行号：")
    if bp is not None and 1 <= bp <= int(code_text.index("end-1c").split(".")[0]):
        breakpoints.add(bp)
        update_ui()
        log(f"已在行号={bp} 设置断点")
    else:
        log("无效的行号")

def clear_breakpoint():
    bp = custom_askinteger("清除断点", "请输入要清除的断点行号：")
    if bp in breakpoints:
        breakpoints.remove(bp)
        update_ui()
        log(f"已清除行号={bp} 的断点")
    else:
        log("该行号没有断点")

def watch_variable():
    var = custom_askstring("监视变量", "请输入变量名（如 %rax 或 0x0FFFFFFF）：")
    if var:
        assembler.watch_variable(var)
        update_ui()
        log(f"开始监视变量: {var}")

def show_about():
    log("汇编模拟器 v1.0\n作者: Li-JunHeng")

def load_code_to_assembler():
    global line_to_pc, pc_to_line
    assembler.instructions.clear()
    assembler.labels.clear()
    assembler.symbols.clear()
    assembler.in_data_section = False
    assembler.pc = 0
    line_to_pc.clear()
    pc_to_line.clear()
    code = code_text.get("1.0", tk.END).strip().splitlines()
    
    pc = 0
    for i, line in enumerate(code):
        line = line.strip()
        if not line:
            continue
        if line.startswith(".data"):
            assembler.in_data_section = True
            log("进入 .data 段")
            continue
        elif line.startswith(".text"):
            assembler.in_data_section = False
            log("进入 .text 段")
            continue
        if assembler.in_data_section:
            parts = line.split(maxsplit=1)
            if len(parts) >= 2 and parts[0].endswith(":"):
                name = parts[0].rstrip(":")
                value = parts[1]
                try:
                    if value.startswith("0x"):
                        val = int(value, 16)
                    elif "." in value:
                        val = float(value)
                    else:
                        val = int(value)
                    assembler.symbols[name] = (assembler.data_address, "int" if isinstance(val, int) else "float", val)
                    assembler.stack.set_value(assembler.data_address, val, is_float=isinstance(val, float))
                    log(f"定义符号 {name} 在 {hex(assembler.data_address)} = {val}")
                    assembler.data_address += 8
                except ValueError:
                    log(f"无法解析数据值: {value}")
            else:
                log(f"无效的数据定义: {line}")
        elif line.startswith("# reg add"):
            parts = line.split()
            if len(parts) >= 4:
                reg_name = parts[3]
                try:
                    bit_width = int(parts[4]) if len(parts) > 4 else 32
                    assembler.registers.add_custom_register(reg_name, bit_width)
                    log(f"添加自定义寄存器: {reg_name} ({bit_width} 位)")
                except ValueError:
                    log(f"位宽无效: {parts[4]}")
            else:
                log(f"无效的 # reg add 命令: {line}")
        else:
            assembler.add_instruction(line)
            line_to_pc[i + 1] = pc
            pc_to_line[pc] = i + 1
            pc += 1

def execute_single_instruction():
    cmd = exec_entry.get().strip()
    if not cmd:
        log("指令为空")
        return
    if cmd.startswith(".data"):
        assembler.in_data_section = True
        log("进入 .data 段")
    elif cmd.startswith(".text"):
        assembler.in_data_section = False
        log("进入 .text 段")
    elif assembler.in_data_section:
        parts = cmd.split(maxsplit=1)
        if len(parts) >= 2 and parts[0].endswith(":"):
            name = parts[0].rstrip(":")
            value = parts[1]
            try:
                if value.startswith("0x"):
                    val = int(value, 16)
                elif "." in value:
                    val = float(value)
                else:
                    val = int(value)
                assembler.symbols[name] = (assembler.data_address, "int" if isinstance(val, int) else "float", val)
                assembler.stack.set_value(assembler.data_address, val, is_float=isinstance(val, float))
                log(f"定义符号 {name} 在 {hex(assembler.data_address)} = {val}")
                assembler.data_address += 8
            except ValueError:
                log(f"无法解析数据值: {value}")
        else:
            log(f"无效的数据定义: {cmd}")
    elif cmd.startswith("# reg add"):
        parts = cmd.split()
        if len(parts) >= 4:
            reg_name = parts[3]
            try:
                bit_width = int(parts[4]) if len(parts) > 4 else 32
                assembler.registers.add_custom_register(reg_name, bit_width)
                log(f"添加自定义寄存器: {reg_name} ({bit_width} 位)")
            except ValueError:
                log(f"位宽无效: {parts[4]}")
        else:
            log(f"无效的 # reg add 命令: {cmd}")
    else:
        assembler.add_instruction(cmd)
        if assembler.execute():
            log(f"执行指令: {cmd}")
        else:
            log(f"执行失败: {cmd}")
    update_ui()
    exec_entry.delete(0, tk.END)

def update_ui():
    registers_tree.delete(*registers_tree.get_children())
    for reg in assembler.registers.register_map:
        if reg in assembler.registers.registers:
            value = assembler.registers.get_register(reg)
            registers_tree.insert("", tk.END, values=(reg, f"{hex(value)} ({value})"), tags=("content",))
        elif reg in assembler.registers.float_registers:
            value = assembler.registers.get_register(reg, as_float=True)
            registers_tree.insert("", tk.END, values=(reg, str(value)), tags=("content",))

    stack_tree.delete(*stack_tree.get_children())
    all_addresses = sorted(set(assembler.stack.int_memory.keys()) | set(assembler.stack.float_memory.keys()), reverse=True)
    for addr in all_addresses:
        if addr in assembler.stack.int_memory:
            value = assembler.stack.int_memory[addr]
            stack_tree.insert("", tk.END, values=(hex(addr), "整数", f"{hex(value)} ({value})"), tags=("content",))
        elif addr in assembler.stack.float_memory:
            value = assembler.stack.float_memory[addr]
            stack_tree.insert("", tk.END, values=(hex(addr), "浮点", str(value)), tags=("content",))

    flags_tree.delete(*flags_tree.get_children())
    flags = [
        ("ZF", assembler.flags.ZF), ("SF", assembler.flags.SF), ("OF", assembler.flags.OF), ("CF", assembler.flags.CF),
        ("FZF", assembler.flags.FZF), ("FSF", assembler.flags.FSF), ("FOF", assembler.flags.FOF),
        ("FUF", assembler.flags.FUF), ("FPE", assembler.flags.FPE)
    ]
    for flag, value in flags:
        flags_tree.insert("", tk.END, values=(flag, int(value)), tags=("content",))

    watch_tree.delete(*watch_tree.get_children())
    if hasattr(assembler, "watched_vars"):
        for var in assembler.watched_vars:
            if var.startswith("%"):
                if var.startswith("%xmm"):
                    value = assembler.registers.get_register(var, as_float=True)
                    watch_tree.insert("", tk.END, values=(var, str(value)), tags=("content",))
                else:
                    value = assembler.registers.get_register(var)
                    watch_tree.insert("", tk.END, values=(var, f"{hex(value)} ({value})"), tags=("content",))
            elif var.startswith("0x"):
                addr = int(var, 16)
                if addr in assembler.stack.float_memory:
                    value = assembler.stack.get_value(addr, as_float=True)
                    watch_tree.insert("", tk.END, values=(var, str(value)), tags=("content",))
                else:
                    value = assembler.stack.get_value(addr)
                    watch_tree.insert("", tk.END, values=(var, f"{hex(value)} ({value})"), tags=("content",))

    breakpoints_tree.delete(*breakpoints_tree.get_children())
    code_lines = code_text.get("1.0", tk.END).strip().splitlines()
    for bp in sorted(breakpoints):
        instr = code_lines[bp - 1] if 1 <= bp <= len(code_lines) else "无效行"
        breakpoints_tree.insert("", tk.END, values=(bp, instr), tags=("content",))

    highlight_current_line_and_breakpoints()

# 初始更新 UI
update_ui()
highlight_code()
update_line_numbers()

# 运行主循环
root.mainloop()