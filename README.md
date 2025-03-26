  汇编模拟器 README body { font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; background-color: #f5f5f5; color: #333; } h1, h2, h3 { color: #2c3e50; } h1 { border-bottom: 2px solid #3498db; padding-bottom: 10px; } h2 { border-bottom: 1px solid #bdc3c7; padding-bottom: 5px; } pre, code { background-color: #ecf0f1; padding: 10px; border-radius: 5px; overflow-x: auto; } pre { margin: 10px 0; } ul, ol { margin: 10px 0; padding-left: 20px; } li { margin: 5px 0; } a { color: #3498db; text-decoration: none; } a:hover { text-decoration: underline; } .note { background-color: #fef9e7; border-left: 4px solid #f1c40f; padding: 10px; margin: 10px 0; border-radius: 3px; }

README
============

本项目是一个基于 Python 实现的 x86\_64 汇编语言模拟器，支持整数和浮点运算，旨在提供一个工具用于模拟和理解汇编指令的执行过程。该模拟器包括栈、寄存器、标志位管理和基本的汇编指令集，并提供交互模式，适用于汇编语言学习、代码调试以及简单程序的模拟运行。

项目概述
----

汇编模拟器由以下核心组件构成：

*   **Stack 类**：负责模拟栈内存，支持整数和浮点数的 push 和 pop 操作，管理栈顶指针 (RSP) 和地址范围。
*   **Registers 类**：实现 x86\_64 寄存器的模拟，包括通用寄存器（如 %rax、%rbx）和浮点寄存器（如 %xmm0-%xmm15），并支持用户自定义寄存器。
*   **Flags 类**：管理整数和浮点运算的标志位，例如 ZF、SF、OF 和浮点标志 FZF、FOF 等。
*   **Assembler 类**：核心类，负责解析和执行汇编指令，支持断点设置、单步执行和状态回溯等功能。
*   **交互模式**：通过命令行界面与用户交互，支持动态添加指令、加载文件和调试操作。

本模拟器的设计目标是为计算机科学相关专业的学生和研究人员提供一个教育工具，帮助深入理解汇编语言的底层机制，同时可作为小型项目的测试平台。

功能特性
----

本模拟器具备以下主要功能特性：

1.  **支持的指令集**：
    *   基本操作：mov、add、sub、cmp、test、jmp、je、jne、jg、jl、call、ret、inc、dec、mul、lea、and、or、shl、shr、sar
    *   栈操作：push、pop
    *   单精度浮点指令：movss、addss、subss、mulss、divss
    *   双精度浮点指令：movsd、addsd、subsd、mulsd、divsd
2.  **寄存器管理**：
    
    内置了对 x86\_64 常用寄存器的支持，包括 8 位、16 位、32 位和 64 位操作，同时允许用户通过自定义方式扩展寄存器。
    
3.  **栈管理**：
    
    栈支持整数和浮点数据的存储，地址从高到低增长，默认起始地址为 0x0FFFFFFF，并包含栈溢出检测机制。
    
4.  **标志位管理**：
    
    每次运算后自动更新整数标志（ZF、SF、OF、CF）和浮点标志（FZF、FSF、FOF、FUF、FPE），以模拟真实 CPU 的标志位行为。
    
5.  **调试功能**：
    *   单步执行（step）：逐条执行指令并显示当前状态。
    *   断点设置（bp）：在指定指令位置暂停执行。
    *   状态回溯（back）：撤销上一步操作，恢复到之前的状态。
    *   变量跟踪（watch）：实时监控寄存器或内存地址的值。
6.  **交互性**：
    
    通过命令行支持用户输入指令、从文件加载汇编代码，并提供栈、寄存器和标志位的实时状态显示。
    

使用方法
----

本节详细说明如何运行和使用汇编模拟器，包括启动方式和支持的命令。

### 启动模拟器

在 Python 环境中运行以下命令以启动模拟器：

    python assembler.py

启动后，系统将显示欢迎信息并列出可用命令。

### 支持的命令

*   **汇编指令**：直接输入汇编指令（如 `mov $5, %rax` 或 `push $3.14`），将被添加到指令列表中。
*   **\# reg add \[bit\_width\]**：定义自定义寄存器，例如 `# reg add myreg 32`。
*   **load** ：从指定文件加载汇编代码，例如 `load test.asm`。
*   **run**：全速运行所有指令，直至程序结束或遇到断点。
*   **step**：单步执行一条指令并显示当前状态。
*   **back**：回退到上一步状态。
*   **bp** ：在指定指令编号处设置断点，例如 `bp 2`。
*   **clear** ：清除指定位置的断点，例如 `clear 2`。
*   **watch** ：跟踪指定变量的值，例如 `watch %rax` 或 `watch 0x0FFFFFFF`。
*   **list**：显示当前指令列表。
*   **exit**：退出模拟器。

### 使用示例

以下是一个简单的加法程序示例，展示如何使用模拟器：

    
    请输入指令> mov $10, %rax
    已添加指令: mov $10, %rax
    请输入指令> mov $20, %rbx
    已添加指令: mov $20, %rbx
    请输入指令> add %rbx, %rax
    已添加指令: add %rbx, %rax
    请输入指令> list
    当前指令列表：
    0: mov $10, %rax
    1: mov $20, %rbx
    2: add %rbx, %rax
    请输入指令> run
    执行指令 (PC=0): mov $10, %rax
    执行指令 (PC=1): mov $20, %rbx
    执行指令 (PC=2): add %rbx, %rax
    程序执行完成
        

执行完成后，寄存器 %rax 的值将变为 30。用户可通过 `step` 命令单步查看执行过程，或使用 `watch %rax` 监控 %rax 的值变化。

代码结构
----

模拟器的代码采用模块化设计，分为以下主要类：

*   **Stack**：管理栈内存，提供 push、pop 和 get\_value 等方法。
*   **Registers**：处理寄存器操作，支持多种位宽和浮点寄存器管理。
*   **Flags**：维护运算标志位状态，根据结果自动更新。
*   **Assembler**：核心类，负责解析和执行汇编指令，并实现调试功能。
*   **interactive\_mode**：交互模式的主函数，处理用户输入和命令执行。

这种结构便于功能扩展，例如通过向 `Assembler.handlers` 添加新方法即可支持更多指令。

注意事项
----

**栈溢出**：若栈顶指针 (RSP) 低于最小地址（默认 0x1000），将抛出 OverflowError 异常。

**浮点指令**：浮点操作（如 movss、addsd）需使用 %xmm 系列寄存器或内存地址作为目标。

**指令格式**：操作数需符合语法规范，例如寄存器以 % 开头，立即数以 $ 开头，内存引用使用括号表示。

局限性与改进建议
--------

当前版本的模拟器功能较为基础，存在以下局限性及改进空间：

*   不支持多线程或复杂的系统调用，仅适用于单线程程序模拟。
*   缺乏对符号表的支持，无法处理高级汇编特性。
*   错误提示较为简单，可能难以定位复杂问题。

未来可考虑以下改进方向：

*   扩展指令集，增加条件移动（cmov）或更多位操作指令。
*   开发图形用户界面，提升交互体验。
*   实现反汇编功能，支持从机器码生成汇编代码。
*   加入异常处理机制，模拟中断或陷阱行为。

许可
--

本项目目前未指定正式许可，仅用于学术和学习目的。如需使用，请尊重作者的知识产权，避免用于商业用途。