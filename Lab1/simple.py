import m5
from m5.objects import *

# 创建系统
system = System()

# 设置时钟频率，指定电压域
system.clk_domain = SrcClockDomain()
system.clk_domain.clock = '1GHz'
system.clk_domain.voltage_domain = VoltageDomain()

# 内存模拟，512MB
system.mem_mode = 'timing'
system.mem_ranges = [AddrRange('512MB')]

# 创建 CPU
system.cpu = X86TimingSimpleCPU()

# 创建内存总线
system.membus = SystemXBar()

# 将 CPU 上的缓存端口连接到内存总线
system.cpu.icache_port = system.membus.cpu_side_ports
system.cpu.dcache_port = system.membus.cpu_side_ports

# 在 CPU 上创建一个 I/O 控制器并将其连接到内存总线
# 将系统中的一个特殊端口连接到 membus，允许系统读写内存。
system.cpu.createInterruptController()
system.cpu.interrupts[0].pio = system.membus.mem_side_ports
system.cpu.interrupts[0].int_requestor = system.membus.cpu_side_ports
system.cpu.interrupts[0].int_responder = system.membus.mem_side_ports

system.system_port = system.membus.cpu_side_ports

# 创建一个内存控制器并将其连接到内存总线，DDR3 控制器
system.mem_ctrl = MemCtrl()
system.mem_ctrl.dram = DDR3_1600_8x8()
system.mem_ctrl.dram.range = system.mem_ranges[0]
system.mem_ctrl.port = system.membus.mem_side_ports

# 完成创建模拟系统
# 接下来，设置 CPU 要执行的进程

# 创建进程
binary = 'tests/test-progs/hello/bin/x86/linux/hello'

# 设置 CPU 使用该进程作为工作负载，并在 CPU 中创建功能执行上下文
system.workload = SEWorkload.init_compatible(binary)

process = Process()
process.cmd = [binary]
system.cpu.workload = process
system.cpu.createThreads()

# 实例化系统并开始执行。
root = Root(full_system = False, system = system)
m5.instantiate()

print("Beginning simulation!")
exit_event = m5.simulate()

# 输出模拟器结束运行的信息
print('Exiting @ tick {} because {}'
      .format(m5.curTick(), exit_event.getCause()))
# m5.curTick() 获取当前模拟的时钟周期
# exit_event.getCause() 获取模拟器退出的原因

