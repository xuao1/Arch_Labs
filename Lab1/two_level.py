import m5
from m5.objects import *
from caches import *

# 使用 OptionParser 模块解析命令行参数
import argparse

parser = argparse.ArgumentParser(description='A simple system with 2-level cache.')
parser.add_argument("binary", default="tests/test-progs/hello/bin/x86/linux/hello", nargs="?", type=str,
                    help="Path to the binary to execute.")
parser.add_argument("--l1i_size",
                    help=f"L1 instruction cache size. Default: 16kB.")
parser.add_argument("--l1d_size",
                    help="L1 data cache size. Default: Default: 64kB.")
parser.add_argument("--l2_size",
                    help="L2 cache size. Default: 256kB.")

options = parser.parse_args()

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

# 创建 L1 Cache
system.cpu.icache = L1ICache(options)
system.cpu.dcache = L1DCache(options)

# 将 Cache 连接到 CPU 端口
system.cpu.icache.connectCPU(system.cpu)
system.cpu.dcache.connectCPU(system.cpu)

# 创建一个 L2 总线，将 L1 Cache 连接到 L2 Cache
system.l2bus = L2XBar()

system.cpu.icache.connectBus(system.l2bus)
system.cpu.dcache.connectBus(system.l2bus)

# 创建内存总线
system.membus = SystemXBar()

# 创建 L2 Cache 并将其连接到 L2 总线和内存总线
system.l2cache = L2Cache(options)
system.l2cache.connectCPUSideBus(system.l2bus)
system.l2cache.connectMemSideBus(system.membus)

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

# 设置 CPU 使用该进程作为工作负载，并在 CPU 中创建功能执行上下文
system.workload = SEWorkload.init_compatible(options.binary)

process = Process()
process.cmd = [options.binary]
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

