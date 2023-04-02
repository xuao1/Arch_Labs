from m5.objects import Cache

# 创建 L1 Cache
class L1Cache(Cache):
    assoc = 2               # 2路组相联
    tag_latency = 2         # 访问缓存行标记所需的时钟周期数
    data_latency = 2        # 访问缓存行数据所需的时钟周期数
    response_latency = 2    # 从缓存返回响应所需的时钟周期数
    mshrs = 4               # MSHR 用于跟踪当前不在缓存中的数据的未完成请求
    tgts_per_mshr = 20      # 每个 MSHR 中可以存储的目标数

    # 构造函数
    def __init__(self, options=None):
        super(L1Cache, self).__init__()
        pass

    # connectCPU 将 CPU 连接到缓存; connectBus 将缓存连接到总线
    def connectCPU(self, cpu):
        raise NotImplementedError

    def connectBus(self, bus):
        self.mem_side = bus.cpu_side_ports


# 添加两个 L1Cache 的子类
class L1ICache(L1Cache):
    size = '16kB'

    def __init__(self, options=None):
        super(L1ICache, self).__init__(options)
        if not options or not options.l1i_size:
            return
        self.size = options.l1i_size

    def connectCPU(self, cpu):
        self.cpu_side = cpu.icache_port

class L1DCache(L1Cache):
    size = '64kB'

    def __init__(self, options=None):
        super(L1DCache, self).__init__(options)
        if not options or not options.l1d_size:
            return
        self.size = options.l1d_size

    def connectCPU(self, cpu):
        self.cpu_side = cpu.dcache_port


# 创建 L2 Cache
class L2Cache(Cache):
    size = '256kB'
    assoc = 8
    tag_latency = 20
    data_latency = 20
    response_latency = 20
    mshrs = 20
    tgts_per_mshr = 12

    def __init__(self, options=None):
        super(L2Cache, self).__init__()
        if not options or not options.l2_size:
            return
        self.size = options.l2_size

    def connectCPUSideBus(self, bus):
        self.cpu_side = bus.mem_side_ports
        # 相对于 L1 和 L2 之间的 BUS, L2 端就是 mem

    def connectMemSideBus(self, bus):
        self.mem_side = bus.cpu_side_ports

