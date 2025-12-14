import structlog
from hyperserve.config import settings

logger = structlog.get_logger()

class BlockAllocator:
    """
    Manages 'Physical' GPU Memory Blocks.
    Simulates the Page Table of an OS.
    """
    def __init__(self):
        self.free_blocks = list(range(settings.MAX_GPU_BLOCKS))
        self.mapping = {} # Logical ID -> Physical ID

    def allocate(self) -> int:
        if not self.free_blocks:
            # In a real system, we would trigger eviction here
            logger.warning("oom_eviction_triggered")
            return -1 # OOM simulation
            
        block = self.free_blocks.pop()
        return block

    def free(self, block_id: int):
        self.free_blocks.append(block_id)   