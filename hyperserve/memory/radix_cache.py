import time
import structlog
from typing import List, Optional, Dict, Tuple
from hyperserve.memory.allocator import BlockAllocator

logger = structlog.get_logger()

class RadixNode:
    def __init__(self, key_tokens: List[int] = None, parent=None):
        self.key = key_tokens or []
        self.children: Dict[int, RadixNode] = {} # Map first_token -> Node
        self.parent = parent
        self.value = None  # Holds the Physical Block ID
        self.last_access = time.time()
        self.lock_count = 0 

class RadixCache:
    """
    Implements a Radix Tree for Prefix Caching.
    Architecture:
    - Tree nodes store token sequences.
    - If a request matches a path, we return the cached Block ID.
    - This allows 'System Prompts' to be computed ONCE and reused forever.
    """
    def __init__(self, allocator: BlockAllocator):
        self.root = RadixNode()
        self.allocator = allocator
        self.total_tokens_saved = 0

    def match_prefix(self, tokens: List[int]) -> Tuple[RadixNode, int]:
        """
        Walks the tree to find the longest cached prefix.
        Returns: (last_matching_node, number_of_matched_tokens)
        """
        node = self.root
        matched_len = 0
        curr_tokens = tokens
        
        while curr_tokens:
            first_tok = curr_tokens[0]
            if first_tok in node.children:
                child = node.children[first_tok]
                
                # Check if the edge fully matches
                if self._match_key(curr_tokens, child.key):
                    node = child
                    match_size = len(child.key)
                    matched_len += match_size
                    curr_tokens = curr_tokens[match_size:]
                    
                    # Update LRU
                    node.last_access = time.time()
                else:
                    # Partial match not implemented for simplicity
                    break
            else:
                break
        
        if matched_len > 0:
            self.total_tokens_saved += matched_len
            
        return node, matched_len

    def insert(self, tokens: List[int], last_node: RadixNode):
        """
        Inserts new tokens into the tree starting from last_node.
        """
        if not tokens:
            return

        # Allocate new memory block
        block_id = self.allocator.allocate()
        
        # Create new node
        new_node = RadixNode(key_tokens=tokens, parent=last_node)
        new_node.value = block_id
        
        # Link to parent
        last_node.children[tokens[0]] = new_node
        
        logger.debug("cache_insert", tokens_added=len(tokens), block_id=block_id)

    def _match_key(self, query, key):
        if len(query) < len(key): return False
        return query[:len(key)] == key