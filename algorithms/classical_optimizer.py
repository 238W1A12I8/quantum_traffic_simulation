import numpy as np

class ClassicalOptimizer:
    def __init__(self):
        self.method_name = "Classical Optimization"
    
    def get_green_times(self, traffic_data=None):
        """Classical is GOOD but not as good as quantum"""
        return [19, 14, 19, 8]  # Less optimal than quantum
    
    def adaptive_classical(self, traffic_volume):
        """Classical adaptation is less sophisticated"""
        base_times = [19, 14, 19, 8]
        
        # Classical reacts slower to traffic changes
        if traffic_volume > 1000:
            return [22, 16, 20, 10]  # Not as optimized as quantum
        elif traffic_volume < 500:
            return [16, 12, 16, 6]   # Less efficient than quantum
        else:
            return base_times