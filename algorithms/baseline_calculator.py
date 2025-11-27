class BaselineCalculator:
    def __init__(self):
        self.method_name = "Fixed Timing"
    
    def get_green_times(self, traffic_data=None):
        return [30, 15, 10, 5]