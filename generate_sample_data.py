import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_sample_traffic_data():
    """Generate sample traffic data for testing"""
    dates = pd.date_range(start='2024-01-01', end='2024-06-01', freq='H')
    
    data = []
    for date in dates:
        hour = date.hour
        day_of_week = date.weekday()
        is_weekend = 1 if day_of_week >= 5 else 0
        
        # Base pattern with noise
        base_volume = 1000
        hour_effect = 500 * np.sin(2 * np.pi * (hour - 6) / 24)  # Peak at 6 AM/PM
        day_effect = 200 if is_weekend else 300  # Higher on weekdays
        noise = np.random.normal(0, 100)
        
        vehicle_count = max(100, base_volume + hour_effect + day_effect + noise)
        
        data.append({
            'timestamp': date,
            'vehicle_count': int(vehicle_count),
            'hour': hour,
            'day_of_week': day_of_week,
            'is_weekend': is_weekend,
            'month': date.month,
            'avg_speed': np.random.uniform(20, 60),
            'congestion_level': np.random.choice(['Low', 'Medium', 'High'], p=[0.6, 0.3, 0.1])
        })
    
    df = pd.DataFrame(data)
    df.to_csv('sample_traffic_data.csv', index=False)
    print("✅ Sample data generated: sample_traffic_data.csv")
    return df

if __name__ == "__main__":
    generate_sample_traffic_data()