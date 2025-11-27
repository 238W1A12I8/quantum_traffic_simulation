import subprocess
import sys
import os

def run_dashboard():
    """Run the Streamlit dashboard"""
    print("🚀 Starting Quantum Traffic Optimization Dashboard...")
    print("📊 Dashboard will open in your browser automatically")
    print("⏹️  Press Ctrl+C to stop the dashboard")
    
    try:
        # Run streamlit
        subprocess.run([sys.executable, "-m", "streamlit", "run", "traffic_dashboard.py"])
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")

if __name__ == "__main__":
    run_dashboard()