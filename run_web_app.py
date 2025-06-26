"""
Web Application Launcher
Simple script to launch the Streamlit web interface
"""
import subprocess
import sys
import os
from pathlib import Path

def check_streamlit():
    """Check if Streamlit is installed"""
    try:
        import streamlit
        return True
    except ImportError:
        return False

def install_streamlit():
    """Install Streamlit if not available"""
    print("Installing Streamlit...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
        return True
    except subprocess.CalledProcessError:
        return False

def launch_app():
    """Launch the Streamlit application"""
    app_path = Path(__file__).parent / "streamlit_app.py"
    
    if not app_path.exists():
        print(f"❌ App file not found: {app_path}")
        return False
    
    print("🚀 Launching Professional Currency Detection Web Interface...")
    print("=" * 60)
    print("📱 The web interface will open in your default browser")
    print("🔧 Use Ctrl+C to stop the server")
    print("=" * 60)
    
    try:
        # Launch Streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(app_path),
            "--server.port=8501",
            "--server.address=localhost"
        ])
        return True
    except KeyboardInterrupt:
        print("\n✅ Web interface stopped by user")
        return True
    except Exception as e:
        print(f"❌ Failed to launch app: {e}")
        return False

def main():
    """Main launcher function"""
    print("💰 Professional Currency Detection System")
    print("Web Interface Launcher")
    print("=" * 50)
    
    # Check if Streamlit is installed
    if not check_streamlit():
        print("⚠️  Streamlit not found. Installing...")
        if not install_streamlit():
            print("❌ Failed to install Streamlit")
            print("Please install manually: pip install streamlit")
            return
        print("✅ Streamlit installed successfully")
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Launch the app
    launch_app()

if __name__ == "__main__":
    main()