#!/usr/bin/env python3
"""
Helper script untuk test dan setup aplikasi Deteksi Sawit
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check Python version"""
    print("🔍 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor} - Requires Python 3.8+")
        return False

def check_model_file():
    """Check if model file exists"""
    print("\n🔍 Checking model file...")
    model_path = Path("best_yolo_model.pt")
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"✅ Model file found: {model_path} ({size_mb:.2f} MB)")
        return True
    else:
        print(f"❌ Model file not found: {model_path}")
        print("   Please ensure 'best_yolo_model.pt' is in the current directory")
        return False

def check_dependencies():
    """Check if required packages are installed"""
    print("\n🔍 Checking dependencies...")
    required_packages = {
        'streamlit': 'Streamlit',
        'torch': 'PyTorch',
        'ultralytics': 'Ultralytics',
        'cv2': 'OpenCV',
        'PIL': 'Pillow',
        'numpy': 'NumPy',
        'pandas': 'Pandas'
    }
    
    missing = []
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✅ {name} - OK")
        except ImportError:
            print(f"❌ {name} - NOT INSTALLED")
            missing.append(package)
    
    return len(missing) == 0, missing

def install_requirements():
    """Install requirements from requirements.txt"""
    print("\n📦 Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install requirements")
        return False

def run_streamlit_app():
    """Run the Streamlit application"""
    print("\n🚀 Starting Streamlit application...")
    print("   Opening browser to http://localhost:8501")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n👋 Application stopped")

def main():
    """Main function"""
    print("=" * 50)
    print("🌴 Deteksi Sawit - Setup & Launch Helper")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check model file
    if not check_model_file():
        print("\n⚠️  Model file is required to run the application!")
        sys.exit(1)
    
    # Check dependencies
    deps_ok, missing = check_dependencies()
    
    if not deps_ok:
        print("\n⚠️  Some dependencies are missing!")
        install = input("\nWould you like to install missing dependencies? (y/n): ")
        if install.lower() == 'y':
            if not install_requirements():
                sys.exit(1)
        else:
            print("❌ Cannot run application without dependencies")
            sys.exit(1)
    
    print("\n" + "=" * 50)
    print("✅ All checks passed! Ready to run application")
    print("=" * 50)
    
    launch = input("\nWould you like to launch the application now? (y/n): ")
    if launch.lower() == 'y':
        run_streamlit_app()
    else:
        print("\n💡 To launch the application manually, run:")
        print("   streamlit run app.py")

if __name__ == "__main__":
    main()
