#!/usr/bin/env python3
"""
Setup Verification Script for Personal Automation Tools

This script verifies that your environment is properly configured
for running the personal automation tools.
"""

import sys
import os
import importlib
import subprocess
from typing import List, Tuple

def check_python_version() -> bool:
    """Check if Python version is 3.8 or higher."""
    version = sys.version_info
    if version >= (3, 8):
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} (supported)")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} (requires 3.8+)")
        return False

def check_package(package_name: str, import_name: str = None) -> bool:
    """Check if a package is installed and importable."""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✅ {package_name} {version}")
        return True
    except ImportError:
        print(f"❌ {package_name} (not installed)")
        return False

def check_required_packages() -> bool:
    """Check all required packages."""
    print("\n📦 Checking Required Packages:")
    
    packages = [
        ('pandas', 'pandas'),
        ('matplotlib', 'matplotlib'),
        ('mplfinance', 'mplfinance'),
        ('requests', 'requests'),
        ('beautifulsoup4', 'bs4'),
        ('google-generativeai', 'google.generativeai'),
        ('pillow', 'PIL'),
        ('yfinance', 'yfinance'),
        ('numpy', 'numpy'),
        ('scipy', 'scipy'),
        ('youtube-transcript-api', 'youtube_transcript_api'),
        ('feedparser', 'feedparser'),
    ]
    
    all_installed = True
    for package_name, import_name in packages:
        if not check_package(package_name, import_name):
            all_installed = False
    
    return all_installed

def check_environment_variables() -> bool:
    """Check if required environment variables are set."""
    print("\n🔐 Checking Environment Variables:")
    
    required_vars = ['GEMINI_API_KEY']
    optional_vars = ['TELEGRAM_BOT_TOKEN', 'TELEGRAM_CHAT_ID', 'YOUTUBE_CHANNEL_ID']
    
    all_required_set = True
    
    # Check required variables
    for var in required_vars:
        value = os.getenv(var)
        if value:
            masked_value = value[:8] + "..." if len(value) > 8 else "***"
            print(f"✅ {var}: {masked_value}")
        else:
            print(f"❌ {var}: Not set (required)")
            all_required_set = False
    
    # Check optional variables
    for var in optional_vars:
        value = os.getenv(var)
        if value:
            masked_value = value[:8] + "..." if len(value) > 8 else "***"
            print(f"✅ {var}: {masked_value}")
        else:
            print(f"⚠️  {var}: Not set (optional)")
    
    return all_required_set

def test_basic_functionality() -> bool:
    """Test basic functionality of key components."""
    print("\n🧪 Testing Basic Functionality:")
    
    success = True
    
    # Test pandas data manipulation
    try:
        import pandas as pd
        df = pd.DataFrame({'test': [1, 2, 3]})
        assert len(df) == 3
        print("✅ pandas data manipulation")
    except Exception as e:
        print(f"❌ pandas test failed: {e}")
        success = False
    
    # Test matplotlib plotting
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])
        plt.close(fig)
        print("✅ matplotlib plotting")
    except Exception as e:
        print(f"❌ matplotlib test failed: {e}")
        success = False
    
    # Test Google AI (if API key is available)
    api_key = os.getenv('GEMINI_API_KEY')
    if api_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            # Don't actually make a request to avoid using quota
            print("✅ Google AI configuration")
        except Exception as e:
            print(f"❌ Google AI test failed: {e}")
            success = False
    else:
        print("⚠️  Google AI: Skipped (no API key)")
    
    # Test requests
    try:
        import requests
        response = requests.get('https://httpbin.org/get', timeout=5)
        assert response.status_code == 200
        print("✅ HTTP requests")
    except Exception as e:
        print(f"❌ HTTP requests test failed: {e}")
        success = False
    
    return success

def check_file_structure() -> bool:
    """Check if the expected file structure exists."""
    print("\n📁 Checking File Structure:")
    
    expected_files = [
        'financial-tools/stock_analysis_ai.py',
        'financial-tools/requirements.txt',
        'financial-tools/README.md',
        'youtube-automation/youtube-summerizer.py',
        'youtube-automation/requirements.txt',
        'youtube-automation/README.md',
        'docs/API_SETUP.md',
        'docs/TROUBLESHOOTING.md',
        'requirements.txt'
    ]
    
    all_exist = True
    for file_path in expected_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} (missing)")
            all_exist = False
    
    return all_exist

def provide_recommendations(results: dict) -> None:
    """Provide recommendations based on verification results."""
    print("\n💡 Recommendations:")
    
    if not results['python_version']:
        print("- Upgrade Python to version 3.8 or higher")
    
    if not results['packages']:
        print("- Install missing packages: pip install -r requirements.txt")
    
    if not results['environment']:
        print("- Set up required environment variables (see docs/API_SETUP.md)")
    
    if not results['functionality']:
        print("- Check error messages above and review docs/TROUBLESHOOTING.md")
    
    if not results['file_structure']:
        print("- Ensure you're running this script from the project root directory")
    
    if all(results.values()):
        print("🎉 Everything looks good! You're ready to use the automation tools.")
        print("\nNext steps:")
        print("- Try: cd financial-tools && python stock_analysis_ai.py --help")
        print("- Try: cd youtube-automation && python youtube-summerizer.py --help")

def main():
    """Main verification function."""
    print("🔍 Personal Automation Tools - Setup Verification")
    print("=" * 60)
    
    results = {
        'python_version': check_python_version(),
        'packages': check_required_packages(),
        'environment': check_environment_variables(),
        'functionality': test_basic_functionality(),
        'file_structure': check_file_structure()
    }
    
    print("\n📋 Summary:")
    total_checks = len(results)
    passed_checks = sum(results.values())
    
    for category, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"- {category.replace('_', ' ').title()}: {status}")
    
    print(f"\nOverall: {passed_checks}/{total_checks} checks passed")
    
    if passed_checks == total_checks:
        print("🎉 All checks passed! Your setup is ready.")
    else:
        print("⚠️  Some checks failed. See recommendations below.")
        provide_recommendations(results)
    
    return passed_checks == total_checks

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
