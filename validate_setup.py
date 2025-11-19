"""
NHL Prediction Model - Setup Validation Script

This script checks if your environment is correctly set up
to run the NHL prediction model.
"""

import sys
import os

print("=" * 60)
print("NHL Prediction Model - Environment Validation")
print("=" * 60)

# Check Python version
print("\n[1/5] Checking Python version...")
py_version = sys.version_info
print(f"  Python version: {py_version.major}.{py_version.minor}.{py_version.micro}")

if py_version.major == 3 and py_version.minor >= 9:
    print("  ✓ Python version is compatible (3.9+)")
else:
    print("  ✗ WARNING: Python 3.9+ recommended")
    print(f"    Current version: {py_version.major}.{py_version.minor}")

# Check required packages
print("\n[2/5] Checking required packages...")

required_packages = {
    'pandas': '2.1.0',
    'numpy': '1.24.3',
    'sklearn': '1.3.0',
    'xgboost': '2.0.0',
    'lightgbm': '4.0.0',
    'matplotlib': '3.7.2',
    'seaborn': '0.12.2'
}

missing_packages = []
version_mismatches = []

for package, expected_version in required_packages.items():
    try:
        if package == 'sklearn':
            import sklearn
            installed_version = sklearn.__version__
            package_name = 'scikit-learn'
        elif package == 'pandas':
            import pandas
            installed_version = pandas.__version__
            package_name = package
        elif package == 'numpy':
            import numpy
            installed_version = numpy.__version__
            package_name = package
        elif package == 'xgboost':
            import xgboost
            installed_version = xgboost.__version__
            package_name = package
        elif package == 'lightgbm':
            import lightgbm
            installed_version = lightgbm.__version__
            package_name = package
        elif package == 'matplotlib':
            import matplotlib
            installed_version = matplotlib.__version__
            package_name = package
        elif package == 'seaborn':
            import seaborn
            installed_version = seaborn.__version__
            package_name = package

        # Check version (allow minor version differences)
        expected_major_minor = '.'.join(expected_version.split('.')[:2])
        installed_major_minor = '.'.join(installed_version.split('.')[:2])

        if expected_major_minor == installed_major_minor:
            print(f"  ✓ {package_name}: {installed_version}")
        else:
            print(f"  ⚠ {package_name}: {installed_version} (expected {expected_version})")
            version_mismatches.append((package_name, installed_version, expected_version))

    except ImportError:
        print(f"  ✗ {package}: NOT INSTALLED")
        missing_packages.append(package)

if missing_packages:
    print(f"\n  ✗ Missing packages: {', '.join(missing_packages)}")
    print("    Run: pip install -r requirements.txt")
else:
    print("\n  ✓ All required packages installed")

if version_mismatches:
    print(f"  ⚠ Version mismatches detected (may cause reproducibility issues)")

# Check input data files
print("\n[3/5] Checking input data files...")

data_files = ['gamedata.csv', 'playergamedata.csv']
data_files_exist = True

for file in data_files:
    if os.path.exists(file):
        file_size = os.path.getsize(file) / (1024 * 1024)  # MB
        print(f"  ✓ {file} ({file_size:.1f} MB)")
    else:
        print(f"  ✗ {file} NOT FOUND")
        data_files_exist = False

if not data_files_exist:
    print("\n  ✗ Missing data files")
    print("    Please place gamedata.csv and playergamedata.csv in this directory")

# Check if prepared data exists
print("\n[4/5] Checking prepared data...")

if os.path.exists('se_assignment1_1_data.csv'):
    file_size = os.path.getsize('se_assignment1_1_data.csv') / (1024 * 1024)
    print(f"  ✓ se_assignment1_1_data.csv exists ({file_size:.1f} MB)")
    print("    You can skip prepare_data.py and run the main script directly")
else:
    print(f"  ⚠ se_assignment1_1_data.csv NOT FOUND")
    if data_files_exist:
        print("    Run: python prepare_data.py")
    else:
        print("    First obtain gamedata.csv and playergamedata.csv")

# Check scripts
print("\n[5/5] Checking script files...")

scripts = {
    'prepare_data.py': 'Data preparation script',
    'se_assignment1_1_code.py': 'Main prediction model script',
    'requirements.txt': 'Package requirements'
}

all_scripts_exist = True
for script, description in scripts.items():
    if os.path.exists(script):
        print(f"  ✓ {script} ({description})")
    else:
        print(f"  ✗ {script} NOT FOUND")
        all_scripts_exist = False

# Summary
print("\n" + "=" * 60)
print("VALIDATION SUMMARY")
print("=" * 60)

issues = []
if py_version.major < 3 or (py_version.major == 3 and py_version.minor < 9):
    issues.append("Python version < 3.9")
if missing_packages:
    issues.append(f"{len(missing_packages)} missing packages")
if not data_files_exist:
    issues.append("Input data files missing")
if not all_scripts_exist:
    issues.append("Script files missing")

if not issues:
    print("✓ Environment is properly configured!")
    print("\nNext steps:")
    if not os.path.exists('se_assignment1_1_data.csv'):
        print("  1. Run: python prepare_data.py")
        print("  2. Run: python se_assignment1_1_code.py")
    else:
        print("  1. Run: python se_assignment1_1_code.py")
else:
    print("✗ Issues detected:")
    for issue in issues:
        print(f"  - {issue}")

    print("\nRecommended actions:")
    if missing_packages:
        print("  1. pip install -r requirements.txt")
    if not data_files_exist:
        print("  2. Obtain gamedata.csv and playergamedata.csv")
    if not all_scripts_exist:
        print("  3. Ensure all script files are present")

print("=" * 60)
