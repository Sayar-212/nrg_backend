"""
Test script to diagnose import issues
"""

import sys
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print("-" * 50)

# Test core imports
imports_to_test = [
    "numpy",
    "pandas", 
    "requests",
    "pathlib",
    "threading",
    "subprocess"
]

for module in imports_to_test:
    try:
        __import__(module)
        print(f"✓ {module} - OK")
    except Exception as e:
        print(f"✗ {module} - ERROR: {e}")

print("-" * 50)

# Test pandas specifically
try:
    import pandas as pd
    print(f"✓ Pandas version: {pd.__version__}")
    
    # Test basic pandas operation
    df = pd.DataFrame({'test': [1, 2, 3]})
    print(f"✓ Pandas DataFrame creation: OK")
    
except Exception as e:
    print(f"✗ Pandas detailed error: {e}")
    print(f"✗ Error type: {type(e).__name__}")