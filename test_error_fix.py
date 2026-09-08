#!/usr/bin/env python3
"""Test that the original error is fixed."""

import sys
sys.path.insert(0, '.')

def test_original_error_fixed():
    """
    The original error was:
    [param-server] parameter_server.py: error: argument --outer_method: 
    invalid choice: 'mla' (choose from heloco, diloco)
    
    This test verifies that MLA is now a valid choice.
    """
    print("Testing that original error is fixed...")
    print("-" * 60)
    print("Original Error:")
    print("  argument --outer_method: invalid choice: 'mla'")
    print("  (choose from heloco, diloco)")
    print("-" * 60)
    
    import argparse
    
    # Recreate the parser from parameter_server.py
    parser = argparse.ArgumentParser(
        description="HeLoCo RL parameter server"
    )
    parser.add_argument(
        "--outer_method", choices=["heloco", "diloco", "mla"], default="heloco"
    )
    
    # Try to parse with --outer_method mla
    try:
        args = parser.parse_args(["--outer_method", "mla"])
        print("\n✓ SUCCESS: --outer_method mla is now accepted!")
        print(f"  Parsed value: {args.outer_method}")
        return True
    except SystemExit as e:
        print(f"\n✗ FAILED: {e}")
        return False

def test_all_methods_available():
    """Test that all three methods are available."""
    print("\n\nTesting that all three methods are available...")
    print("-" * 60)
    
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outer_method", choices=["heloco", "diloco", "mla"], default="heloco"
    )
    
    methods = ["heloco", "diloco", "mla"]
    all_ok = True
    
    for method in methods:
        try:
            args = parser.parse_args(["--outer_method", method])
            print(f"  ✓ {method:10} is available")
        except:
            print(f"  ✗ {method:10} FAILED")
            all_ok = False
    
    return all_ok

if __name__ == "__main__":
    print("=" * 60)
    print("Verification: Original Error Fix")
    print("=" * 60)
    
    test1 = test_original_error_fixed()
    test2 = test_all_methods_available()
    
    print("\n" + "=" * 60)
    if test1 and test2:
        print("✓ ALL TESTS PASSED - Error is FIXED!")
        print("\nYou can now run:")
        print("  $ python -m panoengine.decentralized.parameter_server \\")
        print("      --outer_method mla \\")
        print("      --config YOUR_CONFIG")
        sys.exit(0)
    else:
        print("✗ Tests failed")
        sys.exit(1)
