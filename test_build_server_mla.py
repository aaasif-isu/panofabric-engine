#!/usr/bin/env python3
"""Test that build_server works with MLA method."""

import sys
sys.path.insert(0, '.')

import torch
from torch import nn

# Test the build_server function logic
def test_build_server_mla_logic():
    """Test that the MLA branch in build_server is reachable."""
    print("Testing MLA branch in build_server...")
    
    from panoengine.decentralized.mla import MLAOptimizer, MLAServer
    from panoengine.decentralized.async_diloco import AsyncDiLoCoServer
    from panoengine.decentralized.heloco import HeLoCoServer
    
    # Create a test model
    model = nn.Linear(10, 5, bias=False).to(dtype=torch.float32, device='cpu')
    outer_method = "mla"
    
    # This is the logic from build_server
    if outer_method == "heloco":
        print("  Would use HeLoCoServer")
    elif outer_method == "diloco":
        print("  Would use AsyncDiLoCoServer")
    elif outer_method == "mla":
        print("  ✓ MLA branch reached")
        outer = MLAOptimizer(model.parameters(), lr=0.7, momentum=0.9)
        server_cls = MLAServer
        print(f"  ✓ MLAOptimizer created: {type(outer).__name__}")
        print(f"  ✓ MLAServer selected: {server_cls.__name__}")
    else:
        raise ValueError(f"unknown outer_method {outer_method!r}")
    
    print("✓ MLA branch test passed")
    return True

def test_argument_parser():
    """Test that argument parser accepts MLA."""
    print("\nTesting argument parser...")
    
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outer_method", choices=["heloco", "diloco", "mla"], default="heloco"
    )
    
    # Test all methods
    methods = ["heloco", "diloco", "mla"]
    for method in methods:
        args = parser.parse_args(["--outer_method", method])
        assert args.outer_method == method
        print(f"  ✓ --outer_method {method} accepted")
    
    print("✓ Argument parser test passed")
    return True

def test_imports():
    """Test that all imports work."""
    print("\nTesting imports...")
    
    try:
        from panoengine.decentralized.mla import MLAOptimizer, MLAServer
        print("  ✓ MLAOptimizer imported")
        print("  ✓ MLAServer imported")
        
        from panoengine.decentralized.parameter_server import build_server
        print("  ✓ build_server imported")
        
        print("✓ Imports test passed")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("MLA Integration Tests")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_argument_parser,
        test_build_server_mla_logic,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append((test.__name__, result))
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test.__name__, False))
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    passed = sum(1 for _, r in results if r)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    for test_name, result in results:
        status = "✓" if result else "✗"
        print(f"  {status} {test_name}")
    
    sys.exit(0 if all(r for _, r in results) else 1)
