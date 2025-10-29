"""
Utility script to run all tests and show summary
"""

import subprocess
import sys
import os

def run_tests():
    """Run all supervised learning tests and show results"""
    
    test_files = [
        "test/supervised_learning/test_knn.py",
        "test/supervised_learning/test_logistic_regression.py", 
        "test/supervised_learning/test_regression.py",
        "test/supervised_learning/test_naive_bayes.py"
    ]
    
    print("=" * 80)
    print("ML ALGORITHM PRACTICE FRAMEWORK - TEST RUNNER")
    print("=" * 80)
    print()
    
    results = {}
    
    for test_file in test_files:
        algorithm = os.path.basename(test_file).replace("test_", "").replace(".py", "")
        print(f"Testing {algorithm.upper()}...")
        
        try:
            # Run pytest for this test file
            result = subprocess.run([
                sys.executable, "-m", "pytest", test_file, "-v", "--tb=short"
            ], capture_output=True, text=True, cwd=os.getcwd())
            
            if result.returncode == 0:
                results[algorithm] = "✅ PASSED"
                print(f"  ✅ {algorithm}: All tests passed")
            else:
                results[algorithm] = "❌ FAILED"
                print(f"  ❌ {algorithm}: Some tests failed")
                print(f"     Error: {result.stdout}")
                
        except Exception as e:
            results[algorithm] = f"❌ ERROR: {str(e)}"
            print(f"  ❌ {algorithm}: Error running tests - {str(e)}")
        
        print()
    
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    passed = 0
    total = len(results)
    
    for algorithm, status in results.items():
        print(f"{algorithm:20} : {status}")
        if "PASSED" in status:
            passed += 1
    
    print()
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All algorithms implemented successfully!")
    else:
        print("📝 Keep working on the remaining algorithms.")
        print("💡 Check PRACTICE_README.md for implementation tips.")


if __name__ == "__main__":
    run_tests()