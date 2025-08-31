#!/usr/bin/env python3
"""
Comprehensive test runner script for the AI Career Recommender.
"""
import argparse
import subprocess
import sys
import os
import time
from pathlib import Path
from typing import List, Dict, Optional


class TestRunner:
    """Test runner with comprehensive options and reporting."""
    
    def __init__(self):
        self.backend_dir = Path(__file__).parent.parent
        self.ml_dir = self.backend_dir.parent / "machinelearningmodel"
        self.frontend_dir = self.backend_dir.parent / "frontend"
        self.results = {}
    
    def run_command(self, command: List[str], cwd: Optional[Path] = None, timeout: int = 300) -> Dict:
        """Run a command and return results."""
        start_time = time.time()
        cwd = cwd or self.backend_dir
        
        print(f"Running: {' '.join(command)} in {cwd}")
        
        try:
            result = subprocess.run(
                command,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            duration = time.time() - start_time
            
            return {
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "duration": duration
            }
        
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "returncode": -1,
                "stdout": "",
                "stderr": f"Command timed out after {timeout} seconds",
                "duration": timeout
            }
        
        except Exception as e:
            return {
                "success": False,
                "returncode": -1,
                "stdout": "",
                "stderr": str(e),
                "duration": time.time() - start_time
            }
    
    def run_unit_tests(self, coverage: bool = True, verbose: bool = True) -> bool:
        """Run unit tests for backend services."""
        print("\n[UNIT] Running Backend Unit Tests...")
        
        command = ["python", "-m", "pytest"]
        command.extend([
            "tests/test_simple_unit.py",
            "-m", "unit"
        ])
        
        if verbose:
            command.append("-v")
        
        if coverage:
            command.extend([
                "--cov=app",
                "--cov-report=html:htmlcov",
                "--cov-report=term-missing",
                "--cov-fail-under=80"
            ])
        
        result = self.run_command(command)
        self.results["unit_tests"] = result
        
        if result["success"]:
            print("[PASS] Backend unit tests passed!")
        else:
            print("[FAIL] Backend unit tests failed!")
            print(result["stderr"])
        
        return result["success"]
    
    def run_ml_tests(self, include_slow: bool = False, coverage: bool = True) -> bool:
        """Run ML algorithm tests."""
        print("\n[ML] Running ML Algorithm Tests...")
        
        command = ["python", "-m", "pytest"]
        command.extend([
            "tests/test_ml_algorithms_comprehensive.py",
            "-m", "ml"
        ])
        
        if not include_slow:
            command.extend(["-m", "not slow"])
        
        if coverage:
            command.extend([
                "--cov=.",
                "--cov-report=html:htmlcov",
                "--cov-report=term-missing"
            ])
        
        command.append("-v")
        
        result = self.run_command(command, cwd=self.ml_dir)
        self.results["ml_tests"] = result
        
        if result["success"]:
            print("[PASS] ML algorithm tests passed!")
        else:
            print("[FAIL] ML algorithm tests failed!")
            print(result["stderr"])
        
        return result["success"]
    
    def run_integration_tests(self, setup_db: bool = True) -> bool:
        """Run integration tests."""
        print("\n[INTEGRATION] Running Integration Tests...")
        
        if setup_db:
            print("Setting up test database...")
            db_result = self.run_command([
                "python", "-c", 
                "from app.core.database import engine; from app.models import Base; Base.metadata.create_all(bind=engine)"
            ])
            
            if not db_result["success"]:
                print("[FAIL] Failed to set up test database!")
                return False
        
        command = ["python", "-m", "pytest"]
        command.extend([
            "tests/test_integration_comprehensive.py",
            "-m", "integration",
            "-v",
            "--tb=short"
        ])
        
        result = self.run_command(command)
        self.results["integration_tests"] = result
        
        if result["success"]:
            print("[PASS] Integration tests passed!")
        else:
            print("[FAIL] Integration tests failed!")
            print(result["stderr"])
        
        return result["success"]
    
    def run_performance_tests(self, quick: bool = False) -> bool:
        """Run performance tests."""
        print("\n[PERFORMANCE] Running Performance Tests...")
        
        command = ["python", "-m", "pytest"]
        command.extend([
            "tests/test_performance_comprehensive.py",
            "-m", "performance",
            "-v",
            "--tb=short"
        ])
        
        if quick:
            command.extend(["-m", "not slow"])
        
        # Longer timeout for performance tests
        result = self.run_command(command, timeout=600)
        self.results["performance_tests"] = result
        
        if result["success"]:
            print("[PASS] Performance tests passed!")
        else:
            print("[FAIL] Performance tests failed!")
            print(result["stderr"])
        
        return result["success"]
    
    def run_e2e_tests(self, start_server: bool = True) -> bool:
        """Run end-to-end tests."""
        print("\n[E2E] Running End-to-End Tests...")
        
        server_process = None
        
        if start_server:
            print("Starting test server...")
            server_process = subprocess.Popen([
                "python", "-m", "uvicorn", "app.main:app",
                "--host", "0.0.0.0", "--port", "8000"
            ], cwd=self.backend_dir)
            
            # Wait for server to start
            time.sleep(5)
        
        try:
            command = ["python", "-m", "pytest"]
            command.extend([
                "tests/test_e2e_workflows.py",
                "-m", "e2e",
                "-v",
                "--tb=short"
            ])
            
            result = self.run_command(command, timeout=900)  # 15 minutes timeout
            self.results["e2e_tests"] = result
            
            if result["success"]:
                print("[PASS] End-to-end tests passed!")
            else:
                print("[FAIL] End-to-end tests failed!")
                print(result["stderr"])
            
            return result["success"]
        
        finally:
            if server_process:
                print("Stopping test server...")
                server_process.terminate()
                server_process.wait()
    
    def run_security_tests(self) -> bool:
        """Run security tests."""
        print("\n[SECURITY] Running Security Tests...")
        
        # Install security tools if not present
        security_tools = ["bandit", "safety"]
        for tool in security_tools:
            check_result = self.run_command(["which", tool])
            if not check_result["success"]:
                print(f"Installing {tool}...")
                install_result = self.run_command(["pip", "install", tool])
                if not install_result["success"]:
                    print(f"[FAIL] Failed to install {tool}")
                    return False
        
        # Run Bandit security scan
        print("Running Bandit security scan...")
        bandit_result = self.run_command([
            "bandit", "-r", "app/", "-f", "json", "-o", "bandit-report.json"
        ])
        
        # Run Safety check
        print("Running Safety dependency check...")
        safety_result = self.run_command([
            "safety", "check", "--json", "--output", "safety-report.json"
        ])
        
        # Run pytest security tests
        pytest_result = self.run_command([
            "python", "-m", "pytest",
            "tests/test_security_privacy.py",
            "-v"
        ])
        
        security_success = (
            bandit_result["returncode"] in [0, 1] and  # Bandit returns 1 for issues found
            safety_result["returncode"] in [0, 1] and  # Safety returns 1 for vulnerabilities
            pytest_result["success"]
        )
        
        self.results["security_tests"] = {
            "success": security_success,
            "bandit": bandit_result,
            "safety": safety_result,
            "pytest": pytest_result
        }
        
        if security_success:
            print("[PASS] Security tests completed!")
        else:
            print("[FAIL] Security tests found issues!")
        
        return security_success
    
    def run_all_tests(self, quick: bool = False, skip_slow: bool = False) -> bool:
        """Run all test suites."""
        print("[ALL] Running Comprehensive Test Suite...")
        print("=" * 60)
        
        all_passed = True
        
        # Unit tests
        if not self.run_unit_tests():
            all_passed = False
        
        # ML tests
        if not self.run_ml_tests(include_slow=not skip_slow):
            all_passed = False
        
        # Integration tests
        if not self.run_integration_tests():
            all_passed = False
        
        # Performance tests (skip if quick mode)
        if not quick:
            if not self.run_performance_tests(quick=quick):
                all_passed = False
        
        # E2E tests (skip if quick mode)
        if not quick:
            if not self.run_e2e_tests():
                all_passed = False
        
        # Security tests
        if not self.run_security_tests():
            all_passed = False
        
        return all_passed
    
    def generate_report(self) -> str:
        """Generate a comprehensive test report."""
        report = []
        report.append("# Test Results Report")
        report.append(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        total_duration = 0
        passed_count = 0
        total_count = 0
        
        for test_name, result in self.results.items():
            total_count += 1
            
            if isinstance(result, dict) and "success" in result:
                status = "[PASS]" if result["success"] else "[FAIL]"
                duration = result.get("duration", 0)
                total_duration += duration
                
                if result["success"]:
                    passed_count += 1
                
                report.append(f"## {test_name.replace('_', ' ').title()}")
                report.append(f"Status: {status}")
                report.append(f"Duration: {duration:.2f}s")
                
                if not result["success"] and result.get("stderr"):
                    report.append("### Error Output:")
                    report.append(f"```\n{result['stderr']}\n```")
                
                report.append("")
        
        # Summary
        report.insert(2, f"**Summary: {passed_count}/{total_count} test suites passed**")
        report.insert(3, f"**Total Duration: {total_duration:.2f}s**")
        report.insert(4, "")
        
        return "\n".join(report)
    
    def save_report(self, filename: str = "test_report.md"):
        """Save test report to file."""
        report = self.generate_report()
        
        with open(self.backend_dir / filename, "w", encoding="utf-8") as f:
            f.write(report)
        
        print(f"[REPORT] Test report saved to {filename}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Comprehensive test runner for AI Career Recommender")
    
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--ml", action="store_true", help="Run ML algorithm tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--performance", action="store_true", help="Run performance tests only")
    parser.add_argument("--e2e", action="store_true", help="Run end-to-end tests only")
    parser.add_argument("--security", action="store_true", help="Run security tests only")
    parser.add_argument("--all", action="store_true", help="Run all test suites")
    
    parser.add_argument("--quick", action="store_true", help="Run in quick mode (skip slow tests)")
    parser.add_argument("--no-coverage", action="store_true", help="Skip coverage reporting")
    parser.add_argument("--no-server", action="store_true", help="Don't start test server for E2E tests")
    parser.add_argument("--report", type=str, default="test_report.md", help="Test report filename")
    
    args = parser.parse_args()
    
    runner = TestRunner()
    success = True
    
    # Determine which tests to run
    if args.unit:
        success = runner.run_unit_tests(coverage=not args.no_coverage)
    elif args.ml:
        success = runner.run_ml_tests(include_slow=not args.quick)
    elif args.integration:
        success = runner.run_integration_tests()
    elif args.performance:
        success = runner.run_performance_tests(quick=args.quick)
    elif args.e2e:
        success = runner.run_e2e_tests(start_server=not args.no_server)
    elif args.security:
        success = runner.run_security_tests()
    elif args.all or len(sys.argv) == 1:  # Default to all tests
        success = runner.run_all_tests(quick=args.quick, skip_slow=args.quick)
    
    # Generate and save report
    runner.save_report(args.report)
    
    # Print final summary
    print("\n" + "=" * 60)
    if success:
        print("[SUCCESS] All tests passed successfully!")
        sys.exit(0)
    else:
        print("[ERROR] Some tests failed. Check the report for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()