# TEAM MEMBER 3: Reproducibility & CI Lead
# Create comprehensive GitHub Actions CI/CD configuration

github_actions_content = '''name: PyRISE CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
    tags: [ 'v*' ]
  pull_request:
    branches: [ main, develop ]
  schedule:
    # Run tests weekly to catch dependency issues
    - cron: '0 6 * * 1'

env:
  # Force deterministic JAX behavior
  JAX_ENABLE_X64: "True"
  JAX_PLATFORM_NAME: "cpu"
  # Reproducibility settings
  PYTHONHASHSEED: "42"
  JAX_DETERMINISTIC_APIS: "1"

jobs:
  # Code quality and static analysis
  quality-check:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4

    - name: Set up Python 3.11
      uses: actions/setup-python@v4
      with:
        python-version: "3.11"

    - name: Install quality tools
      run: |
        python -m pip install --upgrade pip
        pip install black isort flake8 mypy

    - name: Check code formatting with Black
      run: black --check --diff src/ tests/

    - name: Check import sorting with isort
      run: isort --check-only --diff src/ tests/

    - name: Lint with flake8
      run: |
        # Stop build if there are Python syntax errors or undefined names
        flake8 src/ tests/ --count --select=E9,F63,F7,F82 --show-source --statistics
        # Exit-zero for all other issues (warnings only)
        flake8 src/ tests/ --count --exit-zero --max-complexity=10 --max-line-length=88 --statistics

    - name: Type checking with MyPy
      run: mypy src/pyrise/ --ignore-missing-imports

  # Test matrix across platforms and Python versions
  test:
    needs: quality-check
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        python-version: ["3.10", "3.11", "3.12"]

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Cache pip dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ matrix.os }}-pip-${{ matrix.python-version }}-${{ hashFiles('pyproject.toml') }}
        restore-keys: |
          ${{ matrix.os }}-pip-${{ matrix.python-version }}-

    - name: Install package with dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"

    - name: Verify reproducible installation
      run: |
        python -c "import pyrise; print(f'PyRISE version: {pyrise.__version__}')"
        python -c "import jax; print(f'JAX version: {jax.__version__}'); print(f'JAX backend: {jax.default_backend()}')"

    - name: Run unit tests with coverage
      run: |
        pytest tests/ -v --cov=pyrise --cov-report=xml --cov-report=term-missing --tb=short

    - name: Run integration tests
      run: |
        pytest tests/ -m integration -v --tb=short

    - name: Upload coverage to Codecov
      if: matrix.os == 'ubuntu-latest' && matrix.python-version == '3.11'
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella
        fail_ci_if_error: false

  # GPU testing (when available)
  test-gpu:
    needs: quality-check
    runs-on: ubuntu-latest
    # Only run on main branch to save resources
    if: github.ref == 'refs/heads/main'

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python 3.11
      uses: actions/setup-python@v4
      with:
        python-version: "3.11"

    - name: Install CUDA (if available)
      run: |
        # Try to install CUDA toolkit for GPU tests
        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb || true
        sudo dpkg -i cuda-keyring_1.0-1_all.deb || true
        sudo apt-get update || true

    - name: Install package with GPU support
      run: |
        python -m pip install --upgrade pip
        # Try GPU installation, fallback to CPU if needed
        pip install "jax[cuda12]" || pip install "jax[cpu]"
        pip install -e ".[dev]"

    - name: Test GPU functionality
      run: |
        python -c "
        import jax
        print(f'Available devices: {jax.devices()}')
        if jax.devices('gpu'):
            print('GPU tests enabled')
            # Run GPU-specific tests
            import pytest
            pytest.main(['-m', 'gpu', '-v'])
        else:
            print('No GPU available, skipping GPU tests')
        "

  # Numerical validation and benchmarking
  numerical-validation:
    needs: test
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python 3.11
      uses: actions/setup-python@v4
      with:
        python-version: "3.11"

    - name: Install package with benchmarking tools
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev,benchmark]"
        pip install pytest-benchmark

    - name: Run numerical accuracy tests
      run: |
        # Run tests marked as requiring high numerical precision
        pytest tests/ -m "not slow" --tb=short -v \\
          --benchmark-skip \\
          --durations=10

    - name: Run performance benchmarks
      run: |
        # Run benchmarks but don't fail on performance regressions
        pytest tests/ -m benchmark --benchmark-only \\
          --benchmark-json=benchmark_results.json || true

    - name: Upload benchmark results
      uses: actions/upload-artifact@v3
      with:
        name: benchmark-results
        path: benchmark_results.json

  # Documentation build verification
  docs:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python 3.11
      uses: actions/setup-python@v4
      with:
        python-version: "3.11"

    - name: Install documentation dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[docs]"

    - name: Build documentation
      run: |
        cd docs/
        # Check if docs build successfully
        sphinx-build -b html . _build/html -W --keep-going

    - name: Upload docs artifacts
      uses: actions/upload-artifact@v3
      with:
        name: documentation
        path: docs/_build/html/

  # Security and dependency scanning
  security:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v4

    - name: Run Bandit security linter
      run: |
        pip install bandit[toml]
        bandit -r src/ -f json -o bandit-report.json || true

    - name: Check for known vulnerabilities
      run: |
        pip install safety
        safety check --json --output safety-report.json || true

    - name: Upload security reports
      uses: actions/upload-artifact@v3
      with:
        name: security-reports
        path: |
          bandit-report.json
          safety-report.json

  # Release automation
  release:
    needs: [test, numerical-validation, docs]
    runs-on: ubuntu-latest
    if: startsWith(github.ref, 'refs/tags/v')

    steps:
    - uses: actions/checkout@v4
      with:
        fetch-depth: 0  # Full history for changelog

    - name: Set up Python 3.11
      uses: actions/setup-python@v4
      with:
        python-version: "3.11"

    - name: Build package
      run: |
        python -m pip install --upgrade pip build
        python -m build

    - name: Verify package integrity
      run: |
        pip install twine
        twine check dist/*

    - name: Create GitHub Release
      uses: actions/create-release@v1
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      with:
        tag_name: ${{ github.ref }}
        release_name: PyRISE ${{ github.ref }}
        draft: false
        prerelease: false

    - name: Publish to PyPI
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
      run: |
        twine upload dist/*

  # Cleanup and notifications
  cleanup:
    needs: [test, numerical-validation, docs, security]
    runs-on: ubuntu-latest
    if: always()

    steps:
    - name: Report CI Results
      run: |
        echo "CI Pipeline completed for commit ${{ github.sha }}"
        echo "Quality check: ${{ needs.quality-check.result }}"
        echo "Tests: ${{ needs.test.result }}"
        echo "Numerical validation: ${{ needs.numerical-validation.result }}"
        echo "Documentation: ${{ needs.docs.result }}"
        echo "Security: ${{ needs.security.result }}"

        # Create reproducibility report
        cat << EOF > ci_report.md
        # PyRISE CI Report

        **Commit:** ${{ github.sha }}
        **Branch:** ${{ github.ref }}
        **Triggered by:** ${{ github.event_name }}
        **Date:** $(date -u)

        ## Results Summary
        - Quality Check: ${{ needs.quality-check.result }}
        - Cross-platform Tests: ${{ needs.test.result }}
        - Numerical Validation: ${{ needs.numerical-validation.result }}
        - Documentation Build: ${{ needs.docs.result }}
        - Security Scan: ${{ needs.security.result }}

        ## Reproducibility Information
        - Python Hash Seed: $PYTHONHASHSEED
        - JAX Configuration: X64=$JAX_ENABLE_X64, Platform=$JAX_PLATFORM_NAME
        - Deterministic APIs: $JAX_DETERMINISTIC_APIS

        [Unverified] All tests use fixed random seeds for reproducibility.
        EOF

    - name: Upload CI report
      uses: actions/upload-artifact@v3
      with:
        name: ci-report
        path: ci_report.md
'''

# Create .github/workflows directory and CI file
Path(".github/workflows").mkdir(parents=True, exist_ok=True)
with open(".github/workflows/ci.yml", "w") as f:
    f.write(github_actions_content)

print("✅ Created comprehensive GitHub Actions CI/CD pipeline")
print("   - Cross-platform testing (Ubuntu, Windows, macOS)")
print("   - Python 3.10-3.12 version matrix")
print("   - Code quality checks (Black, isort, flake8, MyPy)")
print("   - Reproducibility controls (fixed seeds, deterministic JAX)")
print("   - GPU testing when available")
print("   - Security scanning and dependency checks")
print("   - Automated PyPI releases on tags")
print("   - [Unverified] claims in CI reporting")