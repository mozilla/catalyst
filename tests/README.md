# Parser Tests

This directory contains comprehensive tests for the parser functionality in `lib/parser.py`.

## Test Structure

```
tests/
├── __init__.py              # Test package initialization
├── test_parser.py          # Main parser test suite
├── fixtures/               # Test data and mock files
│   ├── test_config.yaml    # Sample experiment config
│   ├── test_config_with_branches.yaml  # Sample rollout config
│   ├── invalid_config.yaml # Config with old format (should fail)
│   ├── mock_probe_index.json   # Mock probe index data
│   └── mock_nimbus_response.json # Mock Nimbus API response
└── README.md               # This file
```

## Running Tests

### Quick Start

1. **Install Just:**
   ```bash
   # macOS
   brew install just
   
   # Windows  
   choco install just
   
   # Rust/Cargo
   cargo install just
   ```

2. **Install dependencies:**
   ```bash
   just install
   ```

3. **Run tests:**
   ```bash
   just test           # Complete test suite (tests + coverage)
   ```

4. **Fix formatting:**
   ```bash
   just lint-fix       # Auto-fix code formatting
   ```

### Legacy Methods
```bash
python tests/test_parser.py     # Direct unittest
./run_tests.sh.legacy          # Original bash script
```

### GitHub Actions (CI/CD)

Tests run automatically on:
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop` branches

The CI pipeline includes:
- **Unit tests** on Python 3.8, 3.9, 3.10, and 3.11
- **Code linting** with flake8
- **Type checking** with mypy
- **Coverage reporting** with pytest-cov
- **Integration tests** with real config files

## Test Coverage

The test suite covers:

### Core Functions
- `checkForLocalFile()` - File loading with error handling
- `parseConfigFile()` - YAML config parsing and validation
- `loadProbeIndex()` - Probe index loading
- `annotateMetrics()` - Complete metric annotation workflow

### Pageload Event Metrics
- `annotatePageloadEventMetrics()` - Schema validation and annotation
- New max parameter format validation
- Error handling for old array format
- Default value handling
- Unknown metric detection

### Histogram Processing
- `annotateHistograms()` - Glean and legacy histogram processing
- Schema lookup and validation
- Platform availability detection

### Nimbus API Integration  
- `retrieveNimbusAPI()` - HTTP requests with caching
- `extractValuesFromAPI()` - API response parsing
- `parseNimbusAPI()` - Complete API workflow
- Error handling (timeouts, HTTP errors, JSON parsing)
- Channel priority selection
- Branch handling

### Edge Cases
- Missing files
- Invalid JSON/YAML
- Network failures
- Malformed API responses
- Unsupported channels
- Missing required fields

## Test Data

### Mock Probe Index
Contains sample Glean and legacy probe definitions for testing histogram annotation.

### Mock Nimbus Response
Contains sample experiment data for testing API integration.

### Sample Configs
- **Experiment config**: Regular A/B test configuration
- **Rollout config**: Configuration with branches (rollout)
- **Invalid config**: Uses old array format for pageload metrics

## Adding New Tests

When adding new functionality to the parser:

1. Add test fixtures to `tests/fixtures/` if needed
2. Add test methods to `TestParser` class in `test_parser.py`
3. Follow naming convention: `test_function_name_condition()`
4. Include docstrings explaining what the test validates
5. Test both success and failure cases
6. Mock external dependencies (HTTP requests, file I/O)

## Dependencies

### Runtime Dependencies
- requests
- PyYAML  
- beautifulsoup4
- numpy

### Development Dependencies
- pytest & pytest-cov (testing)
- flake8 (linting)
- mypy (type checking)
- black (code formatting)
- coverage (coverage reporting)

Install with:
```bash
pip install -r requirements.txt           # Runtime deps
pip install -r requirements-dev.txt       # Development deps
```