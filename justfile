# Catalyst Performance Analysis

# Show help and usage information
help:
    @echo "🚀 Catalyst Performance Analysis - Available Commands"
    @echo ""
    @echo "📦 Development Commands:"
    @echo "   just install                    - Install all dependencies"
    @echo "   just test                       - Run tests with coverage and linting"
    @echo "   just lint                       - Fix code formatting with Black"
    @echo "   just check                      - Validate project setup and dependencies"
    @echo ""
    @echo "🔧 Binary Commands:"
    @echo "   just generate-perf-report ARGS  - Generate performance reports"
    @echo "                                     Example: just generate-perf-report configs/experiment.yaml"
    @echo "                                     Example: just generate-perf-report --config configs/experiment.yaml"
    @echo "   just find-latest-experiment ARGS - Find and process latest experiments"
    @echo "                                      Example: just find-latest-experiment index.html failures.json"
    @echo "   just update-index ARGS          - Update experiment index files"
    @echo "                                     Example: just update-index --reportDir reports"
    @echo "   just update-probe-index ARGS    - Update probe index from telemetry schemas"
    @echo ""
    @echo "🚀 Aliases:"
    @echo "   just run ARGS                   - Alias for generate-perf-report"
    @echo "                                     Example: just run configs/experiment.yaml"
    @echo ""
    @echo "💡 Tips:"
    @echo "   - Use 'just --list' to see all available recipes"
    @echo "   - Use 'just --show COMMAND' to see a specific recipe"
    @echo "   - Add --help to any binary command for detailed usage"

# Alias for help command
usage: help

# Install all dependencies
install:
    pip install -r requirements.txt -r requirements-dev.txt

# Run tests with coverage and linting
test:
    @echo "🧪 Running all tests with coverage..."
    pytest tests/ -v --cov=lib --cov-report=xml --cov-report=html --cov-report=term
    @echo "🎨 Running linting checks..."
    flake8 lib/ tests/ bin/ --max-line-length=88 --extend-ignore=E203,W503,E501 --per-file-ignores="tests/*:E402"
    @echo "✅ All tests and linting passed!"

# Validate project setup and dependencies
check:
    @echo "🔍 Checking project setup..."
    @echo "Python version: $(python --version)"
    @echo "Dependencies: $(pip list | wc -l) packages installed"
    @echo "Config files: $(ls configs/*.yaml | wc -l) YAML configs"
    @echo "Test files: $(find tests/ -name "*.py" | wc -l) test files"
    @echo "✅ Project setup looks good!"

# Fix linting errors automatically
lint:
    @echo "🎨 Fixing code formatting..."
    black lib/ tests/ bin/
    @echo "✅ Code formatted!"

# Generate performance report
generate-perf-report *ARGS:
    @echo "🚀 Running performance report generator..."
    bin/generate-perf-report {{ ARGS }}

# Find latest experiments
find-latest-experiment *ARGS:
    @echo "🔍 Finding latest experiments..."
    bin/find-latest-experiment {{ ARGS }}

# Update experiment index
update-index *ARGS:
    @echo "📄 Updating experiment index..."
    bin/update-index {{ ARGS }}

# Update probe index
update-probe-index *ARGS:
    @echo "🔧 Updating probe index..."
    bin/update-probe-index {{ ARGS }}

# Run the performance report generator (alias for backwards compatibility)
run *ARGS:
    @echo "🚀 Running performance report generator..."
    bin/generate-perf-report {{ ARGS }}
