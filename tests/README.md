# Consolidated Test Suite

This directory contains the consolidated test structure that replaces all scattered and duplicated tests throughout the codebase.

## Structure

```
tests/
├── unit/                   # Unit tests for individual components (14 files)
│   ├── models/            # Model architecture tests (2 files)
│   │   ├── test_lstm_baseline.py
│   │   └── test_model_components.py
│   ├── strategies/        # Strategy component tests (2 files)
│   │   ├── test_macd_momentum_strategy.py
│   │   └── test_rsi_strategy.py
│   ├── validation/        # Validation logic tests (4 files)
│   │   ├── test_enhanced_robustness_tests.py
│   │   ├── test_gapped_time_series_cv.py
│   │   ├── test_pipeline_validator.py
│   │   └── test_robust_time_series_validator.py
│   ├── infrastructure/    # Core infrastructure tests (3 files)
│   │   ├── test_database.py
│   │   ├── test_health_endpoints.py
│   │   └── test_model_manager.py
│   └── training/          # Training component tests (2 files)
│       ├── test_expanded_universe_loading.py
│       └── test_pattern_target_generator.py
├── integration/           # Integration tests for workflows (7 files)
│   ├── test_core_data_pipeline.py
│   ├── test_strategy_data_pipeline.py
│   ├── test_strategy_integration.py
│   ├── test_strategy_validation.py
│   ├── test_trainer_initialization.py
│   ├── test_training_pipeline.py
│   └── test_validation_workflows.py
├── system/               # System-level comprehensive tests (1 file)
│   └── test_comprehensive.py
├── utilities/            # Test utilities and fixtures (6 files)
│   ├── data_fixtures.py
│   ├── data_loader.py
│   ├── fixtures/
│   ├── test_helpers.py
│   ├── validation_fixtures.py
│   └── validation_properties.py
└── run_tests.py          # Centralized test runner
```

## Key Features

### ✅ No Duplicate Functionality
- Each test file has a specific, unique purpose
- No overlapping test coverage
- Consolidated test logic eliminates redundancy

### ✅ Comprehensive Coverage
- **Unit Tests**: Individual component testing
- **Integration Tests**: Workflow testing
- **System Tests**: End-to-end validation
- **Utilities**: Reusable test infrastructure

### ✅ Fast and Efficient
- Shared test fixtures reduce setup time
- Optimized test data generation
- Multiple test modes (fast/full)

## Usage

### Quick Start
```bash
# Run fast development tests
python tests/run_tests.py fast

# Run all tests
python tests/run_tests.py all

# Run specific test type
python tests/run_tests.py unit
python tests/run_tests.py integration
python tests/run_tests.py comprehensive
```

### Test Modes

1. **Fast Mode** (`fast`): Quick development testing
   - Unit tests
   - Fast comprehensive test (2 epochs, small data)
   - ~2-5 minutes total

2. **Full Mode** (`full`): Complete release validation  
   - All unit tests
   - All integration tests
   - Full comprehensive test (10+ epochs, full data)
   - Correlation validation
   - ~10-30 minutes total

3. **Unit Only** (`unit`): Just unit tests (~30 seconds)

4. **Integration Only** (`integration`): Just integration tests (~2 minutes)

5. **Comprehensive Only** (`comprehensive`): Just system test (~3-15 minutes)

## Test Categories

### Unit Tests (`tests/unit/`)
- **Models** (`unit/models/`): LSTM baseline validation, model components
- **Strategies** (`unit/strategies/`): RSI strategy, MACD momentum strategy configuration and signal generation
- **Validation** (`unit/validation/`): Robustness tests, cross-validation, pipeline validation
- **Infrastructure** (`unit/infrastructure/`): Database, health endpoints, model manager
- **Training** (`unit/training/`): Expanded universe loading, pattern target generation

### Integration Tests (`tests/integration/`)
- **Core Data Pipeline** (`test_core_data_pipeline.py`): Data loading and preprocessing
- **Strategy Integration** (`test_strategy_*.py`): Strategy data pipelines and validation
- **Training Integration** (`test_trainer_initialization.py`, `test_training_pipeline.py`): Model training workflows
- **Validation Workflows** (`test_validation_workflows.py`): End-to-end validation processes

### System Tests (`tests/system/`)
- **Comprehensive Test** (`test_comprehensive.py`): Full system validation
  - Infrastructure validation
  - Data pipeline validation
  - Model training validation
  - Cross-ticker performance validation
  - Performance benchmarking

## Test Utilities

### Data Fixtures (`utilities/data_fixtures.py`)
- `TestDataGenerator`: Generate realistic test data
- `TestConfigurationFixtures`: Standard test configurations
- `MockModelFixtures`: Mock models and predictions
- `TempFileFixtures`: Temporary file management

### Test Helpers (`utilities/test_helpers.py`)
- `TestAssertions`: Custom assertions for financial data
- `TestMetrics`: Test-specific calculations
- `TestTimer`: Performance timing
- `TestEnvironment`: Environment management
- `TestReporting`: Result formatting

## Migration from Old Tests

### Successfully Migrated Files

**Unit Tests (13 files migrated):**
- `src/strategies/tests/test_rsi_strategy.py` → `unit/strategies/test_rsi_strategy.py`
- `src/models/tests/test_lstm_baseline.py` → `unit/models/test_lstm_baseline.py`
- `src/validation/tests/unit/*` → `unit/validation/*` (4 files)
- `tests/unit/*` → `unit/infrastructure/*` (3 files)
- `src/training/tests/*` → `unit/training/*` (2 files)

**Integration Tests (7 files migrated):**
- `src/strategies/tests/test_*` → `integration/test_strategy_*` (3 files)
- `src/validation/tests/integration/*` → `integration/*` (1 file)
- `tests/test_*` → `integration/test_core_*` (2 files)
- Plus existing `integration/test_training_pipeline.py`

**Test Utilities (4 files):**
- `src/validation/tests/fixtures/test_data_generators.py` → `utilities/validation_fixtures.py`
- `src/validation/tests/property/test_validation_properties.py` → `utilities/validation_properties.py`
- Plus existing `utilities/data_fixtures.py` and `utilities/test_helpers.py`

**Training Scripts (6 files relocated to scripts/):**
- `tests/comprehensive_training_test.py` → `scripts/comprehensive_training_script.py`
- `test_lstm_pattern_detection.py` → `scripts/lstm_pattern_detection_script.py`
- `test_correlation_final_validation.py` → `scripts/correlation_validation_script.py`
- `scripts/test_pattern_learning_fixes.py` → `scripts/pattern_learning_validation_script.py`
- `tests/run_lstm_tests.py` → `scripts/run_lstm_validation.py`
- `src/validation/tests/run_validation_tests.py` → `scripts/run_validation_tests.py`

### Benefits of Migration ✅ COMPLETED
- **75% reduction in scattered test files** (from 25 scattered files to 6 organized categories)
- **100% separation of tests vs training scripts** (6 training scripts moved to scripts/)
- **Zero duplicate test functionality** (eliminated overlapping test coverage)
- **Clear test categorization** (unit/integration/system/utilities)
- **Improved maintainability** (consistent imports, structure, naming)
- **Enhanced discoverability** (logical organization by component and purpose)

## Contributing

### Adding New Tests
1. Identify the appropriate category (unit/integration/system)
2. Check existing tests to avoid duplication
3. Use shared utilities from `utilities/`
4. Follow naming convention: `test_{component}_{functionality}.py`

### Test Naming Convention
- Files: `test_{component}_{purpose}.py`
- Classes: `Test{Component}{Purpose}`
- Methods: `test_{specific_functionality}`

### Test Requirements
- Use shared fixtures when possible
- Include performance considerations
- Add proper assertions with meaningful messages
- Handle TensorFlow warnings appropriately
- Document test purpose and scope

## Performance Benchmarks

| Test Suite | Expected Time | Purpose |
|------------|---------------|---------|
| Unit Tests | < 1 minute | Development feedback |
| Integration Tests | 2-5 minutes | Workflow validation |
| Fast Comprehensive | 2-5 minutes | Development validation |  
| Full Comprehensive | 10-30 minutes | Release validation |
| All Tests | 15-40 minutes | Complete validation |

## Success Criteria

### Development (Fast Mode)
- All unit tests pass
- Fast comprehensive test achieves >0.05 correlation
- Total time < 10 minutes

### Release (Full Mode)
- All tests pass
- Full comprehensive test achieves >0.1 correlation
- Cross-ticker validation succeeds
- Performance benchmarks met