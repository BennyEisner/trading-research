# Trading Research

## Project Focus

**Goal**: Generate consistent alpha through improved ML models and risk management  
**Target**: 52-55% win rate with Sharpe ratio >1.0  
**Approach**: Cross-sectional ranking, multi-timeframe ensembles, and regime awareness

## Architecture

```
ML Alpha Generation System
├── PostgreSQL + TimescaleDB    # Time-series optimized data storage
├── FastAPI                     # Model serving API
├── Multi-Strategy Models       # Cross-sectional, multi-timeframe, regime-aware
├── Ensemble Strategy Framework # Refactored pipeline orchestration
├── Walk-Forward Backtesting    # Performance validation
├── Risk Management             # Position sizing
└── Docker Deployment           # Containerized setup
```

## Directory Structure

```
├── Infrastructure
│   ├── config/             # YAML configuration
│   ├── database/           # PostgreSQL + TimescaleDB schema
│   ├── ml_api/             # FastAPI model serving
│   ├── docker-compose.yml # Single machine deployment
│   └── Dockerfile         # Python API container
│
├── Pipeline Orchestration
│   ├── pipeline/          # Refactored pipeline components
│   │   ├── orchestration/     # Thin pipeline orchestration (EnsembleDataPipeline)
│   │   ├── coordination/      # Ensemble coordination with graceful degradation
│   │   ├── historical/        # Historical data integration
│   │   └── backtesting/       # Enhanced backtesting framework
│
├── Business Logic
│   ├── strategies/        # strategy implementations
│   │   ├── core/             # Strategy compatibility checker
│   │   ├── implementations/   # RSI, MACD and other strategy implementations
│   │   ├── adapters/         # ENsure coordination and quality checks
│   │   ├── validation/       # Strategy-specific validation
│   │   └── ensemble.py       # Ensemble manager for signal combination
│   ├── backtesting/       # Walk-forward testing framework
│   │   ├── results/       # Performance data
│   │   └── reports/       # Analysis outputs
│   └── models/           # Trained model storage
│       ├── trained/      # Production models
│       └── checkpoints/  # Training snapshots
│
├── ML Pipeline
│   ├── src/
│   │   ├── data/
│   │   ├── features/
│   │   ├── models/
│   │   └── utils/
│   └── scripts/
│
└── Documentation
    └── docs/
```

## Current System Status

### **(Working Foundation**

- **Zero Temporal Loss**: 17,998 training samples
- **Multi-Scale LSTM**: 3.2M parameters with attention mechanisms
- **Advanced Features**: 149 candidates → 24 optimally selected
- **Bias Fixes Applied**: StandardScaler, fillna fixes, removed artificial negative bias
- **Production Pipeline**: 5-stage separation of concerns
- **Performance**: ~ 57% directional accuracy (no magnitude support, combined weekly + daily)
- **Refactored Strategy System**: Clean pipeline orchestration with focused components

### **Completed Infrastructure**

- **Refactored Pipeline Architecture**: focused components
- **Strategy Compatibility Checker**: Robust requirement validation
- **Ensemble Coordinator**: Graceful degradation management
- **Simplified Data Adapter**: Pure data formatting without validation chains
- **API Framework**: FastAPI with model serving endpoints
- **Database**: PostgreSQL + TimescaleDB time-series optimization

### **In Development (Phase 2)**

- **Historical Data Integration**: Enhanced backtesting with refactored components
- **Multi-Strategy Models**: Cross-sectional, multi-timeframe, regime-aware
- **Component-Aware Metrics**: Performance attribution and degradation analysis
- **Enhanced Validation**: Historical validation with bootstrap significance testing
- **Risk Management**: Position sizing, drawdown controls

## Potential Alpha Generation Strategies

### 1. Cross-Sectional Ranking

**Approach**: Predict relative performance within sectors or market cap cohorts  
**Edge**: Remove market-wide noise + daily noise, focus on stock specific alpha  
**Implementation**: Long top 20%, short bottom 20% based on rankings  
**Expected**: 50-53% accuracy through sector neutral predictions

### 2. Multi-Timeframe Ensemble

**Approach**: Combine 1, 3, 5, 10-day predictions with different weights  
**Edge**: Capture momentum (short-term) and mean reversion (longer-term)  
**Implementation**: Sharpe ratio-weighted ensemble with correlation adjustment  
**Expected**: 50-53% accuracy through signal diversification

### 3. Regime-Aware Models

**Approach**: Different models for different market conditions  
**Edge**: Adapt to changing market dynamics (volatility, correlation regimes)  
**Implementation**: HMM regime detection to dynamic model switching  
**Goal**: 52-55+% consistent accuracy through adaptive modeling

## Current Model Architecture Deep Dive

### Multi-Scale LSTM

## Future Performance Targets & Validation

### **Accuracy Targets**

- **Baseline**: 50-53% (random/market efficiency)
- **Cross-sectional**: 50-53% (sector-neutral edge)
- **Multi-timeframe**: 50-54% (signal diversification)
- **Regime-aware**: 50-55% (adaptive modeling)
- **Combined**: 55%+ (ensemble strategies)

### **Risk-Adjusted Returns**

- **Sharpe Ratio**: >1.0
- **Max Drawdown**: <15% (with dynamic position sizing)
- **Calmar Ratio**: >0.8 (return/max drawdown)
- **Volatility Target**: 10-12% annualized

### **Backtesting Framework**

- **Method**: Walk-forward analysis with 252-day training windows
- **Rebalancing**: Weekly (5-day frequency)
- **Transaction Costs**: 10 bps round-trip (for realistic simulation)
- **Position Limits**: 1-10% per position
- **Out of Sample**: Temporal separation

## Development Workflow

### **Research Phase**

1. **Feature Engineering**: Test new indicators in `src/features/`
2. **Model Architecture**: Experiment with new models in `src/models/`

### **Implementation Phase**

1. **Strategy Development**: Add to `strategies/{strategy_name}/`
2. **Backtesting**: Validate with `backtesting/` and `src/validation/` framework
3. **API Integration**: Add endpoints to `ml_api/routes/`

### **Deployment Phase**

1. **Configuration**: Update `config/production.yaml`
2. **Docker Build**: `docker build -t trading-api .`
3. **Deploy**: `docker-compose up -d`

### **Data Pipeline**

- **ETL Pipeline**: Automated data fetching, transformation, and loading from financial APIs
- **Database**: SQLite with SQLAlchemy ORM for data storage and retrieval

### **API Layer**

- **FastAPI**: RESTful API for data access and model predictions
- **Endpoints**: Price data retrieval, ticker management, portfolio formulation, and prediction services
- **Frontend**: Next.js/React.js interface for data visualization and model interaction

### **Machine Learning Framework (in transition period)**

- **Feature Engineering**:
  - Price momentum and volatility metrics
  - Moving averages and Bollinger Bands
  - Volume analysis and market efficiency indicators
  - Support/resistance levels and trend strength
- **Model Architectures**:
  - Multi-branch LSTM with attention mechanisms
  - Convolutional feature extraction
  - Ensemble methods with cross-validation

---

## Key Features

### **Advanced Feature Engineering**

- **Technical Indicators**: RSI, MACD, Bollinger Bands, Stochastic Oscillator
- **Market Microstructure**: Gap analysis, price position within daily range
- **Volatility Metrics**: GARCH, Multi-timeframe volatility ratios and regime detection
- **Volume Analysis**: On-balance volume, volume-price trend correlation
- **Statistical Features**: Returns skewness, kurtosis, and autocorrelation

### **Neural Network**

- **Multi-Head Attention**: Captures temporal dependencies
- **Hybrid Architecture**: Combines LSTM, GRU, and CNN componentularization\*\*: Layer normalization, dropout, and L2 regularization
- **Note**: Currently refactoring LSTM architecture in order to simplify and reduce paramater count to help with overfitting

### **Production Considerations**

- **Monitoring**: Comprehensive logging and performance tracking
- **Error Handling**: data validation and easy failure recovery
- **Configuration Management**: Centralized configuration system

---

## Current Development Focus

### **Phase 1: Code Refactoring & Cleanup**

**Status**: **Completed**

- **Infrastructure Migration**: ML components moved from `ml/` to root level
- **Dependency Updates**: Python 3.12, latest package versions, removed legacy code
- **Clean Architecture**: Separated concerns, removed redundant files and old logs
- **Code Quality**: Standardizing imports, fixing deprecated patterns
- **Documentation**: API completion, proper type hints, docstring standardization
- **Strategy System Refactoring**: Complete pipeline orchestration refactor with focused components

### **Phase 2: Historical Data Integration (Current Priority)**

**Status**: **In Progress**

- **Historical Pipeline Integration**: Connect refactored components with existing DataLoader and TimescaleDB
- **Enhanced Backtesting Framework**: EnsembleBacktestRunner with component-aware metrics
- **Focused Validation Integration**: StrategyCompatibilityChecker and EnsembleCoordinator with historical validation
- **Component Attribution**: Performance analysis with focused component insights
- **API Enhancement**: FastAPI routes for ensemble strategy endpoints
- **Testing Framework**: Comprehensive integration tests for refactored architecture

### **Phase 3: Portfolio Management & Trading (Long-term)**

**Status**: **Planned**

- **Strategy Implementation**: Cross-sectional ranking, multi-timeframe ensembles
- **Backtesting Framework**: Walk-forward analysis with realistic transaction costs
- **Risk Management**: Position sizing, drawdown controls, correlation limits
- **Performance Monitoring**: Real-time tracking, automated rebalancing
- **Production Trading**: Paper trading → live implementation

---

### **Immediate Next Steps (Priority Order)**

1. **Historical Pipeline Adapter** (`src/pipeline/historical/`) - integrate refactored components with existing DataLoader
2. **Pattern Focused LSTM** integrated into ensemble strategy
3. **Voting based confidence** to use technical analysis in paralell to LSTM pattern recognition
4. **Enhanced Backtesting Runner** (`src/pipeline/backtesting/`) - component-aware performance metrics
5. **Focused Validation Pipeline** (`src/validation/historical_validation_pipeline.py`) - no validation chains
6. **Integration Testing** - comprehensive tests for refactored architecture with historical data
7. **API Enhancement** - ensemble strategy endpoints with component attribution

### **Success Metrics by Phase**

**Phase 1 (Refactoring)**: **Completed**

- Clean, maintainable codebase
- Refactored strategy system with focused components
- Pipeline orchestration with 10-line execute method
- No validation chains - direct component ownership

**Phase 2 (Historical Integration)**: **In Progress**

- Historical pipeline adapter functional
- Component-aware backtesting operational
- Enhanced validation with bootstrap significance testing
- Performance attribution with graceful degradation analysis

**Phase 3 (Production Trading)**: **Planned**

- Multi-strategy models (cross-sectional, multi-timeframe, regime-aware)
- 52-55% directional accuracy achieved
- Sharpe ratio >1.0 with ensemble approach
- Component-specific risk management

## Refactored Ensemble Strategy System

### **Key Improvements**

- **No Validation Chains**: Direct PipelineValidator ownership eliminates confusion
- **Focused Components**: Each component has single, clear responsibility
- **Enhanced Testability**: Components can be tested in isolation
- **Graceful Degradation**: EnsembleCoordinator handles strategy failures elegantly

### **Component Responsibilities**

**Pipeline Orchestration** (`src/pipeline/orchestration/ensemble_data_pipeline.py`):

- Coordinates 5 focused execution stages
- Direct PipelineValidator ownership
- Centralized error handling and results packaging

**Strategy Compatibility Checker** (`src/strategies/core/compatibility_checker.py`):

- Validates data against strategy requirements
- Feature quality checks (NaN ratios, RSI ranges, ATR positivity)
- Fallback feature mapping

**Ensemble Coordinator** (`src/pipeline/coordination/ensemble_coordinator.py`):

- Multi-strategy ensemble logic with graceful degradation
- Strategy validation coordination
- Comprehensive failure reporting

**Simplified Data Adapter** (`src/strategies/adapters/data_adapter.py`):

- Pure data formatting and cleaning
- Feature fallback application
- Removed validation and ensemble logic (moved to focused components)

### **Testing and Validation**

**Test Suite**:

- Unit tests for each focused component
- Integration tests for end-to-end pipeline execution
- Component isolation testing for maintainability
- Performance benchmarking of refactored architecture

**Validation Framework Integration**:

- Direct PipelineValidator usage (no chains)
- Strategy-specific validation with StrategyCompatibilityChecker
- Ensemble robustness testing with EnsembleCoordinator
- Integration with existing GappedTimeSeriesCV and bootstrap methods

**Ongoing**:

- Type hint standardization across codebase
- Docstring completion
- Import optimization and dependency cleanup
- Phase 2 historical data integration with refactored architecture
- non linear pattern recognition LSTM creation
