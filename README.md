# ML Alpha Generation System

## Project Focus

**Goal**: Generate consistent alpha through improved ML models and risk management  
**Target**: 52-55% directional accuracy with Sharpe ratio >1.0  
**Approach**: Cross-sectional ranking, multi-timeframe ensembles, and regime awareness

## Architecture

```
ML Alpha Generation System
├── PostgreSQL                  # Single database for all data
├── FastAPI                     # Model serving API
├── Multi-Strategy Models       # Cross-sectional, multi-timeframe, regime-aware
├── Walk-Forward Backtesting    # performance validation
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
├── Business Logic
│   ├── strategies/        # Trading strategy implementations
│   │   ├── cross_sectional/    # Sector-neutral ranking models
│   │   ├── multi_timeframe/    # 1,3,5,10-day ensemble
│   │   └── regime_aware/       # Market condition adaptation
│   ├── backtesting/       # Walk-forward testing framework
│   │   ├── results/       # Performance data
│   │   └── reports/       # Analysis outputs
│   └── models/           # Trained model storage
│       ├── trained/      # Production models
│       └── checkpoints/  # Training snapshots
│
├── ML Pipeline (Enhanced Existing)
│   ├── src/              # Core ML components
│   │   ├── data/         # Zero temporal loss data loading
│   │   ├── features/     # 149+ feature candidates → 24 selected
│   │   ├── models/       # Multi-scale LSTM + directional loss
│   │   └── utils/        # Validation and utilities
│   └── scripts/          # Pipeline orchestration
│
└── Documentation
    └── docs/
```

## Quick Start

### 1. Setup Infrastructure

```bash
# Clone and navigate to project root
cd financial-returns-api

# Start database
docker-compose up database -d

# Install dependenies
pip install -r ml-requirements.txt

# Initialize database schema
python -c "from database.schema import create_tables; from database.connection import get_database_manager; create_tables(get_database_manager().engine)"
```

### 2. Configure Environment

```bash
cp config/development.yaml config/local.yaml

vim config/local.yaml  # Update database URL, tickers, etc.
```

### 3. Train Models

```bash
# Test existing pipeline
python scripts/run_production_pipeline.py --quick-test

# Train cross-sectional ranking model
python -m strategies.cross_sectional.train

# Train multi-timeframe ensemble
python -m strategies.multi_timeframe.train
```

### 4. Start API Server

```bash
# Development server (ML API)
python run_ml_api.py

# Docker deployment
docker-compose up
```

## Current System Status

### **Working Foundation**

- **Zero Temporal Loss**: 17,998 training samples
- **Multi-Scale LSTM**: 3.2M parameters with attention mechanisms
- **Advanced Features**: 149 candidates → 24 optimally selected
- **Bias Fixes Applied**: StandardScaler, fillna fixes, removed artificial negative bias
- **Production Pipeline**: 5-stage separation of concerns
- **Performance**: ~ 57% directional accuracy (no magnitude support, combined weekly + daily)

### **In Development (New Infrastructure)**

- **Multi-Strategy Models**: Cross-sectional, multi-timeframe, regime-aware
- **API Framework**: FastAPI with model serving endpoints
- **Backtesting**: Walk-forward analysis with transaction costs
- **Database**: PostgreSQL + TimescaleDB time-series optimization
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

```python
# Current architecture (3.2M parameters)
Short-term Branch:  10-day patterns (128 LSTM units)
Medium-term Branch: 30-day patterns (512→256 LSTM units)
Long-term Branch:   Subsampled patterns (128 LSTM units)
Attention:          Self + cross-scale attention
Dense Layers:       [256, 128, 64] → single output
Loss Function:      60% MSE + 40% directional loss
```

### Feature Engineering Pipeline

```python
Raw Features:     149 candidates across 8 processors
Selected:         24 optimal features via category-based selection
Processing:       StandardScaler (preserves +/- relationships)
NaN Handling:     Forward/backward fill (to prevent down zero-bias)
Outlier Control:  Quantile clipping with robust scaling
Target:           Daily returns with proper temporal alignment
```

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

## API Endpoints

### Predictions

```bash
# Single stock prediction
POST /predictions/single
{
  "ticker": "AAPL",
  "horizons": [1, 3, 5, 10],
  "strategy": "ensemble"
}

# Batch predictions
POST /predictions/batch
{
  "tickers": ["AAPL", "MSFT", "GOOG"],
  "date": "2025-01-15"
}
```

### Portfolio Management

```bash
# Current positions
GET /portfolio/positions

# Run backtest
POST /portfolio/backtest
{
  "strategy": "cross_sectional"
}
```

### System Health

```bash
# Health check
GET /health

# Model status
GET /info
```

## Version Management

### Package Versions

- **Python**: 3.12+
- **TensorFlow**: 2.19.0
- **FastAPI**: 0.115.0+
- **SQLAlchemy**: 2.0.36+

## Development Workflow

### **Research Phase**

1. **Feature Engineering**: Test new indicators in `src/features/`
2. **Model Architecture**: Experiment with new models in `src/models/`

### **Implementation Phase**

1. **Strategy Development**: Add to `strategies/{strategy_name}/`
2. **Backtesting**: Validate with `backtesting/` framework
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

### **Machine Learning Framework**

- **Feature Engineering**:
  - Price momentum and volatility metrics (GARCH(p,q) focused)
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
- **Hybrid Architecture**: Combines LSTM, GRU, and CNN components
- **Custom Loss Functions**: Directional accuracy penalty for trading focused optimization
- **Regularization**: Layer normalization, dropout, and L2 regularization

### **Production Considerations**

- **Monitoring**: Comprehensive logging and performance tracking
- **Error Handling**: data validation and easy failure recovery
- **Configuration Management**: Centralized configuration system

---

## Current Development Focus

### **Phase 1: Code Refactoring & Cleanup (Current Priority)**

**Status**: **In Progress**

- **Infrastructure Migration**: ML components moved from `ml/` to root level
- **Dependency Updates**: Python 3.12, latest package versions, removed legacy code
- **Clean Architecture**: Separated concerns, removed redundant files and old logs
- **Code Quality**: Standardizing imports, fixing deprecated patterns
- **Documentation**: API completion, proper type hints, docstring standardization

### **Phase 2: Infrastructure Foundation (Next)**

**Status**: **Mostly Planned**

- **API Development**: Complete FastAPI routes and model serving endpoints
- **Database Migration**: Move from SQLite to PostgreSQL + TimescaleDB
- **Configuration System**: Finalize environment-specific config management
- **Testing Framework**: Comprehensive test coverage for ML pipeline
- **Docker Deployment**: Production-ready containerization

### **Phase 3: Portfolio Management & Trading (Long-term)**

**Status**: **Planned**

- **Strategy Implementation**: Cross-sectional ranking, multi-timeframe ensembles
- **Backtesting Framework**: Walk-forward analysis with realistic transaction costs
- **Risk Management**: Position sizing, drawdown controls, correlation limits
- **Performance Monitoring**: Real-time tracking, automated rebalancing
- **Production Trading**: Paper trading → live implementation

---

### **Immediate Next Steps (Priority Order)**

1. **Complete API Routes** (`ml_api/routes/`) - predictions, portfolio, health endpoints
2. **Database Schema Implementation** - migrate from SQLite to PostgreSQL
3. **Model Serving Pipeline** - load/cache models, prediction serving
4. **Testing Infrastructure** - unit tests, integration tests, pipeline validation
5. **Configuration Validation** - ensure all environments work correctly

### **Success Metrics by Phase**

**Phase 1 (Refactoring)**: Complete

- Clean, maintainable codebase
- Standardized development workflow

**Phase 2 (Infrastructure)**:

- API endpoints functional
- Database migration complete
- Docker deployment working
- Test coverage >80%

**Phase 3 (Portfolio Management)**:

- Backtesting framework operational
- 52-55% directional accuracy achieved
- Sharpe ratio >1.0 in paper trading

---

### **Technical Debt & Quality**

**Completed**:

- Removed legacy ML directory and redundant files
- Standardized code formatting (Black, Ruff, pre-commit hooks)

**Ongoing**:

- Type hint standardization across codebase
- Docstring completion for all public APIs
- Import optimization and dependency cleanup

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.
