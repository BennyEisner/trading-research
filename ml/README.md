# Financial ML Alpha Generation System

\*\*Transform 50% directional accuracy into 52-55% alpha through swing trading approach

## Project Focus

**Goal**: Generate consistent alpha through improved ML models and risk management  
**Target**: 52-55% directional accuracy with Sharpe ratio >1.0  
**Approach**: Cross-sectional ranking, multi-timeframe ensembles, regime awareness  
**Scale**: Personal project

## 🏗️ Architecture

```
Financial ML Alpha Generation System
├── PostgreSQL                  # Single database for all data
├── FastAPI                     # Model serving API
├── Multi-Strategy Models       # Cross-sectional, multi-timeframe, regime-aware
├── Walk-Forward Backtesting    # Realistic performance validation
├── Risk Management             # Position sizing, drawdown controls
└── Docker Deployment           # Single-machine containerized setup
```

## 📁 Directory Structure

```
ml/
├── Infrastructure
│   ├── config/             # YAML configuration + type safety
│   ├── database/           # PostgreSQL + TimescaleDB schema
│   ├── api/               # FastAPI model serving
│   ├── docker-compose.yml # Single-machine deployment
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
# Clone and navigate
cd financial-returns-api/ml

# Start database
docker-compose up database -d

# Install dependencies
pip install -r requirements.txt

# Initialize database schema
python -c "from database.schema import create_tables; from database.connection import get_database_manager; create_tables(get_database_manager().engine)"
```

### 2. Configure Environment

```bash
# Copy development config
cp config/development.yaml config/local.yaml

# Edit for your setup
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
# Development server
uvicorn api.app:create_app --factory --reload

# Or full Docker deployment
docker-compose up
```

## Current System Status

### \*\*Working Foundation

- **Zero Temporal Loss**: 17,998 training samples
- **Multi-Scale LSTM**: 3.2M parameters with attention mechanisms
- **Advanced Features**: 149 candidates → 24 optimally selected
- **Bias Fixes Applied**: StandardScaler, fillna fixes, removed artificial negative bias
- **Production Pipeline**: 5-stage separation of concerns

### **In Development (New Infrastructure)**

- **Multi-Strategy Models**: Cross-sectional, multi-timeframe, regime-aware
- **API Framework**: FastAPI with model serving endpoints
- **Backtesting**: Walk-forward analysis with transaction costs
- **Database**: PostgreSQL + TimescaleDB time-series optimization
- **Risk Management**: Position sizing, drawdown controls

## Alpha Generation Strategies

### 1. Cross-Sectional Ranking

**Approach**: Predict relative performance within sectors/market cap cohorts  
**Edge**: Remove market-wide noise, focus on stock specific alpha  
**Implementation**: Long top 20%, short bottom 20% based on rankings  
**Expected**: 52-53% accuracy through sector-neutral predictions

### 2. Multi-Timeframe Ensemble

**Approach**: Combine 1, 3, 5, 10-day predictions with different weights  
**Edge**: Capture momentum (short-term) and mean reversion (longer-term)  
**Implementation**: Sharpe ratio-weighted ensemble with correlation adjustment  
**Expected**: 53-54% accuracy through signal diversification

### 3. Regime-Aware Models

**Approach**: Different models for different market conditions  
**Edge**: Adapt to changing market dynamics (volatility, correlation regimes)  
**Implementation**: HMM regime detection → dynamic model switching  
**Expected**: 54-55% accuracy through adaptive modeling

## Model Architecture Deep Dive

### Enhanced Multi-Scale LSTM

```python
# Current architecture (3.2M parameters)
Short-term Branch:  10-day patterns (128 LSTM units)
Medium-term Branch: 30-day patterns (512→256 LSTM units)
Long-term Branch:   Subsampled patterns (128 LSTM units)
Attention:          Self + cross-scale attention
Dense Layers:       [256, 128, 64]  single output
Loss Function:      60% MSE + 40% directional loss
```

### Feature Engineering Pipeline

```python
Raw Features:     149 candidates across 8 processors
Selected:         24 optimal features via category-based selection
Processing:       StandardScaler (preserves +/- relationships)
NaN Handling:     Forward/backward fill (no zero-bias)
Outlier Control:  Quantile clipping with robust scaling
Target:           Daily returns with proper temporal alignment
```

## Performance Targets & Validation

### **Accuracy Targets**

- **Baseline**: 50% (random/market efficiency)
- **Cross-sectional**: 52-53% (sector-neutral edge)
- **Multi-timeframe**: 53-54% (signal diversification)
- **Regime-aware**: 54-55% (adaptive modeling)
- **Combined**: 55%+ (ensemble of strategies)

### **Risk-Adjusted Returns**

- **Sharpe Ratio**: >1.0 (vs 0.5-0.7 typical)
- **Max Drawdown**: <15% (with dynamic position sizing)
- **Calmar Ratio**: >0.8 (return/max drawdown)
- **Volatility Target**: 12% annualized

### **Backtesting Framework**

- **Method**: Walk-forward analysis with 252-day training windows
- **Rebalancing**: Weekly (5-day frequency)
- **Transaction Costs**: 10 bps round-trip (realistic)
- **Position Limits**: 1-10% per position, correlation controls
- **Out-of-Sample**: Strict temporal separation, no data snooping

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
  "strategy": "cross_sectional",
  "start_date": "2024-01-01",
  "end_date": "2024-12-31"
}
```

### System Health

```bash
# Health check
GET /health

# Model status
GET /info
```

## Cost Structure

### **Development**: Free

- Local PostgreSQL
- System GPU/CPU-only training
- Single-machine deployment

## Development Workflow

### **Research Phase**

1. **Feature Engineering**: Test new indicators in `src/features/`
2. **Model Architecture**: Experiment with new models in `src/models/`

### **Implementation Phase**

1. **Strategy Development**: Add to `strategies/{strategy_name}/`
2. **Backtesting**: Validate with `backtesting/` framework
3. **API Integration**: Add endpoints to `api/routes/`

### **Deployment Phase**

1. **Configuration**: Update `config/production.yaml`
2. **Docker Build**: `docker build -t trading-api .`
3. **Deploy**: `docker-compose up -d`

