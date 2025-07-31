# Financial Pattern Detection API

## Project Focus

**Goal**: Ensemble Trading System with LSTM Pattern Verification for swing trading (2-10 day holding periods)  
**Target**: >55% pattern detection accuracy with meta-labeling for technical indicator signal filtering  
**Approach**: Technical Indicators (RSI, MACD) + LSTM Pattern Verification → Ensemble Signal Integration

## Architecture

```
Ensemble Trading System with LSTM Pattern Verification
├── Technical Indicator Strategies
│   ├── RSI Mean Reversion Strategy
│   ├── MACD Momentum Strategy
│   └── Future Strategies (4-5 total planned)
├── LSTM Pattern Verification Engine
│   ├── 17 Pattern Features        # Specialized pattern detection
│   ├── Shared Backbone LSTM       # Pattern verification (~100-400k params)
│   ├── Multi-Ticker Training      # 27 related securities
│   └── Meta-Labeling Framework    # Filter low confidence technical signals
├── Ensemble Signal Integration
│   ├── Strategy Signal Combination
│   ├── LSTM Pattern Confidence Filtering
│   └── Position Sizing & Risk Management
├── FastAPI Serving                # Unified API endpoints
└── Docker Deployment              # Containerized setup
```

## Directory Structure

```
├── Ensemble Strategy System
│   ├── src/strategies/         # Technical indicator strategies
│   │   ├── implementations/    # RSI, MACD strategy implementations
│   │   ├── ensemble.py         # Strategy combination and weighting
│   │   ├── base.py            # Base strategy framework
│   │   └── validation/        # Strategy specific validation
│   ├── src/pipeline/          # Ensemble orchestration
│   └── src/backtesting/       # Walk-forward testing framework
│
├── LSTM Pattern Verification Engine
│   ├── src/features/           # 17 pattern features system
│   │   ├── pattern_feature_calculator.py
│   │   ├── multi_ticker_engine.py         # Multi-ticker processing
│   │   ├── processors/pattern_features_processor.py  # Processor integration
│   │   └── utils/              # Pattern validation utilities
│   ├── src/models/             # LSTM architecture
│   │   ├── shared_backbone_lstm.py        # Pattern verification specialist
│   │   └── lstm_baseline.py               # Pattern detection validation
│   ├── src/training/           # Training infrastructure
│   │   ├── shared_backbone_trainer.py     # multiticker trainer
│   │   └── pattern_target_generator.py    # Pattern validation targets
│   └── src/validation/         # Robust validation framework
│
├── Infrastructure
│   ├── config/             # System configuration
│   ├── ml_api/             # FastAPI unified serving
│   ├── docker-compose.yml # Deployment setup
│   └── archive/            # Archived legacy code (1200+ lines removed)
│
└── Documentation
    └── docs/
```

## Current System Status

### **Completed Foundation**

**Technical Indicator Strategies**:

- **RSI Mean Reversion Strategy**: Oversold/overbought signal generation with dynamic thresholds
- **MACD Momentum Strategy**: Crossover detection with histogram confirmation
- **Ensemble Framework**: Strategy combination, weighting, and signal integration

**LSTM Pattern Verification Engine**:

- **17 Pattern Features**: Momentum persistence, volatility regimes, trend exhaustion, volume divergences
- **Shared Backbone LSTM**: ~400k parameters, enhanced regularization, pattern verification specialist
- **Multi-Ticker Training**: 27 expanded universe securities → MAG7 specialization
- **Pattern Target Generation**: Binary pattern validation (not return prediction)
- **Overlapping Sequences**: 20-day lookback, 5-day stride for swing trading optimization
- **Clean Architecture**: 1,200+ lines of legacy code archived

### **Ready for Implementation**

**Ensemble Integration**:

- **Technical Indicator Signal Generation**: RSI + MACD primary signals ready
- **LSTM Pattern Confidence Scoring**: Pattern verification for signal filtering
- **Ensemble Signal Combination**: Weighted integration with meta-labeling

**LSTM Pattern Verification**:

- **Pattern Detection Training**: Shared backbone trainer with pattern targets
- **Enhanced Regularization**: Dropout 0.45, L2 0.006, batch normalization
- **Robust Validation**: Bootstrap statistical testing, cross-ticker validation
- **Multi-Ticker Engine**: Parallel processing for expanded universe training efficenccy
- **API Framework**: FastAPI serving infrastructure

### **Next Phase (Ensemble + Meta-Labeling Integration)**

**Ensemble Integration**:

- **LSTM-Strategy Integration**: Connect pattern confidence to RSI/MACD signal filtering
- **Meta-Labeling Implementation**: Filter low-confidence technical indicator signals
- **Additional Technical Strategies**: Expand to 4-5 total technical indicator strategies

**Advanced LSTM Features**:

- **Multi-Task LSTM**: Pattern confidence, direction, position sizing heads
- **MAG7 Specialization**: Transfer learning from expanded universe
- **Production Integration**: Walk-forward validation, ensemble coordination

## Ensemble Strategy Approach

### 1. Technical Indicator Primary Signals

**Philosophy**: RSI and MACD strategies generate primary trading signals based on proven technical analysis  
**Implementation**: RSI mean reversion + MACD momentum crossover strategies  
**Integration**: Ensemble framework combines and weights multiple technical strategies  
**Expansion**: Plans for 4-5 total technical indicator strategies

### 2. LSTM Pattern Verification & Meta-Labeling

**Philosophy**: LSTM as pattern verification specialist, not competing signal generator  
**Core Question**: "When technical indicators signal, are the underlying patterns actually valid?"  
**Implementation**: LSTM provides pattern confidence scores to filter technical indicator signals  
**Edge**: Reduce false signals by identifying when NOT to trade based on pattern confidence  
**Target**: >55% pattern detection accuracy for filtering low-confidence technical signals

### 3. Ensemble Signal Integration

**Approach**: Technical indicators generate signals → LSTM provides pattern confidence → Ensemble combines  
**Signal Flow**: RSI/MACD signals → Pattern confidence filtering → Position sizing → Final trading decision  
**Implementation**: EnsembleManager coordinates multiple strategies with LSTM meta-labeling  
**Expected**: Improved signal quality through confidence-based filtering of technical indicator signals

## System Architecture

### **Technical Indicator Strategies**

**RSI Mean Reversion Strategy**:

- **Logic**: Long when RSI < 30 (oversold), Short when RSI > 70 (overbought)
- **Exit Conditions**: Return to neutral zone (40-60), opposite signals
- **Signal Strength**: Based on RSI extremity with volatility adjustment

**MACD Momentum Strategy**:

- **Logic**: Long on bullish crossover (MACD > Signal), Short on bearish crossover
- **Confirmation**: Histogram momentum confirmation, divergence detection
- **Signal Strength**: Based on histogram magnitude with volatility adjustment

**Ensemble Framework**:

- **Signal Combination**: Weighted average, voting, or confidence-weighted methods
- **Position Sizing**: Signal strength based, equal weight, or volatility adjusted
- **Risk Management**: Maximum position limits, correlation monitoring

### **LSTM Pattern Verification Engine**

**17 Pattern Features**:

- **Non-linear Price Patterns**: Price acceleration, volume-price divergence, volatility regime changes
- **Temporal Dependencies**: Momentum persistence, volatility clustering, trend exhaustion, GARCH forecasting
- **Market Microstructure**: Intraday range expansion, overnight gaps, end-of-day momentum
- **Cross-asset Relationships**: Sector relative strength, market beta instability, VIX term structure
- **Core Context**: Multi-timeframe returns, normalized volume, price levels

**Shared Backbone LSTM**:

- **Architecture**: 2-layer LSTM (64→32 units) + Dense(16) → Pattern confidence output
- **Parameters**: ~400k (reduced from 3.2M to prevent overfitting)
- **Regularization**: Dropout 0.45, L2 0.006, batch normalization, gradient clipping
- **Training**: 27 expanded universe → MAG7 specialization transfer learning
- **Output**: Pattern confidence scores (0-1) for technical signal meta-labeling

**Pattern Validation Framework**:

- **Targets**: Binary pattern resolution validation (not return prediction)
- **Validation**: Bootstrap statistical testing, cross-ticker validation
- **Success Criteria**: >55% pattern detection accuracy + >0.3 correlation with resolution
- **Time Horizons**: 3, 5, 10, 15-day pattern resolution validation

## Development Workflow

### **Ensemble Strategy Pipeline**

1. **Data Loading**: Multi-ticker OHLCV data with technical indicators
2. **Technical Signal Generation**: RSI and MACD strategies generate primary signals
3. **Pattern Feature Calculation**: 17 pattern features via `PatternFeatureCalculator`
4. **Pattern Confidence Scoring**: LSTM provides confidence scores for technical signals
5. **Ensemble Integration**: Combine technical signals with pattern confidence filtering
6. **Validation**: Strategy performance + pattern detection accuracy

### **Implementation Steps**

**Strategy Development**:

1. **Strategy Implementation**: Add strategies to `src/strategies/implementations/`
2. **Ensemble Integration**: Register strategies with `EnsembleManager`
3. **Strategy Validation**: Use `src/strategies/validation/` framework

**LSTM Pattern Verification**:

1. **Feature Development**: Add patterns to `src/features/pattern_feature_calculator.py`
2. **Model Training**: Use `src/training/shared_backbone_trainer.py`
3. **Pattern Validation**: Pattern-specific validation with `src/validation/` framework
4. **Integration**: Connect pattern confidence to ensemble signal filtering

**API Integration**: Serve unified ensemble + pattern verification via FastAPI

### **Deployment**

1. **Configuration**: Ensemble + pattern detection settings in `config/config.py`
2. **Training**: Multi-ticker pattern training + strategy backtesting
3. **Serving**: Unified ensemble strategy + pattern confidence API endpoints
4. **Monitoring**: Strategy performance + pattern detection accuracy tracking

---

## Current Development Status

### **Phase 0: Architecture Refactoring (COMPLETED)**

- **Comprehensive Codebase Audit**: Identified and archived 1,200+ lines of conflicting legacy code
- **Legacy Feature Engineering Removed**: 149-feature system → 17 pattern features
- **Directional Accuracy Eliminated**: Replaced with pattern detection accuracy throughout
- **Complex Models Archived**: Multi-scale LSTM (3.2M params) → Shared backbone (400k params)
- **Clean Pattern Focus**: All components aligned with pattern detection specialist approach

### **Phase 1: Pattern Detection Implementation (READY)**

- **Foundation Complete**: Pattern features, shared backbone LSTM, training infrastructure
- **Target System Ready**: Pattern validation targets (not return prediction)
- **Multi-Ticker Training**: Expanded universe approach for overfitting prevention
- **Enhanced Regularization**: Dropout 0.45, L2 0.006, batch normalization

### **Phase 2: Multi-Task Architecture (NEXT)**

- **Multi-Task LSTM**: Pattern confidence, direction, position sizing heads
- **Meta-Labeling**: Signal filtering for low-confidence patterns
- **MAG7 Specialization**: Transfer learning from expanded universe to MAG7
- **Production Integration**: Walk-forward validation, API endpoints

### **Success Criteria**

**Phase 1**: >55% pattern detection accuracy per pattern type
**Phase 2**: Multi-task architecture with meta-labeling operational
**Phase 3**: MAG7 specialization with production-ready API

---

## Key Technical Decisions

### **Pattern Detection Philosophy**

- **Not Return Prediction**: LSTM validates patterns, doesn't compete with signals
- **Meta-Labeling Focus**: Identify when NOT to trade (low confidence patterns)
- **Pattern Specialization**: Separate accuracy for momentum, volatility, trend, volume patterns

### **Architecture Choices**

**Technical Strategies**:

- **Proven Indicators**: RSI and MACD provide reliable primary signals
- **Ensemble Framework**: Flexible combination of multiple technical strategies
- **Signal Strength Weighting**: Dynamic weighting based on strategy confidence

**LSTM Pattern Verification**:

- **Small Model**: 400k parameters to prevent overfitting (vs. 3.2M legacy)
- **Enhanced Regularization**: Aggressive dropout/L2 for expanded universe training
- **Shared Backbone**: Cross-stock pattern learning with stock-specific adaptation
- **Meta-Labeling Focus**: Pattern confidence for filtering rather than not competing signals

### **Training Strategy**

- **Expanded Universe**: 27 securities for training → MAG7 for specialization
- **Overlapping Sequences**: 20-day lookback, 5-day stride for swing trading
- **Pattern Targets**: Binary validation not returns
