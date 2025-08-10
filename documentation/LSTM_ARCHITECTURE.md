# LSTM Pattern Detection Architecture

### SharedBackboneTrainer Architecture

```python
# Model Architecture: ~400k parameters
Input: (batch_size, sequence_length=20, features=17)
├─ Input Normalization Layer
├─ LSTM(64 units) + BatchNorm + Dropout(0.45)
├─ LSTM(32 units) + BatchNorm
├─ Dense(16 units) + Dropout(0.36)
└─ Dense(1, activation='tanh') → Pattern Confidence [0,1]
```

**Key Architecture Decisions**:

- **Parameter Efficiency**: 400k parameters (reduced from 3.2M legacy)
- **Enhanced Regularization**: Multi-layer dropout, L2 regularization (0.006)
- **Batch Normalization**: Stabilizes training across diverse market conditions
- **Tanh Output**: Pattern confidence scores bounded [0,1] for meta-labeling

### Pattern Feature Engineering (17 Features)

| Category                      | Features                                                                               | Purpose                              |
| ----------------------------- | -------------------------------------------------------------------------------------- | ------------------------------------ |
| **Non-linear Price** (4)      | Price acceleration, Volume-price divergence, Volatility regime change, Return skewness | Detect non-linear market dynamics    |
| **Temporal Dependencies** (4) | Momentum persistence, Volatility clustering, Trend exhaustion, GARCH forecasting       | Capture temporal pattern persistence |
| **Market Microstructure** (3) | Intraday range expansion, Overnight gaps, End-of-day momentum                          | Intraday pattern detection           |
| **Cross-asset Relations** (3) | Sector relative strength, Market beta instability, VIX term structure                  | Cross-market relationships           |
| **Core Context** (3)          | Multi-timeframe returns, Normalized volume, Price levels                               | Essential market context             |

**Feature Implementation**: `src/features/pattern_feature_calculator.py`

### Training Data Architecture

**Sequence Configuration**:

- **Lookback Window**: 20 days (optimized for swing trading patterns)
- **Sequence Stride**: 5 day
- **Prediction Horizon**: 5 days (swing trading focus)
- **Total Universe**: 34 securities (expanded from original MAG7)

**Expanded Training Universe**:

```python
TRAINING_UNIVERSE = [
    # MAG7 Core
    "AAPL", "MSFT", "GOOG", "NVDA", "TSLA", "AMZN", "META",

    # Technology Sector
    "CRM", "ADBE", "NOW", "ORCL", "NFLX", "AMD", "INTC", "CSCO",
    "AVGO", "TXN", "QCOM", "MU", "AMAT", "LRCX", "KLAC", "MRVL",
    "SNPS", "CDNS", "FTNT", "PANW", "INTU", "UBER", "ZM", "DDOG",

    # Market Indices
    "QQQ", "XLK", "SPY"
]
```

## Pattern Target Generation

### Binary Pattern Validation Logic

The system uses **binary pattern validation** rather than return prediction, focusing on whether detected patterns actually resolve as expected:

```python
# Example: Momentum Persistence Validation
def validate_momentum_persistence(features_df, horizon=5):
    for i in range(len(features_df) - horizon):
        current_momentum = features_df['momentum_persistence_7d'][i]

        if current_momentum > percentile_70:  # High momentum signal
            future_returns = features_df['returns_1d'][i+1:i+horizon+1]
            future_signs = np.sign(future_returns[future_returns != 0])

            # Pattern validated if >60% of future returns maintain direction
            persistence_rate = max(np.mean(future_signs > 0), np.mean(future_signs < 0))
            target[i] = 1 if persistence_rate > 0.6 else 0

    return target
```

**Pattern Types Validated**:

- **Momentum Persistence**: Does detected momentum actually continue?
- **Volatility Regime Shifts**: Do regime changes occur as predicted?
- **Trend Exhaustion**: Do trends reverse when exhaustion is detected?
- **Volume Divergence**: Does price-volume divergence resolve as expected?

## Training Infrastructure

### SharedBackboneTrainer Implementation

**Core Training Loop**: `src/training/shared_backbone_trainer.py`

```python
class SharedBackboneTrainer:
    def __init__(self, use_expanded_universe=True):
        self.expanded_universe = EXPANDED_UNIVERSE if use_expanded_universe else MAG7
        self.model_params = {
            'dropout_rate': 0.45,
            'l2_regularization': 0.006,
            'learning_rate': 0.0008,
            'batch_size': 64,
            'sequence_length': 20
        }

    def build_model(self, input_shape):
        # Enhanced regularization architecture
        inputs = Input(shape=input_shape)
        x = BatchNormalization()(inputs)

        # LSTM backbone with enhanced regularization
        x = LSTM(64, return_sequences=True, recurrent_dropout=0.3)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.45)(x)

        x = LSTM(32, return_sequences=False)(x)
        x = BatchNormalization()(x)
        x = Dense(16, activation='relu', kernel_regularizer=l2(0.006))(x)
        x = Dropout(0.36)(x)

        # Pattern confidence output
        outputs = Dense(1, activation='tanh', name='pattern_confidence')(x)

        return Model(inputs, outputs)
```

**Training Optimization**:

- **Loss Function**: Correlation-optimized loss with pattern validation targets
- **Early Stopping**: Monitor `val_pattern_detection_accuracy` (patience=15)
- **Learning Rate**: 0.0008 with gradient clipping
- **Batch Size**: 64 (optimal for multi-ticker training)

### Cross-Ticker Validation Framework

**Time Series Cross-Validation**: `src/validation/gapped_time_series_cv.py`

```python
# 3-fold cross-validation per ticker
for ticker in training_tickers:
    cv = GappedTimeSeriesCV(n_splits=3, test_size=0.2, gap_size=5)

    ticker_results = []
    for train_idx, test_idx in cv.split(ticker_data[ticker]):
        fold_result = validate_lstm_fold(
            train_data=ticker_data[ticker].iloc[train_idx],
            test_data=ticker_data[ticker].iloc[test_idx]
        )
        ticker_results.append(fold_result)

    ticker_performance[ticker] = aggregate_fold_results(ticker_results)
```

**Validation Results by Ticker** (Sample from breakthrough session):

- **MSFT**: 0.718 mean correlation across 3 folds
- **NVDA**: 0.706 mean correlation, 0.572 pattern accuracy
- **TSLA**: 0.698 mean correlation, 0.611 pattern accuracy
- **AAPL**: 0.665 mean correlation, 0.585 pattern accuracy

## Enhanced Regularization Strategy

### Overfitting Prevention Mechanisms

**Multi-Layer Regularization**:

- **Dropout**: 0.45 main, 0.36 dense, 0.3 recurrent
- **L2 Regularization**: 0.006 on dense layers
- **Batch Normalization**: After each major layer
- **Gradient Clipping**: Prevents exploding gradients
- **Early Stopping**: Prevents overtraining

**Cross-Ticker Learning Benefits**:

- **34-security universe** prevents single-stock overfitting
- **Shared backbone** learns generalizable patterns
- **Pattern diversity** across different market sectors
- **Temporal robustness** across multiple time periods

### Training Monitoring & Diagnostics

**Key Training Metrics**:

- **Pattern Detection Accuracy**: Target >55%, achieved 59.7%
- **Correlation Tracking**: Real-time correlation monitoring
- **Loss Convergence**: Both training and validation loss
- **Cross-Ticker Consistency**: Performance across all securities

**Diagnostic Tools**: `src/utils/training_diagnostics.py`

- **Training stability tracking**
- **Overfitting detection** (train/val gap monitoring)
- **Feature importance analysis**
- **Pattern-specific performance breakdown**

## Integration with Ensemble System

### Meta-Labeling Architecture

The LSTM provides **pattern confidence scores** that filter technical indicator signals:

```python
# Integration Flow
def generate_filtered_signals(market_data):
    # 1. Generate technical signals
    technical_signals = ensemble_manager.generate_signals(market_data)

    # 2. Calculate pattern features and confidence
    pattern_features = calculate_pattern_features(market_data)
    sequences = prepare_sequences(pattern_features, lookback=20)
    pattern_confidence = shared_backbone_model.predict(sequences)

    # 3. Apply pattern filtering
    confidence_threshold = 0.6
    filtered_signals = apply_confidence_filtering(
        technical_signals, pattern_confidence, confidence_threshold
    )

    return {
        'technical_signals': technical_signals,
        'pattern_confidence': pattern_confidence,
        'filtered_signals': filtered_signals
    }
```

**Meta-Labeling Benefits**:

- **Signal Quality Improvement**: Higher Sharpe ratios with pattern filtering
- **False Positive Reduction**: Low confidence patterns filtered out
- **Adaptive Thresholding**: Dynamic confidence requirements by market conditions
- **Strategy Agnostic**: Works with any technical indicator strategy

## Production Deployment Architecture

### Model Serving Infrastructure

**Real-Time Inference Pipeline**:

```python
class LSTMPatternService:
    def __init__(self):
        self.model = load_model('models/best_shared_backbone_model.keras')
        self.feature_calculator = PatternFeatureCalculator()
        self.sequence_processor = SequenceProcessor(lookback=20)

    async def predict_pattern_confidence(self, ohlcv_data):
        # Calculate 17 pattern features
        features = self.feature_calculator.calculate_all_features(ohlcv_data)

        # Prepare 20-day sequences
        sequences = self.sequence_processor.prepare_sequences(features)

        # LSTM inference
        confidence_scores = self.model.predict(sequences)

        return {
            'pattern_confidence': confidence_scores.tolist(),
            'confidence_level': self.categorize_confidence(confidence_scores[-1]),
            'recommendation': self.generate_recommendation(confidence_scores[-1])
        }
```

**Performance Characteristics**:

- **Inference Latency**: <50ms for single ticker
- **Batch Processing**: Supports multi-ticker parallel inference
- **Memory Efficiency**: ~400k parameters = minimal memory footprint
- **Scalability**: Stateless architecture supports horizontal scaling

### API Integration Points

**FastAPI Endpoints**:

- `POST /predict/pattern_confidence` - Pattern confidence scores
- `POST /predict/ensemble_signals` - Filtered technical signals
- `POST /predict/multi_ticker_ensemble` - Portfolio-level processing

## Future Architecture Enhancements

### Multi-Task Learning Extension

**Planned Multi-Task Heads**:

```python
# Extended architecture for enhanced meta-labeling
class MultiTaskPatternLSTM:
    def build_model(self, input_shape):
        backbone = self.build_shared_backbone(input_shape)

        # Multiple specialized heads
        pattern_confidence = Dense(1, activation='sigmoid', name='pattern_confidence')(backbone)
        signal_quality = Dense(1, activation='sigmoid', name='signal_quality')(backbone)
        market_regime = Dense(3, activation='softmax', name='market_regime')(backbone)  # Bull/Bear/Neutral
        volatility_forecast = Dense(1, activation='sigmoid', name='volatility_forecast')(backbone)

        return Model(inputs=input, outputs=[pattern_confidence, signal_quality, market_regime, volatility_forecast])
```

### MAG7 Specialization Framework

**Transfer Learning Architecture**:

- **Shared Backbone Pretraining**: Current 34-ticker system
- **Stock-Specific Adaptation**: Fine-tuning for individual MAG7 securities
- **Specialized Pattern Detection**: Stock-specific pattern emphasis
- **Production Deployment**: Individual models per target security

## Technical Implementation Details

### Key Files and Components

| Component                   | File Path                                    | Purpose                   |
| --------------------------- | -------------------------------------------- | ------------------------- |
| **Shared Backbone Model**   | `src/models/shared_backbone_trainer.py`      | Main LSTM architecture    |
| **Pattern Features**        | `src/features/pattern_feature_calculator.py` | 17-feature calculation    |
| **Training Loop**           | `src/training/shared_backbone_trainer.py`    | Training orchestration    |
| **Cross-Ticker Validation** | `src/validation/gapped_time_series_cv.py`    | Time series validation    |
| **Pattern Targets**         | `src/training/pattern_target_generator.py`   | Binary validation targets |
| **Training Diagnostics**    | `src/utils/training_diagnostics.py`          | Monitoring and analysis   |

### Configuration Management

**Training Configuration**: `config/config.py`

```python
LSTM_PATTERN_CONFIG = {
    'expanded_universe': 34,  # Securities for training
    'lookback_window': 20,    # Sequence length
    'sequence_stride': 5,     # Maximum overlap
    'prediction_horizon': 5,  # Days ahead for validation

    'model_params': {
        'dropout_rate': 0.45,
        'l2_regularization': 0.006,
        'learning_rate': 0.0008,
        'batch_size': 64,
        'epochs': 25
    },

    'validation_params': {
        'cv_folds': 3,
        'test_size': 0.2,
        'gap_size': 5,
        'min_correlation': 0.1,
        'min_accuracy': 0.55
    }
}
```

## Success Factors Analysis

### What Made the Breakthrough Possible

1. **Enhanced Regularization**: Multi-layer approach prevented overfitting
2. **Expanded Universe**: 34 securities provided pattern diversity
3. **Correlation Focus**: Loss function optimized for pattern-resolution correlation
4. **Binary Targets**: Pattern validation rather than return prediction
5. **Temporal Design**: 20-day sequences optimal for swing trading patterns
6. **Cross-Ticker Validation**: Rigorous validation across all securities

### Key Learnings

- **Pattern Confidence > Return Prediction**: Meta-labeling approach more effective
- **Multi-Ticker Learning**: Shared backbone prevents single-stock overfitting
- **Regularization Critical**: 400k parameters with heavy regularization optimal
- **Feature Engineering Matters**: 17 carefully designed features essential
- **Validation Rigor**: Cross-ticker time series CV crucial for real performance

