import { useState, useEffect } from 'react';
import { fetchTickers } from '../services/api';
import type { Ticker } from '../types';

interface TickerSelectProps {
    onTickerSelect: (ticker: Ticker | null) => void;
}

const TickerSelector = ({ onTickerSelect }: TickerSelectProps) => {
    const [tickers, setTickers] = useState<Ticker[]>([]);
    const [selectedTicker, setSelectedTicker] = useState<Ticker | null>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const loadTickers = async () => {
        try {
            const fetchedTickers = await fetchTickers();
            setTickers(fetchedTickers);
      } catch (error) {
            console.error('Failed to load tickers:', error);
      } finally {
            setLoading(false);
      }
    };

    loadTickers();
  }, []);

  const handleTickerSelect = (ticker: Ticker) => {
    const newSelection = selectedTicker?.symbol === ticker.symbol ? null : ticker;
    setSelectedTicker(newSelection);
    onTickerSelect(newSelection);
    };

if (loading) {
    return (
      <div>
        <div>Loading tickers...</div>
      </div>
    );
  }

  return (
    <div>
      <h2>Select Ticker</h2>
      
      <div>
        {tickers.map((ticker) => (
          <button
            key={ticker.symbol}
            onClick={() => handleTickerSelect(ticker)}
          >
            {ticker.symbol}
          </button>
        ))}
      </div>

      {selectedTicker && (
        <div>
          <div>Selected:</div>
          <div>{selectedTicker.symbol}</div>
          {selectedTicker.name && (
            <div>{selectedTicker.name}</div>
          )}
        </div>
      )}
    </div>
  );
};

export default TickerSelector;