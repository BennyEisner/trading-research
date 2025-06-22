'use client';

import React, { useState } from 'react';
import TickerSelector from '../components/TickerSelect';
import DateRange from '../components/DateRange';
import { fetchReturns } from '../services/api';
import type { Ticker, ApiResponse } from '../types';

const App = () => {
  const [selectedTicker, setSelectedTicker] = useState<Ticker | null>(null);
  const [dateRange, setDateRange] = useState<{ startDate: string; endDate: string } | null>(null);
  const [apiResults, setApiResults] = useState<ApiResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Handle ticker selection
  const handleTickerSelect = (ticker: Ticker | null) => {
    setSelectedTicker(ticker);
    setApiResults(null);
    setError(null);
  };

  // Handle date range change
  const handleDateRangeChange = (startDate: string, endDate: string) => {
    setDateRange({ startDate, endDate });
    setApiResults(null);
    setError(null);
  };

  // Handle run query
  const handleRunQuery = async () => {
    if (!selectedTicker || !dateRange) {
      setError('Please select both a ticker and date range');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const results = await fetchReturns(
        selectedTicker.symbol,
        dateRange.startDate,
        dateRange.endDate
      );
      setApiResults(results);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch data');
      setApiResults(null);
    } finally {
      setLoading(false);
    }
  };

  // Check if we can run the query
  const canRunQuery = selectedTicker && dateRange && !loading;

  return (
    <div>
      <h1>Financial Returns Analyzer</h1>
      
      {/* Ticker Selection */}
      <div>
        <TickerSelector onTickerSelect={handleTickerSelect} />
      </div>

      {/* Date Range Selection */}
      <div>
        <DateRange 
          onDateRangeChange={handleDateRangeChange}
          disabled={!selectedTicker}
        />
      </div>

      {/* Run Query Button */}
      <div>
        <button
          onClick={handleRunQuery}
          disabled={!canRunQuery}
        >
          {loading ? 'Loading...' : 'Run Analysis'}
        </button>
      </div>

      {/* Error Display */}
      {error && (
        <div>
          <div>Error</div>
          <div>{error}</div>
        </div>
      )}

      {/* Results Display */}
      {apiResults && (
        <div>
          <h2>Results for {apiResults.symbol}</h2>
          
          <table border={1}>
            <thead>
              <tr>
                <th>Date</th>
                <th>Daily Return</th>
              </tr>
            </thead>
            <tbody>
              {apiResults.returns.map((returnData, index) => (
                <tr key={index}>
                  <td>{returnData.date}</td>
                  <td>{(returnData.daily_return * 100).toFixed(2)}%</td>
                </tr>
              ))}
            </tbody>
          </table>

          {/* Summary Stats */}
          {apiResults.returns.length > 0 && (
            <div>
              <div>
                <div>Total Returns</div>
                <div>{apiResults.returns.length} days</div>
              </div>
              <div>
                <div>Avg Daily Return</div>
                <div>
                  {(apiResults.returns.reduce((sum, r) => sum + r.daily_return, 0) / apiResults.returns.length * 100).toFixed(2)}%
                </div>
              </div>
              <div>
                <div>Date Range</div>
                <div>{dateRange?.startDate} to {dateRange?.endDate}</div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default App;