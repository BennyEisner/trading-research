'use client';

import { useEffect, useState } from 'react';

interface ReturnData {
  date: string;
  daily_return: number;
}

interface ApiResponse {
  symbol: string;
  returns: ReturnData[];
}

export default function Home() {
  const [data, setData] = useState<ApiResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      const tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META', 'TSLA', 'NVDA'];
      const startDate = '2024-07-01';
      const endDate = '2024-08-01';
      
      try {
        const promises = tickers.map(ticker => 
          fetch(`http://localhost:8000/prices/${ticker}/${startDate}/${endDate}`)
            .then(res => res.json())
        );
        
        const results = await Promise.all(promises);
        setData(results);
      } catch (err) {
        setError('Failed to fetch data');
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  if (loading) return <div>Loading...</div>;
  if (error) return <div>Error: {error}</div>;

  return (
    <div style={{ padding: '20px' }}>
      <h1>Financial Returns Data</h1>
      
      {data.map(stockData => (
        <div key={stockData.symbol} style={{ marginBottom: '30px' }}>
          <h2>{stockData.symbol}</h2>
          <table border={1} style={{ borderCollapse: 'collapse', width: '100%' }}>
            <thead>
              <tr>
                <th style={{ padding: '8px' }}>Date</th>
                <th style={{ padding: '8px' }}>Daily Return (%)</th>
              </tr>
            </thead>
            <tbody>
              {stockData.returns.map((returnData, index) => (
                <tr key={index}>
                  <td style={{ padding: '8px' }}>{returnData.date}</td>
                  <td style={{ padding: '8px' }}>
                    {(returnData.daily_return * 100).toFixed(2)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ))}
    </div>
  );
}