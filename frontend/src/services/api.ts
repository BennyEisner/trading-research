import type { ApiResponse, DateRange, Ticker } from "../types";

const API_BASE_URL = "http://localhost:8000";

// Fetch all Tickers
export const fetchTickers = async (): Promise<Ticker[]> => {
  const response = await fetch(`${API_BASE_URL}/tickers`);
  if (!response.ok) {
    throw new Error(`Failed to fetch Tickers`);
  }
  return response.json();
};

// Fetch specific Ticker
export const fetchTicker = async (symbol: string): Promise<Ticker> => {
  const response = await fetch(`${API_BASE_URL}/tickers/${symbol}`);
  if (!response.ok) {
    throw new Error(`Failed to fetch ${symbol} Ticker`);
  }
  return response.json();
};
// Fetch returns for a single ticker
export const fetchReturns = async (
  ticker: string,
  startDate: string,
  endDate: string,
): Promise<ApiResponse> => {
  // Convert MM-DD-YYYY to YYYY-MM-DD format
  const formatDate = (dateStr: string) => {
    const [month, day, year] = dateStr.split('-');
    return `${year}-${month.padStart(2, '0')}-${day.padStart(2, '0')}`;
  };
  
  const formattedStartDate = formatDate(startDate);
  const formattedEndDate = formatDate(endDate);
  
  const response = await fetch(
    `${API_BASE_URL}/prices/${ticker}/${formattedStartDate}/${formattedEndDate}`,
  );

  if (!response.ok) {
    throw new Error(`Failed to fetch data for ${ticker}: ${response.status}`);
  }

  return response.json();
};

// Fetch returns for multiple tickers
export const fetchMultipleReturns = async (
  tickers: string[],
  startDate: string,
  endDate: string,
): Promise<ApiResponse[]> => {
  const promises = tickers.map((ticker) =>
    fetchReturns(ticker, startDate, endDate).catch((error) => {
      console.warn(`Failed to fetch ${ticker}:`, error.message);
      return null;
    }),
  );

  const results = await Promise.all(promises);
  return results.filter((result): result is ApiResponse => result !== null);
};

// Available tickers
export const AVAILABLE_TICKERS = [
  "AAPL",
  "MSFT",
  "GOOG",
  "AMZN",
  "META",
  "TSLA",
  "NVDA",
];

export const formatPercentage = (value: number): string => {
  return `${(value * 100).toFixed(2)}%`;
};
