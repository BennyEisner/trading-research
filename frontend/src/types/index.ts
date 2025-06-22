// Ticker information
export interface Ticker {
  symbol: string;
  name?: string;
  sector?: string;
}

// Basic types matching your current API response
export interface ReturnData {
  date: string;
  daily_return: number;
}

export interface ApiResponse {
  symbol: string;
  returns: ReturnData[];
}

// Simple date range for API calls
export interface DateRange {
  start_date: string;
  end_date: string;
}

// Basic error handling
export interface ApiError {
  detail: string;
}