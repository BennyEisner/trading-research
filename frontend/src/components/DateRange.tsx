import { useState } from 'react';

interface DateRangeProps {
  onDateRangeChange: (startDate: string, endDate: string) => void;
  disabled?: boolean;
}

const DateRange = ({ onDateRangeChange, disabled = false }: DateRangeProps) => {
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');

const validateAndNotify = (start: string, end: string) => {
  if (start && end) {
    try {
      const startDateObj = new Date(start);
      const endDateObj = new Date(end);
      
      if (!isNaN(startDateObj.getTime()) && 
          !isNaN(endDateObj.getTime()) && 
          startDateObj <= endDateObj) {
        onDateRangeChange(start, end);
      }
    } catch (error) {
      console.warn('Invalid date format');
    }
  }
};

const handleStartDateChange = (date: string) => {
  setStartDate(date);
  validateAndNotify(date, endDate);
};

const handleEndDateChange = (date: string) => {
  setEndDate(date);
  validateAndNotify(startDate, date);
};
            
  
  return (
    <div>
      <h2>Select Date Range</h2>
      
      <div>
        <div>
            <input  id="end-date" vallue = {startDate} onChange = { (e) => handleStartDateChange(e.target.value)} disabled={disabled}/>
        </div>

        <div>
            <input  id="start-date" vallue = {endDate} onChange = { (e) => handleEndDateChange(e.target.value)} disabled={disabled}/>
        </div>
      </div>

      <div>
          {startDate && endDate ? (
            <p>Selected Range: {startDate} to {endDate}</p>
          ) : (
            <p>Please select both start and end dates</p>
          )}
          
          {/* Show error if start date is after end date */}
          {startDate && endDate && new Date(startDate) > new Date(endDate) && (
            <p style={{color: 'red'}}>Start date must be before or equal to end date</p>
          )}
        </div>   
    </div>
  );
};

export default DateRange;
