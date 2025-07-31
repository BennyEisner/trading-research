#!/usr/bin/env python3

"""
Database verification and data quality assessment
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from src.config.config import Config
from src.data.data_loader import DataLoader  
from src.utils.logging_utils import setup_production_logger
from src.validation.pipeline_validator import create_pipeline_validator

def verify_database_and_data():
    """Comprehensive database and data verification"""
    
    config = Config()
    logger = setup_production_logger({"log_file": "database_verification.log"})
    validator = create_pipeline_validator(logger)
    
    logger.log("=" * 50)
    logger.log("DATABASE VERIFICATION STARTING")
    logger.log("=" * 50)
    
    # Database should be in the root API directory (where pipeline/run.py creates it)
    expected_db_path = "/Users/beneisner/financial-returns-api/returns.db"
    
    logger.log(f"Checking for database at: {expected_db_path}")
    
    if not os.path.exists(expected_db_path):
        logger.log("Database file not found!")
        logger.log("To create the database with data, run:")
        logger.log("  cd /Users/beneisner/financial-returns-api")
        logger.log("  source .venv/bin/activate")
        logger.log("  python -m pipeline.run")
        return False, {"error": "Database file not found", "solution": "Run pipeline.run to populate database"}
    
    logger.log(f"Database found at: {expected_db_path}")
    
    # Update config to point to correct database location
    config.config["database_url"] = f"sqlite:///{expected_db_path}"
    logger.log(f"Database URL: {config.config['database_url']}")
    
    try:
        # 1. Test database connection using the API models
        logger.log("Testing database connection...")
        data_loader = DataLoader(config)
        logger.log("Database connection successful")
        
        # 2. Check available tickers
        tickers = config.get("tickers")
        logger.log(f"Target tickers: {tickers}")
        
        ticker_stats = {}
        for ticker in tickers:
            try:
                # Load sample data
                data = data_loader.load_single_ticker_data(ticker, 1)  # 1 year
                
                # Validate data quality
                is_valid, issues = validator.validate_raw_data(data, ticker)
                
                ticker_stats[ticker] = {
                    "records": len(data),
                    "date_range": f"{data['date'].min()} to {data['date'].max()}",
                    "is_valid": is_valid,
                    "issues": len(issues)
                }
                
                logger.log(f"{ticker}: {len(data)} records, valid: {is_valid}")
                if issues:
                    for issue in issues[:2]:  # Show first 2 issues
                        logger.log(f"   ⚠️  {issue}")
                
            except Exception as e:
                ticker_stats[ticker] = {
                    "error": str(e),
                    "is_valid": False
                }
                logger.log(f"❌ {ticker}: {e}")
        
        # 3. Summary
        valid_tickers = [t for t, stats in ticker_stats.items() 
                        if stats.get("is_valid", False)]
        
        logger.log("\n" + "=" * 50)
        logger.log("VERIFICATION SUMMARY")
        logger.log("=" * 50)
        logger.log(f"Valid tickers: {len(valid_tickers)}/{len(tickers)}")
        logger.log(f"Ready for training: {valid_tickers}")
        
        if len(valid_tickers) >= len(tickers) * 0.7:  # 70% success rate
            logger.log("DATABASE READY FOR TRAINING")
            return True, ticker_stats
        else:
            logger.log("❌ INSUFFICIENT DATA FOR TRAINING")
            return False, ticker_stats
            
    except Exception as e:
        logger.log(f"❌ DATABASE VERIFICATION FAILED: {e}")
        return False, {"error": str(e)}

if __name__ == "__main__":
    success, stats = verify_database_and_data()
    if success:
        print("Database verification passed - Ready for training")
    else:
        print("❌ Database verification failed - Check logs")
    sys.exit(0 if success else 1)