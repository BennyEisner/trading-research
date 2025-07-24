#!/usr/bin/env python3

"""
Main entry point for the ML API server
Requires Python 3.12+
"""

import sys
import uvicorn
from pathlib import Path

# Verify Python version
if sys.version_info < (3, 12):
    raise RuntimeError("This application requires Python 3.12 or higher")

# Add the current directory to the Python path
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

if __name__ == "__main__":
    # Import the app factory from the ml-api module
    from ml_api.app import create_app
    
    app = create_app()
    
    # Run the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )