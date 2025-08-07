"""
Command Line Interface for ARL Feature Selector

This module provides the entry point for the command-line interface.
The actual implementation is in the arl_pipeline.py script.
"""

import sys
import os
from pathlib import Path

def main():
    """
    Main entry point for the CLI.

    This function imports and runs the ARL pipeline script.
    """
    # Add the project root to the path so we can import arl_pipeline
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    try:
        # Import and run the ARL pipeline
        from arl_pipeline import main as pipeline_main
        return pipeline_main()
    except ImportError as e:
        print(f"Error importing ARL pipeline: {e}")
        print("Please ensure arl_pipeline.py is in the project root directory.")
        return 1
    except Exception as e:
        print(f"Error running ARL pipeline: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
