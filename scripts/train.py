#!/usr/bin/env python3
"""
Combine all log files from data/logs/ folder and train a model
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from log_analyzer import LogAnalyzer, train_model

def combine_log_files(logs_dir: Path, output_file: Path):
    """Combine all log files into one, skipping headers."""
    print(f"Combining log files from {logs_dir}...")
    
    # Support both .txt and .log files
    txt_files = sorted(logs_dir.glob("*.txt"))
    log_files_ext = sorted(logs_dir.glob("*.log"))
    log_files = txt_files + log_files_ext
    print(f"Found {len(txt_files)} .txt files and {len(log_files_ext)} .log files ({len(log_files)} total)")
    
    total_lines = 0
    with output_file.open('w', encoding='utf-8') as out:
        for log_file in log_files:
            with log_file.open('r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()
                # Skip header line if it exists
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    # Skip header lines
                    if line.startswith('TIMESTAMP|'):
                        continue
                    # Skip empty lines
                    if not line or line.count('|') < 5:
                        continue
                    out.write(line + '\n')
                    total_lines += 1
    
    print(f"Combined {total_lines} log lines into {output_file}")
    return total_lines

def main():
    project_root = Path(__file__).parent.parent
    logs_dir = project_root / "data" / "logs"
    models_dir = project_root / "data" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    combined_log = project_root / "data" / "combined_logs.txt"
    model_path = models_dir / "trained_model.pkl"
    
    if not logs_dir.exists():
        print(f"Error: {logs_dir} directory not found!")
        return 1
    
    # Combine all log files
    total_lines = combine_log_files(logs_dir, combined_log)
    
    if total_lines == 0:
        print("Error: No log lines found!")
        return 1
    
    print(f"\nTraining model on {total_lines} combined log entries...")
    print("=" * 80)
    
    # Train the model
    result = train_model(
        log_file=combined_log,
        model_path=model_path,
        model_type='random_forest',
        use_rule_labels=True,
        test_size=0.2,
        max_features=2000
    )
    
    # Clean up temporary combined log
    if combined_log.exists():
        combined_log.unlink()
    
    return result

if __name__ == "__main__":
    sys.exit(main())

