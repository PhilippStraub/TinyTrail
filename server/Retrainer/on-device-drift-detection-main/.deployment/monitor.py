#!/usr/bin/env python3
"""
Live monitoring tool for TinyLCM drift detection on Raspberry Pi
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
import psutil
import subprocess

# Add project to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ANSI color codes
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
MAGENTA = '\033[95m'
CYAN = '\033[96m'
WHITE = '\033[97m'
RESET = '\033[0m'
BOLD = '\033[1m'
CLEAR_SCREEN = '\033[2J\033[H'

def get_system_stats():
    """Get current system statistics"""
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    
    # Try to get temperature (Pi specific)
    try:
        temp_output = subprocess.check_output(['vcgencmd', 'measure_temp'], text=True)
        temp = float(temp_output.split('=')[1].split("'")[0])
    except:
        temp = None
    
    return {
        'cpu': cpu_percent,
        'memory': memory.percent,
        'temp': temp
    }

def find_latest_log(scenario_path):
    """Find the latest log file in scenario directory"""
    log_dir = scenario_path / "logs"
    if not log_dir.exists():
        return None
    
    log_files = list(log_dir.glob("*.log"))
    if not log_files:
        return None
    
    return max(log_files, key=lambda f: f.stat().st_mtime)

def parse_log_line(line):
    """Parse a log line and extract relevant information"""
    info = {}
    
    # Check for drift detection
    if "DRIFT DETECTED" in line:
        info['drift'] = True
        info['type'] = 'drift_detected'
    elif "Drift ended" in line:
        info['drift'] = False
        info['type'] = 'drift_ended'
    
    # Extract classification results
    if "Predicted:" in line:
        parts = line.split("Predicted:")[1].strip()
        if "(" in parts:
            label = parts.split("(")[0].strip()
            confidence = float(parts.split("(")[1].split(")")[0])
            info['prediction'] = {'label': label, 'confidence': confidence}
    
    # Extract distance information
    if "distance:" in line:
        try:
            dist = float(line.split("distance:")[1].split()[0])
            info['distance'] = dist
        except:
            pass
    
    # Extract Page-Hinkley values
    if "PH_t=" in line:
        try:
            ph_value = float(line.split("PH_t=")[1].split()[0])
            info['ph_value'] = ph_value
        except:
            pass
    
    return info

def monitor_scenario(scenario_name):
    """Monitor a specific scenario"""
    scenario_path = PROJECT_ROOT / "examples" / "python_examples" / f"scenario_{scenario_name}_drift"
    
    if not scenario_path.exists():
        print(f"{RED}Scenario path not found: {scenario_path}{RESET}")
        return
    
    print(f"{CLEAR_SCREEN}")
    print(f"{BOLD}{CYAN}═══════════════════════════════════════════════════════{RESET}")
    print(f"{BOLD}{WHITE}   TinyLCM Drift Detection Monitor - Scenario {scenario_name.title()}{RESET}")
    print(f"{BOLD}{CYAN}═══════════════════════════════════════════════════════{RESET}\n")
    
    # Find latest log file
    log_file = find_latest_log(scenario_path)
    if not log_file:
        print(f"{YELLOW}No log files found. Waiting for scenario to start...{RESET}")
        while not log_file:
            time.sleep(1)
            log_file = find_latest_log(scenario_path)
    
    print(f"{GREEN}Monitoring: {log_file.name}{RESET}\n")
    
    # Track state
    drift_active = False
    last_prediction = None
    last_distance = None
    last_ph = None
    frame_count = 0
    drift_count = 0
    
    # Open log file and follow
    with open(log_file, 'r') as f:
        # Go to end of file
        f.seek(0, 2)
        
        while True:
            line = f.readline()
            
            if line:
                frame_count += 1
                info = parse_log_line(line)
                
                # Update state
                if 'drift' in info:
                    drift_active = info['drift']
                    if drift_active:
                        drift_count += 1
                
                if 'prediction' in info:
                    last_prediction = info['prediction']
                
                if 'distance' in info:
                    last_distance = info['distance']
                
                if 'ph_value' in info:
                    last_ph = info['ph_value']
                
                # Clear and redraw
                print(f"{CLEAR_SCREEN}")
                print(f"{BOLD}{CYAN}═══════════════════════════════════════════════════════{RESET}")
                print(f"{BOLD}{WHITE}   TinyLCM Drift Detection Monitor - Scenario {scenario_name.title()}{RESET}")
                print(f"{BOLD}{CYAN}═══════════════════════════════════════════════════════{RESET}\n")
                
                # System stats
                stats = get_system_stats()
                print(f"{BOLD}System Status:{RESET}")
                print(f"  CPU: {stats['cpu']:.1f}%  Memory: {stats['memory']:.1f}%", end="")
                if stats['temp']:
                    color = RED if stats['temp'] > 70 else YELLOW if stats['temp'] > 60 else GREEN
                    print(f"  Temp: {color}{stats['temp']:.1f}°C{RESET}")
                else:
                    print()
                print()
                
                # Drift status
                if drift_active:
                    print(f"{RED}{BOLD}⚠️  DRIFT DETECTED ⚠️{RESET}")
                else:
                    print(f"{GREEN}✓ Normal Operation{RESET}")
                print()
                
                # Current prediction
                if last_prediction:
                    conf_color = GREEN if last_prediction['confidence'] > 0.8 else YELLOW if last_prediction['confidence'] > 0.5 else RED
                    print(f"{BOLD}Last Prediction:{RESET}")
                    print(f"  Class: {MAGENTA}{last_prediction['label']}{RESET}")
                    print(f"  Confidence: {conf_color}{last_prediction['confidence']:.3f}{RESET}")
                    print()
                
                # Distance metrics
                if last_distance is not None:
                    dist_color = RED if last_distance > 15 else YELLOW if last_distance > 10 else GREEN
                    print(f"{BOLD}k-NN Distance:{RESET} {dist_color}{last_distance:.3f}{RESET}")
                
                if last_ph is not None:
                    ph_color = RED if last_ph > 8 else YELLOW if last_ph > 5 else GREEN
                    print(f"{BOLD}Page-Hinkley:{RESET} {ph_color}{last_ph:.3f}{RESET}")
                print()
                
                # Statistics
                print(f"{BOLD}Statistics:{RESET}")
                print(f"  Frames: {frame_count}")
                print(f"  Drift Events: {drift_count}")
                print(f"  Time: {datetime.now().strftime('%H:%M:%S')}")
                print()
                print(f"{CYAN}Press Ctrl+C to exit{RESET}")
                
            else:
                time.sleep(0.1)

def main():
    """Main monitoring function"""
    print(f"{BOLD}{CYAN}TinyLCM Drift Detection Monitor{RESET}")
    print(f"{CYAN}══════════════════════════════{RESET}\n")
    
    # Check which scenarios are available
    scenarios = []
    for scenario in ['lego', 'ball']:
        path = PROJECT_ROOT / "examples" / "python_examples" / f"scenario_{scenario}_drift"
        if path.exists():
            scenarios.append(scenario)
    
    if not scenarios:
        print(f"{RED}No scenarios found!{RESET}")
        sys.exit(1)
    
    if len(scenarios) == 1:
        scenario = scenarios[0]
    else:
        print("Available scenarios:")
        for i, s in enumerate(scenarios, 1):
            print(f"  {i}. Scenario {s.title()} Drift")
        
        choice = input("\nSelect scenario (1-{}): ".format(len(scenarios)))
        try:
            scenario = scenarios[int(choice) - 1]
        except:
            print(f"{RED}Invalid choice{RESET}")
            sys.exit(1)
    
    try:
        monitor_scenario(scenario)
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Monitoring stopped{RESET}")
    except Exception as e:
        print(f"\n{RED}Error: {e}{RESET}")
        sys.exit(1)

if __name__ == "__main__":
    main()