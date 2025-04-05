import os
import time
import re
from flask import Flask, Response
import subprocess
import threading
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('metrics-exporter')

app = Flask(__name__)

METRICS_DIR = 'metrics'
IOSTAT_FILE = os.path.join(METRICS_DIR, 'iostat.txt')
MEMINFO_FILE = os.path.join(METRICS_DIR, 'meminfo.txt')

os.makedirs(METRICS_DIR, exist_ok=True)

def collect_iostat():
    """
    Function to collect iostat metrics and write to file every second
    """
    while True:
        try:
            result = subprocess.run(['iostat', '-k'],
                                    capture_output=True, text=True, check=True)

            with open(IOSTAT_FILE, 'w') as f:
                f.write(result.stdout)

            logger.debug("iostat metrics collected")
        except Exception as e:
            logger.error(f"Error collecting iostat metrics: {e}")
        
        time.sleep(1)

def collect_meminfo():
    """
    Function to collect meminfo metrics and write to file every second
    """
    while True:
        try:
            with open('/proc/meminfo', 'r') as src:
                meminfo_content = src.read()

            with open(MEMINFO_FILE, 'w') as f:
                f.write(meminfo_content)

            logger.debug("meminfo metrics collected")
        except Exception as e:
            logger.error(f"Error collecting meminfo metrics: {e}")
        
        time.sleep(1)

def parse_iostat_metrics():
    """
    Parse the iostat file and return metrics in a Prometheus compatible format
    """
    try:
        if not os.path.exists(IOSTAT_FILE):
            return []
        
        with open(IOSTAT_FILE, 'r') as f:
            content = f.read()
        
        # Parse iostat output
        metrics = []
        
        # Extract CPU stats for cpu_avg_percent
        cpu_match = re.search(r'avg-cpu:.*?\n\s*(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)', 
                              content, re.DOTALL)
        if cpu_match:
            user, nice, system, iowait, _, idle = cpu_match.groups()
            # Export CPU metrics in the requested format
            metrics.append(f'cpu_avg_percent{{mode="user"}} {user}')
            metrics.append(f'cpu_avg_percent{{mode="nice"}} {nice}')
            metrics.append(f'cpu_avg_percent{{mode="system"}} {system}')
            metrics.append(f'cpu_avg_percent{{mode="iowait"}} {iowait}')
            metrics.append(f'cpu_avg_percent{{mode="idle"}} {idle}')
        
        # Extract device stats
        device_section = re.search(r'Device.*\s+tps\s+kB_read/s\s+kB_wrtn/s\s+kB_dscd/s\s+kB_read\s+kB_wrtn\s+kB_dscd\n([\s\S]*)', content)
        if device_section:
            device_lines = device_section.group(1).strip().split('\n')
            for line in device_lines:
                parts = line.split()
                if len(parts) >= 7:
                    device = parts[0]
                    # Skip if it doesn't look like a device
                    if not (device.startswith('sd') or device.startswith('xvd') or 
                            device.startswith('nvme') or device.startswith('hd')):
                        continue
                    
                    # Read rate (convert kB/s to bytes/s)
                    read_rate = float(parts[2]) * 1024
                    metrics.append(f'io_read_rate{{device="{device}"}} {read_rate}')
                    
                    # Write rate (convert kB/s to bytes/s)
                    write_rate = float(parts[3]) * 1024
                    metrics.append(f'io_write_rate{{device="{device}"}} {write_rate}')
                    
                    # Total operations per second
                    tps = float(parts[1])
                    metrics.append(f'io_tps{{device="{device}"}} {tps}')
                    
                    # Read bytes (convert kB to bytes)
                    read_bytes = float(parts[5]) * 1024
                    metrics.append(f'io_read_bytes{{device="{device}"}} {read_bytes}')
                    
                    # Write bytes(convert kB to bytes)
                    write_bytes = float(parts[6]) * 1024
                    metrics.append(f'io_write_bytes{{device="{device}"}} {write_bytes}')
        
        return metrics
    except Exception as e:
        logger.error(f"Error parsing iostat metrics: {e}")
        return []

def parse_meminfo_metrics():
    """
    Parse the meminfo file and return metrics in a Prometheus compatible format
    """
    try:
        if not os.path.exists(MEMINFO_FILE):
            return []
        
        with open(MEMINFO_FILE, 'r') as f:
            content = f.readlines()
        
        metrics = []
        
        # Process each line of meminfo
        for line in content:
            # Skip empty lines
            if not line.strip():
                continue
                
            # Parse the line - typical format is "MemTotal:        8167344 kB"
            parts = line.split(':')
            if len(parts) < 2:
                continue
                
            key = parts[0].strip()
            value_part = parts[1].strip()
            
            # Extract the numerical value and unit
            value_match = re.match(r'(\d+)\s*(\w*)', value_part)
            if not value_match:
                continue
                
            value = value_match.group(1)
            unit = value_match.group(2).lower() if value_match.group(2) else ""
            
            # Convert to bytes if unit is kB
            if unit == "kb":
                value = str(int(value) * 1024)
            
            # Create the metric name: convert spaces and parentheses to underscores, lowercase
            metric_name = f"meminfo_{key.lower().replace(' ', '_').replace('(', '_').replace(')', '_')}"
            
            metrics.append(f'{metric_name} {value}')
        
        return metrics
    except Exception as e:
        logger.error(f"Error parsing meminfo metrics: {e}")
        return []

@app.route('/metrics')
def metrics():
    """
    Endpoint to expose iostat metrics
    """
    all_metrics = parse_iostat_metrics() + parse_meminfo_metrics()
    response = f"# Timestamp: {time.time()}\n" + "\n".join(all_metrics) + "\n"
    return Response(response, mimetype='text/plain')

@app.route('/health')
def health():
    """
    Health check endpoint
    """
    return "OK"

if __name__ == '__main__':
    # Start iostat collection in a background thread
    iostat_thread = threading.Thread(target=collect_iostat, daemon=True)
    iostat_thread.start()
    
    # Start meminfo collection in a background thread
    meminfo_thread = threading.Thread(target=collect_meminfo, daemon=True)
    meminfo_thread.start()

    app.run(host='0.0.0.0', port=18000)