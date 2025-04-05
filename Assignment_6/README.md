# Assignment-06 EP21B030

This application collects system metrics from  `/proc/meminfo` and the `iostat` command, making them available via an HTTP endpoint for Prometheus to scrape.

### Requirements
- Flask

### Configuring Prometheus

Add the following to your Prometheus YAML file:

```yaml
job_name: "custom_metrics"
scrape_interval: 2s
static_configs:
  - targets: ["localhost:18000"]
```

### Running the application

```bash
python exporter.py
```

The application will:
- Start collecting iostat and meminfo metrics every second
- Make metrics available at http://localhost:18000/metrics
- Provide a health check at http://localhost:18000/health

### Available Metrics

#### CPU Metrics

- `cpu_avg_percent{mode="user"}` - CPU time spent in user mode
- `cpu_avg_percent{mode="nice"}` - CPU time spent in user mode with low priority 
- `cpu_avg_percent{mode="system"}` - CPU time spent in system mode
- `cpu_avg_percent{mode="iowait"}` - CPU time spent waiting for I/O operations
- `cpu_avg_percent{mode="idle"}` - CPU idle time

#### I/O Metrics

For each device (e.g., sda, nvme0n1):

- `io_read_rate{device="<device>"}` - Read operations per second (in bytes/sec)
- `io_write_rate{device="<device>"}` - Write operations per second (in bytes/sec)
- `io_tps{device="<device>"}` - Transfers per second
- `io_read_bytes{device="<device>"}` - Total bytes read
- `io_write_bytes{device="<device>"}` - Total bytes written

#### Memory Metrics

All metrics from `/proc/meminfo` are exposed with the prefix `meminfo_`, for example:

- `meminfo_memtotal` - Total usable RAM
- `meminfo_memfree` - Flree memory
- `meminfo_buffers` - Memory used by kernel buffers
- `meminfo_cached` - Memory used for page cache
- Many other memory-related metrics