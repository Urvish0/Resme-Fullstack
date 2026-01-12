from prometheus_client import Counter, Gauge, Histogram

### API Metrics ###

API_REQUESTS_TOTAL = Counter(
    "api_request_total",
    "Total number of API requests",
    ["endpoint", "method", "status"],
)

API_ERRORS_TOTAL = Counter(
    "api_errors_total",
    "Total number of API errors",
    ["endpoint", "method"],
)

API_REQUEST_DURATION = Histogram(
    "api_request_duration_seconds",
    "API request latency",
    ["endpoint", "method"],
    buckets=(0.1, 0.3, 0.5, 1, 2, 5),
)

### Async Job Metrics ###

JOBS_ACTIVE = Gauge(
    "jobs_active",
    "Current number of active async jobs",
)

JOBS_STARTED_TOTAL = Counter(
    "jobs_started_total",
    "Total number of async jobs started",
)

JOBS_COMPLETED_TOTAL = Counter(
    "jobs_completed_total",
    "Total number of async jobs completed successfully",
)

JOBS_FAILED_TOTAL = Counter(
    "jobs_failed_total",
    "Total number of async jobs failed",
)

JOB_DURATION_SECONDS = Histogram(
    "job_duration_seconds",
    "Async job execution duration",
    buckets=(1, 2, 5, 10, 30, 60, 120, 300),
)