# Multi-stage build for optimized TensorRT Go API
FROM nvidia/cuda:12.0-devel-ubuntu22.04 AS builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    build-essential \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install Go
RUN wget https://go.dev/dl/go1.21.5.linux-amd64.tar.gz && \
    tar -C /usr/local -xzf go1.21.5.linux-amd64.tar.gz && \
    rm go1.21.5.linux-amd64.tar.gz

ENV PATH="/usr/local/go/bin:${PATH}"

# Install TensorRT
RUN wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/8.6.1/tars/tensorrt-8.6.1.6.linux.x86_64-gnu.cuda-12.0.tar.gz && \
    tar -xzf tensorrt-8.6.1.6.linux.x86_64-gnu.cuda-12.0.tar.gz && \
    cp -r TensorRT-8.6.1.6/include/* /usr/local/include/ && \
    cp -r TensorRT-8.6.1.6/lib/* /usr/local/lib/ && \
    ldconfig && \
    rm -rf TensorRT-8.6.1.6 tensorrt-8.6.1.6.linux.x86_64-gnu.cuda-12.0.tar.gz

# Set working directory
WORKDIR /app

# Copy go mod files
COPY go.mod go.sum ./

# Download dependencies
RUN go mod download

# Copy source code
COPY . .

# Build the application
RUN CGO_ENABLED=1 GOOS=linux GOARCH=amd64 go build -a -installsuffix cgo -o chef-genius-tensorrt-api .

# Final stage - Runtime
FROM nvidia/cuda:12.0-runtime-ubuntu22.04

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy TensorRT libraries from builder
COPY --from=builder /usr/local/lib/libnvinfer* /usr/local/lib/
COPY --from=builder /usr/local/lib/libnvonnxparser* /usr/local/lib/
COPY --from=builder /usr/local/lib/libnvinfer_plugin* /usr/local/lib/

# Update library cache
RUN ldconfig

# Create non-root user
RUN groupadd -r app && useradd -r -g app app

# Set working directory
WORKDIR /app

# Copy binary from builder
COPY --from=builder /app/chef-genius-tensorrt-api .

# Create directories for models and logs
RUN mkdir -p models logs && chown -R app:app /app

# Switch to non-root user
USER app

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Set environment variables
ENV CUDA_VISIBLE_DEVICES=0
ENV PORT=8080
ENV GIN_MODE=release

# Run the application
CMD ["./chef-genius-tensorrt-api"]