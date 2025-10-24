# ChefGenius TensorRT API Makefile

.PHONY: build run test clean deps docker benchmark

# Variables
BINARY_NAME=chef-genius-tensorrt-api
DOCKER_IMAGE=chef-genius-tensorrt-api
VERSION?=1.0.0
LDFLAGS=-ldflags "-X main.Version=$(VERSION)"

# Build the application
build:
	@echo "Building $(BINARY_NAME)..."
	CGO_ENABLED=1 go build $(LDFLAGS) -o bin/$(BINARY_NAME) .
	@echo "Build complete: bin/$(BINARY_NAME)"

# Build for production (optimized)
build-prod:
	@echo "Building $(BINARY_NAME) for production..."
	CGO_ENABLED=1 GOOS=linux GOARCH=amd64 go build -a -installsuffix cgo $(LDFLAGS) -o bin/$(BINARY_NAME) .
	@echo "Production build complete: bin/$(BINARY_NAME)"

# Run the application
run: build
	@echo "Starting $(BINARY_NAME)..."
	./bin/$(BINARY_NAME)

# Run in development mode with hot reload
dev:
	@echo "Starting development server..."
	go run . --env=development

# Install dependencies
deps:
	@echo "Installing dependencies..."
	go mod download
	go mod tidy

# Run tests
test:
	@echo "Running tests..."
	go test -v ./...

# Run benchmarks
benchmark:
	@echo "Running benchmarks..."
	go test -bench=. -benchmem ./...

# Performance test
perf-test: build
	@echo "Running performance tests..."
	./bin/$(BINARY_NAME) &
	sleep 2
	curl -X POST http://localhost:8080/api/v1/benchmark?iterations=100
	pkill $(BINARY_NAME)

# Build Docker image
docker-build:
	@echo "Building Docker image..."
	docker build -t $(DOCKER_IMAGE):$(VERSION) .
	docker tag $(DOCKER_IMAGE):$(VERSION) $(DOCKER_IMAGE):latest

# Run Docker container
docker-run: docker-build
	@echo "Running Docker container..."
	docker run -p 8080:8080 --gpus all $(DOCKER_IMAGE):latest

# Clean build artifacts
clean:
	@echo "Cleaning..."
	rm -rf bin/
	go clean

# Check TensorRT and CUDA installation
check-env:
	@echo "Checking environment..."
	@echo "CUDA version:"
	@nvcc --version || echo "CUDA not found"
	@echo "TensorRT libraries:"
	@ldconfig -p | grep tensorrt || echo "TensorRT not found"
	@echo "Go version:"
	@go version

# Format code
fmt:
	@echo "Formatting code..."
	go fmt ./...

# Lint code
lint:
	@echo "Linting code..."
	golangci-lint run

# Generate code coverage
coverage:
	@echo "Generating code coverage..."
	go test -coverprofile=coverage.out ./...
	go tool cover -html=coverage.out -o coverage.html

# Profile CPU
profile-cpu: build
	@echo "CPU profiling..."
	./bin/$(BINARY_NAME) --profile=cpu &
	sleep 10
	curl -X POST http://localhost:8080/api/v1/generate -d '{"ingredients":["chicken","rice"]}'
	pkill $(BINARY_NAME)

# Profile memory
profile-mem: build
	@echo "Memory profiling..."
	./bin/$(BINARY_NAME) --profile=mem &
	sleep 10
	curl -X POST http://localhost:8080/api/v1/generate -d '{"ingredients":["chicken","rice"]}'
	pkill $(BINARY_NAME)

# Load test with wrk
load-test: build
	@echo "Load testing..."
	./bin/$(BINARY_NAME) &
	sleep 2
	wrk -t12 -c400 -d30s -s loadtest.lua http://localhost:8080/api/v1/generate
	pkill $(BINARY_NAME)

# Install tools
install-tools:
	@echo "Installing development tools..."
	go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest
	go install github.com/cosmtrek/air@latest

# Create deployment package
package: build-prod
	@echo "Creating deployment package..."
	mkdir -p dist
	cp bin/$(BINARY_NAME) dist/
	cp -r configs dist/ || true
	tar -czf dist/$(BINARY_NAME)-$(VERSION).tar.gz -C dist .
	@echo "Package created: dist/$(BINARY_NAME)-$(VERSION).tar.gz"

# Help
help:
	@echo "Available commands:"
	@echo "  build       - Build the application"
	@echo "  build-prod  - Build for production"
	@echo "  run         - Build and run the application"
	@echo "  dev         - Run in development mode"
	@echo "  deps        - Install dependencies"
	@echo "  test        - Run tests"
	@echo "  benchmark   - Run benchmarks"
	@echo "  perf-test   - Run performance tests"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run  - Run Docker container"
	@echo "  clean       - Clean build artifacts"
	@echo "  check-env   - Check TensorRT/CUDA environment"
	@echo "  fmt         - Format code"
	@echo "  lint        - Lint code"
	@echo "  coverage    - Generate code coverage"
	@echo "  profile-cpu - CPU profiling"
	@echo "  profile-mem - Memory profiling"
	@echo "  load-test   - Load test with wrk"
	@echo "  package     - Create deployment package"
	@echo "  help        - Show this help"