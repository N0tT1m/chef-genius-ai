package main

import (
	"fmt"
	"sync"
	"time"
)

// TensorRTPool manages a pool of TensorRT engines for concurrent processing
type TensorRTPool struct {
	engines     chan *TensorRTEngine
	maxEngines  int
	modelPath   string
	mu          sync.RWMutex
	closed      bool
	activeCount int
	totalReqs   int64
	totalTime   time.Duration
}

// PoolStats contains statistics about the pool
type PoolStats struct {
	MaxEngines    int           `json:"max_engines"`
	ActiveEngines int           `json:"active_engines"`
	AvailableEngines int        `json:"available_engines"`
	TotalRequests int64         `json:"total_requests"`
	AvgProcessTime time.Duration `json:"avg_process_time"`
	IsClosed      bool          `json:"is_closed"`
}

// NewTensorRTPool creates a new pool of TensorRT engines
func NewTensorRTPool(maxEngines int) (*TensorRTPool, error) {
	if maxEngines <= 0 {
		maxEngines = 1
	}

	// Use the converted TensorRT engine path
	modelPath := "../models/tensorrt/model.trt"

	pool := &TensorRTPool{
		engines:    make(chan *TensorRTEngine, maxEngines),
		maxEngines: maxEngines,
		modelPath:  modelPath,
	}

	// Pre-create engines to avoid cold start latency
	if err := pool.initializeEngines(); err != nil {
		return nil, fmt.Errorf("failed to initialize engines: %v", err)
	}

	return pool, nil
}

// initializeEngines pre-creates all engines in the pool
func (p *TensorRTPool) initializeEngines() error {
	for i := 0; i < p.maxEngines; i++ {
		engine, err := p.createEngine()
		if err != nil {
			// Cleanup any successfully created engines
			p.Close()
			return fmt.Errorf("failed to create engine %d: %v", i, err)
		}
		p.engines <- engine
	}
	return nil
}

// createEngine creates a new TensorRT engine
func (p *TensorRTPool) createEngine() (*TensorRTEngine, error) {
	// For demonstration, we'll create a mock engine since we don't have actual model files
	// In production, use: return NewTensorRTEngine(p.modelPath)
	return p.createMockEngine(), nil
}

// createMockEngine creates a mock engine for testing
func (p *TensorRTPool) createMockEngine() *TensorRTEngine {
	return &TensorRTEngine{
		cEngine:    nil, // Mock engine
		isLoaded:   true,
		modelPath:  p.modelPath,
		inputSize:  512,
		outputSize: 512 * 50257,
	}
}

// Get retrieves an engine from the pool
func (p *TensorRTPool) Get() *TensorRTEngine {
	p.mu.Lock()
	if p.closed {
		p.mu.Unlock()
		return nil
	}
	p.activeCount++
	p.mu.Unlock()

	select {
	case engine := <-p.engines:
		return engine
	case <-time.After(30 * time.Second): // Timeout after 30 seconds
		p.mu.Lock()
		p.activeCount--
		p.mu.Unlock()
		return nil
	}
}

// Put returns an engine to the pool
func (p *TensorRTPool) Put(engine *TensorRTEngine) {
	if engine == nil {
		return
	}

	p.mu.Lock()
	defer p.mu.Unlock()

	if p.closed {
		engine.Close()
		return
	}

	p.activeCount--
	
	select {
	case p.engines <- engine:
		// Successfully returned to pool
	default:
		// Pool is full, close the engine
		engine.Close()
	}
}

// GetStats returns current pool statistics
func (p *TensorRTPool) GetStats() PoolStats {
	p.mu.RLock()
	defer p.mu.RUnlock()

	var avgProcessTime time.Duration
	if p.totalReqs > 0 {
		avgProcessTime = p.totalTime / time.Duration(p.totalReqs)
	}

	return PoolStats{
		MaxEngines:       p.maxEngines,
		ActiveEngines:    p.activeCount,
		AvailableEngines: len(p.engines),
		TotalRequests:    p.totalReqs,
		AvgProcessTime:   avgProcessTime,
		IsClosed:         p.closed,
	}
}

// RecordRequest records statistics for a request
func (p *TensorRTPool) RecordRequest(duration time.Duration) {
	p.mu.Lock()
	defer p.mu.Unlock()
	
	p.totalReqs++
	p.totalTime += duration
}

// Close closes all engines in the pool
func (p *TensorRTPool) Close() {
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.closed {
		return
	}

	p.closed = true

	// Close all engines in the pool
	close(p.engines)
	for engine := range p.engines {
		engine.Close()
	}
}

// Resize dynamically resizes the pool (experimental)
func (p *TensorRTPool) Resize(newSize int) error {
	if newSize <= 0 {
		return fmt.Errorf("pool size must be positive")
	}

	p.mu.Lock()
	defer p.mu.Unlock()

	if p.closed {
		return fmt.Errorf("pool is closed")
	}

	currentSize := len(p.engines)
	
	if newSize > currentSize {
		// Add more engines
		for i := currentSize; i < newSize; i++ {
			engine, err := p.createEngine()
			if err != nil {
				return fmt.Errorf("failed to create engine during resize: %v", err)
			}
			
			select {
			case p.engines <- engine:
				// Successfully added
			default:
				// Channel full, close the engine
				engine.Close()
				break
			}
		}
	} else if newSize < currentSize {
		// Remove engines
		for i := newSize; i < currentSize; i++ {
			select {
			case engine := <-p.engines:
				engine.Close()
			default:
				// No more engines to remove
				break
			}
		}
	}

	p.maxEngines = newSize
	return nil
}

// HealthCheck performs a health check on all engines in the pool
func (p *TensorRTPool) HealthCheck() map[string]interface{} {
	stats := p.GetStats()
	
	health := map[string]interface{}{
		"status":            "healthy",
		"pool_stats":        stats,
		"timestamp":         time.Now(),
	}

	// Check if any engines are available
	if stats.AvailableEngines == 0 && stats.ActiveEngines == 0 {
		health["status"] = "unhealthy"
		health["reason"] = "no engines available"
	}

	// Check if pool is closed
	if stats.IsClosed {
		health["status"] = "closed"
	}

	return health
}

// WarmUp warms up all engines by running test inference
func (p *TensorRTPool) WarmUp() error {
	testReq := RecipeRequest{
		Ingredients: []string{"test"},
		MaxTokens:   10,
		Temperature: 0.5,
	}

	// Get all engines and warm them up
	engines := make([]*TensorRTEngine, 0, p.maxEngines)
	
	// Collect all engines
	for i := 0; i < p.maxEngines; i++ {
		engine := p.Get()
		if engine == nil {
			// Return collected engines
			for _, e := range engines {
				p.Put(e)
			}
			return fmt.Errorf("failed to get engine for warmup")
		}
		engines = append(engines, engine)
	}

	// Warm up each engine
	var wg sync.WaitGroup
	errors := make(chan error, len(engines))

	for i, engine := range engines {
		wg.Add(1)
		go func(engineIdx int, eng *TensorRTEngine) {
			defer wg.Done()
			
			_, err := eng.GenerateRecipe(testReq)
			if err != nil {
				errors <- fmt.Errorf("engine %d warmup failed: %v", engineIdx, err)
			}
		}(i, engine)
	}

	wg.Wait()
	close(errors)

	// Return all engines to pool
	for _, engine := range engines {
		p.Put(engine)
	}

	// Check for errors
	var warmupErrors []error
	for err := range errors {
		warmupErrors = append(warmupErrors, err)
	}

	if len(warmupErrors) > 0 {
		return fmt.Errorf("warmup completed with %d errors: %v", len(warmupErrors), warmupErrors[0])
	}

	return nil
}