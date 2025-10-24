package main

import (
	"context"
	"encoding/json"
	"log"
	"net/http"
	"os"
	"os/signal"
	"runtime"
	"strconv"
	"sync"
	"syscall"
	"time"

	"github.com/gofiber/fiber/v2"
	"github.com/gofiber/fiber/v2/middleware/compress"
	"github.com/gofiber/fiber/v2/middleware/cors"
	"github.com/gofiber/fiber/v2/middleware/limiter"
	"github.com/gofiber/fiber/v2/middleware/logger"
	"github.com/gofiber/fiber/v2/middleware/monitor"
	"github.com/gofiber/fiber/v2/middleware/recover"
	"github.com/valyala/fasthttp"
)

// Global pool for TensorRT engines
var enginePool *TensorRTPool

func main() {
	// Initialize TensorRT engine pool
	pool, err := NewTensorRTPool(4) // 4 concurrent engines
	if err != nil {
		log.Fatal("Failed to initialize TensorRT pool:", err)
	}
	enginePool = pool
	defer pool.Close()

	// Configure Fiber for maximum performance
	app := fiber.New(fiber.Config{
		ServerHeader:          "ChefGenius-TensorRT-API",
		AppName:               "ChefGenius TensorRT API v1.0",
		DisableStartupMessage: false,
		Prefork:               runtime.NumCPU() > 1, // Enable prefork for multi-core
		CaseSensitive:         true,
		StrictRouting:         true,
		BodyLimit:             4 * 1024 * 1024, // 4MB max body size
		ReadTimeout:           5 * time.Second,
		WriteTimeout:          10 * time.Second,
		IdleTimeout:           120 * time.Second,
		JSONEncoder:           json.Marshal,
		JSONDecoder:           json.Unmarshal,
	})

	// Middleware stack for performance and reliability
	setupMiddleware(app)

	// API routes
	setupRoutes(app)

	// Graceful shutdown
	c := make(chan os.Signal, 1)
	signal.Notify(c, os.Interrupt, syscall.SIGTERM)

	go func() {
		<-c
		log.Println("Gracefully shutting down...")
		_ = app.Shutdown()
	}()

	// Start server
	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	log.Printf("🚀 Server starting on port %s", port)
	log.Fatal(app.Listen(":" + port))
}

func setupMiddleware(app *fiber.App) {
	// Recovery middleware
	app.Use(recover.New())

	// CORS
	app.Use(cors.New(cors.Config{
		AllowOrigins: "*",
		AllowMethods: "GET,POST,OPTIONS",
		AllowHeaders: "Origin,Content-Type,Accept,Authorization",
	}))

	// Compression for better performance
	app.Use(compress.New(compress.Config{
		Level: compress.LevelBestSpeed, // Fast compression
	}))

	// Rate limiting
	app.Use(limiter.New(limiter.Config{
		Max:        100,              // 100 requests
		Expiration: 1 * time.Minute,  // per minute
		KeyGenerator: func(c *fiber.Ctx) string {
			return c.IP()
		},
		LimitReached: func(c *fiber.Ctx) error {
			return c.Status(429).JSON(fiber.Map{
				"error": "Rate limit exceeded",
				"retry_after": "60s",
			})
		},
	}))

	// Request logging
	app.Use(logger.New(logger.Config{
		Format: "${time} ${status} - ${method} ${path} - ${latency}\n",
	}))

	// Performance monitoring endpoint
	app.Get("/metrics", monitor.New(monitor.Config{Title: "ChefGenius TensorRT Metrics"}))
}

func setupRoutes(app *fiber.App) {
	// Health check
	app.Get("/health", func(c *fiber.Ctx) error {
		return c.JSON(fiber.Map{
			"status":    "healthy",
			"timestamp": time.Now().Unix(),
			"version":   "1.0.0",
		})
	})

	api := app.Group("/api/v1")

	// Recipe generation endpoint
	api.Post("/generate", generateRecipe)
	
	// Batch generation endpoint for multiple requests
	api.Post("/generate/batch", generateRecipesBatch)
	
	// Get model info
	api.Get("/model/info", getModelInfo)
	
	// Performance benchmark
	api.Post("/benchmark", runBenchmark)
}

type RecipeRequest struct {
	Ingredients []string `json:"ingredients" validate:"required,min=1"`
	Cuisine     string   `json:"cuisine,omitempty"`
	DietaryReqs []string `json:"dietary_requirements,omitempty"`
	MaxTokens   int      `json:"max_tokens,omitempty"`
	Temperature float32  `json:"temperature,omitempty"`
}

type RecipeResponse struct {
	Recipe      string  `json:"recipe"`
	Confidence  float32 `json:"confidence"`
	ProcessTime float64 `json:"process_time_ms"`
	TokensUsed  int     `json:"tokens_used"`
}

type BatchRequest struct {
	Requests []RecipeRequest `json:"requests" validate:"required,min=1,max=10"`
}

type BatchResponse struct {
	Responses   []RecipeResponse `json:"responses"`
	TotalTime   float64          `json:"total_time_ms"`
	ProcessedAt time.Time        `json:"processed_at"`
}

func generateRecipe(c *fiber.Ctx) error {
	start := time.Now()
	
	var req RecipeRequest
	if err := c.BodyParser(&req); err != nil {
		return c.Status(400).JSON(fiber.Map{
			"error": "Invalid request body",
			"details": err.Error(),
		})
	}

	// Validate request
	if len(req.Ingredients) == 0 {
		return c.Status(400).JSON(fiber.Map{
			"error": "At least one ingredient is required",
		})
	}

	// Set defaults
	if req.MaxTokens == 0 {
		req.MaxTokens = 512
	}
	if req.Temperature == 0 {
		req.Temperature = 0.7
	}

	// Get engine from pool
	engine := enginePool.Get()
	defer enginePool.Put(engine)

	// Generate recipe
	result, err := engine.GenerateRecipe(req)
	if err != nil {
		return c.Status(500).JSON(fiber.Map{
			"error": "Recipe generation failed",
			"details": err.Error(),
		})
	}

	processingTime := float64(time.Since(start).Nanoseconds()) / 1e6

	response := RecipeResponse{
		Recipe:      result.Recipe,
		Confidence:  result.Confidence,
		ProcessTime: processingTime,
		TokensUsed:  result.TokensUsed,
	}

	return c.JSON(response)
}

func generateRecipesBatch(c *fiber.Ctx) error {
	start := time.Now()
	
	var batchReq BatchRequest
	if err := c.BodyParser(&batchReq); err != nil {
		return c.Status(400).JSON(fiber.Map{
			"error": "Invalid batch request body",
			"details": err.Error(),
		})
	}

	if len(batchReq.Requests) == 0 {
		return c.Status(400).JSON(fiber.Map{
			"error": "At least one request is required",
		})
	}

	if len(batchReq.Requests) > 10 {
		return c.Status(400).JSON(fiber.Map{
			"error": "Maximum 10 requests per batch",
		})
	}

	// Process requests concurrently
	responses := make([]RecipeResponse, len(batchReq.Requests))
	var wg sync.WaitGroup
	var mu sync.Mutex

	for i, req := range batchReq.Requests {
		wg.Add(1)
		go func(index int, request RecipeRequest) {
			defer wg.Done()
			
			requestStart := time.Now()
			engine := enginePool.Get()
			defer enginePool.Put(engine)

			// Set defaults
			if request.MaxTokens == 0 {
				request.MaxTokens = 512
			}
			if request.Temperature == 0 {
				request.Temperature = 0.7
			}

			result, err := engine.GenerateRecipe(request)
			
			mu.Lock()
			if err != nil {
				responses[index] = RecipeResponse{
					Recipe:      "Error: " + err.Error(),
					Confidence:  0.0,
					ProcessTime: float64(time.Since(requestStart).Nanoseconds()) / 1e6,
					TokensUsed:  0,
				}
			} else {
				responses[index] = RecipeResponse{
					Recipe:      result.Recipe,
					Confidence:  result.Confidence,
					ProcessTime: float64(time.Since(requestStart).Nanoseconds()) / 1e6,
					TokensUsed:  result.TokensUsed,
				}
			}
			mu.Unlock()
		}(i, req)
	}

	wg.Wait()

	totalTime := float64(time.Since(start).Nanoseconds()) / 1e6

	batchResponse := BatchResponse{
		Responses:   responses,
		TotalTime:   totalTime,
		ProcessedAt: time.Now(),
	}

	return c.JSON(batchResponse)
}

func getModelInfo(c *fiber.Ctx) error {
	engine := enginePool.Get()
	defer enginePool.Put(engine)

	info := engine.GetModelInfo()
	return c.JSON(info)
}

func runBenchmark(c *fiber.Ctx) error {
	iterations := 100
	if iter := c.Query("iterations"); iter != "" {
		if i, err := strconv.Atoi(iter); err == nil && i > 0 && i <= 1000 {
			iterations = i
		}
	}

	engine := enginePool.Get()
	defer enginePool.Put(engine)

	results := engine.Benchmark(iterations)
	return c.JSON(results)
}