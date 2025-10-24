# ML Production Monitoring Guide

A comprehensive guide to monitoring machine learning models in production, with focus on local deployment options.

## Overview

Production ML monitoring is different from training monitoring. Instead of tracking loss curves and gradient norms, you need to monitor:

- **Model Performance**: Accuracy, precision, recall on live data
- **Data Drift**: Changes in input distribution over time
- **Model Drift**: Changes in model predictions for similar inputs
- **System Performance**: Latency, throughput, error rates
- **Business Metrics**: User satisfaction, conversion rates

## Local Monitoring Solutions

### 1. Evidently AI (Recommended)

**Best for**: Comprehensive ML monitoring with minimal setup

#### Installation
```bash
pip install evidently
```

#### Basic Usage
```python
from evidently import ColumnMapping
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab, NumTargetDriftTab
from evidently.model_monitoring import ModelMonitor
import pandas as pd

# Load your reference data (training set sample)
reference_data = pd.read_csv('training_sample.csv')

# Current inference data
current_data = pd.read_csv('recent_predictions.csv')

# Generate drift report
dashboard = Dashboard(tabs=[
    DataDriftTab(), 
    NumTargetDriftTab(),
    ClassificationPerformanceTab()  # if classification
])

dashboard.calculate(reference_data, current_data)
dashboard.save("monitoring_report.html")
```

#### Real-time Monitoring Setup
```python
from evidently.model_monitoring import ModelMonitor
from evidently.monitors import DataDriftMonitor, NumTargetDriftMonitor

# Set up continuous monitoring
monitor = ModelMonitor(
    monitors=[
        DataDriftMonitor(),
        NumTargetDriftMonitor(),
        ClassificationPerformanceMonitor()
    ]
)

# In your inference pipeline
def predict_and_monitor(input_data):
    prediction = model.predict(input_data)
    
    # Log for monitoring
    monitor.log(input_data, prediction)
    
    return prediction
```

#### For Recipe Model Specifically
```python
# Monitor recipe generation quality
from evidently.metrics import TextDescriptorsDistribution

# Track recipe characteristics
recipe_monitor = Dashboard(tabs=[
    DataDriftTab(),  # Ingredient distributions
    TextDescriptorsDistribution()  # Recipe text quality
])

# Custom metrics for recipes
def track_recipe_quality(generated_recipes, user_ratings):
    metrics = {
        'avg_recipe_length': np.mean([len(r) for r in generated_recipes]),
        'avg_user_rating': np.mean(user_ratings),
        'ingredient_diversity': calculate_ingredient_diversity(generated_recipes)
    }
    return metrics
```

### 2. WhyLogs

**Best for**: Lightweight, privacy-preserving monitoring

#### Installation
```bash
pip install whylogs
```

#### Basic Usage
```python
import whylogs as why
import pandas as pd

# Initialize logging session
session = why.logger()

# Log your inference data
def log_inference(input_data, predictions, actuals=None):
    # Combine input and output for profiling
    log_data = pd.DataFrame({
        'input_features': input_data,
        'predictions': predictions,
        'actuals': actuals if actuals else [None] * len(predictions)
    })
    
    profile = session.log_dataframe(log_data)
    return profile

# Generate local reports
profile = log_inference(inputs, predictions)
profile.view().to_html("whylogs_profile.html")

# Save profiles for comparison
profile.write("profiles/")
```

#### Drift Detection with WhyLogs
```python
from whylogs.core.statistics.constraints import SummaryConstraints
from whylogs.core.statistics.constraints.factories import greater_than_number

# Set up constraints for monitoring
constraints = SummaryConstraints(
    greater_than_number(column_name="prediction_confidence", number=0.8)
)

# Monitor in real-time
for batch in inference_batches:
    profile = session.log_dataframe(batch)
    validation_result = constraints.validate(profile.view())
    
    if not validation_result.passed:
        send_alert("Model confidence dropped below threshold")
```

### 3. Grafana + Prometheus Stack

**Best for**: Integration with existing infrastructure monitoring

#### Docker Compose Setup
```yaml
version: '3.8'
services:
  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'

  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-storage:/var/lib/grafana

volumes:
  grafana-storage:
```

#### Prometheus Configuration (prometheus.yml)
```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'ml-model'
    static_configs:
      - targets: ['localhost:8000']  # Your model API
```

#### Custom Metrics in Your Application
```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# Define metrics
PREDICTION_COUNT = Counter('ml_predictions_total', 'Total predictions made')
PREDICTION_LATENCY = Histogram('ml_prediction_duration_seconds', 'Prediction latency')
MODEL_ACCURACY = Gauge('ml_model_accuracy', 'Current model accuracy')
DATA_DRIFT_SCORE = Gauge('ml_data_drift_score', 'Data drift score')

# Instrument your model
@PREDICTION_LATENCY.time()
def predict_with_monitoring(input_data):
    start_time = time.time()
    
    prediction = model.predict(input_data)
    
    # Update metrics
    PREDICTION_COUNT.inc()
    
    # Calculate and update accuracy periodically
    if should_update_accuracy():
        current_accuracy = calculate_recent_accuracy()
        MODEL_ACCURACY.set(current_accuracy)
    
    return prediction

# Start metrics server
start_http_server(8000)
```

## Implementation Strategy for Recipe Model

### 1. Key Metrics to Monitor

```python
class RecipeModelMonitor:
    def __init__(self):
        self.metrics = {
            'generation_latency': [],
            'recipe_coherence_scores': [],
            'ingredient_usage_patterns': {},
            'user_ratings': [],
            'error_rates': []
        }
    
    def track_generation(self, input_ingredients, generated_recipe, user_rating=None):
        # Track ingredient distributions
        self.update_ingredient_patterns(input_ingredients)
        
        # Score recipe coherence
        coherence_score = self.score_recipe_coherence(generated_recipe)
        self.metrics['recipe_coherence_scores'].append(coherence_score)
        
        # Track user satisfaction
        if user_rating:
            self.metrics['user_ratings'].append(user_rating)
    
    def detect_drift(self):
        # Implement custom drift detection
        current_ingredient_dist = self.get_recent_ingredient_distribution()
        baseline_dist = self.load_baseline_distribution()
        
        drift_score = self.calculate_distribution_distance(
            current_ingredient_dist, 
            baseline_dist
        )
        
        return drift_score > self.drift_threshold
```

### 2. Alerting Setup

```python
import smtplib
from email.mime.text import MIMEText

class AlertManager:
    def __init__(self, email_config):
        self.email_config = email_config
        self.alert_thresholds = {
            'accuracy_drop': 0.1,
            'latency_increase': 2.0,
            'error_rate': 0.05,
            'drift_score': 0.3
        }
    
    def check_and_alert(self, metrics):
        alerts = []
        
        if metrics['accuracy'] < (self.baseline_accuracy - self.alert_thresholds['accuracy_drop']):
            alerts.append("Model accuracy dropped significantly")
        
        if metrics['avg_latency'] > (self.baseline_latency * self.alert_thresholds['latency_increase']):
            alerts.append("Response latency increased significantly")
        
        if alerts:
            self.send_alerts(alerts)
    
    def send_alerts(self, alerts):
        message = "\n".join(alerts)
        msg = MIMEText(message)
        msg['Subject'] = 'ML Model Alert'
        msg['From'] = self.email_config['from']
        msg['To'] = self.email_config['to']
        
        # Send email (configure SMTP server)
        with smtplib.SMTP(self.email_config['smtp_server']) as server:
            server.send_message(msg)
```

### 3. Dashboard Setup

For Evidently AI dashboard:
```python
# Create comprehensive monitoring dashboard
def create_recipe_dashboard(reference_data, current_data):
    dashboard = Dashboard(tabs=[
        DataDriftTab(),
        NumTargetDriftTab(),
        RegressionPerformanceTab(),  # If scoring recipes
        DataQualityTab()
    ])
    
    # Add custom recipe-specific metrics
    custom_metrics = {
        'avg_recipe_length': np.mean([len(r) for r in current_data['recipe']]),
        'ingredient_diversity': calculate_diversity(current_data['ingredients']),
        'cuisine_distribution': current_data['cuisine'].value_counts()
    }
    
    dashboard.calculate(reference_data, current_data)
    dashboard.save("recipe_model_dashboard.html")
    
    return dashboard
```

## Getting Started Checklist

1. **Choose Your Tool**:
   - Start with Evidently AI for comprehensive ML monitoring
   - Use WhyLogs if you need lightweight, privacy-preserving solution
   - Go with Grafana+Prometheus if you have existing infrastructure

2. **Set Up Baseline Data**:
   - Export sample of training data as reference
   - Define key metrics for your model
   - Establish performance thresholds

3. **Instrument Your Code**:
   - Add monitoring calls to your inference pipeline
   - Set up automated report generation
   - Configure alerting thresholds

4. **Test Locally**:
   - Generate sample monitoring reports
   - Verify drift detection works
   - Test alert mechanisms

5. **Deploy Monitoring**:
   - Set up automated report generation schedule
   - Configure log aggregation
   - Establish monitoring dashboard access

## Best Practices

- **Start Simple**: Begin with basic drift detection and performance monitoring
- **Automate Everything**: Set up automated report generation and alerting
- **Monitor Business Metrics**: Track user satisfaction, not just technical metrics
- **Version Your Baselines**: Update reference data as your model evolves
- **Test Your Alerts**: Regularly verify that monitoring and alerting works
- **Document Thresholds**: Clearly document why you chose specific alert thresholds

## Cost Considerations

| Tool | Cost | Pros | Cons |
|------|------|------|------|
| Evidently AI | Free (OSS) | Easy setup, comprehensive | Limited enterprise features |
| WhyLogs | Free (OSS) | Lightweight, privacy-preserving | Requires more setup |
| Grafana+Prometheus | Free (OSS) | Integrates with infrastructure | More complex setup |
| WandB Enterprise | $$ | Advanced features | Expensive for small teams |
| Neptune | $$$ | Enterprise ready | High cost |

For most use cases, start with **Evidently AI** locally and scale up as needed.