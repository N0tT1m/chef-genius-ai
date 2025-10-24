#!/usr/bin/env python3
"""
Test script for data pipeline monitoring functionality
"""

import time
import random
from collections import deque

# Mock the DataPipelineMonitor for testing
class MockDataPipelineMonitor:
    """Mock data pipeline monitor for testing."""
    
    def __init__(self):
        self.data_loading_times = deque(maxlen=1000)
        self.collation_times = deque(maxlen=1000)
        self.samples_per_second = deque(maxlen=100)
        self.cache_hits = 0
        self.cache_misses = 0
        self.data_loading_errors = 0
        self.corrupted_samples = 0
        
        self.thresholds = {
            'slow_data_loading': 2.0,
            'slow_collation': 0.5,
            'low_throughput': 10,
            'high_memory_usage': 0.85,
        }
    
    def record_data_loading_time(self, duration: float, batch_size: int):
        """Record data loading performance."""
        self.data_loading_times.append(duration)
        if batch_size > 0:
            self.samples_per_second.append(batch_size / max(duration, 0.001))
        print(f"📊 Data loading: {duration:.3f}s for batch size {batch_size}")
    
    def record_collation_time(self, duration: float, batch_size: int):
        """Record data collation performance."""
        self.collation_times.append(duration)
        print(f"⚙️  Data collation: {duration:.3f}s for batch size {batch_size}")
    
    def record_cache_hit(self):
        """Record cache hit."""
        self.cache_hits += 1
    
    def record_cache_miss(self):
        """Record cache miss."""
        self.cache_misses += 1
    
    def get_bottleneck_analysis(self):
        """Analyze current bottlenecks."""
        analysis = {
            'bottlenecks': [],
            'recommendations': [],
            'severity': 'normal'
        }
        
        # Check data loading performance
        if self.data_loading_times:
            avg_loading = sum(self.data_loading_times) / len(self.data_loading_times)
            if avg_loading > self.thresholds['slow_data_loading']:
                analysis['bottlenecks'].append({
                    'type': 'slow_data_loading',
                    'severity': 'high',
                    'description': f"Data loading is slow: {avg_loading:.2f}s average"
                })
                analysis['recommendations'].extend([
                    "Increase DataLoader num_workers",
                    "Use faster storage (SSD)"
                ])
        
        # Check collation performance
        if self.collation_times:
            avg_collation = sum(self.collation_times) / len(self.collation_times)
            if avg_collation > self.thresholds['slow_collation']:
                analysis['bottlenecks'].append({
                    'type': 'slow_collation',
                    'severity': 'medium',
                    'description': f"Data collation is slow: {avg_collation:.2f}s average"
                })
                analysis['recommendations'].append("Optimize data collation function")
        
        # Check throughput
        if self.samples_per_second:
            current_throughput = sum(list(self.samples_per_second)[-5:]) / min(5, len(self.samples_per_second))
            if current_throughput < self.thresholds['low_throughput']:
                analysis['bottlenecks'].append({
                    'type': 'low_throughput',
                    'severity': 'high',
                    'description': f"Low throughput: {current_throughput:.1f} samples/sec"
                })
        
        # Determine overall severity
        if any(b['severity'] == 'high' for b in analysis['bottlenecks']):
            analysis['severity'] = 'high'
        elif any(b['severity'] == 'medium' for b in analysis['bottlenecks']):
            analysis['severity'] = 'medium'
        
        return analysis
    
    def generate_performance_report(self):
        """Generate a performance report."""
        analysis = self.get_bottleneck_analysis()
        
        # Calculate metrics
        avg_loading = sum(self.data_loading_times) / len(self.data_loading_times) if self.data_loading_times else 0
        avg_collation = sum(self.collation_times) / len(self.collation_times) if self.collation_times else 0
        avg_throughput = sum(self.samples_per_second) / len(self.samples_per_second) if self.samples_per_second else 0
        
        total_requests = self.cache_hits + self.cache_misses
        cache_hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        
        report = ["=" * 60]
        report.append("DATA PIPELINE PERFORMANCE REPORT")
        report.append("=" * 60)
        
        report.append("\n📊 PERFORMANCE METRICS:")
        report.append(f"  • Avg Data Loading Time: {avg_loading:.3f}s")
        report.append(f"  • Avg Collation Time: {avg_collation:.3f}s")
        report.append(f"  • Current Throughput: {avg_throughput:.1f} samples/sec")
        report.append(f"  • Cache Hit Rate: {cache_hit_rate*100:.1f}%")
        report.append(f"  • Data Errors: {self.data_loading_errors}")
        
        if analysis['bottlenecks']:
            report.append(f"\n⚠️  BOTTLENECKS DETECTED ({analysis['severity'].upper()}):")
            for bottleneck in analysis['bottlenecks']:
                report.append(f"  • {bottleneck['description']} ({bottleneck['severity']} severity)")
        
        if analysis['recommendations']:
            report.append("\n💡 RECOMMENDATIONS:")
            for rec in set(analysis['recommendations']):
                report.append(f"  • {rec}")
        
        report.append("\n" + "=" * 60)
        return "\n".join(report)

def simulate_training_with_monitoring():
    """Simulate training with data pipeline monitoring."""
    print("🧪 Testing Data Pipeline Monitoring System")
    print("=" * 50)
    
    monitor = MockDataPipelineMonitor()
    
    # Simulate some training steps
    for epoch in range(3):
        print(f"\n📅 Epoch {epoch + 1}")
        print("-" * 20)
        
        for step in range(5):
            # Simulate varying performance
            batch_size = 32
            
            # Data loading time (sometimes slow)
            loading_time = random.uniform(0.1, 3.5)  # Some steps will be slow
            monitor.record_data_loading_time(loading_time, batch_size)
            
            # Collation time (usually fast, sometimes slow)
            collation_time = random.uniform(0.05, 0.8)  # Some will exceed threshold
            monitor.record_collation_time(collation_time, batch_size)
            
            # Cache hits/misses
            if random.random() > 0.2:  # 80% hit rate
                monitor.record_cache_hit()
            else:
                monitor.record_cache_miss()
            
            time.sleep(0.1)  # Brief pause for demo
        
        # Check for bottlenecks
        analysis = monitor.get_bottleneck_analysis()
        if analysis['bottlenecks']:
            print(f"\n⚠️  Issues detected in epoch {epoch + 1}:")
            for bottleneck in analysis['bottlenecks']:
                print(f"   • {bottleneck['description']}")
    
    # Generate final report
    print("\n" + "=" * 50)
    print("📋 FINAL PERFORMANCE REPORT")
    print("=" * 50)
    
    report = monitor.generate_performance_report()
    print(report)
    
    # Test recommendations
    analysis = monitor.get_bottleneck_analysis()
    if analysis['recommendations']:
        print(f"\n🎯 NEXT STEPS:")
        print("Consider implementing these optimizations:")
        for i, rec in enumerate(set(analysis['recommendations']), 1):
            print(f"   {i}. {rec}")
    
    print("\n✅ Data pipeline monitoring test completed!")

def test_argument_parsing():
    """Test that new command line arguments work."""
    import argparse
    
    print("\n🔧 Testing Command Line Arguments")
    print("-" * 30)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--enable-pipeline-monitoring", action="store_true")
    parser.add_argument("--pipeline-report-interval", type=int, default=5)
    parser.add_argument("--dataloader-num-workers", type=int)
    
    # Test with pipeline monitoring enabled
    test_args = [
        "--enable-pipeline-monitoring",
        "--pipeline-report-interval", "3", 
        "--dataloader-num-workers", "12"
    ]
    
    args = parser.parse_args(test_args)
    
    print(f"✅ Pipeline monitoring enabled: {args.enable_pipeline_monitoring}")
    print(f"✅ Report interval: {args.pipeline_report_interval} epochs")
    print(f"✅ DataLoader workers: {args.dataloader_num_workers}")

if __name__ == "__main__":
    test_argument_parsing()
    simulate_training_with_monitoring()