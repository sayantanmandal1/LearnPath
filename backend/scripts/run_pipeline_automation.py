#!/usr/bin/env python3
"""
Script to run the data pipeline automation system.
This script initializes and starts the complete pipeline automation system.
"""

import asyncio
import sys
import signal
import logging
from pathlib import Path

# Add the backend directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.data_pipeline.pipeline_initializer import get_pipeline_initializer
from app.core.logging import get_logger

logger = get_logger(__name__)


class PipelineAutomationRunner:
    """
    Main runner for the data pipeline automation system
    """
    
    def __init__(self):
        self.initializer = None
        self.running = False
        
    async def start(self):
        """Start the pipeline automation system"""
        try:
            logger.info("Starting Data Pipeline Automation System...")
            
            # Get initializer
            self.initializer = await get_pipeline_initializer()
            
            # Initialize the system
            await self.initializer.initialize_pipeline_system()
            
            self.running = True
            logger.info("✅ Data Pipeline Automation System started successfully!")
            
            # Print system status
            await self.print_system_status()
            
            # Keep the system running
            await self.keep_running()
            
        except Exception as e:
            logger.error(f"❌ Failed to start pipeline automation system: {e}")
            raise
    
    async def stop(self):
        """Stop the pipeline automation system"""
        try:
            logger.info("Stopping Data Pipeline Automation System...")
            
            self.running = False
            
            if self.initializer:
                await self.initializer.shutdown_pipeline_system()
            
            logger.info("✅ Data Pipeline Automation System stopped successfully!")
            
        except Exception as e:
            logger.error(f"❌ Failed to stop pipeline automation system: {e}")
    
    async def print_system_status(self):
        """Print current system status"""
        try:
            if not self.initializer:
                return
            
            status = await self.initializer.get_system_status()
            
            print("\n" + "=" * 60)
            print("📊 PIPELINE AUTOMATION SYSTEM STATUS")
            print("=" * 60)
            
            health = status.get('health_check', {})
            print(f"🏥 Overall Health: {health.get('overall_health', 'unknown').upper()}")
            print(f"📅 Scheduler Status: {health.get('scheduler_status', 'unknown')}")
            print(f"📊 Monitor Status: {health.get('monitor_status', 'unknown')}")
            print(f"⚙️  Scheduled Jobs: {health.get('scheduled_jobs_count', 0)}")
            print(f"🔄 Active Executions: {health.get('active_executions', 0)}")
            
            metrics = status.get('system_metrics', {})
            if metrics:
                print(f"\n📈 System Metrics (24h):")
                print(f"  - Success Rate: {metrics.get('success_rate_24h', 0):.1%}")
                print(f"  - Average Duration: {metrics.get('avg_duration_24h', 0):.1f}s")
                print(f"  - Data Quality Score: {metrics.get('data_quality_score_24h', 0):.2f}")
                print(f"  - Alerts: {metrics.get('alerts_24h', 0)}")
            
            print("=" * 60)
            
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
    
    async def keep_running(self):
        """Keep the system running until interrupted"""
        try:
            logger.info("Pipeline automation system is running. Press Ctrl+C to stop.")
            
            while self.running:
                await asyncio.sleep(60)  # Check every minute
                
                # Optionally print periodic status updates
                if hasattr(self, '_status_counter'):
                    self._status_counter += 1
                else:
                    self._status_counter = 1
                
                # Print status every 10 minutes
                if self._status_counter % 10 == 0:
                    logger.info("System is running normally...")
                    await self.print_system_status()
                    
        except asyncio.CancelledError:
            logger.info("Received shutdown signal")
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}")
            self.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)


async def main():
    """Main function"""
    runner = PipelineAutomationRunner()
    runner.setup_signal_handlers()
    
    try:
        await runner.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"System error: {e}")
        sys.exit(1)
    finally:
        await runner.stop()


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the system
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)