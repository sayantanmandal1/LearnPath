#!/usr/bin/env python3
"""
CLI script to run job market data collection pipeline
"""
import asyncio
import argparse
import logging
import sys
import os
from datetime import datetime
from typing import List, Optional

# Add the parent directory to the path so we can import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.data_collection_pipeline import DataCollectionPipeline, PipelineConfig
from app.services.job_scrapers.base_job_scraper import JobSearchParams
from app.core.database import get_db
from app.core.logging import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


async def run_single_collection(
    keywords: str,
    location: Optional[str] = None,
    platforms: Optional[List[str]] = None,
    limit: int = 100,
    enable_analysis: bool = True
):
    """Run a single job collection"""
    
    if platforms is None:
        platforms = ["linkedin", "indeed"]
    
    pipeline = DataCollectionPipeline()
    
    search_params = JobSearchParams(
        keywords=keywords,
        location=location,
        posted_days=7,
        limit=limit
    )
    
    config = PipelineConfig(
        name=f"cli_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        search_params=search_params,
        platforms=platforms,
        schedule_hours=0,  # Not scheduled
        max_jobs_per_run=limit,
        enable_analysis=enable_analysis,
        enable_trend_analysis=False
    )
    
    logger.info(f"Starting job collection: {keywords}")
    logger.info(f"Platforms: {platforms}")
    logger.info(f"Location: {location or 'Any'}")
    logger.info(f"Limit: {limit}")
    
    try:
        run_result = await pipeline.run_pipeline(config)
        
        print("\n" + "="*60)
        print("JOB COLLECTION RESULTS")
        print("="*60)
        print(f"Status: {run_result.status}")
        print(f"Duration: {(run_result.end_time - run_result.start_time).total_seconds():.1f} seconds")
        print(f"Jobs Scraped: {run_result.jobs_scraped}")
        print(f"Jobs Stored: {run_result.jobs_stored}")
        print(f"Jobs Processed: {run_result.jobs_processed}")
        
        if run_result.errors:
            print(f"\nErrors ({len(run_result.errors)}):")
            for error in run_result.errors:
                print(f"  - {error}")
        
        if run_result.stats:
            print(f"\nDetailed Stats:")
            for category, stats in run_result.stats.items():
                print(f"  {category}: {stats}")
        
        print("="*60)
        
        return run_result.status == 'completed'
        
    except Exception as e:
        logger.error(f"Collection failed: {str(e)}")
        print(f"\nERROR: {str(e)}")
        return False


async def run_scheduled_pipeline(config_names: Optional[List[str]] = None):
    """Run the scheduled pipeline"""
    
    pipeline = DataCollectionPipeline()
    
    # Use specific configs if provided, otherwise use defaults
    if config_names:
        configs = []
        for name in config_names:
            # Create basic configs for specified names
            if name == "tech_jobs":
                config = PipelineConfig(
                    name="tech_jobs_cli",
                    search_params=JobSearchParams(
                        keywords="software engineer developer programmer",
                        posted_days=1,
                        limit=200
                    ),
                    platforms=["linkedin", "indeed"],
                    schedule_hours=6,
                    max_jobs_per_run=500,
                    enable_analysis=True,
                    enable_trend_analysis=False
                )
                configs.append(config)
            elif name == "data_science":
                config = PipelineConfig(
                    name="data_science_cli",
                    search_params=JobSearchParams(
                        keywords="data scientist machine learning AI",
                        posted_days=1,
                        limit=150
                    ),
                    platforms=["linkedin", "indeed"],
                    schedule_hours=8,
                    max_jobs_per_run=300,
                    enable_analysis=True,
                    enable_trend_analysis=False
                )
                configs.append(config)
    else:
        configs = None  # Use defaults
    
    logger.info("Starting scheduled pipeline...")
    print("Starting scheduled job collection pipeline...")
    print("Press Ctrl+C to stop")
    
    try:
        await pipeline.start_pipeline_scheduler(configs)
    except KeyboardInterrupt:
        print("\nPipeline stopped by user")
        logger.info("Pipeline stopped by user")
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        print(f"Pipeline failed: {str(e)}")


async def show_pipeline_status():
    """Show current pipeline status"""
    
    pipeline = DataCollectionPipeline()
    status = await pipeline.get_pipeline_status()
    
    print("\n" + "="*60)
    print("PIPELINE STATUS")
    print("="*60)
    
    running = status.get('running_pipelines', {})
    if running:
        print(f"\nRunning Pipelines ({len(running)}):")
        for name, run_info in running.items():
            print(f"  {name}:")
            print(f"    Status: {run_info['status']}")
            print(f"    Started: {run_info['start_time']}")
            print(f"    Jobs Scraped: {run_info['jobs_scraped']}")
            print(f"    Jobs Stored: {run_info['jobs_stored']}")
            print(f"    Jobs Processed: {run_info['jobs_processed']}")
            if run_info['errors']:
                print(f"    Errors: {len(run_info['errors'])}")
    else:
        print("\nNo pipelines currently running")
    
    recent = status.get('recent_runs', [])
    if recent:
        print(f"\nRecent Runs ({len(recent)}):")
        for run_info in recent[-5:]:  # Show last 5
            duration = run_info.get('duration_minutes')
            duration_str = f"{duration:.1f}min" if duration else "N/A"
            
            print(f"  {run_info['config_name']} - {run_info['status']} ({duration_str})")
            print(f"    Scraped: {run_info['jobs_scraped']}, "
                  f"Stored: {run_info['jobs_stored']}, "
                  f"Processed: {run_info['jobs_processed']}")
            if run_info['error_count'] > 0:
                print(f"    Errors: {run_info['error_count']}")
    
    print("="*60)


async def show_collection_metrics(days: int = 30):
    """Show collection performance metrics"""
    
    pipeline = DataCollectionPipeline()
    metrics = await pipeline.get_collection_metrics(days)
    
    print("\n" + "="*60)
    print(f"COLLECTION METRICS ({days} days)")
    print("="*60)
    
    if metrics['total_runs'] == 0:
        print("No completed runs in the specified period")
        return
    
    metrics_data = metrics['metrics']
    
    print(f"\nOverall Performance:")
    print(f"  Total Runs: {metrics['total_runs']}")
    print(f"  Jobs Scraped: {metrics_data['total_jobs_scraped']}")
    print(f"  Jobs Stored: {metrics_data['total_jobs_stored']}")
    print(f"  Jobs Processed: {metrics_data['total_jobs_processed']}")
    print(f"  Avg Jobs/Run: {metrics_data['avg_jobs_per_run']:.1f}")
    print(f"  Storage Rate: {metrics_data['storage_rate']:.1%}")
    print(f"  Processing Rate: {metrics_data['processing_rate']:.1%}")
    
    platform_perf = metrics_data.get('platform_performance', {})
    if platform_perf:
        print(f"\nPlatform Performance:")
        for platform, stats in platform_perf.items():
            avg_scraped = stats['scraped'] / stats['runs'] if stats['runs'] > 0 else 0
            storage_rate = stats['stored'] / stats['scraped'] if stats['scraped'] > 0 else 0
            
            print(f"  {platform.title()}:")
            print(f"    Runs: {stats['runs']}")
            print(f"    Total Scraped: {stats['scraped']}")
            print(f"    Total Stored: {stats['stored']}")
            print(f"    Avg/Run: {avg_scraped:.1f}")
            print(f"    Storage Rate: {storage_rate:.1%}")
    
    print("="*60)


def main():
    """Main CLI function"""
    
    parser = argparse.ArgumentParser(
        description="Job Market Data Collection Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run single collection
  python run_job_pipeline.py collect "python developer" --location "San Francisco" --limit 100
  
  # Run scheduled pipeline
  python run_job_pipeline.py schedule --configs tech_jobs data_science
  
  # Show pipeline status
  python run_job_pipeline.py status
  
  # Show collection metrics
  python run_job_pipeline.py metrics --days 30
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Collect command
    collect_parser = subparsers.add_parser('collect', help='Run single job collection')
    collect_parser.add_argument('keywords', help='Search keywords')
    collect_parser.add_argument('--location', help='Target location')
    collect_parser.add_argument('--platforms', nargs='+', 
                               choices=['linkedin', 'indeed', 'glassdoor'],
                               default=['linkedin', 'indeed'],
                               help='Platforms to scrape')
    collect_parser.add_argument('--limit', type=int, default=100,
                               help='Maximum jobs to collect')
    collect_parser.add_argument('--no-analysis', action='store_true',
                               help='Skip skill analysis')
    
    # Schedule command
    schedule_parser = subparsers.add_parser('schedule', help='Run scheduled pipeline')
    schedule_parser.add_argument('--configs', nargs='+',
                                choices=['tech_jobs', 'data_science'],
                                help='Specific pipeline configurations to run')
    
    # Status command
    subparsers.add_parser('status', help='Show pipeline status')
    
    # Metrics command
    metrics_parser = subparsers.add_parser('metrics', help='Show collection metrics')
    metrics_parser.add_argument('--days', type=int, default=30,
                               help='Number of days to analyze')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Run the appropriate command
    try:
        if args.command == 'collect':
            success = asyncio.run(run_single_collection(
                keywords=args.keywords,
                location=args.location,
                platforms=args.platforms,
                limit=args.limit,
                enable_analysis=not args.no_analysis
            ))
            sys.exit(0 if success else 1)
            
        elif args.command == 'schedule':
            asyncio.run(run_scheduled_pipeline(args.configs))
            
        elif args.command == 'status':
            asyncio.run(show_pipeline_status())
            
        elif args.command == 'metrics':
            asyncio.run(show_collection_metrics(args.days))
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Command failed: {str(e)}")
        print(f"ERROR: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()