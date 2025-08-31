"""
Backup and Disaster Recovery System
Handles automated backups and disaster recovery procedures.
"""

import asyncio
import os
import shutil
import gzip
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import subprocess
import tempfile

from app.core.logging import get_logger
from app.core.database import get_db
from app.core.redis import get_redis
from app.services.data_pipeline.pipeline_monitor import get_pipeline_monitor
from app.services.data_pipeline.pipeline_config import get_pipeline_config

logger = get_logger(__name__)


class BackupRecoveryManager:
    """
    Manages automated backups and disaster recovery procedures
    """
    
    def __init__(self):
        self.config = get_pipeline_config()
        self.backup_location = Path(self.config.backup_location)
        self.backup_location.mkdir(parents=True, exist_ok=True)
        self.monitor = None
        
    async def execute_backup(self, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a complete system backup"""
        execution_id = metadata.get('execution_id', f"backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}")
        
        self.monitor = await get_pipeline_monitor()
        
        try:
            logger.info(f"Starting backup operation: {execution_id}")
            
            # Initialize metrics
            await self.monitor.update_job_metrics(
                execution_id,
                status="running",
                records_processed=0,
                records_failed=0
            )
            
            backup_type = metadata.get('backup_type', 'full')  # 'full' or 'incremental'
            include_ml_models = metadata.get('include_ml_models', True)
            compress_backup = metadata.get('compress_backup', True)
            
            backup_result = {
                'backup_id': execution_id,
                'backup_type': backup_type,
                'timestamp': datetime.utcnow().isoformat(),
                'components': {},
                'total_size_mb': 0,
                'duration_seconds': 0,
                'success': True,
                'errors': []
            }
            
            start_time = datetime.utcnow()
            
            # Create backup directory
            backup_dir = self.backup_location / execution_id
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            components_backed_up = 0
            components_failed = 0
            
            # Backup database
            try:
                logger.info("Backing up database")
                db_backup_result = await self._backup_database(backup_dir, backup_type)
                backup_result['components']['database'] = db_backup_result
                backup_result['total_size_mb'] += db_backup_result.get('size_mb', 0)
                
                if db_backup_result['success']:
                    components_backed_up += 1
                else:
                    components_failed += 1
                    backup_result['errors'].extend(db_backup_result.get('errors', []))
                    
            except Exception as e:
                logger.error(f"Database backup failed: {e}")
                components_failed += 1
                backup_result['errors'].append(f"Database backup error: {str(e)}")
            
            # Backup Redis data
            try:
                logger.info("Backing up Redis data")
                redis_backup_result = await self._backup_redis(backup_dir)
                backup_result['components']['redis'] = redis_backup_result
                backup_result['total_size_mb'] += redis_backup_result.get('size_mb', 0)
                
                if redis_backup_result['success']:
                    components_backed_up += 1
                else:
                    components_failed += 1
                    backup_result['errors'].extend(redis_backup_result.get('errors', []))
                    
            except Exception as e:
                logger.error(f"Redis backup failed: {e}")
                components_failed += 1
                backup_result['errors'].append(f"Redis backup error: {str(e)}")
            
            # Backup ML models
            if include_ml_models:
                try:
                    logger.info("Backing up ML models")
                    models_backup_result = await self._backup_ml_models(backup_dir)
                    backup_result['components']['ml_models'] = models_backup_result
                    backup_result['total_size_mb'] += models_backup_result.get('size_mb', 0)
                    
                    if models_backup_result['success']:
                        components_backed_up += 1
                    else:
                        components_failed += 1
                        backup_result['errors'].extend(models_backup_result.get('errors', []))
                        
                except Exception as e:
                    logger.error(f"ML models backup failed: {e}")
                    components_failed += 1
                    backup_result['errors'].append(f"ML models backup error: {str(e)}")
            
            # Backup configuration files
            try:
                logger.info("Backing up configuration files")
                config_backup_result = await self._backup_configurations(backup_dir)
                backup_result['components']['configurations'] = config_backup_result
                backup_result['total_size_mb'] += config_backup_result.get('size_mb', 0)
                
                if config_backup_result['success']:
                    components_backed_up += 1
                else:
                    components_failed += 1
                    backup_result['errors'].extend(config_backup_result.get('errors', []))
                    
            except Exception as e:
                logger.error(f"Configuration backup failed: {e}")
                components_failed += 1
                backup_result['errors'].append(f"Configuration backup error: {str(e)}")
            
            # Compress backup if requested
            if compress_backup:
                try:
                    logger.info("Compressing backup")
                    compressed_size = await self._compress_backup(backup_dir)
                    backup_result['compressed_size_mb'] = compressed_size
                except Exception as e:
                    logger.error(f"Backup compression failed: {e}")
                    backup_result['errors'].append(f"Compression error: {str(e)}")
            
            # Calculate final metrics
            end_time = datetime.utcnow()
            backup_result['duration_seconds'] = (end_time - start_time).total_seconds()
            backup_result['success'] = components_failed == 0
            
            # Store backup metadata
            await self._store_backup_metadata(backup_result)
            
            # Clean up old backups
            await self._cleanup_old_backups()
            
            # Update monitoring metrics
            await self.monitor.update_job_metrics(
                execution_id,
                status="completed" if backup_result['success'] else "failed",
                records_processed=components_backed_up,
                records_failed=components_failed,
                data_quality_score=1.0 if backup_result['success'] else 0.5
            )
            
            logger.info(f"Backup operation completed: {execution_id}")
            return backup_result
            
        except Exception as e:
            logger.error(f"Backup operation failed: {e}")
            
            if self.monitor:
                await self.monitor.update_job_metrics(
                    execution_id,
                    status="failed",
                    error_count=1
                )
            
            return {
                'backup_id': execution_id,
                'success': False,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    async def _backup_database(self, backup_dir: Path, backup_type: str) -> Dict[str, Any]:
        """Backup PostgreSQL database"""
        try:
            db_backup_file = backup_dir / "database_backup.sql"
            
            # Get database connection details from environment
            db_host = os.getenv("DATABASE_HOST", "localhost")
            db_port = os.getenv("DATABASE_PORT", "5432")
            db_name = os.getenv("DATABASE_NAME", "career_recommender")
            db_user = os.getenv("DATABASE_USER", "postgres")
            db_password = os.getenv("DATABASE_PASSWORD", "")
            
            # Prepare pg_dump command
            cmd = [
                "pg_dump",
                f"--host={db_host}",
                f"--port={db_port}",
                f"--username={db_user}",
                f"--dbname={db_name}",
                "--no-password",
                "--verbose",
                "--clean",
                "--if-exists",
                "--create"
            ]
            
            # Set password via environment variable
            env = os.environ.copy()
            if db_password:
                env["PGPASSWORD"] = db_password
            
            # Execute backup
            with open(db_backup_file, 'w') as f:
                process = subprocess.run(
                    cmd,
                    stdout=f,
                    stderr=subprocess.PIPE,
                    env=env,
                    text=True,
                    timeout=3600  # 1 hour timeout
                )
            
            if process.returncode == 0:
                # Get file size
                file_size = db_backup_file.stat().st_size / (1024 * 1024)  # MB
                
                return {
                    'success': True,
                    'file_path': str(db_backup_file),
                    'size_mb': file_size,
                    'backup_type': backup_type
                }
            else:
                error_msg = process.stderr if process.stderr else "Unknown pg_dump error"
                return {
                    'success': False,
                    'errors': [f"pg_dump failed: {error_msg}"]
                }
                
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'errors': ["Database backup timed out"]
            }
        except Exception as e:
            return {
                'success': False,
                'errors': [f"Database backup error: {str(e)}"]
            }
    
    async def _backup_redis(self, backup_dir: Path) -> Dict[str, Any]:
        """Backup Redis data"""
        try:
            redis_backup_file = backup_dir / "redis_backup.json"
            
            redis_manager = await get_redis()
            redis_client = redis_manager.redis
            
            # Get all keys and their values
            redis_data = {}
            
            # Get all keys
            keys = await redis_client.keys("*")
            
            for key in keys:
                try:
                    # Get key type
                    key_type = await redis_client.type(key)
                    
                    if key_type == "string":
                        value = await redis_client.get(key)
                        redis_data[key] = {"type": "string", "value": value}
                    elif key_type == "hash":
                        value = await redis_client.hgetall(key)
                        redis_data[key] = {"type": "hash", "value": value}
                    elif key_type == "list":
                        value = await redis_client.lrange(key, 0, -1)
                        redis_data[key] = {"type": "list", "value": value}
                    elif key_type == "set":
                        value = list(await redis_client.smembers(key))
                        redis_data[key] = {"type": "set", "value": value}
                    elif key_type == "zset":
                        value = await redis_client.zrange(key, 0, -1, withscores=True)
                        redis_data[key] = {"type": "zset", "value": value}
                    
                    # Get TTL if exists
                    ttl = await redis_client.ttl(key)
                    if ttl > 0:
                        redis_data[key]["ttl"] = ttl
                        
                except Exception as e:
                    logger.warning(f"Failed to backup Redis key {key}: {e}")
                    continue
            
            # Save to file
            with open(redis_backup_file, 'w') as f:
                json.dump(redis_data, f, indent=2, default=str)
            
            # Get file size
            file_size = redis_backup_file.stat().st_size / (1024 * 1024)  # MB
            
            return {
                'success': True,
                'file_path': str(redis_backup_file),
                'size_mb': file_size,
                'keys_backed_up': len(redis_data)
            }
            
        except Exception as e:
            return {
                'success': False,
                'errors': [f"Redis backup error: {str(e)}"]
            }
    
    async def _backup_ml_models(self, backup_dir: Path) -> Dict[str, Any]:
        """Backup ML models and related files"""
        try:
            models_backup_dir = backup_dir / "ml_models"
            models_backup_dir.mkdir(exist_ok=True)
            
            # Define model directories to backup
            model_paths = [
                Path("machinelearningmodel/models"),
                Path("machinelearningmodel/training/saved_models"),
                Path("/app/models")  # Production model storage
            ]
            
            total_size = 0
            backed_up_models = []
            errors = []
            
            for model_path in model_paths:
                if model_path.exists():
                    try:
                        # Copy model directory
                        dest_path = models_backup_dir / model_path.name
                        shutil.copytree(model_path, dest_path, dirs_exist_ok=True)
                        
                        # Calculate size
                        size = sum(f.stat().st_size for f in dest_path.rglob('*') if f.is_file())
                        total_size += size
                        
                        backed_up_models.append({
                            'path': str(model_path),
                            'size_bytes': size
                        })
                        
                    except Exception as e:
                        errors.append(f"Failed to backup {model_path}: {str(e)}")
            
            # Backup model metadata and version info
            try:
                from machinelearningmodel.training.model_versioning import ModelVersionManager
                version_manager = ModelVersionManager()
                
                model_metadata = await version_manager.get_all_model_metadata()
                metadata_file = models_backup_dir / "model_metadata.json"
                
                with open(metadata_file, 'w') as f:
                    json.dump(model_metadata, f, indent=2, default=str)
                    
            except Exception as e:
                errors.append(f"Failed to backup model metadata: {str(e)}")
            
            return {
                'success': len(errors) == 0,
                'size_mb': total_size / (1024 * 1024),
                'backed_up_models': backed_up_models,
                'errors': errors
            }
            
        except Exception as e:
            return {
                'success': False,
                'errors': [f"ML models backup error: {str(e)}"]
            }
    
    async def _backup_configurations(self, backup_dir: Path) -> Dict[str, Any]:
        """Backup configuration files"""
        try:
            config_backup_dir = backup_dir / "configurations"
            config_backup_dir.mkdir(exist_ok=True)
            
            # Define configuration files and directories to backup
            config_paths = [
                Path(".env"),
                Path("backend/.env"),
                Path("backend/alembic.ini"),
                Path("docker-compose.yml"),
                Path(".kiro/settings"),
                Path("backend/app/core"),
                Path("config")  # If exists
            ]
            
            total_size = 0
            backed_up_configs = []
            errors = []
            
            for config_path in config_paths:
                if config_path.exists():
                    try:
                        dest_path = config_backup_dir / config_path.name
                        
                        if config_path.is_file():
                            shutil.copy2(config_path, dest_path)
                            size = dest_path.stat().st_size
                        else:
                            shutil.copytree(config_path, dest_path, dirs_exist_ok=True)
                            size = sum(f.stat().st_size for f in dest_path.rglob('*') if f.is_file())
                        
                        total_size += size
                        backed_up_configs.append({
                            'path': str(config_path),
                            'size_bytes': size
                        })
                        
                    except Exception as e:
                        errors.append(f"Failed to backup {config_path}: {str(e)}")
            
            # Backup environment variables (sanitized)
            try:
                env_vars = {}
                sensitive_keys = ['password', 'secret', 'key', 'token', 'api_key']
                
                for key, value in os.environ.items():
                    if any(sensitive in key.lower() for sensitive in sensitive_keys):
                        env_vars[key] = "[REDACTED]"
                    else:
                        env_vars[key] = value
                
                env_file = config_backup_dir / "environment_variables.json"
                with open(env_file, 'w') as f:
                    json.dump(env_vars, f, indent=2)
                    
            except Exception as e:
                errors.append(f"Failed to backup environment variables: {str(e)}")
            
            return {
                'success': len(errors) == 0,
                'size_mb': total_size / (1024 * 1024),
                'backed_up_configs': backed_up_configs,
                'errors': errors
            }
            
        except Exception as e:
            return {
                'success': False,
                'errors': [f"Configuration backup error: {str(e)}"]
            }
    
    async def _compress_backup(self, backup_dir: Path) -> float:
        """Compress backup directory"""
        try:
            compressed_file = backup_dir.with_suffix('.tar.gz')
            
            # Create compressed archive
            cmd = [
                "tar",
                "-czf",
                str(compressed_file),
                "-C",
                str(backup_dir.parent),
                backup_dir.name
            ]
            
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minutes timeout
            )
            
            if process.returncode == 0:
                # Remove uncompressed directory
                shutil.rmtree(backup_dir)
                
                # Return compressed size in MB
                return compressed_file.stat().st_size / (1024 * 1024)
            else:
                raise Exception(f"Compression failed: {process.stderr}")
                
        except Exception as e:
            logger.error(f"Failed to compress backup: {e}")
            raise
    
    async def _store_backup_metadata(self, backup_result: Dict[str, Any]):
        """Store backup metadata for tracking"""
        try:
            redis_manager = await get_redis()
            redis_client = redis_manager.redis
            
            # Store detailed backup info
            backup_key = f"backup_metadata:{backup_result['backup_id']}"
            await redis_client.set(
                backup_key,
                json.dumps(backup_result, default=str),
                ex=86400 * self.config.backup_retention_days
            )
            
            # Update backup history
            history_key = "backup_history"
            backup_summary = {
                'backup_id': backup_result['backup_id'],
                'timestamp': backup_result['timestamp'],
                'success': backup_result['success'],
                'size_mb': backup_result['total_size_mb'],
                'duration_seconds': backup_result['duration_seconds']
            }
            
            await redis_client.lpush(history_key, json.dumps(backup_summary, default=str))
            await redis_client.ltrim(history_key, 0, 99)  # Keep last 100 backups
            
        except Exception as e:
            logger.error(f"Failed to store backup metadata: {e}")
    
    async def _cleanup_old_backups(self):
        """Clean up old backup files"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=self.config.backup_retention_days)
            
            for backup_path in self.backup_location.iterdir():
                if backup_path.is_dir() or backup_path.suffix == '.tar.gz':
                    # Extract date from backup name
                    try:
                        # Assuming backup names contain timestamp like backup_20231201_120000
                        date_str = backup_path.name.split('_')[1] + '_' + backup_path.name.split('_')[2]
                        backup_date = datetime.strptime(date_str, '%Y%m%d_%H%M%S')
                        
                        if backup_date < cutoff_date:
                            if backup_path.is_dir():
                                shutil.rmtree(backup_path)
                            else:
                                backup_path.unlink()
                            
                            logger.info(f"Cleaned up old backup: {backup_path.name}")
                            
                    except (ValueError, IndexError):
                        # Skip files that don't match expected naming pattern
                        continue
                        
        except Exception as e:
            logger.error(f"Failed to cleanup old backups: {e}")
    
    async def restore_from_backup(self, backup_id: str, components: List[str] = None) -> Dict[str, Any]:
        """Restore system from a backup"""
        try:
            logger.info(f"Starting restore operation from backup: {backup_id}")
            
            if components is None:
                components = ['database', 'redis', 'ml_models', 'configurations']
            
            # Find backup
            backup_path = self.backup_location / backup_id
            if not backup_path.exists():
                # Check for compressed backup
                compressed_backup = backup_path.with_suffix('.tar.gz')
                if compressed_backup.exists():
                    # Extract compressed backup
                    await self._extract_backup(compressed_backup)
                else:
                    raise FileNotFoundError(f"Backup {backup_id} not found")
            
            restore_result = {
                'backup_id': backup_id,
                'timestamp': datetime.utcnow().isoformat(),
                'components_restored': [],
                'components_failed': [],
                'success': True,
                'errors': []
            }
            
            # Restore each component
            for component in components:
                try:
                    if component == 'database':
                        await self._restore_database(backup_path)
                    elif component == 'redis':
                        await self._restore_redis(backup_path)
                    elif component == 'ml_models':
                        await self._restore_ml_models(backup_path)
                    elif component == 'configurations':
                        await self._restore_configurations(backup_path)
                    
                    restore_result['components_restored'].append(component)
                    logger.info(f"Successfully restored {component}")
                    
                except Exception as e:
                    logger.error(f"Failed to restore {component}: {e}")
                    restore_result['components_failed'].append(component)
                    restore_result['errors'].append(f"{component}: {str(e)}")
            
            restore_result['success'] = len(restore_result['components_failed']) == 0
            
            logger.info(f"Restore operation completed: {backup_id}")
            return restore_result
            
        except Exception as e:
            logger.error(f"Restore operation failed: {e}")
            return {
                'backup_id': backup_id,
                'success': False,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    async def _extract_backup(self, compressed_backup: Path):
        """Extract compressed backup"""
        try:
            cmd = [
                "tar",
                "-xzf",
                str(compressed_backup),
                "-C",
                str(compressed_backup.parent)
            ]
            
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minutes timeout
            )
            
            if process.returncode != 0:
                raise Exception(f"Extraction failed: {process.stderr}")
                
        except Exception as e:
            logger.error(f"Failed to extract backup: {e}")
            raise
    
    async def _restore_database(self, backup_path: Path):
        """Restore database from backup"""
        db_backup_file = backup_path / "database_backup.sql"
        
        if not db_backup_file.exists():
            raise FileNotFoundError("Database backup file not found")
        
        # Get database connection details
        db_host = os.getenv("DATABASE_HOST", "localhost")
        db_port = os.getenv("DATABASE_PORT", "5432")
        db_name = os.getenv("DATABASE_NAME", "career_recommender")
        db_user = os.getenv("DATABASE_USER", "postgres")
        db_password = os.getenv("DATABASE_PASSWORD", "")
        
        # Prepare psql command
        cmd = [
            "psql",
            f"--host={db_host}",
            f"--port={db_port}",
            f"--username={db_user}",
            f"--dbname={db_name}",
            "--no-password",
            "--file",
            str(db_backup_file)
        ]
        
        # Set password via environment variable
        env = os.environ.copy()
        if db_password:
            env["PGPASSWORD"] = db_password
        
        # Execute restore
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=3600  # 1 hour timeout
        )
        
        if process.returncode != 0:
            raise Exception(f"Database restore failed: {process.stderr}")
    
    async def _restore_redis(self, backup_path: Path):
        """Restore Redis data from backup"""
        redis_backup_file = backup_path / "redis_backup.json"
        
        if not redis_backup_file.exists():
            raise FileNotFoundError("Redis backup file not found")
        
        redis_manager = await get_redis()
        redis_client = redis_manager.redis
        
        # Load backup data
        with open(redis_backup_file, 'r') as f:
            redis_data = json.load(f)
        
        # Restore each key
        for key, data in redis_data.items():
            try:
                key_type = data['type']
                value = data['value']
                ttl = data.get('ttl')
                
                if key_type == "string":
                    await redis_client.set(key, value)
                elif key_type == "hash":
                    await redis_client.hset(key, mapping=value)
                elif key_type == "list":
                    await redis_client.lpush(key, *value)
                elif key_type == "set":
                    await redis_client.sadd(key, *value)
                elif key_type == "zset":
                    # Convert list of [member, score] pairs to dict
                    zset_data = {item[0]: item[1] for item in value}
                    await redis_client.zadd(key, zset_data)
                
                # Set TTL if it existed
                if ttl and ttl > 0:
                    await redis_client.expire(key, ttl)
                    
            except Exception as e:
                logger.warning(f"Failed to restore Redis key {key}: {e}")
                continue
    
    async def _restore_ml_models(self, backup_path: Path):
        """Restore ML models from backup"""
        models_backup_dir = backup_path / "ml_models"
        
        if not models_backup_dir.exists():
            raise FileNotFoundError("ML models backup directory not found")
        
        # Restore model files
        for model_dir in models_backup_dir.iterdir():
            if model_dir.is_dir() and model_dir.name != "model_metadata.json":
                dest_path = Path("machinelearningmodel") / model_dir.name
                
                # Remove existing model directory if it exists
                if dest_path.exists():
                    shutil.rmtree(dest_path)
                
                # Copy backup to destination
                shutil.copytree(model_dir, dest_path)
        
        # Restore model metadata
        metadata_file = models_backup_dir / "model_metadata.json"
        if metadata_file.exists():
            try:
                from machinelearningmodel.training.model_versioning import ModelVersionManager
                version_manager = ModelVersionManager()
                
                with open(metadata_file, 'r') as f:
                    model_metadata = json.load(f)
                
                await version_manager.restore_model_metadata(model_metadata)
                
            except Exception as e:
                logger.warning(f"Failed to restore model metadata: {e}")
    
    async def _restore_configurations(self, backup_path: Path):
        """Restore configuration files from backup"""
        config_backup_dir = backup_path / "configurations"
        
        if not config_backup_dir.exists():
            raise FileNotFoundError("Configuration backup directory not found")
        
        # Restore configuration files
        for config_item in config_backup_dir.iterdir():
            if config_item.name == "environment_variables.json":
                continue  # Skip environment variables file
            
            # Determine destination path
            if config_item.name in [".env", "docker-compose.yml"]:
                dest_path = Path(config_item.name)
            elif config_item.name == "alembic.ini":
                dest_path = Path("backend") / config_item.name
            else:
                dest_path = Path(config_item.name)
            
            try:
                if config_item.is_file():
                    # Create parent directories if needed
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(config_item, dest_path)
                else:
                    # Remove existing directory if it exists
                    if dest_path.exists():
                        shutil.rmtree(dest_path)
                    
                    shutil.copytree(config_item, dest_path)
                    
            except Exception as e:
                logger.warning(f"Failed to restore {config_item.name}: {e}")
    
    async def get_backup_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get backup history"""
        try:
            redis_manager = await get_redis()
            redis_client = redis_manager.redis
            
            history_items = await redis_client.lrange("backup_history", 0, limit - 1)
            history = []
            
            for item in history_items:
                try:
                    backup_info = json.loads(item)
                    history.append(backup_info)
                except json.JSONDecodeError:
                    continue
            
            return history
            
        except Exception as e:
            logger.error(f"Failed to get backup history: {e}")
            return []
    
    async def verify_backup_integrity(self, backup_id: str) -> Dict[str, Any]:
        """Verify backup integrity"""
        try:
            backup_path = self.backup_location / backup_id
            compressed_backup = backup_path.with_suffix('.tar.gz')
            
            verification_result = {
                'backup_id': backup_id,
                'timestamp': datetime.utcnow().isoformat(),
                'integrity_check': True,
                'components_verified': [],
                'issues': []
            }
            
            # Check if backup exists
            if not backup_path.exists() and not compressed_backup.exists():
                verification_result['integrity_check'] = False
                verification_result['issues'].append("Backup file not found")
                return verification_result
            
            # If compressed, extract temporarily for verification
            temp_dir = None
            if compressed_backup.exists() and not backup_path.exists():
                temp_dir = tempfile.mkdtemp()
                temp_backup_path = Path(temp_dir) / backup_id
                
                # Extract to temp directory
                cmd = ["tar", "-xzf", str(compressed_backup), "-C", temp_dir]
                process = subprocess.run(cmd, capture_output=True, text=True)
                
                if process.returncode != 0:
                    verification_result['integrity_check'] = False
                    verification_result['issues'].append("Failed to extract compressed backup")
                    return verification_result
                
                backup_path = temp_backup_path
            
            try:
                # Verify database backup
                db_backup_file = backup_path / "database_backup.sql"
                if db_backup_file.exists():
                    if db_backup_file.stat().st_size > 0:
                        verification_result['components_verified'].append('database')
                    else:
                        verification_result['issues'].append("Database backup file is empty")
                
                # Verify Redis backup
                redis_backup_file = backup_path / "redis_backup.json"
                if redis_backup_file.exists():
                    try:
                        with open(redis_backup_file, 'r') as f:
                            json.load(f)
                        verification_result['components_verified'].append('redis')
                    except json.JSONDecodeError:
                        verification_result['issues'].append("Redis backup file is corrupted")
                
                # Verify ML models backup
                models_backup_dir = backup_path / "ml_models"
                if models_backup_dir.exists() and any(models_backup_dir.iterdir()):
                    verification_result['components_verified'].append('ml_models')
                
                # Verify configurations backup
                config_backup_dir = backup_path / "configurations"
                if config_backup_dir.exists() and any(config_backup_dir.iterdir()):
                    verification_result['components_verified'].append('configurations')
                
                verification_result['integrity_check'] = len(verification_result['issues']) == 0
                
            finally:
                # Clean up temp directory
                if temp_dir and Path(temp_dir).exists():
                    shutil.rmtree(temp_dir)
            
            return verification_result
            
        except Exception as e:
            logger.error(f"Failed to verify backup integrity: {e}")
            return {
                'backup_id': backup_id,
                'integrity_check': False,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }