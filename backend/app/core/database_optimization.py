"""
Database optimization utilities for performance monitoring and query optimization
"""
import time
import asyncio
from typing import Dict, List, Any, Optional
from contextlib import asynccontextmanager
import structlog
from sqlalchemy import text, inspect
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.engine import Result
from sqlalchemy.pool import QueuePool

from app.core.database import engine, AsyncSessionLocal
from app.core.config import settings

logger = structlog.get_logger()


class DatabasePerformanceMonitor:
    """Monitor database performance and identify bottlenecks"""
    
    def __init__(self):
        self.query_stats = {}
        self.slow_queries = []
        self.connection_stats = {
            "total_connections": 0,
            "active_connections": 0,
            "pool_size": settings.DATABASE_POOL_SIZE,
            "max_overflow": settings.DATABASE_MAX_OVERFLOW
        }
    
    def record_query(self, query: str, execution_time: float, result_count: int = 0):
        """Record query execution statistics"""
        query_hash = hash(query)
        
        if query_hash not in self.query_stats:
            self.query_stats[query_hash] = {
                "query": query[:200] + "..." if len(query) > 200 else query,
                "execution_count": 0,
                "total_time": 0.0,
                "avg_time": 0.0,
                "max_time": 0.0,
                "min_time": float('inf'),
                "total_results": 0
            }
        
        stats = self.query_stats[query_hash]
        stats["execution_count"] += 1
        stats["total_time"] += execution_time
        stats["avg_time"] = stats["total_time"] / stats["execution_count"]
        stats["max_time"] = max(stats["max_time"], execution_time)
        stats["min_time"] = min(stats["min_time"], execution_time)
        stats["total_results"] += result_count
        
        # Track slow queries (> 1 second)
        if execution_time > 1.0:
            self.slow_queries.append({
                "query": query,
                "execution_time": execution_time,
                "result_count": result_count,
                "timestamp": time.time()
            })
            
            # Keep only last 100 slow queries
            if len(self.slow_queries) > 100:
                self.slow_queries = self.slow_queries[-100:]
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        # Sort queries by total time
        sorted_queries = sorted(
            self.query_stats.values(),
            key=lambda x: x["total_time"],
            reverse=True
        )
        
        return {
            "total_queries": sum(q["execution_count"] for q in self.query_stats.values()),
            "unique_queries": len(self.query_stats),
            "total_execution_time": sum(q["total_time"] for q in self.query_stats.values()),
            "slowest_queries": sorted_queries[:10],
            "recent_slow_queries": self.slow_queries[-10:],
            "connection_stats": self.connection_stats,
            "pool_status": self._get_pool_status()
        }
    
    def _get_pool_status(self) -> Dict[str, Any]:
        """Get connection pool status"""
        try:
            pool = engine.pool
            if hasattr(pool, 'size'):
                pool_status = {
                    "size": pool.size(),
                    "checked_in": pool.checkedin(),
                    "checked_out": pool.checkedout(),
                }
                
                # Only add overflow and invalid if they exist
                if hasattr(pool, 'overflow'):
                    pool_status["overflow"] = pool.overflow()
                if hasattr(pool, 'invalid'):
                    pool_status["invalid"] = pool.invalid()
                    
                return pool_status
        except Exception as e:
            logger.error("Failed to get pool status", error=str(e))
        
        return {"status": "unavailable"}


# Global performance monitor
db_monitor = DatabasePerformanceMonitor()


@asynccontextmanager
async def monitored_query(session: AsyncSession, query: str):
    """Context manager for monitoring query performance"""
    start_time = time.time()
    result = None
    result_count = 0
    
    try:
        result = await session.execute(text(query))
        if hasattr(result, 'rowcount') and result.rowcount is not None:
            result_count = result.rowcount
        elif hasattr(result, 'fetchall'):
            rows = result.fetchall()
            result_count = len(rows)
            # Re-create result with the fetched data
            result = rows
        
        yield result
        
    finally:
        execution_time = time.time() - start_time
        db_monitor.record_query(query, execution_time, result_count)


class DatabaseOptimizer:
    """Database optimization utilities"""
    
    @staticmethod
    async def analyze_table_stats() -> Dict[str, Any]:
        """Analyze table statistics for optimization insights"""
        try:
            async with AsyncSessionLocal() as session:
                # Get table sizes and row counts
                table_stats = {}
                
                # PostgreSQL specific queries
                if "postgresql" in settings.DATABASE_URL:
                    # Table sizes
                    size_query = """
                    SELECT 
                        schemaname,
                        tablename,
                        attname,
                        n_distinct,
                        correlation
                    FROM pg_stats 
                    WHERE schemaname = 'public'
                    ORDER BY tablename, attname;
                    """
                    
                    async with monitored_query(session, size_query) as result:
                        stats_data = result.fetchall() if hasattr(result, 'fetchall') else []
                        
                        for row in stats_data:
                            table_name = row[1]
                            if table_name not in table_stats:
                                table_stats[table_name] = {
                                    "columns": {},
                                    "total_size_mb": 0,
                                    "row_count": 0
                                }
                            
                            table_stats[table_name]["columns"][row[2]] = {
                                "n_distinct": row[3],
                                "correlation": row[4]
                            }
                    
                    # Get table sizes
                    size_query = """
                    SELECT 
                        tablename,
                        pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
                        pg_total_relation_size(schemaname||'.'||tablename) as size_bytes
                    FROM pg_tables 
                    WHERE schemaname = 'public'
                    ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
                    """
                    
                    async with monitored_query(session, size_query) as result:
                        size_data = result.fetchall() if hasattr(result, 'fetchall') else []
                        
                        for row in size_data:
                            table_name = row[0]
                            if table_name in table_stats:
                                table_stats[table_name]["size_pretty"] = row[1]
                                table_stats[table_name]["size_bytes"] = row[2]
                
                return table_stats
                
        except Exception as e:
            logger.error("Failed to analyze table stats", error=str(e))
            return {}
    
    @staticmethod
    async def analyze_index_usage() -> Dict[str, Any]:
        """Analyze index usage and suggest optimizations"""
        try:
            async with AsyncSessionLocal() as session:
                if "postgresql" in settings.DATABASE_URL:
                    # Index usage statistics
                    index_query = """
                    SELECT 
                        schemaname,
                        tablename,
                        indexname,
                        idx_tup_read,
                        idx_tup_fetch,
                        idx_scan
                    FROM pg_stat_user_indexes
                    ORDER BY idx_scan DESC;
                    """
                    
                    async with monitored_query(session, index_query) as result:
                        index_data = result.fetchall() if hasattr(result, 'fetchall') else []
                        
                        indexes = []
                        for row in index_data:
                            indexes.append({
                                "schema": row[0],
                                "table": row[1],
                                "index": row[2],
                                "tuples_read": row[3],
                                "tuples_fetched": row[4],
                                "scans": row[5],
                                "efficiency": row[4] / max(row[3], 1) if row[3] else 0
                            })
                        
                        # Find unused indexes (0 scans)
                        unused_indexes = [idx for idx in indexes if idx["scans"] == 0]
                        
                        # Find inefficient indexes (low efficiency)
                        inefficient_indexes = [
                            idx for idx in indexes 
                            if idx["efficiency"] < 0.1 and idx["scans"] > 0
                        ]
                        
                        return {
                            "total_indexes": len(indexes),
                            "unused_indexes": unused_indexes,
                            "inefficient_indexes": inefficient_indexes,
                            "most_used_indexes": indexes[:10]
                        }
                
                return {"status": "not_supported", "database": "non-postgresql"}
                
        except Exception as e:
            logger.error("Failed to analyze index usage", error=str(e))
            return {"error": str(e)}
    
    @staticmethod
    async def suggest_indexes() -> List[Dict[str, Any]]:
        """Suggest new indexes based on query patterns"""
        suggestions = []
        
        try:
            # Analyze slow queries for index suggestions
            slow_queries = db_monitor.slow_queries
            
            for query_info in slow_queries:
                query = query_info["query"].lower()
                
                # Simple heuristics for index suggestions
                if "where" in query and "order by" in query:
                    suggestions.append({
                        "type": "composite_index",
                        "reason": "Query has WHERE and ORDER BY clauses",
                        "query_sample": query[:100] + "...",
                        "execution_time": query_info["execution_time"]
                    })
                
                elif "join" in query and "on" in query:
                    suggestions.append({
                        "type": "foreign_key_index",
                        "reason": "Query uses JOIN operations",
                        "query_sample": query[:100] + "...",
                        "execution_time": query_info["execution_time"]
                    })
                
                elif "group by" in query:
                    suggestions.append({
                        "type": "grouping_index",
                        "reason": "Query uses GROUP BY clause",
                        "query_sample": query[:100] + "...",
                        "execution_time": query_info["execution_time"]
                    })
            
            return suggestions
            
        except Exception as e:
            logger.error("Failed to suggest indexes", error=str(e))
            return []
    
    @staticmethod
    async def optimize_connection_pool() -> Dict[str, Any]:
        """Analyze and optimize connection pool settings"""
        try:
            pool_status = db_monitor._get_pool_status()
            performance_report = db_monitor.get_performance_report()
            
            recommendations = []
            
            # Check pool utilization
            checked_out = pool_status.get("checked_out", 0)
            if isinstance(checked_out, int) and settings.DATABASE_POOL_SIZE > 0:
                utilization = checked_out / settings.DATABASE_POOL_SIZE
                
                if utilization > 0.8:
                    recommendations.append({
                        "type": "increase_pool_size",
                        "current_size": settings.DATABASE_POOL_SIZE,
                        "suggested_size": settings.DATABASE_POOL_SIZE + 10,
                        "reason": f"High pool utilization: {utilization:.2%}"
                    })
                
                elif utilization < 0.2:
                    recommendations.append({
                        "type": "decrease_pool_size",
                        "current_size": settings.DATABASE_POOL_SIZE,
                        "suggested_size": max(5, settings.DATABASE_POOL_SIZE - 5),
                        "reason": f"Low pool utilization: {utilization:.2%}"
                    })
            
            # Check for connection leaks
            if pool_status.get("overflow", 0) > 0:
                recommendations.append({
                    "type": "investigate_connection_leaks",
                    "overflow_count": pool_status["overflow"],
                    "reason": "Connection pool overflow detected"
                })
            
            return {
                "current_settings": {
                    "pool_size": settings.DATABASE_POOL_SIZE,
                    "max_overflow": settings.DATABASE_MAX_OVERFLOW
                },
                "pool_status": pool_status,
                "recommendations": recommendations,
                "performance_metrics": {
                    "total_queries": performance_report["total_queries"],
                    "avg_query_time": (
                        performance_report["total_execution_time"] / 
                        max(performance_report["total_queries"], 1)
                    )
                }
            }
            
        except Exception as e:
            logger.error("Failed to optimize connection pool", error=str(e))
            return {"error": str(e)}


class QueryOptimizer:
    """Query optimization utilities"""
    
    @staticmethod
    async def explain_query(query: str) -> Dict[str, Any]:
        """Get query execution plan"""
        try:
            async with AsyncSessionLocal() as session:
                explain_query = f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {query}"
                
                async with monitored_query(session, explain_query) as result:
                    plan_data = result.fetchall() if hasattr(result, 'fetchall') else []
                    
                    if plan_data and len(plan_data) > 0:
                        return {
                            "query": query,
                            "execution_plan": plan_data[0][0],
                            "analysis_timestamp": time.time()
                        }
                    
                    return {"error": "No execution plan available"}
                    
        except Exception as e:
            logger.error("Failed to explain query", query=query[:100], error=str(e))
            return {"error": str(e)}
    
    @staticmethod
    def optimize_query_hints(query: str) -> str:
        """Add optimization hints to queries"""
        # Simple query optimization hints
        optimized_query = query
        
        # Add LIMIT if missing for potentially large result sets
        if "select" in query.lower() and "limit" not in query.lower() and "count(" not in query.lower():
            optimized_query += " LIMIT 1000"
        
        # Add index hints for common patterns
        if "order by" in query.lower() and "where" in query.lower():
            # Suggest using indexes for WHERE + ORDER BY
            pass
        
        return optimized_query


# Global optimizer instances
db_optimizer = DatabaseOptimizer()
query_optimizer = QueryOptimizer()


async def get_database_health() -> Dict[str, Any]:
    """Get comprehensive database health status"""
    try:
        health_data = {
            "connection_status": "healthy",
            "performance_report": db_monitor.get_performance_report(),
            "optimization_suggestions": await db_optimizer.suggest_indexes(),
            "pool_optimization": await db_optimizer.optimize_connection_pool(),
            "timestamp": time.time()
        }
        
        # Test database connectivity
        async with AsyncSessionLocal() as session:
            await session.execute(text("SELECT 1"))
        
        return health_data
        
    except Exception as e:
        logger.error("Database health check failed", error=str(e))
        return {
            "connection_status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        }