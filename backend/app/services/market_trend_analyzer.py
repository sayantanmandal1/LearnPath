"""
Market trend analyzer for job market data analysis and predictions
"""
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.cluster import DBSCAN
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_

from app.models.job import JobPosting, JobSkill
from app.models.skill import Skill
from app.repositories.job import JobRepository

logger = logging.getLogger(__name__)


@dataclass
class TrendData:
    """Data structure for trend analysis"""
    dates: List[datetime]
    values: List[float]
    skill_name: str
    metric_type: str  # 'demand', 'salary', 'growth'


@dataclass
class SalaryPrediction:
    """Salary prediction result"""
    skill_name: str
    location: Optional[str]
    experience_level: Optional[str]
    predicted_salary: float
    confidence_interval: Tuple[float, float]
    model_accuracy: float
    sample_size: int


@dataclass
class EmergingSkill:
    """Emerging skill detection result"""
    skill_name: str
    growth_rate: float
    current_demand: int
    trend_score: float
    confidence: float
    related_skills: List[str]


class MarketTrendAnalyzer:
    """Advanced market trend analysis and prediction system"""
    
    def __init__(self):
        self.job_repository = JobRepository()
        self.scaler = StandardScaler()
        
        # Model cache
        self._salary_models = {}
        self._trend_models = {}
    
    async def analyze_skill_demand_trends(
        self,
        db: AsyncSession,
        skill_names: Optional[List[str]] = None,
        days: int = 180,
        time_window: int = 7  # Weekly aggregation
    ) -> List[Dict[str, Any]]:
        """
        Analyze demand trends for skills over time
        
        Args:
            db: Database session
            skill_names: Specific skills to analyze (None for top skills)
            days: Number of days to analyze
            time_window: Aggregation window in days
            
        Returns:
            List of trend analysis results
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Get skills to analyze
        if skill_names is None:
            skill_names = await self._get_top_skills(db, days, limit=50)
        
        trend_results = []
        
        for skill_name in skill_names:
            try:
                # Get time series data for skill demand
                demand_data = await self._get_skill_demand_timeseries(
                    db, skill_name, cutoff_date, time_window
                )
                
                if len(demand_data) < 3:  # Need minimum data points
                    continue
                
                # Perform trend analysis
                trend_analysis = self._analyze_time_series(demand_data, skill_name)
                
                # Calculate additional metrics
                trend_analysis.update({
                    'skill_name': skill_name,
                    'analysis_period_days': days,
                    'data_points': len(demand_data),
                    'current_demand': demand_data[-1]['value'] if demand_data else 0,
                    'peak_demand': max(point['value'] for point in demand_data),
                    'average_demand': np.mean([point['value'] for point in demand_data])
                })
                
                trend_results.append(trend_analysis)
                
            except Exception as e:
                logger.error(f"Failed to analyze trend for {skill_name}: {str(e)}")
                continue
        
        # Sort by trend strength
        trend_results.sort(key=lambda x: abs(x.get('trend_slope', 0)), reverse=True)
        
        return trend_results
    
    async def _get_skill_demand_timeseries(
        self,
        db: AsyncSession,
        skill_name: str,
        start_date: datetime,
        window_days: int
    ) -> List[Dict[str, Any]]:
        """Get time series data for skill demand"""
        
        # Query job postings with this skill over time
        query = select(
            func.date_trunc('week', JobPosting.posted_date).label('week'),
            func.count(JobSkill.id).label('demand_count')
        ).select_from(
            JobPosting.__table__.join(JobSkill.__table__).join(Skill.__table__)
        ).where(
            and_(
                Skill.name == skill_name,
                JobPosting.posted_date >= start_date,
                JobPosting.is_active == True
            )
        ).group_by(
            func.date_trunc('week', JobPosting.posted_date)
        ).order_by(
            func.date_trunc('week', JobPosting.posted_date)
        )
        
        result = await db.execute(query)
        data = result.fetchall()
        
        return [
            {
                'date': row.week,
                'value': float(row.demand_count)
            }
            for row in data
        ]
    
    def _analyze_time_series(
        self,
        data: List[Dict[str, Any]],
        skill_name: str
    ) -> Dict[str, Any]:
        """Analyze time series data for trends"""
        
        if len(data) < 2:
            return {'trend_slope': 0, 'trend_direction': 'stable', 'confidence': 0}
        
        # Convert to numpy arrays
        dates = [point['date'] for point in data]
        values = [point['value'] for point in data]
        
        # Convert dates to numeric (days since first date)
        date_nums = [(date - dates[0]).days for date in dates]
        
        # Linear regression for trend
        X = np.array(date_nums).reshape(-1, 1)
        y = np.array(values)
        
        model = LinearRegression()
        model.fit(X, y)
        
        trend_slope = model.coef_[0]
        r2 = model.score(X, y)
        
        # Determine trend direction
        if abs(trend_slope) < 0.1:
            trend_direction = 'stable'
        elif trend_slope > 0:
            trend_direction = 'growing'
        else:
            trend_direction = 'declining'
        
        # Calculate volatility
        volatility = np.std(values) / np.mean(values) if np.mean(values) > 0 else 0
        
        # Seasonal analysis (if enough data)
        seasonality = self._detect_seasonality(values) if len(values) > 12 else 0
        
        return {
            'trend_slope': float(trend_slope),
            'trend_direction': trend_direction,
            'confidence': float(r2),
            'volatility': float(volatility),
            'seasonality_strength': float(seasonality),
            'growth_rate_weekly': float(trend_slope / np.mean(values)) if np.mean(values) > 0 else 0
        }
    
    def _detect_seasonality(self, values: List[float]) -> float:
        """Detect seasonality in time series data"""
        if len(values) < 12:
            return 0
        
        # Simple autocorrelation at lag 4 (monthly) and lag 12 (quarterly)
        values_array = np.array(values)
        
        # Calculate autocorrelation
        def autocorr(x, lag):
            if len(x) <= lag:
                return 0
            return np.corrcoef(x[:-lag], x[lag:])[0, 1]
        
        monthly_corr = autocorr(values_array, 4)
        quarterly_corr = autocorr(values_array, 12)
        
        return max(abs(monthly_corr), abs(quarterly_corr))
    
    async def predict_salaries(
        self,
        db: AsyncSession,
        skill_names: List[str],
        location: Optional[str] = None,
        experience_level: Optional[str] = None,
        days: int = 365
    ) -> List[SalaryPrediction]:
        """
        Predict salaries for skills using regression models
        
        Args:
            db: Database session
            skill_names: Skills to predict salaries for
            location: Target location
            experience_level: Target experience level
            days: Historical data period
            
        Returns:
            List of salary predictions
        """
        predictions = []
        
        for skill_name in skill_names:
            try:
                prediction = await self._predict_skill_salary(
                    db, skill_name, location, experience_level, days
                )
                if prediction:
                    predictions.append(prediction)
            except Exception as e:
                logger.error(f"Failed to predict salary for {skill_name}: {str(e)}")
                continue
        
        return predictions
    
    async def _predict_skill_salary(
        self,
        db: AsyncSession,
        skill_name: str,
        location: Optional[str],
        experience_level: Optional[str],
        days: int
    ) -> Optional[SalaryPrediction]:
        """Predict salary for a specific skill"""
        
        # Get training data
        training_data = await self._get_salary_training_data(
            db, skill_name, location, experience_level, days
        )
        
        if len(training_data) < 10:  # Need minimum samples
            logger.warning(f"Insufficient data for salary prediction: {skill_name}")
            return None
        
        # Prepare features and target
        df = pd.DataFrame(training_data)
        
        # Feature engineering
        features = self._engineer_salary_features(df)
        target = df['salary'].values
        
        if len(features) == 0:
            return None
        
        # Train model
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42
        )
        
        # Try multiple models and select best
        models = {
            'linear': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        
        best_model = None
        best_score = -np.inf
        
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                score = model.score(X_test, y_test)
                
                if score > best_score:
                    best_score = score
                    best_model = model
            except Exception as e:
                logger.warning(f"Model {name} failed: {str(e)}")
                continue
        
        if best_model is None:
            return None
        
        # Make prediction
        # Use median feature values for prediction
        median_features = np.median(features, axis=0).reshape(1, -1)
        predicted_salary = best_model.predict(median_features)[0]
        
        # Calculate confidence interval
        predictions = best_model.predict(X_test)
        residuals = y_test - predictions
        std_residual = np.std(residuals)
        
        confidence_interval = (
            predicted_salary - 1.96 * std_residual,
            predicted_salary + 1.96 * std_residual
        )
        
        return SalaryPrediction(
            skill_name=skill_name,
            location=location,
            experience_level=experience_level,
            predicted_salary=float(predicted_salary),
            confidence_interval=confidence_interval,
            model_accuracy=float(best_score),
            sample_size=len(training_data)
        )
    
    async def _get_salary_training_data(
        self,
        db: AsyncSession,
        skill_name: str,
        location: Optional[str],
        experience_level: Optional[str],
        days: int
    ) -> List[Dict[str, Any]]:
        """Get training data for salary prediction"""
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Build query
        query = select(
            JobPosting.salary_min,
            JobPosting.salary_max,
            JobPosting.salary_period,
            JobPosting.location,
            JobPosting.experience_level,
            JobPosting.remote_type,
            JobPosting.company,
            JobPosting.posted_date
        ).select_from(
            JobPosting.__table__.join(JobSkill.__table__).join(Skill.__table__)
        ).where(
            and_(
                Skill.name == skill_name,
                JobPosting.posted_date >= cutoff_date,
                or_(
                    JobPosting.salary_min.isnot(None),
                    JobPosting.salary_max.isnot(None)
                )
            )
        )
        
        # Add filters
        if location:
            query = query.where(JobPosting.location.ilike(f"%{location}%"))
        
        if experience_level:
            query = query.where(JobPosting.experience_level == experience_level)
        
        result = await db.execute(query)
        data = result.fetchall()
        
        # Process salary data
        training_data = []
        for row in data:
            # Calculate average salary
            if row.salary_min and row.salary_max:
                avg_salary = (row.salary_min + row.salary_max) / 2
            else:
                avg_salary = row.salary_min or row.salary_max
            
            if not avg_salary:
                continue
            
            # Normalize to yearly salary
            if row.salary_period == 'hourly':
                avg_salary *= 2080  # 40 hours/week * 52 weeks
            elif row.salary_period == 'monthly':
                avg_salary *= 12
            
            training_data.append({
                'salary': avg_salary,
                'location': row.location or 'Unknown',
                'experience_level': row.experience_level or 'Unknown',
                'remote_type': row.remote_type or 'Unknown',
                'company': row.company,
                'posted_date': row.posted_date
            })
        
        return training_data
    
    def _engineer_salary_features(self, df: pd.DataFrame) -> np.ndarray:
        """Engineer features for salary prediction"""
        
        features = []
        
        # Experience level encoding
        exp_mapping = {'entry': 1, 'mid': 2, 'senior': 3, 'executive': 4}
        exp_features = df['experience_level'].map(
            lambda x: exp_mapping.get(x, 2)
        ).values
        features.append(exp_features)
        
        # Remote type encoding
        remote_mapping = {'remote': 2, 'hybrid': 1, 'onsite': 0}
        remote_features = df['remote_type'].map(
            lambda x: remote_mapping.get(x, 0)
        ).values
        features.append(remote_features)
        
        # Time features (month, year)
        df['month'] = pd.to_datetime(df['posted_date']).dt.month
        df['year'] = pd.to_datetime(df['posted_date']).dt.year
        features.append(df['month'].values)
        features.append(df['year'].values)
        
        # Location features (simplified - could be enhanced with geo-encoding)
        # For now, just use a simple hash
        location_hash = df['location'].apply(lambda x: hash(x) % 1000).values
        features.append(location_hash)
        
        if len(features) == 0:
            return np.array([])
        
        # Stack features
        feature_matrix = np.column_stack(features)
        
        # Scale features
        return self.scaler.fit_transform(feature_matrix)
    
    async def detect_emerging_skills(
        self,
        db: AsyncSession,
        days: int = 90,
        min_growth_rate: float = 0.1,
        min_sample_size: int = 5
    ) -> List[EmergingSkill]:
        """
        Detect emerging skills using anomaly detection
        
        Args:
            db: Database session
            days: Analysis period
            min_growth_rate: Minimum growth rate to consider
            min_sample_size: Minimum number of job postings
            
        Returns:
            List of emerging skills
        """
        # Get skill trends
        trends = await self.analyze_skill_demand_trends(db, days=days)
        
        # Filter for potential emerging skills
        candidates = [
            trend for trend in trends
            if (trend.get('growth_rate_weekly', 0) > min_growth_rate and
                trend.get('data_points', 0) >= min_sample_size and
                trend.get('trend_direction') == 'growing')
        ]
        
        if not candidates:
            return []
        
        # Extract features for anomaly detection
        features = []
        for trend in candidates:
            features.append([
                trend.get('growth_rate_weekly', 0),
                trend.get('current_demand', 0),
                trend.get('volatility', 0),
                trend.get('confidence', 0)
            ])
        
        features_array = np.array(features)
        
        # Apply DBSCAN for anomaly detection
        # Skills with high growth and low volatility are potential emerging skills
        clustering = DBSCAN(eps=0.3, min_samples=2)
        clusters = clustering.fit_predict(features_array)
        
        emerging_skills = []
        
        for i, trend in enumerate(candidates):
            # Calculate trend score
            trend_score = (
                trend.get('growth_rate_weekly', 0) * 0.4 +
                trend.get('confidence', 0) * 0.3 +
                (1 - trend.get('volatility', 1)) * 0.3
            )
            
            # Get related skills
            related_skills = await self._find_related_skills(
                db, trend['skill_name']
            )
            
            emerging_skill = EmergingSkill(
                skill_name=trend['skill_name'],
                growth_rate=trend.get('growth_rate_weekly', 0),
                current_demand=int(trend.get('current_demand', 0)),
                trend_score=trend_score,
                confidence=trend.get('confidence', 0),
                related_skills=related_skills
            )
            
            emerging_skills.append(emerging_skill)
        
        # Sort by trend score
        emerging_skills.sort(key=lambda x: x.trend_score, reverse=True)
        
        return emerging_skills[:20]  # Return top 20
    
    async def _find_related_skills(
        self,
        db: AsyncSession,
        skill_name: str,
        limit: int = 5
    ) -> List[str]:
        """Find skills that commonly appear with the given skill"""
        
        # Get job IDs that require this skill
        job_ids_query = select(JobSkill.job_posting_id).where(
            JobSkill.skill_id.in_(
                select(Skill.id).where(Skill.name == skill_name)
            )
        )
        
        result = await db.execute(job_ids_query)
        job_ids = [row[0] for row in result.fetchall()]
        
        if not job_ids:
            return []
        
        # Find other skills in these jobs
        related_query = select(
            Skill.name,
            func.count(JobSkill.id).label('co_occurrence')
        ).select_from(
            JobSkill.__table__.join(Skill.__table__)
        ).where(
            and_(
                JobSkill.job_posting_id.in_(job_ids),
                Skill.name != skill_name
            )
        ).group_by(
            Skill.id, Skill.name
        ).order_by(
            func.count(JobSkill.id).desc()
        ).limit(limit)
        
        result = await db.execute(related_query)
        related = result.fetchall()
        
        return [row.name for row in related]
    
    async def _get_top_skills(
        self,
        db: AsyncSession,
        days: int,
        limit: int = 50
    ) -> List[str]:
        """Get top skills by demand in the specified period"""
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        query = select(
            Skill.name,
            func.count(JobSkill.id).label('demand_count')
        ).select_from(
            JobSkill.__table__.join(Skill.__table__).join(JobPosting.__table__)
        ).where(
            JobPosting.posted_date >= cutoff_date
        ).group_by(
            Skill.id, Skill.name
        ).order_by(
            func.count(JobSkill.id).desc()
        ).limit(limit)
        
        result = await db.execute(query)
        skills = result.fetchall()
        
        return [row.name for row in skills]
    
    async def generate_market_report(
        self,
        db: AsyncSession,
        days: int = 90
    ) -> Dict[str, Any]:
        """
        Generate comprehensive market analysis report
        
        Args:
            db: Database session
            days: Analysis period
            
        Returns:
            Comprehensive market report
        """
        logger.info(f"Generating market report for {days} days")
        
        # Get skill trends
        skill_trends = await self.analyze_skill_demand_trends(db, days=days)
        
        # Get emerging skills
        emerging_skills = await self.detect_emerging_skills(db, days=days)
        
        # Get salary predictions for top skills
        top_skills = [trend['skill_name'] for trend in skill_trends[:20]]
        salary_predictions = await self.predict_salaries(db, top_skills, days=days)
        
        # Calculate market statistics
        total_jobs = await self._count_total_jobs(db, days)
        total_skills = len(skill_trends)
        
        # Growth analysis
        growing_skills = [
            trend for trend in skill_trends
            if trend.get('trend_direction') == 'growing'
        ]
        declining_skills = [
            trend for trend in skill_trends
            if trend.get('trend_direction') == 'declining'
        ]
        
        return {
            'report_generated': datetime.utcnow(),
            'analysis_period_days': days,
            'market_overview': {
                'total_jobs_analyzed': total_jobs,
                'total_skills_tracked': total_skills,
                'growing_skills_count': len(growing_skills),
                'declining_skills_count': len(declining_skills),
                'emerging_skills_count': len(emerging_skills)
            },
            'skill_trends': skill_trends[:50],
            'emerging_skills': [
                {
                    'skill_name': skill.skill_name,
                    'growth_rate': skill.growth_rate,
                    'current_demand': skill.current_demand,
                    'trend_score': skill.trend_score,
                    'confidence': skill.confidence,
                    'related_skills': skill.related_skills
                }
                for skill in emerging_skills
            ],
            'salary_predictions': [
                {
                    'skill_name': pred.skill_name,
                    'predicted_salary': pred.predicted_salary,
                    'confidence_interval': pred.confidence_interval,
                    'model_accuracy': pred.model_accuracy,
                    'sample_size': pred.sample_size
                }
                for pred in salary_predictions
            ],
            'top_growing_skills': growing_skills[:10],
            'top_declining_skills': declining_skills[:10]
        }
    
    async def _count_total_jobs(self, db: AsyncSession, days: int) -> int:
        """Count total jobs in analysis period"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        query = select(func.count(JobPosting.id)).where(
            JobPosting.posted_date >= cutoff_date
        )
        
        result = await db.execute(query)
        return result.scalar() or 0