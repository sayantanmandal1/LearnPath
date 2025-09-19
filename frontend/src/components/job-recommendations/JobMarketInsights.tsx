/**
 * Job Market Insights Component
 * Displays market trends and insights for Indian tech jobs
 */
import React, { useState, useEffect } from 'react';
import { motion } from 'motion/react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Badge } from '../ui/badge';
import { Button } from '../ui/button';
import { Progress } from '../ui/progress';
import { 
  TrendingUp, 
  TrendingDown, 
  MapPin, 
  Building, 
  DollarSign, 
  Users, 
  Calendar,
  BarChart3,
  PieChart,
  Activity,
  Target,
  Briefcase,
  RefreshCw
} from 'lucide-react';

import { jobRecommendationService } from '../../services/jobRecommendationService';

interface JobMarketInsightsProps {
  targetRole?: string;
  preferredCities?: string[];
  className?: string;
}

interface MarketInsight {
  total_jobs: number;
  cities_distribution: { [key: string]: number };
  companies: Array<{ company: string; job_count: number }>;
  skills_demand: { [key: string]: number };
  experience_levels: { [key: string]: number };
  salary_insights: {
    count: number;
    min: number;
    max: number;
    median: number;
    average: number;
  };
  posting_trends: { [key: string]: number };
}

export const JobMarketInsights: React.FC<JobMarketInsightsProps> = ({
  targetRole = 'Software Developer',
  preferredCities = ['Bangalore', 'Hyderabad', 'Pune'],
  className = '',
}) => {
  const [insights, setInsights] = useState<MarketInsight | null>(null);
  const [loading, setLoading] = useState(false);
  const [selectedCity, setSelectedCity] = useState<string>('all');

  useEffect(() => {
    loadMarketInsights();
  }, [targetRole, preferredCities]);

  const loadMarketInsights = async () => {
    setLoading(true);
    try {
      const data = await jobRecommendationService.getIndianTechMarketInsights(
        targetRole,
        preferredCities
      );
      setInsights(data);
    } catch (error) {
      console.error('Error loading market insights:', error);
    } finally {
      setLoading(false);
    }
  };

  const formatSalary = (amount: number) => {
    if (amount >= 100000) {
      return `₹${(amount / 100000).toFixed(1)}L`;
    }
    return `₹${(amount / 1000).toFixed(0)}K`;
  };

  const getTopSkills = () => {
    if (!insights?.skills_demand) return [];
    return Object.entries(insights.skills_demand)
      .sort(([,a], [,b]) => b - a)
      .slice(0, 10);
  };

  const getTopCities = () => {
    if (!insights?.cities_distribution) return [];
    return Object.entries(insights.cities_distribution)
      .sort(([,a], [,b]) => b - a)
      .slice(0, 8);
  };

  const getTrendDirection = (current: number, previous: number) => {
    if (current > previous) return { direction: 'up', color: 'text-green-600', icon: TrendingUp };
    if (current < previous) return { direction: 'down', color: 'text-red-600', icon: TrendingDown };
    return { direction: 'stable', color: 'text-gray-600', icon: Activity };
  };

  if (loading) {
    return (
      <div className={`space-y-6 ${className}`}>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {[1, 2, 3].map((i) => (
            <Card key={i} className="animate-pulse">
              <CardContent className="p-6">
                <div className="h-4 bg-gray-200 rounded w-3/4 mb-2"></div>
                <div className="h-8 bg-gray-200 rounded w-1/2 mb-2"></div>
                <div className="h-2 bg-gray-200 rounded w-full"></div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    );
  }

  if (!insights) {
    return (
      <Card className={className}>
        <CardContent className="text-center py-12">
          <BarChart3 className="w-12 h-12 mx-auto mb-4 text-gray-400" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">No market data available</h3>
          <p className="text-gray-600 mb-4">
            Unable to load market insights at this time.
          </p>
          <Button onClick={loadMarketInsights}>
            <RefreshCw className="w-4 h-4 mr-2" />
            Retry
          </Button>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-gray-900">Market Insights</h2>
          <p className="text-gray-600">
            Indian tech job market trends for {targetRole}
          </p>
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={loadMarketInsights}
          disabled={loading}
          className="flex items-center space-x-2"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          <span>Refresh</span>
        </Button>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
        >
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600">Total Jobs</p>
                  <p className="text-3xl font-bold text-blue-600">{insights.total_jobs}</p>
                  <p className="text-xs text-gray-500">Available positions</p>
                </div>
                <Briefcase className="w-8 h-8 text-blue-500" />
              </div>
            </CardContent>
          </Card>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: 0.1 }}
        >
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600">Avg Salary</p>
                  <p className="text-3xl font-bold text-green-600">
                    {insights.salary_insights?.average ? formatSalary(insights.salary_insights.average) : 'N/A'}
                  </p>
                  <p className="text-xs text-gray-500">Per annum</p>
                </div>
                <DollarSign className="w-8 h-8 text-green-500" />
              </div>
            </CardContent>
          </Card>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: 0.2 }}
        >
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600">Top Cities</p>
                  <p className="text-3xl font-bold text-purple-600">{getTopCities().length}</p>
                  <p className="text-xs text-gray-500">Hiring locations</p>
                </div>
                <MapPin className="w-8 h-8 text-purple-500" />
              </div>
            </CardContent>
          </Card>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: 0.3 }}
        >
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600">Companies</p>
                  <p className="text-3xl font-bold text-orange-600">{insights.companies?.length || 0}</p>
                  <p className="text-xs text-gray-500">Actively hiring</p>
                </div>
                <Building className="w-8 h-8 text-orange-500" />
              </div>
            </CardContent>
          </Card>
        </motion.div>
      </div>

      {/* Charts and Detailed Insights */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* City Distribution */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <MapPin className="w-5 h-5" />
              <span>Jobs by City</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {getTopCities().map(([city, count], index) => (
                <motion.div
                  key={city}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.3, delay: index * 0.1 }}
                  className="flex items-center justify-between"
                >
                  <div className="flex items-center space-x-3">
                    <div className="w-3 h-3 rounded-full bg-blue-500"></div>
                    <span className="font-medium capitalize">{city}</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <div className="w-24">
                      <Progress 
                        value={(count / insights.total_jobs) * 100} 
                        className="h-2" 
                      />
                    </div>
                    <span className="text-sm font-medium w-8">{count}</span>
                  </div>
                </motion.div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Top Skills in Demand */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Target className="w-5 h-5" />
              <span>Skills in Demand</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {getTopSkills().map(([skill, count], index) => (
                <motion.div
                  key={skill}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.3, delay: index * 0.1 }}
                  className="flex items-center justify-between"
                >
                  <div className="flex items-center space-x-3">
                    <Badge variant="secondary" className="text-xs">
                      {skill}
                    </Badge>
                  </div>
                  <div className="flex items-center space-x-2">
                    <div className="w-20">
                      <Progress 
                        value={(count / Math.max(...Object.values(insights.skills_demand))) * 100} 
                        className="h-2" 
                      />
                    </div>
                    <span className="text-sm font-medium w-8">{count}</span>
                  </div>
                </motion.div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Top Hiring Companies */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Building className="w-5 h-5" />
              <span>Top Hiring Companies</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {insights.companies?.slice(0, 8).map((company, index) => (
                <motion.div
                  key={company.company}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.3, delay: index * 0.1 }}
                  className="flex items-center justify-between p-3 bg-gray-50 rounded-lg"
                >
                  <div className="flex items-center space-x-3">
                    <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center">
                      <span className="text-xs font-bold text-blue-600">
                        {company.company.charAt(0).toUpperCase()}
                      </span>
                    </div>
                    <span className="font-medium">{company.company}</span>
                  </div>
                  <Badge variant="outline">
                    {company.job_count} jobs
                  </Badge>
                </motion.div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Experience Level Distribution */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Users className="w-5 h-5" />
              <span>Experience Level Demand</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {Object.entries(insights.experience_levels || {}).map(([level, count], index) => (
                <motion.div
                  key={level}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.3, delay: index * 0.1 }}
                  className="flex items-center justify-between"
                >
                  <div className="flex items-center space-x-3">
                    <div className="w-3 h-3 rounded-full bg-green-500"></div>
                    <span className="font-medium capitalize">{level.replace('_', ' ')}</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <div className="w-24">
                      <Progress 
                        value={(count / insights.total_jobs) * 100} 
                        className="h-2" 
                      />
                    </div>
                    <span className="text-sm font-medium w-8">{count}</span>
                  </div>
                </motion.div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Salary Insights */}
      {insights.salary_insights && insights.salary_insights.count > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <DollarSign className="w-5 h-5" />
              <span>Salary Distribution</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
              <div className="text-center">
                <p className="text-2xl font-bold text-green-600">
                  {formatSalary(insights.salary_insights.min)}
                </p>
                <p className="text-sm text-gray-600">Minimum</p>
              </div>
              <div className="text-center">
                <p className="text-2xl font-bold text-blue-600">
                  {formatSalary(insights.salary_insights.median)}
                </p>
                <p className="text-sm text-gray-600">Median</p>
              </div>
              <div className="text-center">
                <p className="text-2xl font-bold text-purple-600">
                  {formatSalary(insights.salary_insights.average)}
                </p>
                <p className="text-sm text-gray-600">Average</p>
              </div>
              <div className="text-center">
                <p className="text-2xl font-bold text-orange-600">
                  {formatSalary(insights.salary_insights.max)}
                </p>
                <p className="text-sm text-gray-600">Maximum</p>
              </div>
              <div className="text-center">
                <p className="text-2xl font-bold text-gray-600">
                  {insights.salary_insights.count}
                </p>
                <p className="text-sm text-gray-600">Data Points</p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Market Recommendations */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Activity className="w-5 h-5" />
            <span>Market Recommendations</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-medium text-gray-900 mb-3">Skills to Focus On</h4>
              <div className="space-y-2">
                {getTopSkills().slice(0, 5).map(([skill]) => (
                  <div key={skill} className="flex items-center space-x-2">
                    <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                    <span className="text-sm">{skill}</span>
                  </div>
                ))}
              </div>
            </div>
            <div>
              <h4 className="font-medium text-gray-900 mb-3">Best Cities to Target</h4>
              <div className="space-y-2">
                {getTopCities().slice(0, 5).map(([city]) => (
                  <div key={city} className="flex items-center space-x-2">
                    <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                    <span className="text-sm capitalize">{city}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default JobMarketInsights;