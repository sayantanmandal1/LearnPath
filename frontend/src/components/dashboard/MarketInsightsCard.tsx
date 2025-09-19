/**
 * Market Insights Card with current industry trends
 */
import React, { useState } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Badge } from '../ui/badge';
import { Button } from '../ui/button';
import { Progress } from '../ui/progress';
import { 
  TrendingUp, 
  TrendingDown, 
  DollarSign, 
  MapPin, 
  Users, 
  Briefcase,
  RefreshCw,
  AlertCircle,
  CheckCircle,
  Star,
  Calendar,
  BarChart3,
  Globe,
  Building
} from 'lucide-react';

interface MarketTrend {
  id: string;
  title: string;
  description: string;
  trend_direction: 'up' | 'down' | 'stable';
  impact_level: 'High' | 'Medium' | 'Low';
  relevance_score: number;
  category: 'technology' | 'salary' | 'demand' | 'skills' | 'location';
  time_period: string;
  data_points?: {
    current_value: number;
    previous_value: number;
    change_percentage: number;
  };
}

interface SalaryInsight {
  role: string;
  location: string;
  salary_range: {
    min: number;
    max: number;
    median: number;
    currency: string;
  };
  experience_level: string;
  growth_rate: number;
  market_demand: 'High' | 'Medium' | 'Low';
}

interface SkillDemand {
  skill: string;
  demand_score: number;
  growth_rate: number;
  avg_salary_impact: number;
  job_postings_count: number;
  trend_direction: 'up' | 'down' | 'stable';
}

interface MarketInsightsCardProps {
  trends: MarketTrend[];
  salaryInsights: SalaryInsight[];
  skillDemands: SkillDemand[];
  loading?: boolean;
  onRefresh?: () => void;
  className?: string;
}

export const MarketInsightsCard: React.FC<MarketInsightsCardProps> = ({
  trends = [],
  salaryInsights = [],
  skillDemands = [],
  loading = false,
  onRefresh,
  className = '',
}) => {
  const [activeTab, setActiveTab] = useState<'trends' | 'salary' | 'skills'>('trends');

  const getTrendIcon = (direction: string) => {
    switch (direction) {
      case 'up':
        return <TrendingUp className="w-4 h-4 text-green-600" />;
      case 'down':
        return <TrendingDown className="w-4 h-4 text-red-600" />;
      default:
        return <BarChart3 className="w-4 h-4 text-blue-600" />;
    }
  };

  const getImpactColor = (level: string) => {
    switch (level.toLowerCase()) {
      case 'high':
        return 'text-red-600 bg-red-50 border-red-200';
      case 'medium':
        return 'text-orange-600 bg-orange-50 border-orange-200';
      case 'low':
        return 'text-green-600 bg-green-50 border-green-200';
      default:
        return 'text-gray-600 bg-gray-50 border-gray-200';
    }
  };

  const getDemandColor = (demand: string) => {
    switch (demand.toLowerCase()) {
      case 'high':
        return 'text-green-600 bg-green-50';
      case 'medium':
        return 'text-orange-600 bg-orange-50';
      case 'low':
        return 'text-red-600 bg-red-50';
      default:
        return 'text-gray-600 bg-gray-50';
    }
  };

  const formatSalary = (amount: number, currency: string = 'INR') => {
    if (currency === 'INR') {
      if (amount >= 100000) {
        return `₹${(amount / 100000).toFixed(1)}L`;
      }
      return `₹${(amount / 1000).toFixed(0)}K`;
    }
    return `$${(amount / 1000).toFixed(0)}K`;
  };

  const formatPercentage = (value: number) => {
    const sign = value >= 0 ? '+' : '';
    return `${sign}${value.toFixed(1)}%`;
  };

  const getRelevanceColor = (score: number) => {
    if (score >= 8) return 'text-green-600';
    if (score >= 6) return 'text-blue-600';
    return 'text-orange-600';
  };

  return (
    <Card className={`h-full ${className}`}>
      <CardHeader className="pb-4">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center space-x-2">
            <TrendingUp className="w-5 h-5 text-blue-600" />
            <span>Market Insights</span>
            <Badge variant="secondary" className="ml-2">
              Live Data
            </Badge>
          </CardTitle>
          <Button
            variant="ghost"
            size="sm"
            onClick={onRefresh}
            disabled={loading}
            className="flex items-center space-x-1"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
            <span>Refresh</span>
          </Button>
        </div>
        
        <div className="flex items-center space-x-4 text-sm text-muted-foreground mt-2">
          <div className="flex items-center space-x-1">
            <Globe className="w-4 h-4" />
            <span>Indian Tech Market</span>
          </div>
          <div className="flex items-center space-x-1">
            <Calendar className="w-4 h-4" />
            <span>Updated daily</span>
          </div>
        </div>

        {/* Tab Navigation */}
        <div className="flex space-x-1 bg-gray-100 rounded-lg p-1 mt-4">
          {[
            { key: 'trends', label: 'Trends', icon: TrendingUp },
            { key: 'salary', label: 'Salary', icon: DollarSign },
            { key: 'skills', label: 'Skills', icon: Star },
          ].map(({ key, label, icon: Icon }) => (
            <button
              key={key}
              onClick={() => setActiveTab(key as any)}
              className={`flex items-center space-x-2 px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                activeTab === key
                  ? 'bg-white text-blue-600 shadow-sm'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              <Icon className="w-4 h-4" />
              <span>{label}</span>
            </button>
          ))}
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {loading ? (
          <div className="space-y-4">
            {[1, 2, 3].map((i) => (
              <div key={i} className="animate-pulse">
                <div className="h-4 bg-gray-200 rounded w-3/4 mb-2"></div>
                <div className="h-3 bg-gray-200 rounded w-1/2 mb-2"></div>
                <div className="h-2 bg-gray-200 rounded w-full"></div>
              </div>
            ))}
          </div>
        ) : (
          <div className="max-h-80 overflow-y-auto">
            <AnimatePresence mode="wait">
              {/* Market Trends Tab */}
              {activeTab === 'trends' && (
                <motion.div
                  key="trends"
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  className="space-y-4"
                >
                  {trends.length === 0 ? (
                    <div className="text-center py-8 text-muted-foreground">
                      <TrendingUp className="w-12 h-12 mx-auto mb-4 opacity-50" />
                      <p>No market trends available</p>
                    </div>
                  ) : (
                    trends.map((trend, index) => (
                      <motion.div
                        key={trend.id}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: index * 0.1 }}
                        className="border border-gray-200 rounded-lg p-4"
                      >
                        <div className="flex items-start justify-between mb-3">
                          <div className="flex items-start space-x-3">
                            {getTrendIcon(trend.trend_direction)}
                            <div className="flex-1">
                              <h4 className="font-semibold text-gray-900 mb-1">{trend.title}</h4>
                              <p className="text-sm text-muted-foreground mb-2">
                                {trend.description}
                              </p>
                            </div>
                          </div>
                          <div className="flex flex-col items-end space-y-1">
                            <Badge className={`${getImpactColor(trend.impact_level)} border text-xs`}>
                              {trend.impact_level} Impact
                            </Badge>
                            <div className={`text-xs font-medium ${getRelevanceColor(trend.relevance_score)}`}>
                              {trend.relevance_score}/10 relevance
                            </div>
                          </div>
                        </div>

                        {trend.data_points && (
                          <div className="bg-gray-50 rounded-lg p-3">
                            <div className="flex items-center justify-between text-sm">
                              <span className="text-muted-foreground">Change:</span>
                              <div className="flex items-center space-x-2">
                                <span className="font-medium">
                                  {trend.data_points.current_value}
                                </span>
                                <span className={`text-xs ${
                                  trend.data_points.change_percentage >= 0 
                                    ? 'text-green-600' 
                                    : 'text-red-600'
                                }`}>
                                  {formatPercentage(trend.data_points.change_percentage)}
                                </span>
                              </div>
                            </div>
                          </div>
                        )}

                        <div className="flex items-center justify-between text-xs text-muted-foreground mt-3">
                          <span className="capitalize">{trend.category}</span>
                          <span>{trend.time_period}</span>
                        </div>
                      </motion.div>
                    ))
                  )}
                </motion.div>
              )}

              {/* Salary Insights Tab */}
              {activeTab === 'salary' && (
                <motion.div
                  key="salary"
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  className="space-y-4"
                >
                  {salaryInsights.length === 0 ? (
                    <div className="text-center py-8 text-muted-foreground">
                      <DollarSign className="w-12 h-12 mx-auto mb-4 opacity-50" />
                      <p>No salary insights available</p>
                    </div>
                  ) : (
                    salaryInsights.map((insight, index) => (
                      <motion.div
                        key={`${insight.role}-${insight.location}`}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: index * 0.1 }}
                        className="border border-gray-200 rounded-lg p-4"
                      >
                        <div className="flex items-start justify-between mb-3">
                          <div>
                            <h4 className="font-semibold text-gray-900 mb-1">{insight.role}</h4>
                            <div className="flex items-center space-x-2 text-sm text-muted-foreground">
                              <MapPin className="w-4 h-4" />
                              <span>{insight.location}</span>
                              <span>•</span>
                              <span>{insight.experience_level}</span>
                            </div>
                          </div>
                          <Badge className={`${getDemandColor(insight.market_demand)} border text-xs`}>
                            {insight.market_demand} Demand
                          </Badge>
                        </div>

                        <div className="grid grid-cols-2 gap-4 mb-3">
                          <div className="bg-green-50 rounded-lg p-3">
                            <div className="text-xs text-muted-foreground mb-1">Salary Range</div>
                            <div className="font-semibold text-green-700">
                              {formatSalary(insight.salary_range.min, insight.salary_range.currency)} - {' '}
                              {formatSalary(insight.salary_range.max, insight.salary_range.currency)}
                            </div>
                            <div className="text-xs text-muted-foreground">
                              Median: {formatSalary(insight.salary_range.median, insight.salary_range.currency)}
                            </div>
                          </div>
                          <div className="bg-blue-50 rounded-lg p-3">
                            <div className="text-xs text-muted-foreground mb-1">Growth Rate</div>
                            <div className={`font-semibold ${
                              insight.growth_rate >= 0 ? 'text-green-700' : 'text-red-700'
                            }`}>
                              {formatPercentage(insight.growth_rate)}
                            </div>
                            <div className="text-xs text-muted-foreground">Year over year</div>
                          </div>
                        </div>
                      </motion.div>
                    ))
                  )}
                </motion.div>
              )}

              {/* Skills Demand Tab */}
              {activeTab === 'skills' && (
                <motion.div
                  key="skills"
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  className="space-y-4"
                >
                  {skillDemands.length === 0 ? (
                    <div className="text-center py-8 text-muted-foreground">
                      <Star className="w-12 h-12 mx-auto mb-4 opacity-50" />
                      <p>No skill demand data available</p>
                    </div>
                  ) : (
                    skillDemands.map((skill, index) => (
                      <motion.div
                        key={skill.skill}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: index * 0.1 }}
                        className="border border-gray-200 rounded-lg p-4"
                      >
                        <div className="flex items-start justify-between mb-3">
                          <div className="flex items-center space-x-3">
                            {getTrendIcon(skill.trend_direction)}
                            <div>
                              <h4 className="font-semibold text-gray-900">{skill.skill}</h4>
                              <div className="flex items-center space-x-2 text-sm text-muted-foreground">
                                <Briefcase className="w-4 h-4" />
                                <span>{skill.job_postings_count.toLocaleString()} jobs</span>
                              </div>
                            </div>
                          </div>
                          <div className="text-right">
                            <div className="text-sm font-medium">
                              Demand Score: {skill.demand_score}/10
                            </div>
                            <Progress value={skill.demand_score * 10} className="h-2 w-20 mt-1" />
                          </div>
                        </div>

                        <div className="grid grid-cols-2 gap-4">
                          <div className="bg-blue-50 rounded-lg p-3">
                            <div className="text-xs text-muted-foreground mb-1">Growth Rate</div>
                            <div className={`font-semibold ${
                              skill.growth_rate >= 0 ? 'text-green-700' : 'text-red-700'
                            }`}>
                              {formatPercentage(skill.growth_rate)}
                            </div>
                          </div>
                          <div className="bg-green-50 rounded-lg p-3">
                            <div className="text-xs text-muted-foreground mb-1">Salary Impact</div>
                            <div className="font-semibold text-green-700">
                              +{formatPercentage(skill.avg_salary_impact)}
                            </div>
                          </div>
                        </div>
                      </motion.div>
                    ))
                  )}
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        )}

        {/* Summary Stats */}
        <div className="border-t border-gray-200 pt-4 mt-4">
          <div className="grid grid-cols-3 gap-4 text-center">
            <div>
              <div className="text-lg font-semibold text-green-600">
                {trends.filter(t => t.trend_direction === 'up').length}
              </div>
              <div className="text-xs text-muted-foreground">Growing Trends</div>
            </div>
            <div>
              <div className="text-lg font-semibold text-blue-600">
                {salaryInsights.filter(s => s.market_demand === 'High').length}
              </div>
              <div className="text-xs text-muted-foreground">High Demand Roles</div>
            </div>
            <div>
              <div className="text-lg font-semibold text-purple-600">
                {skillDemands.filter(s => s.demand_score >= 8).length}
              </div>
              <div className="text-xs text-muted-foreground">Hot Skills</div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default MarketInsightsCard;