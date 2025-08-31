'use client';

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  BarChart3, 
  Target, 
  TrendingUp, 
  Users, 
  Settings,
  Download,
  Share,
  RefreshCw,
  Eye,
  EyeOff
} from 'lucide-react';

import SkillRadarChart from './SkillRadarChart';
import CareerRoadmapVisualization from './CareerRoadmapVisualization';
import SkillGapAnalysis from './SkillGapAnalysis';
import JobCompatibilityDashboard from './JobCompatibilityDashboard';

const VisualizationDashboard = ({ 
  userData = {},
  className = "" 
}) => {
  const [activeTab, setActiveTab] = useState('overview');
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [refreshing, setRefreshing] = useState(false);

  const tabs = [
    { id: 'overview', label: 'Overview', icon: BarChart3 },
    { id: 'skills', label: 'Skills Analysis', icon: Target },
    { id: 'career', label: 'Career Path', icon: TrendingUp },
    { id: 'jobs', label: 'Job Matching', icon: Users }
  ];

  const handleRefresh = async () => {
    setRefreshing(true);
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 2000));
    setRefreshing(false);
  };

  const handleExport = () => {
    // Export functionality
    console.log('Exporting dashboard data...');
  };

  const handleShare = () => {
    // Share functionality
    console.log('Sharing dashboard...');
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.6 }}
      className={`${isFullscreen ? 'fixed inset-0 z-50 bg-white dark:bg-gray-900' : ''} ${className}`}
    >
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="bg-white dark:bg-gray-900 rounded-t-xl shadow-sm p-6 border-b border-gray-200 dark:border-gray-700"
      >
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-3xl font-bold text-gray-900 dark:text-white">
              Career Analytics Dashboard
            </h2>
            <p className="text-gray-600 dark:text-gray-400 mt-1">
              Comprehensive analysis of your career profile and opportunities
            </p>
          </div>
          
          <div className="flex items-center gap-2">
            <button
              onClick={handleRefresh}
              disabled={refreshing}
              className="p-2 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 transition-colors"
              title="Refresh Data"
            >
              <RefreshCw className={`w-5 h-5 ${refreshing ? 'animate-spin' : ''}`} />
            </button>
            
            <button
              onClick={() => setIsFullscreen(!isFullscreen)}
              className="p-2 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 transition-colors"
              title={isFullscreen ? "Exit Fullscreen" : "Enter Fullscreen"}
            >
              {isFullscreen ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
            </button>
            
            <button
              onClick={handleShare}
              className="p-2 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 transition-colors"
              title="Share Dashboard"
            >
              <Share className="w-5 h-5" />
            </button>
            
            <button
              onClick={handleExport}
              className="p-2 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 transition-colors"
              title="Export Data"
            >
              <Download className="w-5 h-5" />
            </button>
            
            <button className="p-2 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 transition-colors">
              <Settings className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Tab Navigation */}
        <div className="flex space-x-1 bg-gray-100 dark:bg-gray-800 rounded-lg p-1">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition-all duration-200 ${
                  activeTab === tab.id
                    ? 'bg-white dark:bg-gray-700 text-blue-600 dark:text-blue-400 shadow-sm'
                    : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200'
                }`}
              >
                <Icon className="w-4 h-4" />
                {tab.label}
              </button>
            );
          })}
        </div>
      </motion.div>

      {/* Content */}
      <div className={`${isFullscreen ? 'h-full overflow-auto' : ''} bg-gray-50 dark:bg-gray-800`}>
        <AnimatePresence mode="wait">
          {activeTab === 'overview' && (
            <motion.div
              key="overview"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={{ duration: 0.3 }}
              className="p-6"
            >
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <SkillRadarChart
                  skills={userData.skills || []}
                  title="Current Skill Profile"
                  showComparison={true}
                  className="lg:col-span-1"
                />
                
                <div className="space-y-6">
                  <div className="bg-white dark:bg-gray-900 rounded-xl shadow-lg p-6">
                    <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-4">
                      Quick Stats
                    </h3>
                    <div className="grid grid-cols-2 gap-4">
                      <div className="text-center">
                        <p className="text-3xl font-bold text-blue-600 dark:text-blue-400">
                          {userData.skillCount || 0}
                        </p>
                        <p className="text-sm text-gray-600 dark:text-gray-400">Skills</p>
                      </div>
                      <div className="text-center">
                        <p className="text-3xl font-bold text-green-600 dark:text-green-400">
                          {userData.matchingJobs || 0}
                        </p>
                        <p className="text-sm text-gray-600 dark:text-gray-400">Matching Jobs</p>
                      </div>
                      <div className="text-center">
                        <p className="text-3xl font-bold text-yellow-600 dark:text-yellow-400">
                          {userData.careerProgress || 0}%
                        </p>
                        <p className="text-sm text-gray-600 dark:text-gray-400">Career Progress</p>
                      </div>
                      <div className="text-center">
                        <p className="text-3xl font-bold text-purple-600 dark:text-purple-400">
                          {userData.learningHours || 0}h
                        </p>
                        <p className="text-sm text-gray-600 dark:text-gray-400">Learning Time</p>
                      </div>
                    </div>
                  </div>

                  <CareerRoadmapVisualization
                    roadmapData={userData.careerRoadmap || []}
                    currentPosition={userData.currentPosition || 0}
                    className="h-64"
                  />
                </div>
              </div>
            </motion.div>
          )}

          {activeTab === 'skills' && (
            <motion.div
              key="skills"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={{ duration: 0.3 }}
              className="p-6"
            >
              <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
                <div className="xl:col-span-2">
                  <SkillGapAnalysis
                    skillGaps={userData.skillGaps || []}
                    targetRole={userData.targetRole || "Software Engineer"}
                  />
                </div>
                
                <div className="space-y-6">
                  <SkillRadarChart
                    skills={userData.skills || []}
                    title="Skill Radar"
                    useD3={true}
                    className="h-96"
                  />
                  
                  <div className="bg-white dark:bg-gray-900 rounded-xl shadow-lg p-6">
                    <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">
                      Skill Recommendations
                    </h3>
                    <div className="space-y-3">
                      {(userData.skillRecommendations || []).slice(0, 5).map((skill, index) => (
                        <div key={index} className="flex items-center justify-between">
                          <span className="text-sm text-gray-700 dark:text-gray-300">
                            {skill.name}
                          </span>
                          <span className="text-xs bg-blue-100 dark:bg-blue-900/20 text-blue-800 dark:text-blue-200 px-2 py-1 rounded-full">
                            {skill.priority}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          )}

          {activeTab === 'career' && (
            <motion.div
              key="career"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={{ duration: 0.3 }}
              className="p-6"
            >
              <CareerRoadmapVisualization
                roadmapData={userData.careerRoadmap || []}
                currentPosition={userData.currentPosition || 0}
                className="mb-6"
              />
              
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                <div className="bg-white dark:bg-gray-900 rounded-xl shadow-lg p-6">
                  <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">
                    Career Recommendations
                  </h3>
                  <div className="space-y-3">
                    {(userData.careerRecommendations || []).slice(0, 3).map((career, index) => (
                      <div key={index} className="p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
                        <h4 className="font-medium text-gray-900 dark:text-white">
                          {career.title}
                        </h4>
                        <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                          {career.matchScore}% match
                        </p>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="bg-white dark:bg-gray-900 rounded-xl shadow-lg p-6">
                  <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">
                    Learning Paths
                  </h3>
                  <div className="space-y-3">
                    {(userData.learningPaths || []).slice(0, 3).map((path, index) => (
                      <div key={index} className="p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
                        <h4 className="font-medium text-gray-900 dark:text-white">
                          {path.title}
                        </h4>
                        <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                          {path.duration} • {path.difficulty}
                        </p>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="bg-white dark:bg-gray-900 rounded-xl shadow-lg p-6">
                  <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">
                    Market Trends
                  </h3>
                  <div className="space-y-3">
                    {(userData.marketTrends || []).slice(0, 3).map((trend, index) => (
                      <div key={index} className="p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
                        <h4 className="font-medium text-gray-900 dark:text-white">
                          {trend.skill}
                        </h4>
                        <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                          {trend.growth > 0 ? '+' : ''}{trend.growth}% demand
                        </p>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </motion.div>
          )}

          {activeTab === 'jobs' && (
            <motion.div
              key="jobs"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={{ duration: 0.3 }}
              className="p-6"
            >
              <JobCompatibilityDashboard
                jobs={userData.jobs || []}
                userSkills={userData.skills || []}
              />
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </motion.div>
  );
};

export default VisualizationDashboard;