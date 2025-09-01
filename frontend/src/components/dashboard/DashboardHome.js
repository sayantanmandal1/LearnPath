'use client';

import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { useAuth } from '../../contexts/AuthContext';
import GlassCard from '../ui/GlassCard';
import AnimatedButton from '../ui/AnimatedButton';
import AnimatedBackground from '../ui/AnimatedBackground';
import {
  ChartBarIcon,
  BriefcaseIcon,
  AcademicCapIcon,
  UserIcon,
  StarIcon,
  ClockIcon,
  CheckCircleIcon,
  ArrowRightIcon,
  PlusIcon,
  BellIcon
} from '@heroicons/react/24/outline';
import anime from 'animejs';

const DashboardHome = () => {
  const { user, userProfile, updateAnalytics } = useAuth();
  const [dashboardData, setDashboardData] = useState({
    stats: {
      profileCompletion: 85,
      jobMatches: 24,
      skillsAssessed: 12,
      learningProgress: 67
    },
    recentActivity: [],
    recommendations: [],
    upcomingTasks: []
  });

  useEffect(() => {
    // Update analytics
    updateAnalytics({ profileViews: 1 });

    // Animate dashboard elements
    anime({
      targets: '.dashboard-card',
      translateY: [30, 0],
      opacity: [0, 1],
      duration: 800,
      delay: anime.stagger(100),
      easing: 'easeOutExpo'
    });

    // Simulate loading dashboard data
    setTimeout(() => {
      setDashboardData({
        stats: {
          profileCompletion: 85,
          jobMatches: 24,
          skillsAssessed: 12,
          learningProgress: 67
        },
        recentActivity: [
          {
            id: 1,
            type: 'job_match',
            title: 'New job match found',
            description: 'Senior Software Engineer at TechCorp',
            time: '2 hours ago',
            icon: BriefcaseIcon,
            color: 'text-blue-400'
          },
          {
            id: 2,
            type: 'skill_assessment',
            title: 'Skill assessment completed',
            description: 'React.js - Advanced level achieved',
            time: '1 day ago',
            icon: CheckCircleIcon,
            color: 'text-green-400'
          },
          {
            id: 3,
            type: 'learning',
            title: 'Course progress updated',
            description: 'Machine Learning Fundamentals - 75% complete',
            time: '2 days ago',
            icon: AcademicCapIcon,
            color: 'text-purple-400'
          }
        ],
        recommendations: [
          {
            id: 1,
            type: 'job',
            title: 'Frontend Developer',
            company: 'Startup Inc.',
            match: 92,
            salary: '$80k - $120k',
            location: 'Remote'
          },
          {
            id: 2,
            type: 'job',
            title: 'Full Stack Engineer',
            company: 'BigTech Corp',
            match: 88,
            salary: '$100k - $150k',
            location: 'San Francisco'
          },
          {
            id: 3,
            type: 'learning',
            title: 'Advanced React Patterns',
            provider: 'TechEd',
            duration: '6 weeks',
            rating: 4.8
          }
        ],
        upcomingTasks: [
          {
            id: 1,
            title: 'Complete Python skill assessment',
            dueDate: 'Today',
            priority: 'high'
          },
          {
            id: 2,
            title: 'Update portfolio with recent projects',
            dueDate: 'Tomorrow',
            priority: 'medium'
          },
          {
            id: 3,
            title: 'Review job applications',
            dueDate: 'This week',
            priority: 'low'
          }
        ]
      });
    }, 1000);
  }, [updateAnalytics]);

  const StatCard = ({ icon: Icon, title, value, change, color }) => (
    <GlassCard className="p-6 dashboard-card" hover>
      <div className="flex items-center justify-between">
        <div>
          <p className="text-gray-400 text-sm mb-1">{title}</p>
          <p className="text-2xl font-bold text-white">{value}</p>
          {change && (
            <div className={`flex items-center mt-2 text-sm ${change > 0 ? 'text-green-400' : 'text-red-400'}`}>
              <ChartBarIcon className="w-4 h-4 mr-1" />
              <span>{change > 0 ? '+' : ''}{change}%</span>
            </div>
          )}
        </div>
        <div className={`w-12 h-12 rounded-xl bg-gradient-to-r ${color} flex items-center justify-center`}>
          <Icon className="w-6 h-6 text-white" />
        </div>
      </div>
    </GlassCard>
  );

  const ActivityItem = ({ activity }) => (
    <motion.div
      className="flex items-start space-x-3 p-4 rounded-lg hover:bg-white/5 transition-colors cursor-pointer"
      whileHover={{ x: 5 }}
    >
      <div className={`w-10 h-10 rounded-full bg-gradient-to-r from-primary-500 to-secondary-500 flex items-center justify-center`}>
        <activity.icon className={`w-5 h-5 ${activity.color}`} />
      </div>
      <div className="flex-1">
        <h4 className="text-white font-medium">{activity.title}</h4>
        <p className="text-gray-400 text-sm">{activity.description}</p>
        <p className="text-gray-500 text-xs mt-1">{activity.time}</p>
      </div>
    </motion.div>
  );

  const RecommendationCard = ({ recommendation }) => (
    <GlassCard className="p-6" hover>
      <div className="flex items-start justify-between mb-4">
        <div>
          <h4 className="text-white font-semibold">{recommendation.title}</h4>
          {recommendation.company && (
            <p className="text-gray-400 text-sm">{recommendation.company}</p>
          )}
          {recommendation.provider && (
            <p className="text-gray-400 text-sm">{recommendation.provider}</p>
          )}
        </div>
        {recommendation.match && (
          <div className="text-right">
            <div className="text-green-400 font-bold">{recommendation.match}%</div>
            <div className="text-xs text-gray-400">Match</div>
          </div>
        )}
      </div>
      
      <div className="space-y-2 text-sm text-gray-300">
        {recommendation.salary && (
          <div className="flex justify-between">
            <span>Salary:</span>
            <span>{recommendation.salary}</span>
          </div>
        )}
        {recommendation.location && (
          <div className="flex justify-between">
            <span>Location:</span>
            <span>{recommendation.location}</span>
          </div>
        )}
        {recommendation.duration && (
          <div className="flex justify-between">
            <span>Duration:</span>
            <span>{recommendation.duration}</span>
          </div>
        )}
        {recommendation.rating && (
          <div className="flex justify-between items-center">
            <span>Rating:</span>
            <div className="flex items-center">
              <StarIcon className="w-4 h-4 text-yellow-400 mr-1" />
              <span>{recommendation.rating}</span>
            </div>
          </div>
        )}
      </div>
      
      <AnimatedButton
        variant="ghost"
        size="sm"
        className="w-full mt-4"
        icon={<ArrowRightIcon className="w-4 h-4" />}
        iconPosition="right"
      >
        View Details
      </AnimatedButton>
    </GlassCard>
  );

  const TaskItem = ({ task }) => (
    <motion.div
      className="flex items-center space-x-3 p-3 rounded-lg hover:bg-white/5 transition-colors cursor-pointer"
      whileHover={{ x: 5 }}
    >
      <div className={`w-3 h-3 rounded-full ${
        task.priority === 'high' ? 'bg-red-500' :
        task.priority === 'medium' ? 'bg-yellow-500' :
        'bg-green-500'
      }`} />
      <div className="flex-1">
        <p className="text-white text-sm">{task.title}</p>
        <p className="text-gray-400 text-xs">{task.dueDate}</p>
      </div>
      <CheckCircleIcon className="w-5 h-5 text-gray-400 hover:text-green-400 transition-colors" />
    </motion.div>
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-dark-900 via-dark-800 to-dark-900 relative">
      <AnimatedBackground variant="particles" />
      
      <div className="relative z-10 p-6">
        <div className="max-w-7xl mx-auto">
          {/* Header */}
          <motion.div
            className="mb-8"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <div className="flex items-center justify-between">
              <div>
                <h1 className="text-4xl font-bold text-white mb-2">
                  Welcome back, {userProfile?.profile?.firstName || user?.displayName || 'User'}! 👋
                </h1>
                <p className="text-gray-400">
                  Here's what's happening with your career journey today.
                </p>
              </div>
              <div className="flex items-center space-x-4">
                <motion.div
                  className="relative"
                  whileHover={{ scale: 1.05 }}
                >
                  <BellIcon className="w-6 h-6 text-gray-400 cursor-pointer hover:text-white transition-colors" />
                  <div className="absolute -top-1 -right-1 w-3 h-3 bg-red-500 rounded-full"></div>
                </motion.div>
                <AnimatedButton
                  variant="primary"
                  size="sm"
                  icon={<PlusIcon className="w-4 h-4" />}
                >
                  Quick Action
                </AnimatedButton>
              </div>
            </div>
          </motion.div>

          {/* Stats Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            <StatCard
              icon={UserIcon}
              title="Profile Completion"
              value={`${dashboardData.stats.profileCompletion}%`}
              change={5}
              color="from-blue-500 to-cyan-500"
            />
            <StatCard
              icon={BriefcaseIcon}
              title="Job Matches"
              value={dashboardData.stats.jobMatches}
              change={12}
              color="from-green-500 to-emerald-500"
            />
            <StatCard
              icon={CheckCircleIcon}
              title="Skills Assessed"
              value={dashboardData.stats.skillsAssessed}
              change={8}
              color="from-purple-500 to-pink-500"
            />
            <StatCard
              icon={AcademicCapIcon}
              title="Learning Progress"
              value={`${dashboardData.stats.learningProgress}%`}
              change={15}
              color="from-orange-500 to-red-500"
            />
          </div>

          {/* Main Content Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            {/* Recent Activity */}
            <div className="lg:col-span-1">
              <GlassCard className="p-6 dashboard-card">
                <div className="flex items-center justify-between mb-6">
                  <h3 className="text-xl font-semibold text-white">Recent Activity</h3>
                  <ClockIcon className="w-5 h-5 text-gray-400" />
                </div>
                <div className="space-y-2">
                  {dashboardData.recentActivity.map((activity) => (
                    <ActivityItem key={activity.id} activity={activity} />
                  ))}
                </div>
                <AnimatedButton
                  variant="ghost"
                  size="sm"
                  className="w-full mt-4"
                >
                  View All Activity
                </AnimatedButton>
              </GlassCard>
            </div>

            {/* Recommendations */}
            <div className="lg:col-span-2">
              <div className="mb-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-xl font-semibold text-white">Recommendations for You</h3>
                  <AnimatedButton variant="ghost" size="sm">
                    View All
                  </AnimatedButton>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {dashboardData.recommendations.slice(0, 4).map((recommendation) => (
                    <div key={recommendation.id} className="dashboard-card">
                      <RecommendationCard recommendation={recommendation} />
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Bottom Section */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mt-8">
            {/* Upcoming Tasks */}
            <GlassCard className="p-6 dashboard-card">
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-xl font-semibold text-white">Upcoming Tasks</h3>
                <PlusIcon className="w-5 h-5 text-gray-400 cursor-pointer hover:text-white transition-colors" />
              </div>
              <div className="space-y-3">
                {dashboardData.upcomingTasks.map((task) => (
                  <TaskItem key={task.id} task={task} />
                ))}
              </div>
            </GlassCard>

            {/* Quick Actions */}
            <GlassCard className="p-6 dashboard-card">
              <h3 className="text-xl font-semibold text-white mb-6">Quick Actions</h3>
              <div className="grid grid-cols-2 gap-4">
                <AnimatedButton
                  variant="glass"
                  className="h-20 flex-col"
                  icon={<BriefcaseIcon className="w-6 h-6 mb-2" />}
                >
                  Find Jobs
                </AnimatedButton>
                <AnimatedButton
                  variant="glass"
                  className="h-20 flex-col"
                  icon={<AcademicCapIcon className="w-6 h-6 mb-2" />}
                >
                  Take Assessment
                </AnimatedButton>
                <AnimatedButton
                  variant="glass"
                  className="h-20 flex-col"
                  icon={<ChartBarIcon className="w-6 h-6 mb-2" />}
                >
                  View Analytics
                </AnimatedButton>
                <AnimatedButton
                  variant="glass"
                  className="h-20 flex-col"
                  icon={<UserIcon className="w-6 h-6 mb-2" />}
                >
                  Update Profile
                </AnimatedButton>
              </div>
            </GlassCard>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DashboardHome;