'use client';

import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { useAuth } from '../../contexts/AuthContext';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import {
  ChartBarIcon,
  BriefcaseIcon,
  AcademicCapIcon,
  UserIcon,
  BellIcon,
  Cog6ToothIcon,
  ArrowRightIcon,
  TrophyIcon,
  SparklesIcon,
  RocketLaunchIcon,
  ClockIcon,
  CheckCircleIcon
} from '@heroicons/react/24/outline';

const DashboardPage = () => {
  const { user, signOut } = useAuth();
  const router = useRouter();
  const [stats, setStats] = useState({
    profileCompletion: 75,
    jobMatches: 24,
    skillsAnalyzed: 12,
    learningProgress: 60
  });

  useEffect(() => {
    if (!user) {
      router.push('/login');
    }
  }, [user, router]);

  const handleSignOut = async () => {
    try {
      await signOut();
      router.push('/');
    } catch (error) {
      console.error('Error signing out:', error);
    }
  };

  const quickActions = [
    {
      title: 'Analyze Career',
      description: 'Get AI-powered career insights',
      icon: SparklesIcon,
      href: '/analyze',
      color: 'from-primary-500 to-primary-600',
      bgColor: 'bg-primary-50 dark:bg-primary-900/20'
    },
    {
      title: 'View Jobs',
      description: 'Browse recommended positions',
      icon: BriefcaseIcon,
      href: '/jobs',
      color: 'from-accent-500 to-accent-600',
      bgColor: 'bg-accent-50 dark:bg-accent-900/20'
    },
    {
      title: 'Learning Path',
      description: 'Continue skill development',
      icon: AcademicCapIcon,
      href: '/learning',
      color: 'from-green-500 to-green-600',
      bgColor: 'bg-green-50 dark:bg-green-900/20'
    },
    {
      title: 'Profile',
      description: 'Update your information',
      icon: UserIcon,
      href: '/profile',
      color: 'from-blue-500 to-blue-600',
      bgColor: 'bg-blue-50 dark:bg-blue-900/20'
    }
  ];

  const recentActivity = [
    {
      type: 'job_match',
      title: 'New job match found',
      description: 'Senior Software Engineer at TechCorp',
      time: '2 hours ago',
      icon: BriefcaseIcon,
      color: 'text-primary-600 dark:text-primary-400'
    },
    {
      type: 'skill_analysis',
      title: 'Skill analysis completed',
      description: 'React.js proficiency updated',
      time: '1 day ago',
      icon: ChartBarIcon,
      color: 'text-accent-600 dark:text-accent-400'
    },
    {
      type: 'learning',
      title: 'Course progress',
      description: 'Advanced JavaScript - 80% complete',
      time: '2 days ago',
      icon: AcademicCapIcon,
      color: 'text-green-600 dark:text-green-400'
    }
  ];

  if (!user) {
    return (
      <div className="min-h-screen bg-white dark:bg-gray-900 flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600 dark:border-accent-400"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* Navigation */}
      <nav className="bg-white dark:bg-gray-800 shadow-sm border-b border-gray-200 dark:border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center">
              <Link href="/dashboard" className="text-2xl font-bold gradient-text">
                CareerAI
              </Link>
            </div>
            
            <div className="flex items-center space-x-4">
              <button className="p-2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 transition-colors">
                <BellIcon className="w-6 h-6" />
              </button>
              <button className="p-2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 transition-colors">
                <Cog6ToothIcon className="w-6 h-6" />
              </button>
              <div className="relative">
                <button className="flex items-center space-x-2 text-gray-700 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white transition-colors">
                  <div className="w-8 h-8 bg-gradient-to-r from-primary-500 to-accent-500 rounded-full flex items-center justify-center">
                    <span className="text-white text-sm font-medium">
                      {user.email?.charAt(0).toUpperCase()}
                    </span>
                  </div>
                  <span className="hidden md:block">{user.email}</span>
                </button>
              </div>
              <button
                onClick={handleSignOut}
                className="text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors"
              >
                Sign Out
              </button>
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Welcome Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="mb-8"
        >
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
            Welcome back, {user.displayName || user.email?.split('@')[0]}!
          </h1>
          <p className="text-gray-600 dark:text-gray-400">
            Here's what's happening with your career journey today.
          </p>
        </motion.div>

        {/* Stats Grid */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.1 }}
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8"
        >
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Profile Completion</p>
                <p className="text-2xl font-bold text-gray-900 dark:text-white">{stats.profileCompletion}%</p>
              </div>
              <div className="w-12 h-12 bg-primary-100 dark:bg-primary-900/20 rounded-lg flex items-center justify-center">
                <UserIcon className="w-6 h-6 text-primary-600 dark:text-primary-400" />
              </div>
            </div>
            <div className="mt-4 w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
              <div
                className="bg-gradient-to-r from-primary-500 to-primary-600 h-2 rounded-full transition-all duration-300"
                style={{ width: `${stats.profileCompletion}%` }}
              ></div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Job Matches</p>
                <p className="text-2xl font-bold text-gray-900 dark:text-white">{stats.jobMatches}</p>
              </div>
              <div className="w-12 h-12 bg-accent-100 dark:bg-accent-900/20 rounded-lg flex items-center justify-center">
                <BriefcaseIcon className="w-6 h-6 text-accent-600 dark:text-accent-400" />
              </div>
            </div>
            <p className="text-sm text-green-600 dark:text-green-400 mt-2">+3 new this week</p>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Skills Analyzed</p>
                <p className="text-2xl font-bold text-gray-900 dark:text-white">{stats.skillsAnalyzed}</p>
              </div>
              <div className="w-12 h-12 bg-green-100 dark:bg-green-900/20 rounded-lg flex items-center justify-center">
                <ChartBarIcon className="w-6 h-6 text-green-600 dark:text-green-400" />
              </div>
            </div>
            <p className="text-sm text-blue-600 dark:text-blue-400 mt-2">Last updated today</p>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Learning Progress</p>
                <p className="text-2xl font-bold text-gray-900 dark:text-white">{stats.learningProgress}%</p>
              </div>
              <div className="w-12 h-12 bg-blue-100 dark:bg-blue-900/20 rounded-lg flex items-center justify-center">
                <AcademicCapIcon className="w-6 h-6 text-blue-600 dark:text-blue-400" />
              </div>
            </div>
            <div className="mt-4 w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
              <div
                className="bg-gradient-to-r from-blue-500 to-blue-600 h-2 rounded-full transition-all duration-300"
                style={{ width: `${stats.learningProgress}%` }}
              ></div>
            </div>
          </div>
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Quick Actions */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="lg:col-span-2"
          >
            <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-6">Quick Actions</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {quickActions.map((action, index) => (
                <Link
                  key={index}
                  href={action.href}
                  className="group bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700 hover:shadow-md transition-all duration-300"
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className={`w-12 h-12 ${action.bgColor} rounded-lg flex items-center justify-center mb-4`}>
                        <action.icon className={`w-6 h-6 bg-gradient-to-r ${action.color} bg-clip-text text-transparent`} />
                      </div>
                      <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
                        {action.title}
                      </h3>
                      <p className="text-gray-600 dark:text-gray-400 text-sm">
                        {action.description}
                      </p>
                    </div>
                    <ArrowRightIcon className="w-5 h-5 text-gray-400 group-hover:text-gray-600 dark:group-hover:text-gray-300 transition-colors" />
                  </div>
                </Link>
              ))}
            </div>
          </motion.div>

          {/* Recent Activity */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.3 }}
          >
            <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-6">Recent Activity</h2>
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700">
              <div className="p-6">
                <div className="space-y-4">
                  {recentActivity.map((activity, index) => (
                    <div key={index} className="flex items-start space-x-3">
                      <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${activity.color === 'text-primary-600 dark:text-primary-400' ? 'bg-primary-100 dark:bg-primary-900/20' : activity.color === 'text-accent-600 dark:text-accent-400' ? 'bg-accent-100 dark:bg-accent-900/20' : 'bg-green-100 dark:bg-green-900/20'}`}>
                        <activity.icon className={`w-4 h-4 ${activity.color}`} />
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium text-gray-900 dark:text-white">
                          {activity.title}
                        </p>
                        <p className="text-sm text-gray-600 dark:text-gray-400">
                          {activity.description}
                        </p>
                        <p className="text-xs text-gray-500 dark:text-gray-500 mt-1 flex items-center">
                          <ClockIcon className="w-3 h-3 mr-1" />
                          {activity.time}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
              <div className="px-6 py-3 bg-gray-50 dark:bg-gray-700/50 rounded-b-xl">
                <Link href="/activity" className="text-sm text-primary-600 dark:text-accent-400 hover:text-primary-500 dark:hover:text-accent-300 font-medium transition-colors">
                  View all activity →
                </Link>
              </div>
            </div>
          </motion.div>
        </div>

        {/* Achievement Banner */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.4 }}
          className="mt-8 bg-gradient-to-r from-primary-600 to-accent-600 dark:from-accent-600 dark:to-primary-600 rounded-xl p-6 text-white"
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="w-12 h-12 bg-white/20 rounded-lg flex items-center justify-center">
                <TrophyIcon className="w-6 h-6 text-white" />
              </div>
              <div>
                <h3 className="text-lg font-semibold">Congratulations!</h3>
                <p className="text-white/90">You've completed 75% of your profile. Keep going!</p>
              </div>
            </div>
            <Link
              href="/profile"
              className="bg-white/20 hover:bg-white/30 text-white px-4 py-2 rounded-lg font-medium transition-colors flex items-center"
            >
              Complete Profile
              <ArrowRightIcon className="w-4 h-4 ml-2" />
            </Link>
          </div>
        </motion.div>
      </div>
    </div>
  );
};

export default DashboardPage;