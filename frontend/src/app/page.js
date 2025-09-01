'use client';

import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { useAuth } from '../contexts/AuthContext';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import {
  SparklesIcon,
  RocketLaunchIcon,
  ChartBarIcon,
  UserGroupIcon,
  CheckCircleIcon,
  ArrowRightIcon,
  PlayIcon,
  SunIcon,
  MoonIcon,
  Bars3Icon,
  XMarkIcon,
  CpuChipIcon,
  LightBulbIcon,
  TrophyIcon
} from '@heroicons/react/24/outline';



const HomePage = () => {
  const { user } = useAuth();
  const router = useRouter();
  const [darkMode, setDarkMode] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  useEffect(() => {
    if (user) {
      router.push('/dashboard');
    }
  }, [user, router]);

  useEffect(() => {
    // Check for saved theme preference or default to light mode
    const savedTheme = localStorage.getItem('theme');
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    
    if (savedTheme === 'dark' || (!savedTheme && prefersDark)) {
      setDarkMode(true);
      document.documentElement.classList.add('dark');
    }
  }, []);

  const toggleDarkMode = () => {
    setDarkMode(!darkMode);
    if (!darkMode) {
      document.documentElement.classList.add('dark');
      localStorage.setItem('theme', 'dark');
    } else {
      document.documentElement.classList.remove('dark');
      localStorage.setItem('theme', 'light');
    }
  };

  const features = [
    {
      icon: CpuChipIcon,
      title: 'AI-Powered Analysis',
      description: 'Advanced machine learning algorithms analyze your skills, experience, and career goals to provide personalized recommendations.',
      gradient: 'from-primary-500 to-primary-600'
    },
    {
      icon: ChartBarIcon,
      title: 'Career Insights',
      description: 'Get detailed analytics on market trends, salary expectations, and growth opportunities in your field.',
      gradient: 'from-accent-500 to-accent-600'
    },
    {
      icon: LightBulbIcon,
      title: 'Smart Recommendations',
      description: 'Receive tailored job suggestions, skill development paths, and career advancement strategies.',
      gradient: 'from-primary-600 to-accent-500'
    },
    {
      icon: TrophyIcon,
      title: 'Success Tracking',
      description: 'Monitor your progress with detailed metrics and celebrate your career milestones.',
      gradient: 'from-accent-600 to-primary-500'
    }
  ];

  const stats = [
    { label: 'Active Users', value: '50K+', description: 'Professionals trust our platform' },
    { label: 'Job Matches', value: '1.2M+', description: 'Successful career connections' },
    { label: 'Success Rate', value: '94%', description: 'Users find better opportunities' },
    { label: 'Companies', value: '5K+', description: 'Partner organizations' }
  ];

  return (
    <div className="min-h-screen bg-white dark:bg-gray-900 transition-colors duration-300">
      {/* Navigation */}
      <nav className="fixed top-0 w-full z-50 bg-white dark:bg-gray-900 border-b border-gray-200 dark:border-gray-700 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            {/* Logo */}
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <h1 className="text-2xl font-bold gradient-text">CareerAI</h1>
              </div>
            </div>

            {/* Desktop Navigation */}
            <div className="hidden md:block">
              <div className="ml-10 flex items-baseline space-x-4">
                <Link href="#features" className="text-gray-700 dark:text-gray-300 hover:text-primary-600 dark:hover:text-accent-400 px-3 py-2 rounded-md text-sm font-medium transition-colors">
                  Features
                </Link>
                <Link href="#how-it-works" className="text-gray-700 dark:text-gray-300 hover:text-primary-600 dark:hover:text-accent-400 px-3 py-2 rounded-md text-sm font-medium transition-colors">
                  How It Works
                </Link>
                <Link href="#pricing" className="text-gray-700 dark:text-gray-300 hover:text-primary-600 dark:hover:text-accent-400 px-3 py-2 rounded-md text-sm font-medium transition-colors">
                  Pricing
                </Link>
              </div>
            </div>

            {/* Right side buttons */}
            <div className="hidden md:flex items-center space-x-4">
              <button
                onClick={toggleDarkMode}
                className="p-2 rounded-lg bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
              >
                {darkMode ? <SunIcon className="w-5 h-5" /> : <MoonIcon className="w-5 h-5" />}
              </button>
              <Link href="/login" className="text-gray-700 dark:text-gray-300 hover:text-primary-600 dark:hover:text-accent-400 px-4 py-2 rounded-md text-sm font-medium transition-colors">
                Sign In
              </Link>
              <Link href="/signup" className="bg-primary-600 dark:bg-accent-500 hover:bg-primary-700 dark:hover:bg-accent-600 text-white px-6 py-2 rounded-lg text-sm font-medium transition-colors shadow-lg">
                Get Started
              </Link>
            </div>

            {/* Mobile menu button */}
            <div className="md:hidden flex items-center space-x-2">
              <button
                onClick={toggleDarkMode}
                className="p-2 rounded-lg bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400"
              >
                {darkMode ? <SunIcon className="w-5 h-5" /> : <MoonIcon className="w-5 h-5" />}
              </button>
              <button
                onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
                className="p-2 rounded-lg bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400"
              >
                {mobileMenuOpen ? <XMarkIcon className="w-6 h-6" /> : <Bars3Icon className="w-6 h-6" />}
              </button>
            </div>
          </div>
        </div>

        {/* Mobile menu */}
        {mobileMenuOpen && (
          <div className="md:hidden bg-white dark:bg-gray-900 border-t border-gray-200 dark:border-gray-700">
            <div className="px-2 pt-2 pb-3 space-y-1">
              <Link href="#features" className="block px-3 py-2 text-gray-700 dark:text-gray-300 hover:text-primary-600 dark:hover:text-accent-400">
                Features
              </Link>
              <Link href="#how-it-works" className="block px-3 py-2 text-gray-700 dark:text-gray-300 hover:text-primary-600 dark:hover:text-accent-400">
                How It Works
              </Link>
              <Link href="#pricing" className="block px-3 py-2 text-gray-700 dark:text-gray-300 hover:text-primary-600 dark:hover:text-accent-400">
                Pricing
              </Link>
              <div className="border-t border-gray-200 dark:border-gray-700 pt-4">
                <Link href="/login" className="block px-3 py-2 text-gray-700 dark:text-gray-300 hover:text-primary-600 dark:hover:text-accent-400">
                  Sign In
                </Link>
                <Link href="/signup" className="block px-3 py-2 bg-primary-600 dark:bg-accent-500 text-white rounded-lg mt-2 text-center">
                  Get Started
                </Link>
              </div>
            </div>
          </div>
        )}
      </nav>

      {/* Hero Section */}
      <section className="pt-20 pb-16 px-4 sm:px-6 lg:px-8 min-h-screen flex items-center">
        <div className="max-w-7xl mx-auto w-full">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
            {/* Left side - Content */}
            <motion.div
              initial={{ opacity: 0, x: -50 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8 }}
              className="text-center lg:text-left"
            >
              <div className="inline-flex items-center px-4 py-2 bg-primary-100 dark:bg-accent-900/30 rounded-full mb-6">
                <SparklesIcon className="w-5 h-5 text-primary-600 dark:text-accent-400 mr-2" />
                <span className="text-sm font-medium text-primary-700 dark:text-accent-300">AI-Powered Career Platform</span>
              </div>
              
              <h1 className="text-4xl lg:text-6xl font-bold mb-6 leading-tight">
                <span className="gradient-text">Transform</span>
                <br />
                Your Career
                <br />
                <span className="text-gray-900 dark:text-white">Journey</span>
              </h1>
              
              <p className="text-lg text-gray-600 dark:text-gray-300 mb-8 leading-relaxed max-w-xl">
                Discover your perfect career path with AI-powered recommendations, 
                personalized learning, and data-driven insights that accelerate your professional growth.
              </p>

              <div className="flex flex-col sm:flex-row gap-4 mb-12">
                <Link href="/signup" className="bg-primary-600 dark:bg-accent-500 hover:bg-primary-700 dark:hover:bg-accent-600 text-white px-6 py-3 rounded-lg font-semibold transition-all duration-300 shadow-lg hover:shadow-xl flex items-center justify-center">
                  <RocketLaunchIcon className="w-5 h-5 mr-2" />
                  Start Your Journey
                </Link>
                <button className="border-2 border-primary-600 dark:border-accent-500 text-primary-600 dark:text-accent-400 hover:bg-primary-50 dark:hover:bg-accent-900/20 px-6 py-3 rounded-lg font-semibold transition-all duration-300 flex items-center justify-center">
                  <PlayIcon className="w-5 h-5 mr-2" />
                  Watch Demo
                </button>
              </div>

              {/* Stats */}
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-6">
                {stats.map((stat, index) => (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.6, delay: index * 0.1 }}
                    className="text-center lg:text-left"
                  >
                    <div className="text-2xl font-bold text-primary-600 dark:text-accent-400 mb-1">
                      {stat.value}
                    </div>
                    <div className="text-sm font-medium text-gray-900 dark:text-white mb-1">
                      {stat.label}
                    </div>
                    <div className="text-xs text-gray-600 dark:text-gray-400">
                      {stat.description}
                    </div>
                  </motion.div>
                ))}
              </div>
            </motion.div>

            {/* Right side - Visual */}
            <motion.div
              initial={{ opacity: 0, x: 50 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8, delay: 0.2 }}
              className="flex items-center justify-center lg:justify-end"
            >
              <div className="relative w-96 h-96">
                <div className="w-full h-full bg-gradient-to-br from-primary-100 to-primary-200 dark:from-accent-900/20 dark:to-accent-800/20 rounded-3xl flex items-center justify-center shadow-xl border border-primary-200 dark:border-accent-700">
                  <div className="text-center">
                    <div className="w-20 h-20 bg-gradient-to-br from-primary-500 to-primary-600 dark:from-accent-500 dark:to-accent-600 rounded-2xl flex items-center justify-center mx-auto mb-4 shadow-lg">
                      <SparklesIcon className="w-10 h-10 text-white" />
                    </div>
                    <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-2">AI-Powered</h3>
                    <p className="text-gray-600 dark:text-gray-400 text-sm">Career Intelligence</p>
                  </div>
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="py-20 px-4 sm:px-6 lg:px-8 bg-gray-50 dark:bg-gray-800/50">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl lg:text-5xl font-bold mb-6">
              <span className="gradient-text">Powerful Features</span>
            </h2>
            <p className="text-lg text-gray-600 dark:text-gray-300 max-w-2xl mx-auto">
              Everything you need to accelerate your career and achieve your professional goals
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {features.map((feature, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                viewport={{ once: true }}
                className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-lg hover:shadow-xl transition-all duration-300 border border-gray-200 dark:border-gray-700"
              >
                <div className={`w-16 h-16 bg-gradient-to-r ${feature.gradient} rounded-2xl flex items-center justify-center mb-6`}>
                  <feature.icon className="w-8 h-8 text-white" />
                </div>
                <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">{feature.title}</h3>
                <p className="text-gray-600 dark:text-gray-300 leading-relaxed">{feature.description}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section id="how-it-works" className="py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl lg:text-5xl font-bold mb-6">
              <span className="gradient-text">How It Works</span>
            </h2>
            <p className="text-lg text-gray-600 dark:text-gray-300 max-w-2xl mx-auto">
              Simple steps to transform your career with AI-powered insights
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {[
              {
                step: '01',
                title: 'Create Your Profile',
                description: 'Upload your resume and connect your professional accounts for comprehensive analysis.',
                icon: UserGroupIcon
              },
              {
                step: '02',
                title: 'AI Analysis',
                description: 'Our advanced AI analyzes your skills, experience, and career goals to create your unique profile.',
                icon: SparklesIcon
              },
              {
                step: '03',
                title: 'Get Recommendations',
                description: 'Receive personalized job matches, learning paths, and career guidance tailored to your goals.',
                icon: RocketLaunchIcon
              }
            ].map((item, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.2 }}
                viewport={{ once: true }}
                className="text-center"
              >
                <div className="relative mb-8">
                  <div className="w-16 h-16 bg-gradient-to-r from-primary-500 to-accent-500 dark:from-accent-500 dark:to-primary-500 rounded-full flex items-center justify-center mx-auto mb-4 shadow-lg">
                    <item.icon className="w-8 h-8 text-white" />
                  </div>
                  <div className="absolute -top-2 -right-2 w-8 h-8 bg-primary-600 dark:bg-accent-500 rounded-full flex items-center justify-center text-white font-bold text-sm shadow-lg">
                    {item.step}
                  </div>
                </div>
                <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">{item.title}</h3>
                <p className="text-gray-600 dark:text-gray-300 leading-relaxed">{item.description}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 px-4 sm:px-6 lg:px-8 bg-gradient-to-r from-primary-600 to-accent-600 dark:from-accent-600 dark:to-primary-600">
        <div className="max-w-4xl mx-auto text-center">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
          >
            <h2 className="text-3xl lg:text-4xl font-bold text-white mb-6">
              Ready to Transform Your Career?
            </h2>
            <p className="text-lg text-white/90 mb-8 leading-relaxed max-w-2xl mx-auto">
              Join thousands of professionals who are already using AI to accelerate their careers. 
              Start your journey today and unlock your full potential.
            </p>
            
            <div className="flex flex-col sm:flex-row gap-4 justify-center items-center mb-8">
              <Link href="/signup" className="bg-white text-primary-600 dark:text-accent-600 hover:bg-gray-100 px-6 py-3 rounded-lg font-semibold transition-all duration-300 shadow-lg hover:shadow-xl flex items-center">
                <RocketLaunchIcon className="w-5 h-5 mr-2" />
                Get Started Free
              </Link>
              <Link href="/demo" className="border-2 border-white text-white hover:bg-white hover:text-primary-600 dark:hover:text-accent-600 px-6 py-3 rounded-lg font-semibold transition-all duration-300 flex items-center">
                <ArrowRightIcon className="w-5 h-5 mr-2" />
                Learn More
              </Link>
            </div>

            <div className="flex flex-wrap justify-center gap-6 text-sm text-white/80">
              <div className="flex items-center">
                <CheckCircleIcon className="w-5 h-5 mr-2" />
                <span>Free to start</span>
              </div>
              <div className="flex items-center">
                <CheckCircleIcon className="w-5 h-5 mr-2" />
                <span>No credit card required</span>
              </div>
              <div className="flex items-center">
                <CheckCircleIcon className="w-5 h-5 mr-2" />
                <span>Setup in 2 minutes</span>
              </div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-gray-900 dark:bg-gray-950 text-white py-12 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
            <div className="col-span-1 md:col-span-2">
              <h3 className="text-2xl font-bold gradient-text mb-4">CareerAI</h3>
              <p className="text-gray-400 mb-4 max-w-md">
                Transform your career with AI-powered recommendations and personalized insights.
              </p>
            </div>
            <div>
              <h4 className="text-lg font-semibold mb-4">Product</h4>
              <ul className="space-y-2 text-gray-400">
                <li><Link href="/features" className="hover:text-white transition-colors">Features</Link></li>
                <li><Link href="/pricing" className="hover:text-white transition-colors">Pricing</Link></li>
                <li><Link href="/demo" className="hover:text-white transition-colors">Demo</Link></li>
              </ul>
            </div>
            <div>
              <h4 className="text-lg font-semibold mb-4">Company</h4>
              <ul className="space-y-2 text-gray-400">
                <li><Link href="/about" className="hover:text-white transition-colors">About</Link></li>
                <li><Link href="/contact" className="hover:text-white transition-colors">Contact</Link></li>
                <li><Link href="/privacy" className="hover:text-white transition-colors">Privacy</Link></li>
              </ul>
            </div>
          </div>
          <div className="border-t border-gray-800 mt-8 pt-8 text-center text-gray-400">
            <p>&copy; 2024 CareerAI. All rights reserved.</p>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default HomePage;