'use client';

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useAuth } from '../../contexts/AuthContext.jsx';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import {
  ArrowLeftIcon,
  SparklesIcon,
  DocumentTextIcon,
  BriefcaseIcon,
  AcademicCapIcon,
  ChartBarIcon,
  RocketLaunchIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon,
  LightBulbIcon,
  TrophyIcon,
  ArrowRightIcon
} from '@heroicons/react/24/outline';



const AnalyzePage = () => {
  const { user } = useAuth();
  const router = useRouter();
  const [currentStep, setCurrentStep] = useState(1);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisComplete, setAnalysisComplete] = useState(false);
  const [formData, setFormData] = useState({
    // Personal Information
    currentRole: '',
    experience: '',
    industry: '',
    location: '',

    // Career Goals
    desiredRole: '',
    careerGoals: '',
    timeframe: '',
    salaryExpectation: '',

    // Skills & Education
    skills: '',
    education: '',
    certifications: '',
    languages: '',

    // Work Preferences
    workType: '',
    companySize: '',
    workCulture: '',
    benefits: []
  });

  const [analysisResults, setAnalysisResults] = useState(null);

  useEffect(() => {
    if (!user) {
      router.push('/login');
    }
  }, [user, router]);

  const handleInputChange = (e) => {
    const { name, value, type, checked } = e.target;

    if (type === 'checkbox') {
      if (name === 'benefits') {
        setFormData(prev => ({
          ...prev,
          benefits: checked
            ? [...prev.benefits, value]
            : prev.benefits.filter(benefit => benefit !== value)
        }));
      }
    } else {
      setFormData(prev => ({
        ...prev,
        [name]: value
      }));
    }
  };

  const handleNext = () => {
    if (currentStep < 4) {
      setCurrentStep(currentStep + 1);
    }
  };

  const handlePrevious = () => {
    if (currentStep > 1) {
      setCurrentStep(currentStep - 1);
    }
  };

  const handleAnalyze = async () => {
    setIsAnalyzing(true);

    // Simulate AI analysis
    setTimeout(() => {
      setAnalysisResults({
        overallScore: 85,
        strengths: [
          'Strong technical skills in React and JavaScript',
          'Excellent problem-solving abilities',
          'Good communication skills',
          'Leadership experience'
        ],
        improvements: [
          'Consider learning cloud technologies (AWS, Azure)',
          'Develop project management skills',
          'Gain experience in machine learning',
          'Build a stronger professional network'
        ],
        recommendations: [
          {
            type: 'job',
            title: 'Senior Frontend Developer',
            company: 'TechCorp Inc.',
            match: 92,
            salary: '$95,000 - $120,000',
            location: 'San Francisco, CA'
          },
          {
            type: 'job',
            title: 'Full Stack Engineer',
            company: 'StartupXYZ',
            match: 88,
            salary: '$85,000 - $110,000',
            location: 'Austin, TX'
          },
          {
            type: 'job',
            title: 'React Developer',
            company: 'Digital Agency',
            match: 85,
            salary: '$75,000 - $95,000',
            location: 'Remote'
          }
        ],
        learningPaths: [
          {
            title: 'Cloud Computing Fundamentals',
            provider: 'AWS',
            duration: '6 weeks',
            difficulty: 'Intermediate'
          },
          {
            title: 'Advanced React Patterns',
            provider: 'Frontend Masters',
            duration: '4 weeks',
            difficulty: 'Advanced'
          },
          {
            title: 'Machine Learning Basics',
            provider: 'Coursera',
            duration: '8 weeks',
            difficulty: 'Beginner'
          }
        ],
        marketInsights: {
          demandTrend: 'High',
          salaryGrowth: '+12% YoY',
          topSkills: ['React', 'TypeScript', 'Node.js', 'AWS'],
          competitionLevel: 'Medium'
        }
      });

      setIsAnalyzing(false);
      setAnalysisComplete(true);
    }, 5000);
  };

  const renderStep = () => {
    switch (currentStep) {
      case 1:
        return (
          <div className="space-y-6">
            <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">Personal Information</h3>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Current Role
                </label>
                <input
                  type="text"
                  name="currentRole"
                  value={formData.currentRole}
                  onChange={handleInputChange}
                  className="w-full px-4 py-3 bg-gray-50 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-primary-500 dark:focus:ring-accent-500 focus:border-transparent transition-all duration-300"
                  placeholder="e.g., Software Developer"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Years of Experience
                </label>
                <select
                  name="experience"
                  value={formData.experience}
                  onChange={handleInputChange}
                  className="w-full px-4 py-3 bg-gray-50 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-primary-500 dark:focus:ring-accent-500 focus:border-transparent transition-all duration-300"
                >
                  <option value="">Select experience</option>
                  <option value="0-1">0-1 years</option>
                  <option value="2-3">2-3 years</option>
                  <option value="4-6">4-6 years</option>
                  <option value="7-10">7-10 years</option>
                  <option value="10+">10+ years</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Industry
                </label>
                <input
                  type="text"
                  name="industry"
                  value={formData.industry}
                  onChange={handleInputChange}
                  className="w-full px-4 py-3 bg-gray-50 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-primary-500 dark:focus:ring-accent-500 focus:border-transparent transition-all duration-300"
                  placeholder="e.g., Technology, Healthcare"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Location
                </label>
                <input
                  type="text"
                  name="location"
                  value={formData.location}
                  onChange={handleInputChange}
                  className="w-full px-4 py-3 bg-gray-50 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-primary-500 dark:focus:ring-accent-500 focus:border-transparent transition-all duration-300"
                  placeholder="e.g., San Francisco, CA"
                />
              </div>
            </div>
          </div>
        );

      case 2:
        return (
          <div className="space-y-6">
            <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">Career Goals</h3>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Desired Role
                </label>
                <input
                  type="text"
                  name="desiredRole"
                  value={formData.desiredRole}
                  onChange={handleInputChange}
                  className="w-full px-4 py-3 bg-gray-50 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-primary-500 dark:focus:ring-accent-500 focus:border-transparent transition-all duration-300"
                  placeholder="e.g., Senior Software Engineer"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Timeframe
                </label>
                <select
                  name="timeframe"
                  value={formData.timeframe}
                  onChange={handleInputChange}
                  className="w-full px-4 py-3 bg-gray-50 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-primary-500 dark:focus:ring-accent-500 focus:border-transparent transition-all duration-300"
                >
                  <option value="">Select timeframe</option>
                  <option value="immediate">Immediate (0-3 months)</option>
                  <option value="short">Short term (3-6 months)</option>
                  <option value="medium">Medium term (6-12 months)</option>
                  <option value="long">Long term (1-2 years)</option>
                </select>
              </div>

              <div className="md:col-span-2">
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Career Goals
                </label>
                <textarea
                  name="careerGoals"
                  value={formData.careerGoals}
                  onChange={handleInputChange}
                  rows={4}
                  className="w-full px-4 py-3 bg-gray-50 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-primary-500 dark:focus:ring-accent-500 focus:border-transparent transition-all duration-300"
                  placeholder="Describe your career aspirations and goals..."
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Salary Expectation
                </label>
                <input
                  type="text"
                  name="salaryExpectation"
                  value={formData.salaryExpectation}
                  onChange={handleInputChange}
                  className="w-full px-4 py-3 bg-gray-50 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-primary-500 dark:focus:ring-accent-500 focus:border-transparent transition-all duration-300"
                  placeholder="e.g., $80,000 - $100,000"
                />
              </div>
            </div>
          </div>
        );

      case 3:
        return (
          <div className="space-y-6">
            <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">Skills & Education</h3>

            <div className="space-y-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Technical Skills
                </label>
                <textarea
                  name="skills"
                  value={formData.skills}
                  onChange={handleInputChange}
                  rows={3}
                  className="w-full px-4 py-3 bg-gray-50 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-primary-500 dark:focus:ring-accent-500 focus:border-transparent transition-all duration-300"
                  placeholder="e.g., JavaScript, React, Node.js, Python, AWS..."
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Education
                </label>
                <input
                  type="text"
                  name="education"
                  value={formData.education}
                  onChange={handleInputChange}
                  className="w-full px-4 py-3 bg-gray-50 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-primary-500 dark:focus:ring-accent-500 focus:border-transparent transition-all duration-300"
                  placeholder="e.g., Bachelor's in Computer Science"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Certifications
                </label>
                <input
                  type="text"
                  name="certifications"
                  value={formData.certifications}
                  onChange={handleInputChange}
                  className="w-full px-4 py-3 bg-gray-50 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-primary-500 dark:focus:ring-accent-500 focus:border-transparent transition-all duration-300"
                  placeholder="e.g., AWS Certified Developer, Google Cloud Professional"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Languages
                </label>
                <input
                  type="text"
                  name="languages"
                  value={formData.languages}
                  onChange={handleInputChange}
                  className="w-full px-4 py-3 bg-gray-50 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-primary-500 dark:focus:ring-accent-500 focus:border-transparent transition-all duration-300"
                  placeholder="e.g., English (Native), Spanish (Conversational)"
                />
              </div>
            </div>
          </div>
        );

      case 4:
        return (
          <div className="space-y-6">
            <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">Work Preferences</h3>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Work Type
                </label>
                <select
                  name="workType"
                  value={formData.workType}
                  onChange={handleInputChange}
                  className="w-full px-4 py-3 bg-gray-50 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-primary-500 dark:focus:ring-accent-500 focus:border-transparent transition-all duration-300"
                >
                  <option value="">Select work type</option>
                  <option value="remote">Remote</option>
                  <option value="hybrid">Hybrid</option>
                  <option value="onsite">On-site</option>
                  <option value="flexible">Flexible</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Company Size
                </label>
                <select
                  name="companySize"
                  value={formData.companySize}
                  onChange={handleInputChange}
                  className="w-full px-4 py-3 bg-gray-50 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-primary-500 dark:focus:ring-accent-500 focus:border-transparent transition-all duration-300"
                >
                  <option value="">Select company size</option>
                  <option value="startup">Startup (1-50)</option>
                  <option value="small">Small (51-200)</option>
                  <option value="medium">Medium (201-1000)</option>
                  <option value="large">Large (1000+)</option>
                </select>
              </div>

              <div className="md:col-span-2">
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Work Culture Preferences
                </label>
                <textarea
                  name="workCulture"
                  value={formData.workCulture}
                  onChange={handleInputChange}
                  rows={3}
                  className="w-full px-4 py-3 bg-gray-50 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-primary-500 dark:focus:ring-accent-500 focus:border-transparent transition-all duration-300"
                  placeholder="Describe your ideal work environment and culture..."
                />
              </div>

              <div className="md:col-span-2">
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-4">
                  Important Benefits (Select all that apply)
                </label>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                  {[
                    'Health Insurance',
                    'Dental Insurance',
                    'Vision Insurance',
                    '401(k) Matching',
                    'Flexible PTO',
                    'Remote Work',
                    'Professional Development',
                    'Stock Options',
                    'Gym Membership',
                    'Commuter Benefits',
                    'Parental Leave',
                    'Mental Health Support'
                  ].map((benefit) => (
                    <label key={benefit} className="flex items-center">
                      <input
                        type="checkbox"
                        name="benefits"
                        value={benefit}
                        checked={formData.benefits.includes(benefit)}
                        onChange={handleInputChange}
                        className="h-4 w-4 text-primary-600 dark:text-accent-500 focus:ring-primary-500 dark:focus:ring-accent-500 border-gray-300 dark:border-gray-600 rounded"
                      />
                      <span className="ml-2 text-sm text-gray-700 dark:text-gray-300">{benefit}</span>
                    </label>
                  ))}
                </div>
              </div>
            </div>
          </div>
        );

      default:
        return null;
    }
  };

  if (!user) {
    return (
      <div className="min-h-screen bg-white dark:bg-gray-900 flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600 dark:border-accent-400"></div>
      </div>
    );
  }

  if (analysisComplete && analysisResults) {
    return (
      <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
        {/* Navigation */}
        <nav className="bg-white dark:bg-gray-800 shadow-sm border-b border-gray-200 dark:border-gray-700">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex justify-between items-center h-16">
              <Link href="/dashboard" className="flex items-center text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors">
                <ArrowLeftIcon className="w-5 h-5 mr-2" />
                Back to Dashboard
              </Link>
              <h1 className="text-xl font-semibold text-gray-900 dark:text-white">Career Analysis Results</h1>
              <div></div>
            </div>
          </div>
        </nav>

        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {/* Results Header */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-center mb-12"
          >
            <div className="inline-flex items-center justify-center w-20 h-20 bg-gradient-to-r from-primary-500 to-accent-500 dark:from-accent-500 dark:to-primary-500 rounded-full mb-6">
              <TrophyIcon className="w-10 h-10 text-white" />
            </div>
            <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">
              Your Career Analysis is Complete!
            </h1>
            <p className="text-xl text-gray-600 dark:text-gray-400 max-w-2xl mx-auto">
              Based on your profile, we've generated personalized insights and recommendations to accelerate your career growth.
            </p>
          </motion.div>

          {/* Overall Score */}
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.2 }}
            className="bg-gradient-to-r from-primary-600 to-accent-600 dark:from-accent-600 dark:to-primary-600 rounded-2xl p-8 text-white mb-8"
          >
            <div className="text-center">
              <h2 className="text-2xl font-bold mb-4">Overall Career Score</h2>
              <div className="text-6xl font-bold mb-2">{analysisResults.overallScore}</div>
              <div className="text-xl opacity-90">out of 100</div>
              <p className="mt-4 text-lg opacity-90">
                Excellent! You have a strong foundation for career growth.
              </p>
            </div>
          </motion.div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
            {/* Strengths */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.3 }}
              className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700"
            >
              <div className="flex items-center mb-4">
                <CheckCircleIcon className="w-6 h-6 text-green-500 mr-2" />
                <h3 className="text-xl font-bold text-gray-900 dark:text-white">Your Strengths</h3>
              </div>
              <ul className="space-y-3">
                {analysisResults.strengths.map((strength, index) => (
                  <li key={index} className="flex items-start">
                    <div className="w-2 h-2 bg-green-500 rounded-full mt-2 mr-3 flex-shrink-0"></div>
                    <span className="text-gray-700 dark:text-gray-300">{strength}</span>
                  </li>
                ))}
              </ul>
            </motion.div>

            {/* Areas for Improvement */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.4 }}
              className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700"
            >
              <div className="flex items-center mb-4">
                <LightBulbIcon className="w-6 h-6 text-accent-500 dark:text-accent-400 mr-2" />
                <h3 className="text-xl font-bold text-gray-900 dark:text-white">Growth Opportunities</h3>
              </div>
              <ul className="space-y-3">
                {analysisResults.improvements.map((improvement, index) => (
                  <li key={index} className="flex items-start">
                    <div className="w-2 h-2 bg-accent-500 dark:bg-accent-400 rounded-full mt-2 mr-3 flex-shrink-0"></div>
                    <span className="text-gray-700 dark:text-gray-300">{improvement}</span>
                  </li>
                ))}
              </ul>
            </motion.div>
          </div>

          {/* Job Recommendations */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 }}
            className="mb-8"
          >
            <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">Recommended Jobs</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {analysisResults.recommendations.map((job, index) => (
                <div key={index} className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700 hover:shadow-md transition-shadow">
                  <div className="flex items-center justify-between mb-4">
                    <BriefcaseIcon className="w-8 h-8 text-primary-600 dark:text-accent-400" />
                    <span className="bg-green-100 dark:bg-green-900/20 text-green-800 dark:text-green-400 px-2 py-1 rounded-full text-sm font-medium">
                      {job.match}% Match
                    </span>
                  </div>
                  <h4 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">{job.title}</h4>
                  <p className="text-gray-600 dark:text-gray-400 mb-2">{job.company}</p>
                  <p className="text-sm text-gray-500 dark:text-gray-500 mb-2">{job.location}</p>
                  <p className="text-primary-600 dark:text-accent-400 font-medium">{job.salary}</p>
                </div>
              ))}
            </div>
          </motion.div>

          {/* Learning Paths */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.6 }}
            className="mb-8"
          >
            <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">Recommended Learning Paths</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {analysisResults.learningPaths.map((course, index) => (
                <div key={index} className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
                  <AcademicCapIcon className="w-8 h-8 text-accent-600 dark:text-accent-400 mb-4" />
                  <h4 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">{course.title}</h4>
                  <p className="text-gray-600 dark:text-gray-400 mb-2">{course.provider}</p>
                  <div className="flex justify-between text-sm text-gray-500 dark:text-gray-500">
                    <span>{course.duration}</span>
                    <span className={`px-2 py-1 rounded-full text-xs ${course.difficulty === 'Beginner' ? 'bg-green-100 dark:bg-green-900/20 text-green-800 dark:text-green-400' :
                      course.difficulty === 'Intermediate' ? 'bg-yellow-100 dark:bg-yellow-900/20 text-yellow-800 dark:text-yellow-400' :
                        'bg-red-100 dark:bg-red-900/20 text-red-800 dark:text-red-400'
                      }`}>
                      {course.difficulty}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </motion.div>

          {/* Action Buttons */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.7 }}
            className="flex flex-col sm:flex-row gap-4 justify-center"
          >
            <Link
              href="/dashboard"
              className="bg-primary-600 dark:bg-accent-500 hover:bg-primary-700 dark:hover:bg-accent-600 text-white px-8 py-3 rounded-lg font-semibold transition-colors flex items-center justify-center"
            >
              <RocketLaunchIcon className="w-5 h-5 mr-2" />
              Continue Journey
            </Link>
            <button
              onClick={() => window.print()}
              className="border-2 border-primary-600 dark:border-accent-500 text-primary-600 dark:text-accent-400 hover:bg-primary-50 dark:hover:bg-accent-900/20 px-8 py-3 rounded-lg font-semibold transition-colors flex items-center justify-center"
            >
              <DocumentTextIcon className="w-5 h-5 mr-2" />
              Download Report
            </button>
          </motion.div>
        </div>
      </div>
    );
  }

  if (isAnalyzing) {
    return (
      <div className="min-h-screen bg-gray-50 dark:bg-gray-900 flex items-center justify-center relative">
        {/* Background Animation */}
        <div className="absolute inset-0 opacity-20">
          <div className="absolute inset-0 bg-gradient-to-br from-primary-500 to-accent-500 dark:from-accent-500 dark:to-primary-500 animate-pulse"></div>
        </div>

        <div className="relative z-10 text-center">
          <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            className="bg-white dark:bg-gray-800 rounded-2xl p-12 shadow-2xl border border-gray-200 dark:border-gray-700 max-w-md w-full"
          >
            <div className="w-20 h-20 bg-gradient-to-r from-primary-500 to-accent-500 dark:from-accent-500 dark:to-primary-500 rounded-full flex items-center justify-center mx-auto mb-6">
              <SparklesIcon className="w-10 h-10 text-white animate-pulse" />
            </div>

            <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-4">
              Analyzing Your Career Profile
            </h2>

            <p className="text-gray-600 dark:text-gray-400 mb-8 max-w-md">
              Our AI is processing your information to generate personalized insights and recommendations.
            </p>

            <div className="space-y-4">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: "100%" }}
                transition={{ duration: 5, ease: "easeInOut" }}
                className="h-2 bg-gradient-to-r from-primary-500 to-accent-500 dark:from-accent-500 dark:to-primary-500 rounded-full"
              />

              <div className="flex justify-center space-x-2">
                {[0, 1, 2].map((i) => (
                  <motion.div
                    key={i}
                    animate={{
                      scale: [1, 1.2, 1],
                      opacity: [0.5, 1, 0.5]
                    }}
                    transition={{
                      duration: 1.5,
                      repeat: Infinity,
                      delay: i * 0.2
                    }}
                    className="w-3 h-3 bg-primary-500 dark:bg-accent-400 rounded-full"
                  />
                ))}
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* Navigation */}
      <nav className="bg-white dark:bg-gray-800 shadow-sm border-b border-gray-200 dark:border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <Link href="/dashboard" className="flex items-center text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors">
              <ArrowLeftIcon className="w-5 h-5 mr-2" />
              Back to Dashboard
            </Link>
            <h1 className="text-xl font-semibold text-gray-900 dark:text-white">Career Analysis</h1>
            <div></div>
          </div>
        </div>
      </nav>

      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Progress Bar */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-4">
            <span className="text-sm font-medium text-gray-600 dark:text-gray-400">
              Step {currentStep} of 4
            </span>
            <span className="text-sm font-medium text-gray-600 dark:text-gray-400">
              {Math.round((currentStep / 4) * 100)}% Complete
            </span>
          </div>
          <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${(currentStep / 4) * 100}%` }}
              transition={{ duration: 0.5 }}
              className="bg-gradient-to-r from-primary-500 to-accent-500 dark:from-accent-500 dark:to-primary-500 h-2 rounded-full"
            ></motion.div>
          </div>
        </div>

        {/* Form */}
        <motion.div
          key={currentStep}
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: -20 }}
          transition={{ duration: 0.3 }}
          className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-lg border border-gray-200 dark:border-gray-700 mb-8"
        >
          {renderStep()}
        </motion.div>

        {/* Navigation Buttons */}
        <div className="flex justify-between">
          <button
            onClick={handlePrevious}
            disabled={currentStep === 1}
            className="px-6 py-3 border-2 border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 rounded-lg font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-50 dark:hover:bg-gray-700"
          >
            Previous
          </button>

          {currentStep < 4 ? (
            <button
              onClick={handleNext}
              className="px-6 py-3 bg-gradient-to-r from-primary-600 to-accent-600 dark:from-accent-500 dark:to-primary-500 hover:from-primary-700 hover:to-accent-700 dark:hover:from-accent-600 dark:hover:to-primary-600 text-white rounded-lg font-medium transition-all duration-300 shadow-lg hover:shadow-xl flex items-center"
            >
              Next
              <ArrowRightIcon className="w-5 h-5 ml-2" />
            </button>
          ) : (
            <motion.button
              onClick={handleAnalyze}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="px-8 py-3 bg-gradient-to-r from-primary-600 to-accent-600 dark:from-accent-500 dark:to-primary-500 hover:from-primary-700 hover:to-accent-700 dark:hover:from-accent-600 dark:hover:to-primary-600 text-white rounded-lg font-semibold transition-all duration-300 shadow-lg hover:shadow-xl flex items-center"
            >
              <SparklesIcon className="w-6 h-6 mr-2 animate-pulse" />
              Analyze My Career
            </motion.button>
          )}
        </div>
      </div>
    </div>
  );
};

export default AnalyzePage;