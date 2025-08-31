'use client';

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  TrendingUp, 
  TrendingDown, 
  Target, 
  Clock, 
  BookOpen, 
  CheckCircle,
  AlertCircle,
  BarChart3
} from 'lucide-react';

const SkillGapAnalysis = ({ 
  skillGaps = [], 
  targetRole = "Software Engineer",
  className = "" 
}) => {
  const [selectedSkill, setSelectedSkill] = useState(null);
  const [animationComplete, setAnimationComplete] = useState(false);

  useEffect(() => {
    const timer = setTimeout(() => setAnimationComplete(true), 1000);
    return () => clearTimeout(timer);
  }, []);

  const getPriorityColor = (priority) => {
    switch (priority) {
      case 'high': return 'bg-red-500';
      case 'medium': return 'bg-yellow-500';
      case 'low': return 'bg-green-500';
      default: return 'bg-gray-500';
    }
  };

  const getPriorityTextColor = (priority) => {
    switch (priority) {
      case 'high': return 'text-red-600 dark:text-red-400';
      case 'medium': return 'text-yellow-600 dark:text-yellow-400';
      case 'low': return 'text-green-600 dark:text-green-400';
      default: return 'text-gray-600 dark:text-gray-400';
    }
  };

  const getGapSize = (current, required) => {
    const gap = required - current;
    if (gap <= 0.2) return 'small';
    if (gap <= 0.5) return 'medium';
    return 'large';
  };

  const getGapIcon = (gapSize) => {
    switch (gapSize) {
      case 'small': return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'medium': return <AlertCircle className="w-4 h-4 text-yellow-500" />;
      case 'large': return <TrendingUp className="w-4 h-4 text-red-500" />;
      default: return <BarChart3 className="w-4 h-4 text-gray-500" />;
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
      className={`bg-white dark:bg-gray-900 rounded-xl shadow-lg p-6 ${className}`}
    >
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="flex items-center justify-between mb-6"
      >
        <h3 className="text-2xl font-bold text-gray-900 dark:text-white">
          Skill Gap Analysis
        </h3>
        <div className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          <Target className="w-4 h-4" />
          <span>Target: {targetRole}</span>
        </div>
      </motion.div>

      {/* Summary Cards */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6"
      >
        <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-4">
          <div className="flex items-center gap-2 mb-2">
            <TrendingUp className="w-5 h-5 text-red-500" />
            <span className="font-semibold text-red-700 dark:text-red-300">High Priority</span>
          </div>
          <p className="text-2xl font-bold text-red-600 dark:text-red-400">
            {skillGaps.filter(skill => skill.priority === 'high').length}
          </p>
          <p className="text-sm text-red-600 dark:text-red-400">Skills to focus on</p>
        </div>

        <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-4">
          <div className="flex items-center gap-2 mb-2">
            <Clock className="w-5 h-5 text-yellow-500" />
            <span className="font-semibold text-yellow-700 dark:text-yellow-300">Medium Priority</span>
          </div>
          <p className="text-2xl font-bold text-yellow-600 dark:text-yellow-400">
            {skillGaps.filter(skill => skill.priority === 'medium').length}
          </p>
          <p className="text-sm text-yellow-600 dark:text-yellow-400">Skills to improve</p>
        </div>

        <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-4">
          <div className="flex items-center gap-2 mb-2">
            <CheckCircle className="w-5 h-5 text-green-500" />
            <span className="font-semibold text-green-700 dark:text-green-300">Low Priority</span>
          </div>
          <p className="text-2xl font-bold text-green-600 dark:text-green-400">
            {skillGaps.filter(skill => skill.priority === 'low').length}
          </p>
          <p className="text-sm text-green-600 dark:text-green-400">Skills to maintain</p>
        </div>
      </motion.div>

      {/* Skill Gap List */}
      <div className="space-y-4">
        {skillGaps.map((skill, index) => {
          const gapSize = getGapSize(skill.currentLevel, skill.requiredLevel);
          const progressPercentage = (skill.currentLevel / skill.requiredLevel) * 100;
          
          return (
            <motion.div
              key={skill.name}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.4 + index * 0.1 }}
              className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4 cursor-pointer hover:shadow-md transition-shadow"
              onClick={() => setSelectedSkill(selectedSkill === index ? null : index)}
            >
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-3">
                  {getGapIcon(gapSize)}
                  <h4 className="font-semibold text-gray-900 dark:text-white">
                    {skill.name}
                  </h4>
                  <span className={`px-2 py-1 rounded-full text-xs font-medium ${getPriorityTextColor(skill.priority)} bg-opacity-20`}>
                    {skill.priority} priority
                  </span>
                </div>
                <div className="text-right">
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    {Math.round(skill.currentLevel * 100)}% / {Math.round(skill.requiredLevel * 100)}%
                  </p>
                  <p className="text-xs text-gray-500 dark:text-gray-500">
                    Current / Required
                  </p>
                </div>
              </div>

              {/* Progress Bar */}
              <div className="mb-3">
                <div className="flex justify-between text-xs text-gray-600 dark:text-gray-400 mb-1">
                  <span>Progress</span>
                  <span>{Math.round(progressPercentage)}%</span>
                </div>
                <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: animationComplete ? `${Math.min(progressPercentage, 100)}%` : 0 }}
                    transition={{ duration: 1, delay: 0.5 + index * 0.1, ease: "easeOut" }}
                    className={`h-2 rounded-full ${getPriorityColor(skill.priority)}`}
                  />
                </div>
              </div>

              {/* Gap Indicator */}
              <div className="flex items-center justify-between text-sm">
                <div className="flex items-center gap-2">
                  <span className="text-gray-600 dark:text-gray-400">Gap:</span>
                  <span className={`font-medium ${getPriorityTextColor(skill.priority)}`}>
                    {Math.round((skill.requiredLevel - skill.currentLevel) * 100)}%
                  </span>
                </div>
                <div className="flex items-center gap-2 text-gray-500 dark:text-gray-400">
                  <Clock className="w-3 h-3" />
                  <span>{skill.estimatedLearningTime}</span>
                </div>
              </div>

              {/* Expanded Details */}
              <AnimatePresence>
                {selectedSkill === index && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    transition={{ duration: 0.3 }}
                    className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700"
                  >
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div>
                        <h5 className="font-medium text-gray-900 dark:text-white mb-2">
                          Learning Resources
                        </h5>
                        <div className="space-y-2">
                          {skill.learningResources?.map((resource, resourceIndex) => (
                            <div key={resourceIndex} className="flex items-center gap-2">
                              <BookOpen className="w-3 h-3 text-blue-500" />
                              <span className="text-sm text-gray-600 dark:text-gray-400">
                                {resource.title}
                              </span>
                              <span className="text-xs text-gray-500 dark:text-gray-500">
                                ({resource.type})
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>
                      
                      <div>
                        <h5 className="font-medium text-gray-900 dark:text-white mb-2">
                          Skill Breakdown
                        </h5>
                        <div className="space-y-2">
                          {skill.subSkills?.map((subSkill, subIndex) => (
                            <div key={subIndex} className="flex items-center justify-between">
                              <span className="text-sm text-gray-600 dark:text-gray-400">
                                {subSkill.name}
                              </span>
                              <div className="flex items-center gap-2">
                                <div className="w-16 bg-gray-200 dark:bg-gray-700 rounded-full h-1">
                                  <div 
                                    className={`h-1 rounded-full ${getPriorityColor(skill.priority)}`}
                                    style={{ width: `${subSkill.level * 100}%` }}
                                  />
                                </div>
                                <span className="text-xs text-gray-500 dark:text-gray-500 w-8">
                                  {Math.round(subSkill.level * 100)}%
                                </span>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>

                    {/* Action Buttons */}
                    <div className="mt-4 flex gap-2">
                      <button className="px-3 py-1 bg-blue-500 text-white rounded-md text-sm hover:bg-blue-600 transition-colors">
                        Start Learning
                      </button>
                      <button className="px-3 py-1 bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-md text-sm hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors">
                        View Resources
                      </button>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>
          );
        })}
      </div>

      {/* Overall Progress */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.8 }}
        className="mt-6 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg"
      >
        <div className="flex items-center justify-between mb-2">
          <span className="font-medium text-blue-700 dark:text-blue-300">
            Overall Readiness for {targetRole}
          </span>
          <span className="text-lg font-bold text-blue-600 dark:text-blue-400">
            {Math.round(skillGaps.reduce((acc, skill) => acc + (skill.currentLevel / skill.requiredLevel), 0) / skillGaps.length * 100)}%
          </span>
        </div>
        <div className="w-full bg-blue-200 dark:bg-blue-800 rounded-full h-3">
          <motion.div
            initial={{ width: 0 }}
            animate={{ 
              width: animationComplete ? 
                `${Math.round(skillGaps.reduce((acc, skill) => acc + (skill.currentLevel / skill.requiredLevel), 0) / skillGaps.length * 100)}%` : 
                0 
            }}
            transition={{ duration: 1.5, delay: 1, ease: "easeOut" }}
            className="bg-blue-500 h-3 rounded-full"
          />
        </div>
        <p className="text-sm text-blue-600 dark:text-blue-400 mt-2">
          Focus on high-priority skills to improve your readiness score
        </p>
      </motion.div>
    </motion.div>
  );
};

export default SkillGapAnalysis;