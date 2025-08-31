'use client';

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronRight, MapPin, Clock, TrendingUp, Star, CheckCircle } from 'lucide-react';

const CareerRoadmapVisualization = ({ 
  roadmapData = [], 
  currentPosition = 0,
  className = ""
}) => {
  const [selectedStep, setSelectedStep] = useState(null);
  const [hoveredStep, setHoveredStep] = useState(null);

  const getStepStatus = (index) => {
    if (index < currentPosition) return 'completed';
    if (index === currentPosition) return 'current';
    return 'upcoming';
  };

  const getStepColor = (status) => {
    switch (status) {
      case 'completed': return 'bg-green-500';
      case 'current': return 'bg-blue-500';
      default: return 'bg-gray-300 dark:bg-gray-600';
    }
  };

  const getConnectorColor = (index) => {
    return index < currentPosition ? 'bg-green-500' : 'bg-gray-300 dark:bg-gray-600';
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.6 }}
      className={`bg-white dark:bg-gray-900 rounded-xl shadow-lg p-6 ${className}`}
    >
      <motion.h3
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="text-2xl font-bold text-gray-900 dark:text-white mb-8 text-center"
      >
        Career Roadmap
      </motion.h3>

      {/* Desktop/Tablet View - Horizontal Timeline */}
      <div className="hidden md:block">
        <div className="relative">
          {/* Timeline Line */}
          <div className="absolute top-1/2 left-0 right-0 h-1 bg-gray-200 dark:bg-gray-700 transform -translate-y-1/2" />
          
          {/* Progress Line */}
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${(currentPosition / (roadmapData.length - 1)) * 100}%` }}
            transition={{ duration: 1.5, ease: "easeInOut" }}
            className="absolute top-1/2 left-0 h-1 bg-blue-500 transform -translate-y-1/2 z-10"
          />

          {/* Timeline Steps */}
          <div className="relative flex justify-between items-center">
            {roadmapData.map((step, index) => {
              const status = getStepStatus(index);
              return (
                <motion.div
                  key={step.id || index}
                  initial={{ opacity: 0, y: 50 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.3 + index * 0.1 }}
                  className="relative flex flex-col items-center cursor-pointer group"
                  onMouseEnter={() => setHoveredStep(index)}
                  onMouseLeave={() => setHoveredStep(null)}
                  onClick={() => setSelectedStep(selectedStep === index ? null : index)}
                >
                  {/* Step Circle */}
                  <motion.div
                    whileHover={{ scale: 1.2 }}
                    whileTap={{ scale: 0.95 }}
                    className={`w-12 h-12 rounded-full ${getStepColor(status)} flex items-center justify-center shadow-lg z-20 border-4 border-white dark:border-gray-900`}
                  >
                    {status === 'completed' ? (
                      <CheckCircle className="w-6 h-6 text-white" />
                    ) : status === 'current' ? (
                      <Star className="w-6 h-6 text-white" />
                    ) : (
                      <span className="text-white font-bold text-sm">{index + 1}</span>
                    )}
                  </motion.div>

                  {/* Step Label */}
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.5 + index * 0.1 }}
                    className={`mt-4 text-center max-w-32 ${
                      status === 'current' ? 'text-blue-600 dark:text-blue-400 font-semibold' : 
                      'text-gray-700 dark:text-gray-300'
                    }`}
                  >
                    <p className="text-sm font-medium">{step.title}</p>
                    <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                      {step.duration}
                    </p>
                  </motion.div>

                  {/* Hover/Selected Details */}
                  <AnimatePresence>
                    {(hoveredStep === index || selectedStep === index) && (
                      <motion.div
                        initial={{ opacity: 0, y: 10, scale: 0.9 }}
                        animate={{ opacity: 1, y: 0, scale: 1 }}
                        exit={{ opacity: 0, y: 10, scale: 0.9 }}
                        className="absolute top-16 left-1/2 transform -translate-x-1/2 bg-white dark:bg-gray-800 rounded-lg shadow-xl p-4 border border-gray-200 dark:border-gray-700 z-30 w-64"
                      >
                        <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
                          {step.title}
                        </h4>
                        <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                          {step.description}
                        </p>
                        
                        {step.skills && (
                          <div className="mb-3">
                            <p className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">
                              Key Skills:
                            </p>
                            <div className="flex flex-wrap gap-1">
                              {step.skills.slice(0, 3).map((skill, skillIndex) => (
                                <span
                                  key={skillIndex}
                                  className="px-2 py-1 bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200 text-xs rounded-full"
                                >
                                  {skill}
                                </span>
                              ))}
                            </div>
                          </div>
                        )}

                        <div className="flex items-center justify-between text-xs text-gray-500 dark:text-gray-400">
                          <div className="flex items-center gap-1">
                            <Clock className="w-3 h-3" />
                            <span>{step.duration}</span>
                          </div>
                          {step.salary && (
                            <div className="flex items-center gap-1">
                              <TrendingUp className="w-3 h-3" />
                              <span>{step.salary}</span>
                            </div>
                          )}
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </motion.div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Mobile View - Vertical Timeline */}
      <div className="md:hidden">
        <div className="relative">
          {roadmapData.map((step, index) => {
            const status = getStepStatus(index);
            const isLast = index === roadmapData.length - 1;
            
            return (
              <motion.div
                key={step.id || index}
                initial={{ opacity: 0, x: -50 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.2 + index * 0.1 }}
                className="relative flex items-start mb-8"
              >
                {/* Vertical Line */}
                {!isLast && (
                  <div className={`absolute left-6 top-12 w-0.5 h-16 ${getConnectorColor(index)}`} />
                )}

                {/* Step Circle */}
                <motion.div
                  whileTap={{ scale: 0.95 }}
                  className={`w-12 h-12 rounded-full ${getStepColor(status)} flex items-center justify-center shadow-lg z-10 border-4 border-white dark:border-gray-900 flex-shrink-0`}
                  onClick={() => setSelectedStep(selectedStep === index ? null : index)}
                >
                  {status === 'completed' ? (
                    <CheckCircle className="w-6 h-6 text-white" />
                  ) : status === 'current' ? (
                    <Star className="w-6 h-6 text-white" />
                  ) : (
                    <span className="text-white font-bold text-sm">{index + 1}</span>
                  )}
                </motion.div>

                {/* Step Content */}
                <div className="ml-4 flex-1">
                  <h4 className={`font-semibold ${
                    status === 'current' ? 'text-blue-600 dark:text-blue-400' : 
                    'text-gray-900 dark:text-white'
                  }`}>
                    {step.title}
                  </h4>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                    {step.description}
                  </p>
                  
                  <div className="flex items-center gap-4 mt-2 text-xs text-gray-500 dark:text-gray-400">
                    <div className="flex items-center gap-1">
                      <Clock className="w-3 h-3" />
                      <span>{step.duration}</span>
                    </div>
                    {step.salary && (
                      <div className="flex items-center gap-1">
                        <TrendingUp className="w-3 h-3" />
                        <span>{step.salary}</span>
                      </div>
                    )}
                  </div>

                  {step.skills && (
                    <div className="mt-2">
                      <div className="flex flex-wrap gap-1">
                        {step.skills.slice(0, 3).map((skill, skillIndex) => (
                          <span
                            key={skillIndex}
                            className="px-2 py-1 bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200 text-xs rounded-full"
                          >
                            {skill}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </motion.div>
            );
          })}
        </div>
      </div>

      {/* Progress Summary */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.8 }}
        className="mt-8 p-4 bg-gray-50 dark:bg-gray-800 rounded-lg"
      >
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
            Progress
          </span>
          <span className="text-sm font-bold text-blue-600 dark:text-blue-400">
            {Math.round((currentPosition / (roadmapData.length - 1)) * 100)}%
          </span>
        </div>
        <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${(currentPosition / (roadmapData.length - 1)) * 100}%` }}
            transition={{ duration: 1.5, ease: "easeInOut" }}
            className="bg-blue-500 h-2 rounded-full"
          />
        </div>
      </motion.div>
    </motion.div>
  );
};

export default CareerRoadmapVisualization;