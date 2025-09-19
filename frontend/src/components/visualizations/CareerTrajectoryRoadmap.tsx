import React, { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { CheckCircle, Circle, ArrowRight, Calendar, Target, Trophy, BookOpen } from 'lucide-react';

interface RoadmapStep {
  id: string;
  title: string;
  description: string;
  status: 'completed' | 'current' | 'upcoming';
  timeframe: string;
  skills: string[];
  type: 'milestone' | 'learning' | 'project' | 'certification';
}

interface CareerTrajectoryRoadmapProps {
  steps: RoadmapStep[];
  title?: string;
  className?: string;
}

export function CareerTrajectoryRoadmap({ 
  steps, 
  title = "Career Trajectory", 
  className = "" 
}: CareerTrajectoryRoadmapProps) {
  const [visibleSteps, setVisibleSteps] = useState<number>(0);
  const [selectedStep, setSelectedStep] = useState<string | null>(null);

  useEffect(() => {
    // Animate steps appearing one by one
    const timer = setInterval(() => {
      setVisibleSteps(prev => {
        if (prev < steps.length) {
          return prev + 1;
        }
        clearInterval(timer);
        return prev;
      });
    }, 200);

    return () => clearInterval(timer);
  }, [steps.length]);

  const getStepIcon = (type: string, status: string) => {
    const iconClass = `w-5 h-5 ${
      status === 'completed' ? 'text-green-600' : 
      status === 'current' ? 'text-blue-600' : 'text-gray-400'
    }`;

    switch (type) {
      case 'milestone':
        return <Trophy className={iconClass} />;
      case 'learning':
        return <BookOpen className={iconClass} />;
      case 'project':
        return <Target className={iconClass} />;
      case 'certification':
        return <CheckCircle className={iconClass} />;
      default:
        return <Circle className={iconClass} />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'bg-green-500';
      case 'current':
        return 'bg-blue-500';
      default:
        return 'bg-gray-300';
    }
  };

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        duration: 0.6,
        staggerChildren: 0.1
      }
    }
  };

  const stepVariants = {
    hidden: { opacity: 0, x: -50, scale: 0.8 },
    visible: {
      opacity: 1,
      x: 0,
      scale: 1,
      transition: {
        duration: 0.5,
        ease: "easeOut"
      }
    }
  };

  return (
    <motion.div
      className={`bg-white rounded-lg shadow-lg p-6 ${className}`}
      variants={containerVariants}
      initial="hidden"
      animate="visible"
    >
      <motion.h3
        className="text-xl font-semibold text-gray-800 mb-6"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        {title}
      </motion.h3>

      <div className="relative">
        {/* Progress line */}
        <div className="absolute left-8 top-0 bottom-0 w-0.5 bg-gray-200"></div>
        
        {/* Animated progress line */}
        <motion.div
          className="absolute left-8 top-0 w-0.5 bg-blue-500"
          initial={{ height: 0 }}
          animate={{ 
            height: `${(steps.filter(s => s.status === 'completed').length / steps.length) * 100}%` 
          }}
          transition={{ duration: 2, delay: 0.5, ease: "easeOut" }}
        />

        <div className="space-y-6">
          {steps.map((step, index) => (
            <AnimatePresence key={step.id}>
              {index < visibleSteps && (
                <motion.div
                  variants={stepVariants}
                  initial="hidden"
                  animate="visible"
                  className="relative flex items-start space-x-4 cursor-pointer"
                  onClick={() => setSelectedStep(selectedStep === step.id ? null : step.id)}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                >
                  {/* Step indicator */}
                  <motion.div
                    className={`relative z-10 w-16 h-16 rounded-full ${getStatusColor(step.status)} 
                      flex items-center justify-center shadow-lg`}
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    transition={{ delay: index * 0.2 + 0.3, duration: 0.4, type: "spring" }}
                  >
                    {getStepIcon(step.type, step.status)}
                    
                    {/* Pulse animation for current step */}
                    {step.status === 'current' && (
                      <motion.div
                        className="absolute inset-0 rounded-full bg-blue-500 opacity-30"
                        animate={{ scale: [1, 1.2, 1] }}
                        transition={{ duration: 2, repeat: Infinity }}
                      />
                    )}
                  </motion.div>

                  {/* Step content */}
                  <motion.div
                    className="flex-1 min-w-0 pb-6"
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.2 + 0.5, duration: 0.4 }}
                  >
                    <div className="bg-gray-50 rounded-lg p-4 hover:bg-gray-100 transition-colors">
                      <div className="flex items-center justify-between mb-2">
                        <h4 className="text-lg font-medium text-gray-900">{step.title}</h4>
                        <div className="flex items-center text-sm text-gray-500">
                          <Calendar className="w-4 h-4 mr-1" />
                          {step.timeframe}
                        </div>
                      </div>
                      
                      <p className="text-gray-600 mb-3">{step.description}</p>
                      
                      {/* Skills tags */}
                      <div className="flex flex-wrap gap-2 mb-3">
                        {step.skills.map((skill, skillIndex) => (
                          <motion.span
                            key={skill}
                            className="px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded-full"
                            initial={{ opacity: 0, scale: 0 }}
                            animate={{ opacity: 1, scale: 1 }}
                            transition={{ delay: index * 0.2 + 0.7 + skillIndex * 0.1, duration: 0.3 }}
                          >
                            {skill}
                          </motion.span>
                        ))}
                      </div>

                      {/* Expandable details */}
                      <AnimatePresence>
                        {selectedStep === step.id && (
                          <motion.div
                            initial={{ opacity: 0, height: 0 }}
                            animate={{ opacity: 1, height: "auto" }}
                            exit={{ opacity: 0, height: 0 }}
                            transition={{ duration: 0.3 }}
                            className="border-t border-gray-200 pt-3 mt-3"
                          >
                            <div className="space-y-2">
                              <div className="flex items-center justify-between text-sm">
                                <span className="text-gray-600">Progress:</span>
                                <div className="flex items-center space-x-2">
                                  <div className="w-20 h-2 bg-gray-200 rounded-full overflow-hidden">
                                    <motion.div
                                      className={`h-full ${
                                        step.status === 'completed' ? 'bg-green-500' :
                                        step.status === 'current' ? 'bg-blue-500' : 'bg-gray-300'
                                      }`}
                                      initial={{ width: 0 }}
                                      animate={{ 
                                        width: step.status === 'completed' ? '100%' : 
                                               step.status === 'current' ? '60%' : '0%' 
                                      }}
                                      transition={{ duration: 1, delay: 0.5 }}
                                    />
                                  </div>
                                  <span className="text-gray-500">
                                    {step.status === 'completed' ? '100%' : 
                                     step.status === 'current' ? '60%' : '0%'}
                                  </span>
                                </div>
                              </div>
                              
                              {step.status === 'current' && (
                                <div className="text-sm text-blue-600 font-medium">
                                  🎯 Currently working on this step
                                </div>
                              )}
                              
                              {step.status === 'upcoming' && (
                                <div className="text-sm text-gray-500">
                                  📅 Scheduled to start soon
                                </div>
                              )}
                            </div>
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </div>
                  </motion.div>

                  {/* Arrow connector (except for last item) */}
                  {index < steps.length - 1 && (
                    <motion.div
                      className="absolute left-8 top-16 transform -translate-x-1/2"
                      initial={{ opacity: 0, y: -10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: index * 0.2 + 0.8, duration: 0.3 }}
                    >
                      <ArrowRight className="w-4 h-4 text-gray-400 rotate-90" />
                    </motion.div>
                  )}
                </motion.div>
              )}
            </AnimatePresence>
          ))}
        </div>
      </div>

      {/* Summary stats */}
      <motion.div
        className="mt-6 p-4 bg-gradient-to-r from-blue-50 to-green-50 rounded-lg"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 2, duration: 0.5 }}
      >
        <div className="grid grid-cols-3 gap-4 text-center">
          <div>
            <div className="text-2xl font-bold text-green-600">
              {steps.filter(s => s.status === 'completed').length}
            </div>
            <div className="text-sm text-gray-600">Completed</div>
          </div>
          <div>
            <div className="text-2xl font-bold text-blue-600">
              {steps.filter(s => s.status === 'current').length}
            </div>
            <div className="text-sm text-gray-600">In Progress</div>
          </div>
          <div>
            <div className="text-2xl font-bold text-gray-600">
              {steps.filter(s => s.status === 'upcoming').length}
            </div>
            <div className="text-sm text-gray-600">Upcoming</div>
          </div>
        </div>
      </motion.div>
    </motion.div>
  );
}