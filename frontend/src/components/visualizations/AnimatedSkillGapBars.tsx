import React, { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { TrendingUp, TrendingDown, Minus, Target, AlertCircle } from 'lucide-react';

interface SkillGap {
  skill: string;
  current: number;
  required: number;
  gap: number;
  priority: 'high' | 'medium' | 'low';
  category: string;
  timeToClose: string;
  resources: string[];
}

interface AnimatedSkillGapBarsProps {
  skillGaps: SkillGap[];
  title?: string;
  className?: string;
  showDetails?: boolean;
}

export function AnimatedSkillGapBars({ 
  skillGaps, 
  title = "Skill Gap Analysis", 
  className = "",
  showDetails = true 
}: AnimatedSkillGapBarsProps) {
  const [animatedGaps, setAnimatedGaps] = useState<SkillGap[]>([]);
  const [selectedSkill, setSelectedSkill] = useState<string | null>(null);
  const [sortBy, setSortBy] = useState<'gap' | 'priority' | 'category'>('gap');

  useEffect(() => {
    // Initialize with zero values for animation
    const initialGaps = skillGaps.map(gap => ({
      ...gap,
      current: 0,
      required: 0,
      gap: 0
    }));
    setAnimatedGaps(initialGaps);

    // Animate to actual values
    const timer = setTimeout(() => {
      setAnimatedGaps(skillGaps);
    }, 300);

    return () => clearTimeout(timer);
  }, [skillGaps]);

  const sortedGaps = [...animatedGaps].sort((a, b) => {
    switch (sortBy) {
      case 'gap':
        return Math.abs(b.gap) - Math.abs(a.gap);
      case 'priority':
        const priorityOrder = { high: 3, medium: 2, low: 1 };
        return priorityOrder[b.priority] - priorityOrder[a.priority];
      case 'category':
        return a.category.localeCompare(b.category);
      default:
        return 0;
    }
  });

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'high':
        return 'bg-red-500';
      case 'medium':
        return 'bg-yellow-500';
      case 'low':
        return 'bg-green-500';
      default:
        return 'bg-gray-500';
    }
  };

  const getGapIcon = (gap: number) => {
    if (gap > 0) return <TrendingUp className="w-4 h-4 text-red-500" />;
    if (gap < 0) return <TrendingDown className="w-4 h-4 text-green-500" />;
    return <Minus className="w-4 h-4 text-gray-500" />;
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

  const barVariants = {
    hidden: { opacity: 0, x: -50 },
    visible: {
      opacity: 1,
      x: 0,
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
      {/* Header */}
      <div className="flex justify-between items-center mb-6">
        <motion.h3
          className="text-xl font-semibold text-gray-800"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          {title}
        </motion.h3>
        
        <motion.div
          className="flex space-x-2"
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value as 'gap' | 'priority' | 'category')}
            className="text-sm border border-gray-300 rounded px-2 py-1 focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="gap">Sort by Gap</option>
            <option value="priority">Sort by Priority</option>
            <option value="category">Sort by Category</option>
          </select>
        </motion.div>
      </div>

      {/* Summary stats */}
      <motion.div
        className="grid grid-cols-3 gap-4 mb-6 p-4 bg-gray-50 rounded-lg"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3, duration: 0.5 }}
      >
        <div className="text-center">
          <div className="text-2xl font-bold text-red-600">
            {skillGaps.filter(g => g.priority === 'high').length}
          </div>
          <div className="text-sm text-gray-600">High Priority</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-yellow-600">
            {skillGaps.filter(g => g.priority === 'medium').length}
          </div>
          <div className="text-sm text-gray-600">Medium Priority</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-gray-600">
            {Math.round(skillGaps.reduce((acc, gap) => acc + Math.abs(gap.gap), 0) / skillGaps.length)}
          </div>
          <div className="text-sm text-gray-600">Avg Gap</div>
        </div>
      </motion.div>

      {/* Skill gap bars */}
      <div className="space-y-4">
        {sortedGaps.map((skillGap, index) => (
          <motion.div
            key={skillGap.skill}
            variants={barVariants}
            className="border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow cursor-pointer"
            onClick={() => setSelectedSkill(selectedSkill === skillGap.skill ? null : skillGap.skill)}
            whileHover={{ scale: 1.01 }}
            whileTap={{ scale: 0.99 }}
          >
            {/* Skill header */}
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center space-x-3">
                <div className={`w-3 h-3 rounded-full ${getPriorityColor(skillGap.priority)}`} />
                <h4 className="font-medium text-gray-900">{skillGap.skill}</h4>
                <span className="text-xs text-gray-500 bg-gray-100 px-2 py-1 rounded">
                  {skillGap.category}
                </span>
              </div>
              <div className="flex items-center space-x-2">
                {getGapIcon(skillGap.gap)}
                <span className="text-sm font-medium text-gray-600">
                  {skillGap.gap > 0 ? `+${skillGap.gap}` : skillGap.gap} points
                </span>
              </div>
            </div>

            {/* Progress bars */}
            <div className="space-y-2">
              {/* Current level bar */}
              <div className="flex items-center space-x-3">
                <span className="text-sm text-gray-600 w-16">Current</span>
                <div className="flex-1 bg-gray-200 rounded-full h-3 overflow-hidden">
                  <motion.div
                    className="h-full bg-blue-500 rounded-full"
                    initial={{ width: 0 }}
                    animate={{ width: `${skillGap.current}%` }}
                    transition={{ duration: 1.5, delay: index * 0.1 + 0.5, ease: "easeOut" }}
                  />
                </div>
                <span className="text-sm font-medium text-gray-700 w-12">
                  {skillGap.current}%
                </span>
              </div>

              {/* Required level bar */}
              <div className="flex items-center space-x-3">
                <span className="text-sm text-gray-600 w-16">Required</span>
                <div className="flex-1 bg-gray-200 rounded-full h-3 overflow-hidden">
                  <motion.div
                    className="h-full bg-green-500 rounded-full"
                    initial={{ width: 0 }}
                    animate={{ width: `${skillGap.required}%` }}
                    transition={{ duration: 1.5, delay: index * 0.1 + 0.7, ease: "easeOut" }}
                  />
                </div>
                <span className="text-sm font-medium text-gray-700 w-12">
                  {skillGap.required}%
                </span>
              </div>

              {/* Gap visualization */}
              <div className="flex items-center space-x-3">
                <span className="text-sm text-gray-600 w-16">Gap</span>
                <div className="flex-1 bg-gray-200 rounded-full h-3 overflow-hidden">
                  <motion.div
                    className={`h-full rounded-full ${
                      skillGap.gap > 0 ? 'bg-red-500' : 'bg-green-500'
                    }`}
                    initial={{ width: 0 }}
                    animate={{ width: `${Math.abs(skillGap.gap)}%` }}
                    transition={{ duration: 1.5, delay: index * 0.1 + 0.9, ease: "easeOut" }}
                  />
                </div>
                <span className={`text-sm font-medium w-12 ${
                  skillGap.gap > 0 ? 'text-red-600' : 'text-green-600'
                }`}>
                  {Math.abs(skillGap.gap)}%
                </span>
              </div>
            </div>

            {/* Time to close gap */}
            <motion.div
              className="mt-3 flex items-center justify-between text-sm text-gray-600"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: index * 0.1 + 1.2, duration: 0.3 }}
            >
              <div className="flex items-center space-x-1">
                <Target className="w-4 h-4" />
                <span>Est. time to close: {skillGap.timeToClose}</span>
              </div>
              {skillGap.priority === 'high' && (
                <div className="flex items-center space-x-1 text-red-600">
                  <AlertCircle className="w-4 h-4" />
                  <span>High Priority</span>
                </div>
              )}
            </motion.div>

            {/* Expandable details */}
            <AnimatePresence>
              {selectedSkill === skillGap.skill && showDetails && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  exit={{ opacity: 0, height: 0 }}
                  transition={{ duration: 0.3 }}
                  className="mt-4 pt-4 border-t border-gray-200"
                >
                  <h5 className="font-medium text-gray-800 mb-2">Recommended Resources:</h5>
                  <div className="space-y-2">
                    {skillGap.resources.map((resource, resourceIndex) => (
                      <motion.div
                        key={resource}
                        className="flex items-center space-x-2 text-sm text-gray-600"
                        initial={{ opacity: 0, x: -10 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: resourceIndex * 0.1, duration: 0.3 }}
                      >
                        <div className="w-2 h-2 bg-blue-500 rounded-full" />
                        <span>{resource}</span>
                      </motion.div>
                    ))}
                  </div>
                  
                  {/* Action button */}
                  <motion.button
                    className="mt-3 px-4 py-2 bg-blue-500 text-white text-sm rounded-lg hover:bg-blue-600 transition-colors"
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.3, duration: 0.3 }}
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                  >
                    Start Learning Plan
                  </motion.button>
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>
        ))}
      </div>

      {/* Overall progress indicator */}
      <motion.div
        className="mt-6 p-4 bg-gradient-to-r from-blue-50 to-green-50 rounded-lg"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 1.5, duration: 0.5 }}
      >
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-medium text-gray-700">Overall Skill Readiness</span>
          <span className="text-sm text-gray-600">
            {Math.round(100 - (skillGaps.reduce((acc, gap) => acc + Math.abs(gap.gap), 0) / skillGaps.length))}%
          </span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
          <motion.div
            className="h-full bg-gradient-to-r from-blue-500 to-green-500 rounded-full"
            initial={{ width: 0 }}
            animate={{ 
              width: `${100 - (skillGaps.reduce((acc, gap) => acc + Math.abs(gap.gap), 0) / skillGaps.length)}%` 
            }}
            transition={{ duration: 2, delay: 1.8, ease: "easeOut" }}
          />
        </div>
      </motion.div>
    </motion.div>
  );
}