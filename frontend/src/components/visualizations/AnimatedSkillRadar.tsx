import React, { useEffect, useState } from 'react';
import { motion } from 'motion/react';
import { RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, ResponsiveContainer } from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';
import { RefreshCw, Star } from 'lucide-react';

interface SkillData {
  skill: string;
  current: number;
  target: number;
  fullMark: number;
}

interface AnimatedSkillRadarProps {
  skills: SkillData[];
  title?: string;
  subtitle?: string;
  className?: string;
  loading?: boolean;
  onRefresh?: () => void;
}

export function AnimatedSkillRadar({ 
  skills, 
  title = "Skill Assessment", 
  subtitle,
  className = "",
  loading = false,
  onRefresh
}: AnimatedSkillRadarProps) {
  const [animatedData, setAnimatedData] = useState<SkillData[]>([]);
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    // Initialize with zero values for animation
    const initialData = skills.map(skill => ({
      ...skill,
      current: 0,
      target: 0
    }));
    setAnimatedData(initialData);

    // Trigger animation after component mounts
    const timer = setTimeout(() => {
      setIsVisible(true);
      setAnimatedData(skills);
    }, 300);

    return () => clearTimeout(timer);
  }, [skills]);

  const containerVariants = {
    hidden: { opacity: 0, scale: 0.8 },
    visible: {
      opacity: 1,
      scale: 1,
      transition: {
        duration: 0.6,
        ease: [0.4, 0, 0.2, 1] as const
      }
    }
  };

  const titleVariants = {
    hidden: { opacity: 0, y: -20 },
    visible: {
      opacity: 1,
      y: 0,
      transition: {
        duration: 0.5,
        delay: 0.2
      }
    }
  };

  if (loading) {
    return (
      <Card className={className}>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Star className="w-5 h-5" />
            <span>{title}</span>
          </CardTitle>
          {subtitle && <p className="text-sm text-muted-foreground">{subtitle}</p>}
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center h-96">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!skills || skills.length === 0) {
    return (
      <Card className={className}>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center space-x-2">
              <Star className="w-5 h-5" />
              <span>{title}</span>
            </CardTitle>
            {onRefresh && (
              <Button variant="ghost" size="sm" onClick={onRefresh}>
                <RefreshCw className="w-4 h-4" />
              </Button>
            )}
          </div>
          {subtitle && <p className="text-sm text-muted-foreground">{subtitle}</p>}
        </CardHeader>
        <CardContent>
          <div className="flex flex-col items-center justify-center h-96 text-muted-foreground">
            <Star className="w-12 h-12 mb-4 opacity-50" />
            <p>No skill data available</p>
            <p className="text-sm">Connect your platforms to see skill analysis</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className={className}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center space-x-2">
              <Star className="w-5 h-5" />
              <span>{title}</span>
              <Badge variant="secondary" className="ml-2">
                {skills.length} skills
              </Badge>
            </CardTitle>
            {subtitle && <p className="text-sm text-muted-foreground mt-1">{subtitle}</p>}
          </div>
          {onRefresh && (
            <Button variant="ghost" size="sm" onClick={onRefresh}>
              <RefreshCw className="w-4 h-4" />
            </Button>
          )}
        </div>
      </CardHeader>
      <CardContent>
        <motion.div
          variants={containerVariants}
          initial="hidden"
          animate="visible"
        >
      
      <div className="relative">
        <ResponsiveContainer width="100%" height={400}>
          <RadarChart data={animatedData} margin={{ top: 20, right: 80, bottom: 20, left: 80 }}>
            <PolarGrid 
              stroke="#e5e7eb" 
              strokeWidth={1}
            />
            <PolarAngleAxis 
              dataKey="skill" 
              tick={{ fontSize: 12, fill: '#6b7280' }}
              className="text-sm"
            />
            <PolarRadiusAxis 
              angle={90} 
              domain={[0, 100]} 
              tick={{ fontSize: 10, fill: '#9ca3af' }}
              tickCount={6}
            />
            
            {/* Current skill level */}
            <Radar
              name="Current Level"
              dataKey="current"
              stroke="#3b82f6"
              fill="#3b82f6"
              fillOpacity={0.3}
              strokeWidth={2}
              dot={{ fill: '#3b82f6', strokeWidth: 2, r: 4 }}
              animationBegin={0}
              animationDuration={2000}
              animationEasing="ease-out"
            />
            
            {/* Target skill level */}
            <Radar
              name="Target Level"
              dataKey="target"
              stroke="#10b981"
              fill="transparent"
              strokeWidth={2}
              strokeDasharray="5,5"
              dot={{ fill: '#10b981', strokeWidth: 2, r: 3 }}
              animationBegin={500}
              animationDuration={2000}
              animationEasing="ease-out"
            />
          </RadarChart>
        </ResponsiveContainer>

        {/* Legend */}
        <motion.div
          className="flex justify-center space-x-6 mt-4"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1, duration: 0.5 }}
        >
          <div className="flex items-center space-x-2">
            <div className="w-4 h-4 bg-blue-500 rounded-full opacity-60"></div>
            <span className="text-sm text-gray-600">Current Level</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-4 h-1 bg-green-500 border-dashed border-2 border-green-500"></div>
            <span className="text-sm text-gray-600">Target Level</span>
          </div>
          </motion.div>
        </div>

        {/* Skill improvement suggestions */}
        <motion.div
          className="mt-6 p-4 bg-gray-50 rounded-lg"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1.5, duration: 0.5 }}
        >
          <h4 className="text-sm font-medium text-gray-700 mb-2">Focus Areas</h4>
          <div className="space-y-1">
            {skills
              .filter(skill => skill.target - skill.current > 20)
              .slice(0, 3)
              .map((skill, index) => (
                <motion.div
                  key={skill.skill}
                  className="text-xs text-gray-600 flex justify-between"
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 1.7 + index * 0.1, duration: 0.3 }}
                >
                  <span>{skill.skill}</span>
                  <span className="text-orange-600 font-medium">
                    +{skill.target - skill.current} points needed
                  </span>
                </motion.div>
              ))}
          </div>
        </motion.div>
        </motion.div>
      </CardContent>
    </Card>
  );
}