import React, { useState } from 'react';
import { motion } from 'motion/react';
import { AnimatedSkillRadar } from './visualizations/AnimatedSkillRadar';
import { AnimatedSkillGapBars } from './visualizations/AnimatedSkillGapBars';
import { CareerTrajectoryRoadmap } from './visualizations/CareerTrajectoryRoadmap';
import { JobCompatibilityIndicators } from './visualizations/JobCompatibilityIndicators';

const demoData = {
  // For AnimatedSkillRadar
  skillRadar: [
    { skill: 'JavaScript', current: 85, target: 90, fullMark: 100 },
    { skill: 'React', current: 80, target: 85, fullMark: 100 },
    { skill: 'Python', current: 70, target: 80, fullMark: 100 },
    { skill: 'Node.js', current: 75, target: 75, fullMark: 100 },
    { skill: 'SQL', current: 60, target: 70, fullMark: 100 },
    { skill: 'AWS', current: 45, target: 65, fullMark: 100 }
  ],
  // For AnimatedSkillGapBars
  skillGaps: [
    { 
      skill: 'JavaScript', 
      current: 85, 
      required: 90, 
      gap: 5, 
      priority: 'medium' as const, 
      category: 'Frontend',
      timeToClose: '2 weeks',
      resources: ['MDN Docs', 'JavaScript.info']
    },
    { 
      skill: 'React', 
      current: 80, 
      required: 85, 
      gap: 5, 
      priority: 'medium' as const, 
      category: 'Frontend',
      timeToClose: '3 weeks',
      resources: ['React Docs', 'React Patterns']
    },
    { 
      skill: 'AWS', 
      current: 45, 
      required: 65, 
      gap: 20, 
      priority: 'high' as const, 
      category: 'Cloud',
      timeToClose: '2 months',
      resources: ['AWS Training', 'Cloud Practitioner']
    }
  ],
  // For CareerTrajectoryRoadmap
  careerSteps: [
    {
      id: '1',
      title: 'Junior Developer',
      description: 'Started career in web development',
      status: 'completed' as const,
      timeframe: '2022',
      skills: ['HTML', 'CSS', 'JavaScript'],
      type: 'milestone' as const
    },
    {
      id: '2',
      title: 'Mid-level Developer',
      description: 'Gained experience with React and Node.js',
      status: 'completed' as const,
      timeframe: '2023',
      skills: ['React', 'Node.js', 'MongoDB'],
      type: 'milestone' as const
    },
    {
      id: '3',
      title: 'Senior Developer',
      description: 'Leading projects and mentoring juniors',
      status: 'current' as const,
      timeframe: '2024',
      skills: ['System Design', 'Leadership', 'AWS'],
      type: 'milestone' as const
    },
    {
      id: '4',
      title: 'Tech Lead',
      description: 'Technical leadership and architecture decisions',
      status: 'upcoming' as const,
      timeframe: '2025',
      skills: ['Architecture', 'Team Management', 'Strategy'],
      type: 'milestone' as const
    }
  ],
  // For JobCompatibilityIndicators
  jobMatches: [
    {
      id: '1',
      title: 'Frontend Developer',
      company: 'TechCorp',
      location: 'San Francisco, CA',
      salary: '$120k - $150k',
      matchScore: 92,
      skillMatches: ['React', 'JavaScript', 'CSS'],
      skillGaps: ['TypeScript'],
      experience: '3-5 years',
      type: 'Full-time',
      postedDate: '2 days ago',
      url: '#'
    },
    {
      id: '2',
      title: 'Full Stack Developer',
      company: 'StartupXYZ',
      location: 'Remote',
      salary: '$100k - $130k',
      matchScore: 85,
      skillMatches: ['React', 'Node.js', 'JavaScript'],
      skillGaps: ['GraphQL', 'Docker'],
      experience: '2-4 years',
      type: 'Full-time',
      postedDate: '1 week ago',
      url: '#'
    },
    {
      id: '3',
      title: 'React Developer',
      company: 'WebSolutions',
      location: 'New York, NY',
      salary: '$110k - $140k',
      matchScore: 88,
      skillMatches: ['React', 'JavaScript', 'Redux'],
      skillGaps: ['Next.js'],
      experience: '3+ years',
      type: 'Full-time',
      postedDate: '3 days ago',
      url: '#'
    }
  ]
};

export function AnimatedVisualizationsDemo() {
  const [activeDemo, setActiveDemo] = useState('radar');

  const demos = [
    { id: 'radar', title: 'Skill Radar Chart', description: 'Interactive radar chart showing skill levels vs requirements' },
    { id: 'gaps', title: 'Skill Gap Analysis', description: 'Animated bars showing skill gaps and progress' },
    { id: 'roadmap', title: 'Career Roadmap', description: 'Timeline visualization of career progression' },
    { id: 'compatibility', title: 'Job Compatibility', description: 'Job matching indicators with compatibility scores' }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50 py-12">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-12"
        >
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            Interactive Career Visualizations
          </h1>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Explore our advanced visualization components that help users understand their career progress,
            skill gaps, and job compatibility through interactive and animated charts.
          </p>
        </motion.div>

        {/* Demo Navigation */}
        <div className="flex flex-wrap justify-center gap-4 mb-8">
          {demos.map((demo) => (
            <motion.button
              key={demo.id}
              onClick={() => setActiveDemo(demo.id)}
              className={`px-6 py-3 rounded-lg font-medium transition-all ${
                activeDemo === demo.id
                  ? 'bg-blue-600 text-white shadow-lg'
                  : 'bg-white text-gray-700 hover:bg-gray-50 shadow-md'
              }`}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              {demo.title}
            </motion.button>
          ))}
        </div>

        {/* Demo Content */}
        <motion.div
          key={activeDemo}
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.3 }}
          className="bg-white rounded-xl shadow-xl p-8"
        >
          <div className="mb-6">
            <h2 className="text-2xl font-bold text-gray-900 mb-2">
              {demos.find(d => d.id === activeDemo)?.title}
            </h2>
            <p className="text-gray-600">
              {demos.find(d => d.id === activeDemo)?.description}
            </p>
          </div>

          <div className="min-h-[500px] flex items-center justify-center">
            {activeDemo === 'radar' && (
              <div className="w-full max-w-2xl">
                <AnimatedSkillRadar skills={demoData.skillRadar} />
              </div>
            )}

            {activeDemo === 'gaps' && (
              <div className="w-full max-w-3xl">
                <AnimatedSkillGapBars skillGaps={demoData.skillGaps} />
              </div>
            )}

            {activeDemo === 'roadmap' && (
              <div className="w-full max-w-4xl">
                <CareerTrajectoryRoadmap steps={demoData.careerSteps} />
              </div>
            )}

            {activeDemo === 'compatibility' && (
              <div className="w-full max-w-3xl">
                <JobCompatibilityIndicators jobMatches={demoData.jobMatches} />
              </div>
            )}
          </div>
        </motion.div>

        {/* Features Grid */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="mt-16 grid md:grid-cols-2 lg:grid-cols-4 gap-6"
        >
          {[
            {
              icon: '📊',
              title: 'Interactive Charts',
              description: 'Hover and click interactions for detailed insights'
            },
            {
              icon: '🎨',
              title: 'Smooth Animations',
              description: 'Fluid transitions and engaging visual effects'
            },
            {
              icon: '📱',
              title: 'Responsive Design',
              description: 'Optimized for all screen sizes and devices'
            },
            {
              icon: '⚡',
              title: 'Real-time Updates',
              description: 'Dynamic data updates with live visualizations'
            }
          ].map((feature, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 + index * 0.1 }}
              className="bg-white rounded-lg p-6 shadow-md hover:shadow-lg transition-shadow"
            >
              <div className="text-3xl mb-3">{feature.icon}</div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">
                {feature.title}
              </h3>
              <p className="text-gray-600 text-sm">
                {feature.description}
              </p>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </div>
  );
}