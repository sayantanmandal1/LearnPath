'use client';

import React, { useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Search, 
  Filter, 
  MapPin, 
  DollarSign, 
  Clock, 
  TrendingUp,
  Star,
  Building,
  Users,
  CheckCircle,
  XCircle,
  AlertTriangle,
  ChevronDown,
  ChevronUp,
  ExternalLink
} from 'lucide-react';

const JobCompatibilityDashboard = ({ 
  jobs = [], 
  userSkills = [],
  className = "" 
}) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedFilters, setSelectedFilters] = useState({
    location: '',
    salaryRange: '',
    experienceLevel: '',
    compatibility: ''
  });
  const [sortBy, setSortBy] = useState('compatibility');
  const [expandedJob, setExpandedJob] = useState(null);

  // Filter and sort jobs
  const filteredJobs = useMemo(() => {
    let filtered = jobs.filter(job => {
      const matchesSearch = job.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
                           job.company.toLowerCase().includes(searchTerm.toLowerCase());
      
      const matchesLocation = !selectedFilters.location || 
                             job.location.toLowerCase().includes(selectedFilters.location.toLowerCase());
      
      const matchesSalary = !selectedFilters.salaryRange || 
                           (job.salaryRange && 
                            job.salaryRange.min >= parseInt(selectedFilters.salaryRange.split('-')[0]) &&
                            job.salaryRange.max <= parseInt(selectedFilters.salaryRange.split('-')[1]));
      
      const matchesExperience = !selectedFilters.experienceLevel || 
                               job.experienceLevel === selectedFilters.experienceLevel;
      
      const matchesCompatibility = !selectedFilters.compatibility ||
                                  getCompatibilityLevel(job.compatibilityScore) === selectedFilters.compatibility;

      return matchesSearch && matchesLocation && matchesSalary && matchesExperience && matchesCompatibility;
    });

    // Sort jobs
    filtered.sort((a, b) => {
      switch (sortBy) {
        case 'compatibility':
          return b.compatibilityScore - a.compatibilityScore;
        case 'salary':
          return (b.salaryRange?.max || 0) - (a.salaryRange?.max || 0);
        case 'date':
          return new Date(b.postedDate) - new Date(a.postedDate);
        default:
          return 0;
      }
    });

    return filtered;
  }, [jobs, searchTerm, selectedFilters, sortBy]);

  const getCompatibilityLevel = (score) => {
    if (score >= 0.8) return 'high';
    if (score >= 0.6) return 'medium';
    return 'low';
  };

  const getCompatibilityColor = (score) => {
    if (score >= 0.8) return 'text-green-600 dark:text-green-400';
    if (score >= 0.6) return 'text-yellow-600 dark:text-yellow-400';
    return 'text-red-600 dark:text-red-400';
  };

  const getCompatibilityBg = (score) => {
    if (score >= 0.8) return 'bg-green-100 dark:bg-green-900/20';
    if (score >= 0.6) return 'bg-yellow-100 dark:bg-yellow-900/20';
    return 'bg-red-100 dark:bg-red-900/20';
  };

  const getCompatibilityIcon = (score) => {
    if (score >= 0.8) return <CheckCircle className="w-5 h-5 text-green-500" />;
    if (score >= 0.6) return <AlertTriangle className="w-5 h-5 text-yellow-500" />;
    return <XCircle className="w-5 h-5 text-red-500" />;
  };

  const calculateSkillMatch = (jobSkills, userSkills) => {
    const matchedSkills = jobSkills.filter(jobSkill => 
      userSkills.some(userSkill => 
        userSkill.name.toLowerCase() === jobSkill.toLowerCase()
      )
    );
    return {
      matched: matchedSkills.length,
      total: jobSkills.length,
      percentage: Math.round((matchedSkills.length / jobSkills.length) * 100)
    };
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
      className={`bg-white dark:bg-gray-900 rounded-xl shadow-lg p-6 ${className}`}
    >
      <motion.h3
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="text-2xl font-bold text-gray-900 dark:text-white mb-6"
      >
        Job Compatibility Dashboard
      </motion.h3>

      {/* Search and Filters */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        className="mb-6 space-y-4"
      >
        {/* Search Bar */}
        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
          <input
            type="text"
            placeholder="Search jobs by title or company..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-10 pr-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
        </div>

        {/* Filters */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <select
            value={selectedFilters.location}
            onChange={(e) => setSelectedFilters({...selectedFilters, location: e.target.value})}
            className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500"
          >
            <option value="">All Locations</option>
            <option value="remote">Remote</option>
            <option value="new york">New York</option>
            <option value="san francisco">San Francisco</option>
            <option value="london">London</option>
          </select>

          <select
            value={selectedFilters.experienceLevel}
            onChange={(e) => setSelectedFilters({...selectedFilters, experienceLevel: e.target.value})}
            className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500"
          >
            <option value="">All Experience Levels</option>
            <option value="entry">Entry Level</option>
            <option value="mid">Mid Level</option>
            <option value="senior">Senior Level</option>
          </select>

          <select
            value={selectedFilters.compatibility}
            onChange={(e) => setSelectedFilters({...selectedFilters, compatibility: e.target.value})}
            className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500"
          >
            <option value="">All Compatibility</option>
            <option value="high">High Match (80%+)</option>
            <option value="medium">Medium Match (60-79%)</option>
            <option value="low">Low Match (&lt;60%)</option>
          </select>

          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value)}
            className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500"
          >
            <option value="compatibility">Sort by Compatibility</option>
            <option value="salary">Sort by Salary</option>
            <option value="date">Sort by Date Posted</option>
          </select>
        </div>
      </motion.div>

      {/* Results Summary */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="mb-6 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg"
      >
        <div className="flex items-center justify-between">
          <span className="text-blue-700 dark:text-blue-300">
            Found {filteredJobs.length} matching jobs
          </span>
          <div className="flex items-center gap-4 text-sm text-blue-600 dark:text-blue-400">
            <span>High Match: {filteredJobs.filter(job => job.compatibilityScore >= 0.8).length}</span>
            <span>Medium Match: {filteredJobs.filter(job => job.compatibilityScore >= 0.6 && job.compatibilityScore < 0.8).length}</span>
            <span>Low Match: {filteredJobs.filter(job => job.compatibilityScore < 0.6).length}</span>
          </div>
        </div>
      </motion.div>

      {/* Job List */}
      <div className="space-y-4">
        {filteredJobs.map((job, index) => {
          const skillMatch = calculateSkillMatch(job.requiredSkills, userSkills);
          const isExpanded = expandedJob === index;
          
          return (
            <motion.div
              key={job.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 + index * 0.1 }}
              className={`border border-gray-200 dark:border-gray-700 rounded-lg p-4 hover:shadow-md transition-shadow ${getCompatibilityBg(job.compatibilityScore)}`}
            >
              <div className="flex items-start justify-between mb-3">
                <div className="flex-1">
                  <div className="flex items-center gap-3 mb-2">
                    <h4 className="text-lg font-semibold text-gray-900 dark:text-white">
                      {job.title}
                    </h4>
                    {getCompatibilityIcon(job.compatibilityScore)}
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${getCompatibilityColor(job.compatibilityScore)}`}>
                      {Math.round(job.compatibilityScore * 100)}% Match
                    </span>
                  </div>
                  
                  <div className="flex items-center gap-4 text-sm text-gray-600 dark:text-gray-400 mb-2">
                    <div className="flex items-center gap-1">
                      <Building className="w-4 h-4" />
                      <span>{job.company}</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <MapPin className="w-4 h-4" />
                      <span>{job.location}</span>
                    </div>
                    {job.salaryRange && (
                      <div className="flex items-center gap-1">
                        <DollarSign className="w-4 h-4" />
                        <span>${job.salaryRange.min}k - ${job.salaryRange.max}k</span>
                      </div>
                    )}
                    <div className="flex items-center gap-1">
                      <Clock className="w-4 h-4" />
                      <span>{job.experienceLevel}</span>
                    </div>
                  </div>

                  <div className="flex items-center gap-4 text-sm">
                    <div className="flex items-center gap-2">
                      <span className="text-gray-600 dark:text-gray-400">Skills Match:</span>
                      <span className={`font-medium ${getCompatibilityColor(skillMatch.percentage / 100)}`}>
                        {skillMatch.matched}/{skillMatch.total} ({skillMatch.percentage}%)
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      <Users className="w-4 h-4 text-gray-400" />
                      <span className="text-gray-600 dark:text-gray-400">{job.applicants || 0} applicants</span>
                    </div>
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  <button
                    onClick={() => setExpandedJob(isExpanded ? null : index)}
                    className="p-2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 transition-colors"
                  >
                    {isExpanded ? <ChevronUp className="w-5 h-5" /> : <ChevronDown className="w-5 h-5" />}
                  </button>
                  <button className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors flex items-center gap-2">
                    Apply
                    <ExternalLink className="w-4 h-4" />
                  </button>
                </div>
              </div>

              {/* Skill Match Progress Bar */}
              <div className="mb-3">
                <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${skillMatch.percentage}%` }}
                    transition={{ duration: 1, delay: 0.6 + index * 0.1 }}
                    className={`h-2 rounded-full ${
                      skillMatch.percentage >= 80 ? 'bg-green-500' :
                      skillMatch.percentage >= 60 ? 'bg-yellow-500' : 'bg-red-500'
                    }`}
                  />
                </div>
              </div>

              {/* Expanded Details */}
              <AnimatePresence>
                {isExpanded && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    transition={{ duration: 0.3 }}
                    className="border-t border-gray-200 dark:border-gray-700 pt-4 mt-4"
                  >
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      {/* Job Description */}
                      <div>
                        <h5 className="font-semibold text-gray-900 dark:text-white mb-2">
                          Job Description
                        </h5>
                        <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
                          {job.description}
                        </p>

                        <h5 className="font-semibold text-gray-900 dark:text-white mb-2">
                          Required Skills
                        </h5>
                        <div className="flex flex-wrap gap-2">
                          {job.requiredSkills.map((skill, skillIndex) => {
                            const hasSkill = userSkills.some(userSkill => 
                              userSkill.name.toLowerCase() === skill.toLowerCase()
                            );
                            return (
                              <span
                                key={skillIndex}
                                className={`px-2 py-1 rounded-full text-xs font-medium ${
                                  hasSkill 
                                    ? 'bg-green-100 dark:bg-green-900/20 text-green-800 dark:text-green-200'
                                    : 'bg-red-100 dark:bg-red-900/20 text-red-800 dark:text-red-200'
                                }`}
                              >
                                {skill} {hasSkill ? '✓' : '✗'}
                              </span>
                            );
                          })}
                        </div>
                      </div>

                      {/* Compatibility Analysis */}
                      <div>
                        <h5 className="font-semibold text-gray-900 dark:text-white mb-2">
                          Compatibility Analysis
                        </h5>
                        
                        <div className="space-y-3">
                          <div className="flex items-center justify-between">
                            <span className="text-sm text-gray-600 dark:text-gray-400">Overall Match</span>
                            <span className={`font-medium ${getCompatibilityColor(job.compatibilityScore)}`}>
                              {Math.round(job.compatibilityScore * 100)}%
                            </span>
                          </div>
                          
                          <div className="flex items-center justify-between">
                            <span className="text-sm text-gray-600 dark:text-gray-400">Skills Match</span>
                            <span className={`font-medium ${getCompatibilityColor(skillMatch.percentage / 100)}`}>
                              {skillMatch.percentage}%
                            </span>
                          </div>
                          
                          <div className="flex items-center justify-between">
                            <span className="text-sm text-gray-600 dark:text-gray-400">Experience Level</span>
                            <span className={`font-medium ${
                              job.experienceLevel === 'entry' ? 'text-green-600 dark:text-green-400' :
                              job.experienceLevel === 'mid' ? 'text-yellow-600 dark:text-yellow-400' :
                              'text-red-600 dark:text-red-400'
                            }`}>
                              {job.experienceLevel}
                            </span>
                          </div>

                          {job.salaryRange && (
                            <div className="flex items-center justify-between">
                              <span className="text-sm text-gray-600 dark:text-gray-400">Salary Range</span>
                              <span className="font-medium text-gray-900 dark:text-white">
                                ${job.salaryRange.min}k - ${job.salaryRange.max}k
                              </span>
                            </div>
                          )}
                        </div>

                        {/* Missing Skills */}
                        {skillMatch.matched < skillMatch.total && (
                          <div className="mt-4">
                            <h6 className="font-medium text-gray-900 dark:text-white mb-2">
                              Skills to Develop
                            </h6>
                            <div className="flex flex-wrap gap-2">
                              {job.requiredSkills
                                .filter(skill => !userSkills.some(userSkill => 
                                  userSkill.name.toLowerCase() === skill.toLowerCase()
                                ))
                                .map((skill, skillIndex) => (
                                  <span
                                    key={skillIndex}
                                    className="px-2 py-1 bg-orange-100 dark:bg-orange-900/20 text-orange-800 dark:text-orange-200 rounded-full text-xs font-medium"
                                  >
                                    {skill}
                                  </span>
                                ))
                              }
                            </div>
                          </div>
                        )}
                      </div>
                    </div>

                    {/* Action Buttons */}
                    <div className="mt-4 flex gap-2">
                      <button className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors">
                        Apply Now
                      </button>
                      <button className="px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 transition-colors">
                        Save Job
                      </button>
                      <button className="px-4 py-2 bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors">
                        View Company
                      </button>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>
          );
        })}
      </div>

      {/* No Results */}
      {filteredJobs.length === 0 && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="text-center py-12"
        >
          <Search className="w-12 h-12 text-gray-400 mx-auto mb-4" />
          <h4 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
            No jobs found
          </h4>
          <p className="text-gray-600 dark:text-gray-400">
            Try adjusting your search criteria or filters
          </p>
        </motion.div>
      )}
    </motion.div>
  );
};

export default JobCompatibilityDashboard;