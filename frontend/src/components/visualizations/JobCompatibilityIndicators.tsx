import React, { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { 
  Briefcase, 
  MapPin, 
  DollarSign, 
  Clock, 
  Star, 
  TrendingUp, 
  CheckCircle, 
  XCircle,
  AlertTriangle,
  ExternalLink,
  Heart,
  Bookmark
} from 'lucide-react';

interface JobMatch {
  id: string;
  title: string;
  company: string;
  location: string;
  salary: string;
  matchScore: number;
  skillMatches: string[];
  skillGaps: string[];
  experience: string;
  type: string;
  postedDate: string;
  url: string;
  description: string;
  requirements: string[];
  benefits: string[];
  isBookmarked?: boolean;
  isLiked?: boolean;
}

interface JobCompatibilityIndicatorsProps {
  jobMatches: JobMatch[];
  title?: string;
  className?: string;
  maxJobs?: number;
}

export function JobCompatibilityIndicators({ 
  jobMatches, 
  title = "Job Compatibility Analysis", 
  className = "",
  maxJobs = 5
}: JobCompatibilityIndicatorsProps) {
  const [animatedJobs, setAnimatedJobs] = useState<JobMatch[]>([]);
  const [selectedJob, setSelectedJob] = useState<string | null>(null);
  const [sortBy, setSortBy] = useState<'match' | 'salary' | 'date'>('match');
  const [bookmarkedJobs, setBookmarkedJobs] = useState<Set<string>>(new Set());
  const [likedJobs, setLikedJobs] = useState<Set<string>>(new Set());

  useEffect(() => {
    // Initialize with zero match scores for animation
    const initialJobs = jobMatches.slice(0, maxJobs).map(job => ({
      ...job,
      matchScore: 0
    }));
    setAnimatedJobs(initialJobs);

    // Animate to actual scores
    const timer = setTimeout(() => {
      setAnimatedJobs(jobMatches.slice(0, maxJobs));
    }, 500);

    return () => clearTimeout(timer);
  }, [jobMatches, maxJobs]);

  const sortedJobs = [...animatedJobs].sort((a, b) => {
    switch (sortBy) {
      case 'match':
        return b.matchScore - a.matchScore;
      case 'salary':
        // Simple salary comparison (assuming format like "$50k-70k")
        const getSalaryValue = (salary: string) => {
          const match = salary.match(/\$(\d+)/);
          return match ? parseInt(match[1]) : 0;
        };
        return getSalaryValue(b.salary) - getSalaryValue(a.salary);
      case 'date':
        return new Date(b.postedDate).getTime() - new Date(a.postedDate).getTime();
      default:
        return 0;
    }
  });

  const getMatchColor = (score: number) => {
    if (score >= 80) return 'text-green-600 bg-green-100';
    if (score >= 60) return 'text-yellow-600 bg-yellow-100';
    return 'text-red-600 bg-red-100';
  };

  const getMatchIcon = (score: number) => {
    if (score >= 80) return <CheckCircle className="w-5 h-5 text-green-600" />;
    if (score >= 60) return <AlertTriangle className="w-5 h-5 text-yellow-600" />;
    return <XCircle className="w-5 h-5 text-red-600" />;
  };

  const toggleBookmark = (jobId: string) => {
    setBookmarkedJobs(prev => {
      const newSet = new Set(prev);
      if (newSet.has(jobId)) {
        newSet.delete(jobId);
      } else {
        newSet.add(jobId);
      }
      return newSet;
    });
  };

  const toggleLike = (jobId: string) => {
    setLikedJobs(prev => {
      const newSet = new Set(prev);
      if (newSet.has(jobId)) {
        newSet.delete(jobId);
      } else {
        newSet.add(jobId);
      }
      return newSet;
    });
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

  const jobCardVariants = {
    hidden: { opacity: 0, y: 50, scale: 0.9 },
    visible: {
      opacity: 1,
      y: 0,
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
          className="flex items-center space-x-3"
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value as 'match' | 'salary' | 'date')}
            className="text-sm border border-gray-300 rounded px-3 py-1 focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="match">Best Match</option>
            <option value="salary">Highest Salary</option>
            <option value="date">Most Recent</option>
          </select>
        </motion.div>
      </div>

      {/* Summary stats */}
      <motion.div
        className="grid grid-cols-4 gap-4 mb-6 p-4 bg-gradient-to-r from-blue-50 to-green-50 rounded-lg"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3, duration: 0.5 }}
      >
        <div className="text-center">
          <div className="text-2xl font-bold text-green-600">
            {jobMatches.filter(j => j.matchScore >= 80).length}
          </div>
          <div className="text-sm text-gray-600">Excellent Match</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-yellow-600">
            {jobMatches.filter(j => j.matchScore >= 60 && j.matchScore < 80).length}
          </div>
          <div className="text-sm text-gray-600">Good Match</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-blue-600">
            {Math.round(jobMatches.reduce((acc, job) => acc + job.matchScore, 0) / jobMatches.length)}%
          </div>
          <div className="text-sm text-gray-600">Avg Match</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-purple-600">
            {bookmarkedJobs.size}
          </div>
          <div className="text-sm text-gray-600">Bookmarked</div>
        </div>
      </motion.div>

      {/* Job cards */}
      <div className="space-y-4">
        {sortedJobs.map((job, index) => (
          <motion.div
            key={job.id}
            variants={jobCardVariants}
            className="border border-gray-200 rounded-lg p-5 hover:shadow-lg transition-all duration-300 cursor-pointer"
            onClick={() => setSelectedJob(selectedJob === job.id ? null : job.id)}
            whileHover={{ scale: 1.01, y: -2 }}
            whileTap={{ scale: 0.99 }}
          >
            {/* Job header */}
            <div className="flex items-start justify-between mb-4">
              <div className="flex-1">
                <div className="flex items-center space-x-3 mb-2">
                  <h4 className="text-lg font-semibold text-gray-900">{job.title}</h4>
                  <div className="flex items-center space-x-1">
                    {getMatchIcon(job.matchScore)}
                  </div>
                </div>
                <div className="flex items-center space-x-4 text-sm text-gray-600">
                  <div className="flex items-center space-x-1">
                    <Briefcase className="w-4 h-4" />
                    <span>{job.company}</span>
                  </div>
                  <div className="flex items-center space-x-1">
                    <MapPin className="w-4 h-4" />
                    <span>{job.location}</span>
                  </div>
                  <div className="flex items-center space-x-1">
                    <DollarSign className="w-4 h-4" />
                    <span>{job.salary}</span>
                  </div>
                  <div className="flex items-center space-x-1">
                    <Clock className="w-4 h-4" />
                    <span>{job.postedDate}</span>
                  </div>
                </div>
              </div>
              
              {/* Actions */}
              <div className="flex items-center space-x-2">
                <motion.button
                  onClick={(e) => {
                    e.stopPropagation();
                    toggleLike(job.id);
                  }}
                  className={`p-2 rounded-full transition-colors ${
                    likedJobs.has(job.id) ? 'text-red-500 bg-red-50' : 'text-gray-400 hover:text-red-500'
                  }`}
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.9 }}
                >
                  <Heart className={`w-5 h-5 ${likedJobs.has(job.id) ? 'fill-current' : ''}`} />
                </motion.button>
                
                <motion.button
                  onClick={(e) => {
                    e.stopPropagation();
                    toggleBookmark(job.id);
                  }}
                  className={`p-2 rounded-full transition-colors ${
                    bookmarkedJobs.has(job.id) ? 'text-blue-500 bg-blue-50' : 'text-gray-400 hover:text-blue-500'
                  }`}
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.9 }}
                >
                  <Bookmark className={`w-5 h-5 ${bookmarkedJobs.has(job.id) ? 'fill-current' : ''}`} />
                </motion.button>
              </div>
            </div>

            {/* Match score visualization */}
            <div className="mb-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium text-gray-700">Compatibility Score</span>
                <span className={`text-sm font-bold px-2 py-1 rounded-full ${getMatchColor(job.matchScore)}`}>
                  {job.matchScore}%
                </span>
              </div>
              
              <div className="relative">
                <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
                  <motion.div
                    className={`h-full rounded-full ${
                      job.matchScore >= 80 ? 'bg-gradient-to-r from-green-400 to-green-600' :
                      job.matchScore >= 60 ? 'bg-gradient-to-r from-yellow-400 to-yellow-600' :
                      'bg-gradient-to-r from-red-400 to-red-600'
                    }`}
                    initial={{ width: 0 }}
                    animate={{ width: `${job.matchScore}%` }}
                    transition={{ duration: 1.5, delay: index * 0.2 + 0.5, ease: "easeOut" }}
                  />
                </div>
                
                {/* Score indicator */}
                <motion.div
                  className="absolute top-0 h-3 w-1 bg-white border-2 border-gray-400 rounded-full"
                  initial={{ left: 0 }}
                  animate={{ left: `${job.matchScore}%` }}
                  transition={{ duration: 1.5, delay: index * 0.2 + 0.5, ease: "easeOut" }}
                  style={{ transform: 'translateX(-50%)' }}
                />
              </div>
            </div>

            {/* Skills match/gap indicators */}
            <div className="grid grid-cols-2 gap-4 mb-4">
              <div>
                <h5 className="text-sm font-medium text-green-700 mb-2 flex items-center">
                  <CheckCircle className="w-4 h-4 mr-1" />
                  Matching Skills ({job.skillMatches.length})
                </h5>
                <div className="flex flex-wrap gap-1">
                  {job.skillMatches.slice(0, 4).map((skill, skillIndex) => (
                    <motion.span
                      key={skill}
                      className="px-2 py-1 bg-green-100 text-green-800 text-xs rounded-full"
                      initial={{ opacity: 0, scale: 0 }}
                      animate={{ opacity: 1, scale: 1 }}
                      transition={{ delay: index * 0.2 + 0.8 + skillIndex * 0.1, duration: 0.3 }}
                    >
                      {skill}
                    </motion.span>
                  ))}
                  {job.skillMatches.length > 4 && (
                    <span className="px-2 py-1 bg-gray-100 text-gray-600 text-xs rounded-full">
                      +{job.skillMatches.length - 4} more
                    </span>
                  )}
                </div>
              </div>
              
              <div>
                <h5 className="text-sm font-medium text-red-700 mb-2 flex items-center">
                  <XCircle className="w-4 h-4 mr-1" />
                  Skill Gaps ({job.skillGaps.length})
                </h5>
                <div className="flex flex-wrap gap-1">
                  {job.skillGaps.slice(0, 4).map((skill, skillIndex) => (
                    <motion.span
                      key={skill}
                      className="px-2 py-1 bg-red-100 text-red-800 text-xs rounded-full"
                      initial={{ opacity: 0, scale: 0 }}
                      animate={{ opacity: 1, scale: 1 }}
                      transition={{ delay: index * 0.2 + 1.0 + skillIndex * 0.1, duration: 0.3 }}
                    >
                      {skill}
                    </motion.span>
                  ))}
                  {job.skillGaps.length > 4 && (
                    <span className="px-2 py-1 bg-gray-100 text-gray-600 text-xs rounded-full">
                      +{job.skillGaps.length - 4} more
                    </span>
                  )}
                </div>
              </div>
            </div>

            {/* Expandable details */}
            <AnimatePresence>
              {selectedJob === job.id && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  exit={{ opacity: 0, height: 0 }}
                  transition={{ duration: 0.3 }}
                  className="border-t border-gray-200 pt-4 mt-4"
                >
                  <div className="space-y-4">
                    {/* Job description */}
                    <div>
                      <h5 className="font-medium text-gray-800 mb-2">Job Description</h5>
                      <p className="text-sm text-gray-600 leading-relaxed">
                        {job.description}
                      </p>
                    </div>

                    {/* Requirements */}
                    <div>
                      <h5 className="font-medium text-gray-800 mb-2">Requirements</h5>
                      <ul className="space-y-1">
                        {job.requirements.map((req, reqIndex) => (
                          <motion.li
                            key={req}
                            className="text-sm text-gray-600 flex items-start space-x-2"
                            initial={{ opacity: 0, x: -10 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: reqIndex * 0.1, duration: 0.3 }}
                          >
                            <div className="w-1.5 h-1.5 bg-blue-500 rounded-full mt-2 flex-shrink-0" />
                            <span>{req}</span>
                          </motion.li>
                        ))}
                      </ul>
                    </div>

                    {/* Benefits */}
                    <div>
                      <h5 className="font-medium text-gray-800 mb-2">Benefits</h5>
                      <div className="flex flex-wrap gap-2">
                        {job.benefits.map((benefit, benefitIndex) => (
                          <motion.span
                            key={benefit}
                            className="px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded-full"
                            initial={{ opacity: 0, scale: 0 }}
                            animate={{ opacity: 1, scale: 1 }}
                            transition={{ delay: benefitIndex * 0.1, duration: 0.3 }}
                          >
                            {benefit}
                          </motion.span>
                        ))}
                      </div>
                    </div>

                    {/* Action buttons */}
                    <div className="flex space-x-3 pt-2">
                      <motion.a
                        href={job.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="flex items-center space-x-2 px-4 py-2 bg-blue-500 text-white text-sm rounded-lg hover:bg-blue-600 transition-colors"
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                      >
                        <ExternalLink className="w-4 h-4" />
                        <span>View Job</span>
                      </motion.a>
                      
                      <motion.button
                        className="flex items-center space-x-2 px-4 py-2 border border-gray-300 text-gray-700 text-sm rounded-lg hover:bg-gray-50 transition-colors"
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                      >
                        <TrendingUp className="w-4 h-4" />
                        <span>Improve Match</span>
                      </motion.button>
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>
        ))}
      </div>

      {/* Load more button */}
      {jobMatches.length > maxJobs && (
        <motion.div
          className="text-center mt-6"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.5, duration: 0.5 }}
        >
          <button className="px-6 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition-colors">
            Load More Jobs ({jobMatches.length - maxJobs} remaining)
          </button>
        </motion.div>
      )}
    </motion.div>
  );
}