'use client';

import React from 'react';
import { VisualizationDashboard } from '../../components/visualizations';

// Mock data for demonstration
const mockUserData = {
  skills: [
    { name: 'JavaScript', level: 0.9, required: 0.95 },
    { name: 'React', level: 0.85, required: 0.9 },
    { name: 'Node.js', level: 0.7, required: 0.8 },
    { name: 'Python', level: 0.6, required: 0.85 },
    { name: 'SQL', level: 0.8, required: 0.75 },
    { name: 'AWS', level: 0.4, required: 0.7 },
    { name: 'Docker', level: 0.5, required: 0.8 },
    { name: 'GraphQL', level: 0.3, required: 0.6 }
  ],
  skillCount: 15,
  matchingJobs: 42,
  careerProgress: 68,
  learningHours: 120,
  targetRole: "Senior Full Stack Developer",
  currentPosition: 2,
  careerRoadmap: [
    {
      id: 1,
      title: "Junior Developer",
      description: "Entry-level position focusing on basic programming skills",
      duration: "6-12 months",
      salary: "$50k-70k",
      skills: ["HTML", "CSS", "JavaScript", "Git"]
    },
    {
      id: 2,
      title: "Mid-Level Developer",
      description: "Developing more complex applications and working with frameworks",
      duration: "1-2 years",
      salary: "$70k-90k",
      skills: ["React", "Node.js", "Database Design", "Testing"]
    },
    {
      id: 3,
      title: "Senior Developer",
      description: "Leading projects and mentoring junior developers",
      duration: "2-3 years",
      salary: "$90k-120k",
      skills: ["System Architecture", "Leadership", "DevOps", "Performance Optimization"]
    },
    {
      id: 4,
      title: "Tech Lead",
      description: "Technical leadership and strategic decision making",
      duration: "3+ years",
      salary: "$120k-150k",
      skills: ["Team Management", "Strategic Planning", "Stakeholder Communication"]
    },
    {
      id: 5,
      title: "Engineering Manager",
      description: "Managing engineering teams and driving technical vision",
      duration: "Ongoing",
      salary: "$150k+",
      skills: ["People Management", "Business Strategy", "Technical Vision"]
    }
  ],
  skillGaps: [
    {
      name: "AWS Cloud Services",
      currentLevel: 0.4,
      requiredLevel: 0.8,
      priority: "high",
      estimatedLearningTime: "3-4 months",
      learningResources: [
        { title: "AWS Solutions Architect Course", type: "course", provider: "AWS" },
        { title: "Cloud Practitioner Certification", type: "certification", provider: "AWS" }
      ],
      subSkills: [
        { name: "EC2", level: 0.5 },
        { name: "S3", level: 0.6 },
        { name: "Lambda", level: 0.2 },
        { name: "RDS", level: 0.3 }
      ]
    },
    {
      name: "Docker & Containerization",
      currentLevel: 0.5,
      requiredLevel: 0.8,
      priority: "high",
      estimatedLearningTime: "2-3 months",
      learningResources: [
        { title: "Docker Mastery Course", type: "course", provider: "Udemy" },
        { title: "Kubernetes Fundamentals", type: "course", provider: "Linux Foundation" }
      ],
      subSkills: [
        { name: "Docker Basics", level: 0.7 },
        { name: "Docker Compose", level: 0.4 },
        { name: "Kubernetes", level: 0.2 }
      ]
    },
    {
      name: "System Design",
      currentLevel: 0.3,
      requiredLevel: 0.7,
      priority: "medium",
      estimatedLearningTime: "4-6 months",
      learningResources: [
        { title: "System Design Interview Course", type: "course", provider: "Educative" },
        { title: "Designing Data-Intensive Applications", type: "book", provider: "O'Reilly" }
      ],
      subSkills: [
        { name: "Scalability", level: 0.3 },
        { name: "Load Balancing", level: 0.2 },
        { name: "Caching", level: 0.4 }
      ]
    }
  ],
  jobs: [
    {
      id: 1,
      title: "Senior Full Stack Developer",
      company: "TechCorp Inc.",
      location: "San Francisco, CA",
      salaryRange: { min: 120, max: 150 },
      experienceLevel: "senior",
      compatibilityScore: 0.85,
      requiredSkills: ["JavaScript", "React", "Node.js", "AWS", "Docker", "GraphQL"],
      description: "We're looking for a senior full stack developer to join our growing team...",
      applicants: 45,
      postedDate: "2024-01-15"
    },
    {
      id: 2,
      title: "Frontend Developer",
      company: "StartupXYZ",
      location: "Remote",
      salaryRange: { min: 90, max: 110 },
      experienceLevel: "mid",
      compatibilityScore: 0.92,
      requiredSkills: ["JavaScript", "React", "CSS", "TypeScript"],
      description: "Join our remote team building the next generation of web applications...",
      applicants: 23,
      postedDate: "2024-01-12"
    },
    {
      id: 3,
      title: "DevOps Engineer",
      company: "CloudTech Solutions",
      location: "New York, NY",
      salaryRange: { min: 110, max: 140 },
      experienceLevel: "senior",
      compatibilityScore: 0.45,
      requiredSkills: ["AWS", "Docker", "Kubernetes", "Python", "Terraform"],
      description: "Looking for a DevOps engineer to help scale our infrastructure...",
      applicants: 67,
      postedDate: "2024-01-10"
    }
  ],
  careerRecommendations: [
    { title: "Senior Full Stack Developer", matchScore: 85 },
    { title: "Frontend Architect", matchScore: 78 },
    { title: "Technical Lead", matchScore: 72 }
  ],
  learningPaths: [
    { title: "AWS Cloud Mastery", duration: "3 months", difficulty: "Intermediate" },
    { title: "System Design Fundamentals", duration: "4 months", difficulty: "Advanced" },
    { title: "Leadership Skills", duration: "2 months", difficulty: "Beginner" }
  ],
  marketTrends: [
    { skill: "AI/ML", growth: 45 },
    { skill: "Cloud Computing", growth: 32 },
    { skill: "Cybersecurity", growth: 28 }
  ],
  skillRecommendations: [
    { name: "TypeScript", priority: "high" },
    { name: "Kubernetes", priority: "medium" },
    { name: "GraphQL", priority: "medium" },
    { name: "Microservices", priority: "low" },
    { name: "Machine Learning", priority: "low" }
  ]
};

export default function VisualizationsPage() {
  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      <div className="container mx-auto px-4 py-8">
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-2">
            Career Analytics Visualizations
          </h1>
          <p className="text-lg text-gray-600 dark:text-gray-400">
            Interactive dashboard showcasing your career development progress and opportunities
          </p>
        </div>
        
        <VisualizationDashboard userData={mockUserData} />
      </div>
    </div>
  );
}