'use client';

import React, { useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import { useAuth } from '../contexts/AuthContext';
import { useRouter } from 'next/navigation';
import AnimatedButton from '../components/ui/AnimatedButton';
import GlassCard from '../components/ui/GlassCard';
import AnimatedBackground from '../components/ui/AnimatedBackground';
import Link from 'next/link';
import {
  SparklesIcon,
  RocketLaunchIcon,
  ChartBarIcon,
  AcademicCapIcon,
  BriefcaseIcon,
  UserGroupIcon,
  CheckCircleIcon,
  ArrowRightIcon,
  PlayIcon
} from '@heroicons/react/24/outline';
import anime from 'animejs';

const HomePage = () => {
  const { user } = useAuth();
  const router = useRouter();
  const heroRef = useRef(null);
  const statsRef = useRef(null);

  useEffect(() => {
    if (user) {
      router.push('/dashboard');
    }
  }, [user, router]);

  useEffect(() => {
    // Hero section animations
    anime({
      targets: '.hero-element',
      translateY: [50, 0],
      opacity: [0, 1],
      duration: 1000,
      delay: anime.stagger(200),
      easing: 'easeOutExpo'
    });

    // Stats counter animation
    anime({
      targets: '.stat-number',
      innerHTML: (el) => [0, parseInt(el.getAttribute('data-count'))],
      duration: 2000,
      delay: 1000,
      easing: 'easeOutExpo',
      round: 1
    });

    // Feature cards animation
    anime({
      targets: '.feature-card',
      scale: [0.8, 1],
      opacity: [0, 1],
      duration: 800,
      delay: anime.stagger(100, { start: 1500 }),
      easing: 'easeOutBack'
    });
  }, []);

  const features = [
    {
      icon: SparklesIcon,
      title: 'AI-Powered Matching',
      description: 'Advanced machine learning algorithms analyze your skills and match you with perfect career opportunities.',
      gradient: 'from-blue-500 to-cyan-500'
    },
    {
      icon: ChartBarIcon,
      title: 'Career Analytics',
      description: 'Comprehensive insights into your career progression with detailed analytics and market trends.',
      gradient: 'from-purple-500 to-pink-500'
    },
    {
      icon: AcademicCapIcon,
      title: 'Personalized Learning',
      description: 'Custom learning paths designed to bridge your skill gaps and accelerate your career growth.',
      gradient: 'from-orange-500 to-red-500'
    },
    {
      icon: BriefcaseIcon,
      title: 'Job Recommendations',
      description: 'Get tailored job recommendations based on your profile, preferences, and career goals.',
      gradient: 'from-green-500 to-emerald-500'
    },
    {
      icon: UserGroupIcon,
      title: 'Network Building',
      description: 'Connect with industry professionals and expand your network for better opportunities.',
      gradient: 'from-indigo-500 to-blue-500'
    },
    {
      icon: RocketLaunchIcon,
      title: 'Career Acceleration',
      description: 'Fast-track your career with strategic guidance and actionable insights from AI analysis.',
      gradient: 'from-pink-500 to-rose-500'
    }
  ];

  const stats = [
    { label: 'Active Users', value: '50000', suffix: '+' },
    { label: 'Job Matches', value: '1200000', suffix: '+' },
    { label: 'Success Rate', value: '94', suffix: '%' },
    { label: 'Companies', value: '5000', suffix: '+' }
  ];

  const testimonials = [
    {
      name: 'Sarah Chen',
      role: 'Software Engineer at Google',
      image: '/testimonials/sarah.jpg',
      content: 'CareerAI helped me transition from marketing to tech. The personalized learning path was incredible!'
    },
    {
      name: 'Marcus Johnson',
      role: 'Data Scientist at Microsoft',
      image: '/testimonials/marcus.jpg',
      content: 'The AI recommendations were spot-on. I found my dream job within 2 weeks of using the platform.'
    },
    {
      name: 'Emily Rodriguez',
      role: 'Product Manager at Stripe',
      image: '/testimonials/emily.jpg',
      content: 'The career analytics gave me insights I never had before. It completely changed my approach to job hunting.'
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-dark-900 via-dark-800 to-dark-900 relative overflow-hidden">
      <AnimatedBackground variant="neural" />
      
      {/* Hero Section */}
      <section className="relative z-10 min-h-screen flex items-center justify-center px-4 py-20">
        <div className="max-w-7xl mx-auto text-center">
          <motion.div
            className="hero-element mb-8"
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
          >
            <div className="inline-flex items-center px-4 py-2 bg-white/10 backdrop-blur-sm rounded-full border border-white/20 mb-6">
              <SparklesIcon className="w-5 h-5 text-primary-400 mr-2" />
              <span className="text-sm text-gray-300">Powered by Advanced AI</span>
            </div>
            
            <h1 className="text-6xl md:text-8xl font-bold mb-6 gradient-text font-display">
              Transform Your
              <br />
              <span className="animate-typewriter">Career Journey</span>
            </h1>
            
            <p className="text-xl md:text-2xl text-gray-300 mb-8 max-w-3xl mx-auto leading-relaxed">
              Discover your perfect career path with AI-powered recommendations, 
              personalized learning, and data-driven insights that accelerate your professional growth.
            </p>
          </motion.div>

          <motion.div
            className="hero-element flex flex-col sm:flex-row gap-4 justify-center items-center mb-12"
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.3 }}
          >
            <Link href="/auth/signup">
              <AnimatedButton
                variant="primary"
                size="lg"
                glow
                gradient
                icon={<RocketLaunchIcon className="w-6 h-6" />}
              >
                Start Your Journey
              </AnimatedButton>
            </Link>
            
            <AnimatedButton
              variant="glass"
              size="lg"
              icon={<PlayIcon className="w-6 h-6" />}
            >
              Watch Demo
            </AnimatedButton>
          </motion.div>

          {/* Stats */}
          <motion.div
            ref={statsRef}
            className="hero-element grid grid-cols-2 md:grid-cols-4 gap-8 max-w-4xl mx-auto"
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.6 }}
          >
            {stats.map((stat, index) => (
              <GlassCard key={index} className="p-6 text-center">
                <div className="text-3xl md:text-4xl font-bold text-white mb-2">
                  <span className="stat-number" data-count={stat.value}>0</span>
                  <span className="text-primary-400">{stat.suffix}</span>
                </div>
                <p className="text-gray-400 text-sm">{stat.label}</p>
              </GlassCard>
            ))}
          </motion.div>
        </div>
      </section>

      {/* Features Section */}
      <section className="relative z-10 py-20 px-4">
        <div className="max-w-7xl mx-auto">
          <motion.div
            className="text-center mb-16"
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
          >
            <h2 className="text-4xl md:text-6xl font-bold mb-6 gradient-text font-display">
              Powerful Features
            </h2>
            <p className="text-xl text-gray-300 max-w-3xl mx-auto">
              Everything you need to accelerate your career and achieve your professional goals
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {features.map((feature, index) => (
              <motion.div
                key={index}
                className="feature-card"
                initial={{ opacity: 0, scale: 0.8 }}
                whileInView={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                viewport={{ once: true }}
              >
                <GlassCard className="p-8 h-full hover:scale-105 transition-transform duration-300" hover glow>
                  <div className={`w-16 h-16 bg-gradient-to-r ${feature.gradient} rounded-2xl flex items-center justify-center mb-6`}>
                    <feature.icon className="w-8 h-8 text-white" />
                  </div>
                  <h3 className="text-2xl font-bold text-white mb-4">{feature.title}</h3>
                  <p className="text-gray-300 leading-relaxed">{feature.description}</p>
                </GlassCard>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section className="relative z-10 py-20 px-4">
        <div className="max-w-7xl mx-auto">
          <motion.div
            className="text-center mb-16"
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
          >
            <h2 className="text-4xl md:text-6xl font-bold mb-6 gradient-text font-display">
              How It Works
            </h2>
            <p className="text-xl text-gray-300 max-w-3xl mx-auto">
              Simple steps to transform your career with AI-powered insights
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {[
              {
                step: '01',
                title: 'Create Your Profile',
                description: 'Upload your resume and connect your professional accounts for comprehensive analysis.',
                icon: UserGroupIcon
              },
              {
                step: '02',
                title: 'AI Analysis',
                description: 'Our advanced AI analyzes your skills, experience, and career goals to create your unique profile.',
                icon: SparklesIcon
              },
              {
                step: '03',
                title: 'Get Recommendations',
                description: 'Receive personalized job matches, learning paths, and career guidance tailored to your goals.',
                icon: RocketLaunchIcon
              }
            ].map((item, index) => (
              <motion.div
                key={index}
                className="text-center"
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.2 }}
                viewport={{ once: true }}
              >
                <div className="relative mb-8">
                  <div className="w-24 h-24 bg-gradient-to-r from-primary-500 to-secondary-500 rounded-full flex items-center justify-center mx-auto mb-4">
                    <item.icon className="w-12 h-12 text-white" />
                  </div>
                  <div className="absolute -top-2 -right-2 w-8 h-8 bg-accent-500 rounded-full flex items-center justify-center text-white font-bold text-sm">
                    {item.step}
                  </div>
                </div>
                <h3 className="text-2xl font-bold text-white mb-4">{item.title}</h3>
                <p className="text-gray-300 leading-relaxed">{item.description}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Testimonials Section */}
      <section className="relative z-10 py-20 px-4">
        <div className="max-w-7xl mx-auto">
          <motion.div
            className="text-center mb-16"
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
          >
            <h2 className="text-4xl md:text-6xl font-bold mb-6 gradient-text font-display">
              Success Stories
            </h2>
            <p className="text-xl text-gray-300 max-w-3xl mx-auto">
              Join thousands of professionals who have transformed their careers with CareerAI
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {testimonials.map((testimonial, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                viewport={{ once: true }}
              >
                <GlassCard className="p-8 h-full" hover>
                  <div className="flex items-center mb-6">
                    <div className="w-12 h-12 bg-gradient-to-r from-primary-400 to-secondary-400 rounded-full flex items-center justify-center mr-4">
                      <span className="text-white font-bold text-lg">
                        {testimonial.name.charAt(0)}
                      </span>
                    </div>
                    <div>
                      <h4 className="text-white font-semibold">{testimonial.name}</h4>
                      <p className="text-gray-400 text-sm">{testimonial.role}</p>
                    </div>
                  </div>
                  <p className="text-gray-300 leading-relaxed italic">
                    "{testimonial.content}"
                  </p>
                </GlassCard>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="relative z-10 py-20 px-4">
        <div className="max-w-4xl mx-auto text-center">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
          >
            <GlassCard className="p-12" glow>
              <h2 className="text-4xl md:text-5xl font-bold mb-6 gradient-text font-display">
                Ready to Transform Your Career?
              </h2>
              <p className="text-xl text-gray-300 mb-8 leading-relaxed">
                Join thousands of professionals who are already using AI to accelerate their careers. 
                Start your journey today and unlock your full potential.
              </p>
              
              <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
                <Link href="/auth/signup">
                  <AnimatedButton
                    variant="primary"
                    size="lg"
                    glow
                    gradient
                    icon={<RocketLaunchIcon className="w-6 h-6" />}
                  >
                    Get Started Free
                  </AnimatedButton>
                </Link>
                
                <Link href="/features">
                  <AnimatedButton
                    variant="ghost"
                    size="lg"
                    icon={<ArrowRightIcon className="w-6 h-6" />}
                    iconPosition="right"
                  >
                    Learn More
                  </AnimatedButton>
                </Link>
              </div>

              <div className="mt-8 flex items-center justify-center space-x-6 text-sm text-gray-400">
                <div className="flex items-center">
                  <CheckCircleIcon className="w-5 h-5 text-green-400 mr-2" />
                  <span>Free to start</span>
                </div>
                <div className="flex items-center">
                  <CheckCircleIcon className="w-5 h-5 text-green-400 mr-2" />
                  <span>No credit card required</span>
                </div>
                <div className="flex items-center">
                  <CheckCircleIcon className="w-5 h-5 text-green-400 mr-2" />
                  <span>Setup in 2 minutes</span>
                </div>
              </div>
            </GlassCard>
          </motion.div>
        </div>
      </section>
    </div>
  );
};

export default HomePage;