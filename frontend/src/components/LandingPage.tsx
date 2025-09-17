import React from 'react';
import { motion } from 'motion/react';
import { Link } from 'react-router-dom';
import { Button } from './ui/button';
import { Card, CardContent } from './ui/card';
import { Badge } from './ui/badge';
import { ImageWithFallback } from './figma/ImageWithFallback';
import { 
  ArrowRight, 
  Star, 
  Users, 
  TrendingUp, 
  Brain, 
  Target, 
  Sparkles,
  CheckCircle,
  Github,
  Linkedin,
  Mail
} from 'lucide-react';

interface LandingPageProps {
  onLogin: () => void;
}

export function LandingPage({ onLogin }: LandingPageProps) {
  const stats = [
    { value: '10K+', label: 'Career Paths Recommended' },
    { value: '5K+', label: 'Professionals Guided' },
    { value: '95%', label: 'Success Rate' },
    { value: '24/7', label: 'AI Support' },
  ];

  const features = [
    {
      icon: Brain,
      title: 'AI-Powered Analysis',
      description: 'Get personalized career recommendations based on your skills and goals'
    },
    {
      icon: Target,
      title: 'Skill Gap Analysis',
      description: 'Identify areas for improvement and get learning recommendations'
    },
    {
      icon: TrendingUp,
      title: 'Market Insights',
      description: 'Stay updated with industry trends and salary benchmarks'
    },
    {
      icon: Users,
      title: 'Community Support',
      description: 'Connect with professionals and mentors in your field'
    }
  ];

  return (
    <div className="pt-16">
      {/* Hero Section */}
      <section className="relative min-h-screen flex items-center justify-center overflow-hidden">
        {/* Animated Background */}
        <div className="absolute inset-0 bg-gradient-to-br from-blue-50 via-white to-purple-50">
          <motion.div
            className="absolute top-20 left-10 w-72 h-72 bg-blue-200 rounded-full mix-blend-multiply filter blur-xl opacity-70"
            animate={{
              x: [0, 100, 0],
              y: [0, -100, 0],
            }}
            transition={{
              duration: 20,
              repeat: Infinity,
              repeatType: "reverse"
            }}
          />
          <motion.div
            className="absolute top-40 right-10 w-72 h-72 bg-purple-200 rounded-full mix-blend-multiply filter blur-xl opacity-70"
            animate={{
              x: [0, -100, 0],
              y: [0, 100, 0],
            }}
            transition={{
              duration: 15,
              repeat: Infinity,
              repeatType: "reverse"
            }}
          />
          <motion.div
            className="absolute bottom-20 left-20 w-72 h-72 bg-pink-200 rounded-full mix-blend-multiply filter blur-xl opacity-70"
            animate={{
              x: [0, -50, 0],
              y: [0, -50, 0],
            }}
            transition={{
              duration: 25,
              repeat: Infinity,
              repeatType: "reverse"
            }}
          />
        </div>

        <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <motion.div
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
          >
            <Badge variant="secondary" className="mb-6 px-4 py-2">
              <Sparkles className="w-4 h-4 mr-2" />
              AI-Powered Career Guidance
            </Badge>
            
            <h1 className="text-4xl md:text-6xl lg:text-7xl font-bold text-gray-900 mb-6">
              Navigate Your
              <span className="block text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-purple-600">
                Career Journey
              </span>
            </h1>
            
            <p className="text-xl md:text-2xl text-gray-600 mb-8 max-w-3xl mx-auto">
              Discover personalized career paths, identify skill gaps, and accelerate your professional growth with AI-powered insights.
            </p>
            
            <div className="flex flex-col sm:flex-row gap-4 justify-center mb-12">
              <Button 
                size="lg" 
                onClick={onLogin}
                className="bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white px-8 py-3"
              >
                Get Started Free
                <ArrowRight className="ml-2 w-5 h-5" />
              </Button>
              <Button variant="outline" size="lg" asChild>
                <Link to="/features">Learn More</Link>
              </Button>
            </div>

            {/* Login Options */}
            <div className="flex flex-col items-center space-y-4">
              <p className="text-sm text-gray-500">Sign up with</p>
              <div className="flex space-x-4">
                <Button 
                  variant="outline" 
                  size="sm"
                  onClick={onLogin}
                  className="flex items-center space-x-2"
                >
                  <Github className="w-4 h-4" />
                  <span>GitHub</span>
                </Button>
                <Button 
                  variant="outline" 
                  size="sm"
                  onClick={onLogin}
                  className="flex items-center space-x-2"
                >
                  <Linkedin className="w-4 h-4" />
                  <span>LinkedIn</span>
                </Button>
                <Button 
                  variant="outline" 
                  size="sm"
                  onClick={onLogin}
                  className="flex items-center space-x-2"
                >
                  <Mail className="w-4 h-4" />
                  <span>Email</span>
                </Button>
              </div>
            </div>
          </motion.div>
        </div>

        {/* Hero Image */}
        <motion.div 
          className="absolute bottom-10 right-10 hidden lg:block"
          initial={{ opacity: 0, x: 100 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 1, delay: 0.5 }}
        >
          <ImageWithFallback
            src="https://images.unsplash.com/photo-1738566061505-556830f8b8f5?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxjYXJlZXIlMjBwcm9mZXNzaW9uYWwlMjBzdWNjZXNzfGVufDF8fHx8MTc1Nzg0MTQyMHww&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral"
            alt="Career Success"
            className="w-64 h-64 object-cover rounded-2xl shadow-2xl"
          />
        </motion.div>
      </section>

      {/* Stats Section */}
      <section className="py-20 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            {stats.map((stat, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                className="text-center"
              >
                <div className="text-3xl md:text-4xl font-bold text-gray-900 mb-2">{stat.value}</div>
                <div className="text-gray-600">{stat.label}</div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
              Powerful Features for Career Growth
            </h2>
            <p className="text-xl text-gray-600 max-w-2xl mx-auto">
              Everything you need to make informed career decisions and accelerate your professional journey.
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
            {features.map((feature, index) => {
              const Icon = feature.icon;
              return (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.6, delay: index * 0.1 }}
                  whileHover={{ y: -5 }}
                >
                  <Card className="h-full hover:shadow-lg transition-shadow">
                    <CardContent className="p-6 text-center">
                      <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center mx-auto mb-4">
                        <Icon className="w-6 h-6 text-blue-600" />
                      </div>
                      <h3 className="text-lg font-semibold mb-2">{feature.title}</h3>
                      <p className="text-gray-600">{feature.description}</p>
                    </CardContent>
                  </Card>
                </motion.div>
              );
            })}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 bg-gradient-to-r from-blue-600 to-purple-600">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center text-white">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <h2 className="text-3xl md:text-4xl font-bold mb-4">
              Ready to Transform Your Career?
            </h2>
            <p className="text-xl mb-8 opacity-90">
              Join thousands of professionals who have already discovered their ideal career path.
            </p>
            <Button 
              size="lg" 
              onClick={onLogin}
              className="bg-white text-blue-600 hover:bg-gray-100 px-8 py-3"
            >
              Start Your Journey
              <ArrowRight className="ml-2 w-5 h-5" />
            </Button>
          </motion.div>
        </div>
      </section>
    </div>
  );
}