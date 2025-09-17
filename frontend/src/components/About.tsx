import React from 'react';
import { motion } from 'motion/react';
import { Card, CardContent } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { ImageWithFallback } from './figma/ImageWithFallback';
import { 
  Target, 
  Brain, 
  Users, 
  Award, 
  TrendingUp,
  Heart,
  Lightbulb,
  Shield,
  Clock,
  ArrowRight
} from 'lucide-react';

export function About() {
  const values = [
    {
      icon: Target,
      title: 'Personalized Guidance',
      description: 'Every recommendation is tailored to your unique skills, experience, and career goals.'
    },
    {
      icon: Brain,
      title: 'AI-Powered Insights',
      description: 'Leveraging advanced AI to analyze market trends and provide data-driven career advice.'
    },
    {
      icon: Users,
      title: 'Community Driven',
      description: 'Learn from a community of professionals who have successfully navigated their career journeys.'
    },
    {
      icon: Shield,
      title: 'Privacy First',
      description: 'Your data is secure and private. We never share your personal information without consent.'
    }
  ];

  const team = [
    {
      name: 'Sarah Chen',
      role: 'CEO & Co-founder',
      bio: 'Former Google PM with 10+ years in tech career development',
      image: 'https://images.unsplash.com/photo-1719038423831-c4a7c78c9a54?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MXwxfHNlYXJjaHwxfHxwcm9mZXNzaW9uYWwlMjB3b21hbnxlbnwwfHx8fDE3NTc4NDU1ODN8MA&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral'
    },
    {
      name: 'Marcus Rodriguez',
      role: 'CTO & Co-founder',
      bio: 'AI/ML expert, former Microsoft engineer with expertise in career analytics',
      image: 'https://images.unsplash.com/photo-1654289152769-c7a41a0c5d5c?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MXwxfHNlYXJjaHwxfHxwcm9mZXNzaW9uYWwlMjBtYW58ZW58MHx8fHwxNzU3ODQ1NjY1fDA&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral'
    },
    {
      name: 'Dr. Emily Watson',
      role: 'Head of Career Psychology',
      bio: 'PhD in Organizational Psychology, 15+ years in career counseling',
      image: 'https://images.unsplash.com/photo-1718220216044-006f43e3a9b1?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxtb2Rlcm4lMjBvZmZpY2UlMjB3b3Jrc3BhY2V8ZW58MXx8fHwxNzU3NzQ0NTUwfDA&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral'
    }
  ];

  const stats = [
    { value: '50K+', label: 'Professionals Helped' },
    { value: '95%', label: 'User Satisfaction' },
    { value: '200+', label: 'Partner Companies' },
    { value: '24/7', label: 'AI Support' }
  ];

  const milestones = [
    { year: '2022', event: 'CareerPilot Founded', description: 'Started with a vision to democratize career guidance' },
    { year: '2023', event: 'AI Engine Launch', description: 'Launched our proprietary career matching algorithm' },
    { year: '2024', event: '10K Users Milestone', description: 'Reached 10,000 active users and expanded globally' },
    { year: '2024', event: 'Series A Funding', description: 'Raised $15M to accelerate product development' }
  ];

  return (
    <div className="pt-16 min-h-screen bg-white">
      {/* Hero Section */}
      <section className="py-20 bg-gradient-to-br from-blue-50 via-white to-purple-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="text-center"
          >
            <Badge variant="secondary" className="mb-6 px-4 py-2">
              <Heart className="w-4 h-4 mr-2" />
              Our Mission
            </Badge>
            <h1 className="text-4xl md:text-6xl font-bold text-gray-900 mb-6">
              Empowering Careers Through
              <span className="block text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-purple-600">
                AI-Driven Insights
              </span>
            </h1>
            <p className="text-xl text-gray-600 mb-8 max-w-3xl mx-auto">
              We believe everyone deserves a fulfilling career. CareerPilot combines cutting-edge AI with human expertise 
              to provide personalized career guidance that adapts to the changing job market.
            </p>
          </motion.div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="py-16 bg-white">
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
                <div className="text-3xl md:text-4xl font-bold text-primary mb-2">{stat.value}</div>
                <div className="text-gray-600">{stat.label}</div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Our Story */}
      <section className="py-20 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6 }}
            >
              <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-6">Our Story</h2>
              <div className="space-y-4 text-gray-600">
                <p>
                  CareerPilot was born from a simple observation: too many talented professionals 
                  struggle to navigate their career paths effectively. Traditional career counseling 
                  was expensive, time-consuming, and often generic.
                </p>
                <p>
                  Our founders, having experienced these challenges firsthand in their careers at 
                  top tech companies, envisioned a world where personalized career guidance could 
                  be accessible to everyone, powered by AI but grounded in human understanding.
                </p>
                <p>
                  Today, we're proud to have helped thousands of professionals discover their ideal 
                  career paths, develop in-demand skills, and achieve their professional goals.
                </p>
              </div>
              <Button className="mt-6">
                Learn More About Our Journey
                <ArrowRight className="w-4 h-4 ml-2" />
              </Button>
            </motion.div>
            
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6, delay: 0.2 }}
              className="relative"
            >
              <ImageWithFallback
                src="https://images.unsplash.com/photo-1718220216044-006f43e3a9b1?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxtb2Rlcm4lMjBvZmZpY2UlMjB3b3Jrc3BhY2V8ZW58MXx8fHwxNzU3NzQ0NTUwfDA&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral"
                alt="Team collaboration"
                className="w-full h-96 object-cover rounded-2xl shadow-2xl"
              />
              <div className="absolute -bottom-4 -left-4 bg-white p-4 rounded-lg shadow-lg">
                <div className="flex items-center space-x-2">
                  <Lightbulb className="w-6 h-6 text-yellow-500" />
                  <div>
                    <div className="text-sm font-semibold">Innovation First</div>
                    <div className="text-xs text-gray-600">Constantly improving</div>
                  </div>
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Values */}
      <section className="py-20 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">Our Values</h2>
            <p className="text-xl text-gray-600 max-w-2xl mx-auto">
              The principles that guide everything we do at CareerPilot
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {values.map((value, index) => {
              const Icon = value.icon;
              return (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.6, delay: index * 0.1 }}
                >
                  <Card className="h-full hover:shadow-lg transition-shadow">
                    <CardContent className="p-6">
                      <div className="flex items-start space-x-4">
                        <div className="p-3 bg-blue-100 rounded-lg">
                          <Icon className="w-6 h-6 text-blue-600" />
                        </div>
                        <div>
                          <h3 className="text-lg font-semibold mb-2">{value.title}</h3>
                          <p className="text-gray-600">{value.description}</p>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>
              );
            })}
          </div>
        </div>
      </section>

      {/* Team */}
      <section className="py-20 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">Meet Our Team</h2>
            <p className="text-xl text-gray-600 max-w-2xl mx-auto">
              Passionate experts dedicated to transforming career development
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {team.map((member, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
              >
                <Card className="text-center hover:shadow-lg transition-shadow">
                  <CardContent className="p-6">
                    <ImageWithFallback
                      src={member.image}
                      alt={member.name}
                      className="w-24 h-24 rounded-full mx-auto mb-4 object-cover"
                    />
                    <h3 className="text-lg font-semibold mb-1">{member.name}</h3>
                    <p className="text-primary font-medium mb-2">{member.role}</p>
                    <p className="text-sm text-gray-600">{member.bio}</p>
                  </CardContent>
                </Card>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Timeline */}
      <section className="py-20 bg-white">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">Our Journey</h2>
            <p className="text-xl text-gray-600">Key milestones in our mission to transform careers</p>
          </motion.div>

          <div className="space-y-8">
            {milestones.map((milestone, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, x: index % 2 === 0 ? -20 : 20 }}
                whileInView={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                className="flex items-center space-x-6"
              >
                <div className="flex-shrink-0 w-20 text-right">
                  <div className="text-lg font-bold text-primary">{milestone.year}</div>
                </div>
                <div className="flex-shrink-0">
                  <div className="w-4 h-4 bg-primary rounded-full"></div>
                </div>
                <Card className="flex-1">
                  <CardContent className="p-4">
                    <h3 className="font-semibold text-gray-900 mb-1">{milestone.event}</h3>
                    <p className="text-gray-600">{milestone.description}</p>
                  </CardContent>
                </Card>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="py-20 bg-gradient-to-r from-blue-600 to-purple-600">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center text-white">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <h2 className="text-3xl md:text-4xl font-bold mb-4">
              Ready to Start Your Journey?
            </h2>
            <p className="text-xl mb-8 opacity-90">
              Join thousands of professionals who have transformed their careers with CareerPilot
            </p>
            <Button size="lg" className="bg-white text-blue-600 hover:bg-gray-100">
              Get Started Today
              <ArrowRight className="w-5 h-5 ml-2" />
            </Button>
          </motion.div>
        </div>
      </section>
    </div>
  );
}