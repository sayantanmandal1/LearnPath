import React from 'react';
import { motion } from 'motion/react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { ImageWithFallback } from './figma/ImageWithFallback';
import { 
  Brain, 
  Target, 
  TrendingUp, 
  Users, 
  BookOpen, 
  Award,
  Sparkles,
  ChartBar,
  MessageSquare,
  Shield,
  Zap,
  Globe,
  Clock,
  ArrowRight,
  CheckCircle
} from 'lucide-react';

export function Features() {
  const mainFeatures = [
    {
      icon: Brain,
      title: 'AI-Powered Career Analysis',
      description: 'Advanced machine learning algorithms analyze your skills, experience, and market trends to provide personalized career recommendations.',
      image: 'https://images.unsplash.com/photo-1744782211816-c5224434614f?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxkYXRhJTIwdmlzdWFsaXphdGlvbiUyMGNoYXJ0c3xlbnwxfHx8fDE3NTc3NjM3MjZ8MA&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral',
      features: ['Skill gap analysis', 'Market demand insights', 'Personalized recommendations', 'Real-time updates']
    },
    {
      icon: Target,
      title: 'Smart Skill Development',
      description: 'Identify skill gaps and get curated learning paths tailored to your career goals and current market demands.',
      image: 'https://images.unsplash.com/photo-1718220216044-006f43e3a9b1?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxtb2Rlcm4lMjBvZmZpY2UlMjB3b3Jrc3BhY2V8ZW58MXx8fHwxNzU3NzQ0NTUwfDA&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral',
      features: ['Curated learning paths', 'Progress tracking', 'Certification guidance', 'Industry benchmarks']
    },
    {
      icon: TrendingUp,
      title: 'Market Intelligence',
      description: 'Stay ahead with real-time job market data, salary insights, and emerging technology trends in your field.',
      image: 'https://images.unsplash.com/photo-1738566061505-556830f8b8f5?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxjYXJlZXIlMjBwcm9mZXNzaW9uYWwlMjBzdWNjZXNzfGVufDF8fHx8MTc1Nzg0MTQyMHww&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral',
      features: ['Salary benchmarks', 'Job market trends', 'Industry forecasts', 'Location-based insights']
    }
  ];

  const additionalFeatures = [
    {
      icon: Users,
      title: 'Community & Networking',
      description: 'Connect with professionals in your field and learn from their experiences'
    },
    {
      icon: ChartBar,
      title: 'Progress Analytics',
      description: 'Track your career development with detailed analytics and insights'
    },
    {
      icon: MessageSquare,
      title: '24/7 AI Assistant',
      description: 'Get instant answers to your career questions anytime, anywhere'
    },
    {
      icon: Shield,
      title: 'Privacy Protected',
      description: 'Your data is encrypted and secure. We never share without permission'
    },
    {
      icon: Zap,
      title: 'Instant Insights',
      description: 'Get immediate feedback and recommendations as you input your information'
    },
    {
      icon: Globe,
      title: 'Global Opportunities',
      description: 'Explore career opportunities across different countries and markets'
    }
  ];

  const pricingTiers = [
    {
      name: 'Starter',
      price: 'Free',
      description: 'Perfect for exploring career options',
      features: [
        'Basic career analysis',
        'Skill assessment',
        'General recommendations',
        'Community access'
      ],
      cta: 'Get Started',
      popular: false
    },
    {
      name: 'Professional',
      price: '$19/month',
      description: 'For serious career development',
      features: [
        'Advanced AI analysis',
        'Personalized learning paths',
        'Market intelligence',
        'Priority support',
        'Resume optimization',
        'Interview preparation'
      ],
      cta: 'Start Free Trial',
      popular: true
    },
    {
      name: 'Enterprise',
      price: 'Custom',
      description: 'For teams and organizations',
      features: [
        'Everything in Professional',
        'Team analytics',
        'Custom integrations',
        'Dedicated support',
        'Bulk licensing',
        'Advanced reporting'
      ],
      cta: 'Contact Sales',
      popular: false
    }
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
              <Sparkles className="w-4 h-4 mr-2" />
              Powerful Features
            </Badge>
            <h1 className="text-4xl md:text-6xl font-bold text-gray-900 mb-6">
              Everything You Need for
              <span className="block text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-purple-600">
                Career Success
              </span>
            </h1>
            <p className="text-xl text-gray-600 mb-8 max-w-3xl mx-auto">
              Discover how CareerPilot's advanced AI and comprehensive tools can accelerate your professional growth
              and help you achieve your career aspirations.
            </p>
            <Button size="lg" className="bg-gradient-to-r from-blue-600 to-purple-600">
              Explore Features
              <ArrowRight className="w-5 h-5 ml-2" />
            </Button>
          </motion.div>
        </div>
      </section>

      {/* Main Features */}
      <section className="py-20 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="space-y-20">
            {mainFeatures.map((feature, index) => {
              const Icon = feature.icon;
              const isEven = index % 2 === 0;
              
              return (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.6, delay: index * 0.1 }}
                  className={`grid grid-cols-1 lg:grid-cols-2 gap-12 items-center ${
                    !isEven ? 'lg:grid-flow-col-dense' : ''
                  }`}
                >
                  <div className={isEven ? 'lg:order-1' : 'lg:order-2'}>
                    <div className="flex items-center space-x-3 mb-4">
                      <div className="p-3 bg-blue-100 rounded-lg">
                        <Icon className="w-6 h-6 text-blue-600" />
                      </div>
                      <h2 className="text-2xl md:text-3xl font-bold text-gray-900">{feature.title}</h2>
                    </div>
                    <p className="text-lg text-gray-600 mb-6">{feature.description}</p>
                    <div className="grid grid-cols-2 gap-3 mb-6">
                      {feature.features.map((item, itemIndex) => (
                        <div key={itemIndex} className="flex items-center space-x-2">
                          <CheckCircle className="w-4 h-4 text-green-600" />
                          <span className="text-sm text-gray-700">{item}</span>
                        </div>
                      ))}
                    </div>
                    <Button variant="outline">
                      Learn More
                      <ArrowRight className="w-4 h-4 ml-2" />
                    </Button>
                  </div>
                  
                  <div className={isEven ? 'lg:order-2' : 'lg:order-1'}>
                    <ImageWithFallback
                      src={feature.image}
                      alt={feature.title}
                      className="w-full h-96 object-cover rounded-2xl shadow-2xl"
                    />
                  </div>
                </motion.div>
              );
            })}
          </div>
        </div>
      </section>

      {/* Additional Features Grid */}
      <section className="py-20 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
              And Much More
            </h2>
            <p className="text-xl text-gray-600 max-w-2xl mx-auto">
              Comprehensive tools and features designed to support every aspect of your career journey
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {additionalFeatures.map((feature, index) => {
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

      {/* How It Works */}
      <section className="py-20 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
              How CareerPilot Works
            </h2>
            <p className="text-xl text-gray-600 max-w-2xl mx-auto">
              Simple steps to transform your career journey
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {[
              {
                step: '01',
                title: 'Input Your Profile',
                description: 'Share your skills, experience, goals, and preferences with our intelligent system'
              },
              {
                step: '02',
                title: 'AI Analysis',
                description: 'Our advanced AI analyzes your profile against market data and career patterns'
              },
              {
                step: '03',
                title: 'Get Recommendations',
                description: 'Receive personalized career paths, skill development plans, and actionable insights'
              }
            ].map((step, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.2 }}
                className="text-center"
              >
                <div className="relative mb-8">
                  <div className="w-16 h-16 bg-gradient-to-r from-blue-600 to-purple-600 rounded-full flex items-center justify-center mx-auto mb-4">
                    <span className="text-white font-bold text-lg">{step.step}</span>
                  </div>
                  {index < 2 && (
                    <div className="hidden md:block absolute top-8 left-full w-full h-0.5 bg-gray-300 transform -translate-y-0.5" />
                  )}
                </div>
                <h3 className="text-xl font-semibold mb-2">{step.title}</h3>
                <p className="text-gray-600">{step.description}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Pricing */}
      <section className="py-20 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
              Choose Your Plan
            </h2>
            <p className="text-xl text-gray-600 max-w-2xl mx-auto">
              Flexible pricing options to match your career development needs
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {pricingTiers.map((tier, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                className="relative"
              >
                {tier.popular && (
                  <Badge className="absolute -top-3 left-1/2 transform -translate-x-1/2 bg-gradient-to-r from-blue-600 to-purple-600">
                    Most Popular
                  </Badge>
                )}
                <Card className={`h-full ${tier.popular ? 'border-2 border-blue-500 shadow-lg' : ''}`}>
                  <CardHeader className="text-center pb-4">
                    <CardTitle className="text-xl mb-2">{tier.name}</CardTitle>
                    <div className="text-3xl font-bold text-gray-900 mb-2">{tier.price}</div>
                    <p className="text-gray-600">{tier.description}</p>
                  </CardHeader>
                  <CardContent className="pt-0">
                    <ul className="space-y-3 mb-6">
                      {tier.features.map((feature, featureIndex) => (
                        <li key={featureIndex} className="flex items-center space-x-2">
                          <CheckCircle className="w-4 h-4 text-green-600" />
                          <span className="text-sm text-gray-700">{feature}</span>
                        </li>
                      ))}
                    </ul>
                    <Button className={`w-full ${tier.popular ? 'bg-gradient-to-r from-blue-600 to-purple-600' : ''}`}>
                      {tier.cta}
                    </Button>
                  </CardContent>
                </Card>
              </motion.div>
            ))}
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
              Ready to Pilot Your Career?
            </h2>
            <p className="text-xl mb-8 opacity-90">
              Start your journey today with our free plan and discover what's possible for your career
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Button size="lg" className="bg-white text-blue-600 hover:bg-gray-100">
                Start Free Trial
                <ArrowRight className="w-5 h-5 ml-2" />
              </Button>
              <Button size="lg" variant="outline" className="border-white text-white hover:bg-white hover:text-blue-600">
                Schedule Demo
              </Button>
            </div>
          </motion.div>
        </div>
      </section>
    </div>
  );
}