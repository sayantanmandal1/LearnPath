import React from 'react';
import { motion } from 'motion/react';
import { 
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from './ui/accordion';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { 
  HelpCircle, 
  MessageSquare, 
  Mail, 
  Phone,
  ArrowRight,
  Clock,
  Shield,
  Users,
  Brain
} from 'lucide-react';

export function FAQ() {
  const faqCategories = [
    {
      title: 'Getting Started',
      icon: HelpCircle,
      faqs: [
        {
          question: 'How do I create an account?',
          answer: 'You can sign up for free using your email, GitHub, or LinkedIn account. Simply click the "Get Started" button and follow the quick registration process.'
        },
        {
          question: 'Is CareerPilot really free to start?',
          answer: 'Yes! Our Starter plan is completely free and includes basic career analysis, skill assessment, and community access. You can upgrade anytime for advanced features.'
        },
        {
          question: 'How accurate are the career recommendations?',
          answer: 'Our AI analyzes thousands of career paths and market data points to provide recommendations with 85%+ accuracy. The more information you provide, the more personalized and accurate your recommendations become.'
        },
        {
          question: 'How long does the career analysis take?',
          answer: 'The initial analysis takes about 10-15 minutes to complete. You\'ll get instant preliminary results, with more detailed insights generated within 24 hours.'
        }
      ]
    },
    {
      title: 'Features & Functionality',
      icon: Brain,
      faqs: [
        {
          question: 'What makes CareerPilot different from other career tools?',
          answer: 'CareerPilot combines advanced AI with real-time market data to provide personalized, actionable career guidance. Unlike generic advice, our recommendations are tailored to your specific skills, experience, and goals.'
        },
        {
          question: 'Can I track my progress over time?',
          answer: 'Absolutely! Your dashboard provides detailed analytics on your skill development, completed courses, and career progression. You can set goals and monitor your advancement toward them.'
        },
        {
          question: 'How often is the job market data updated?',
          answer: 'We update our job market data daily, pulling from hundreds of job boards, company websites, and salary databases to ensure you have the most current information.'
        },
        {
          question: 'Can I export my career analysis results?',
          answer: 'Yes, Professional and Enterprise users can export their analysis results, learning paths, and progress reports in PDF format for interviews or personal records.'
        }
      ]
    },
    {
      title: 'Privacy & Security',
      icon: Shield,
      faqs: [
        {
          question: 'How is my personal data protected?',
          answer: 'We use industry-standard encryption and security measures to protect your data. Your information is never shared with third parties without your explicit consent, and you have full control over your privacy settings.'
        },
        {
          question: 'Can I delete my account and data?',
          answer: 'Yes, you can delete your account and all associated data at any time from your account settings. This action is permanent and cannot be undone.'
        },
        {
          question: 'Do you share my information with employers?',
          answer: 'No, we never share your personal information with employers unless you explicitly opt-in to our job matching service and give permission for specific opportunities.'
        },
        {
          question: 'Where is my data stored?',
          answer: 'Your data is securely stored in SOC 2 compliant data centers with 99.9% uptime. We maintain backups and use enterprise-grade security protocols.'
        }
      ]
    },
    {
      title: 'Billing & Plans',
      icon: Users,
      faqs: [
        {
          question: 'Can I change my plan anytime?',
          answer: 'Yes, you can upgrade or downgrade your plan at any time. Changes take effect immediately, and billing is prorated accordingly.'
        },
        {
          question: 'What payment methods do you accept?',
          answer: 'We accept all major credit cards (Visa, MasterCard, American Express), PayPal, and bank transfers for annual plans.'
        },
        {
          question: 'Do you offer discounts for students?',
          answer: 'Yes! Students get 50% off our Professional plan with a valid student email address. Contact support for verification and discount code.'
        },
        {
          question: 'What happens if I cancel my subscription?',
          answer: 'You can use your paid features until the end of your billing period. After that, your account reverts to the free Starter plan, and your data remains accessible.'
        }
      ]
    }
  ];

  const supportOptions = [
    {
      icon: MessageSquare,
      title: 'Live Chat',
      description: 'Get instant help from our support team',
      availability: '24/7 for Pro users',
      action: 'Start Chat'
    },
    {
      icon: Mail,
      title: 'Email Support',
      description: 'Send us your questions and we\'ll respond quickly',
      availability: 'Response within 24 hours',
      action: 'Send Email'
    },
    {
      icon: Phone,
      title: 'Phone Support',
      description: 'Talk directly with our career experts',
      availability: 'Mon-Fri, 9AM-6PM EST',
      action: 'Schedule Call'
    }
  ];

  return (
    <div className="pt-16 min-h-screen bg-white">
      {/* Hero Section */}
      <section className="py-20 bg-gradient-to-br from-blue-50 via-white to-purple-50">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <Badge variant="secondary" className="mb-6 px-4 py-2">
              <HelpCircle className="w-4 h-4 mr-2" />
              Help Center
            </Badge>
            <h1 className="text-4xl md:text-5xl font-bold text-gray-900 mb-6">
              Frequently Asked Questions
            </h1>
            <p className="text-xl text-gray-600 mb-8">
              Find answers to common questions about CareerPilot, or reach out to our support team for personalized help.
            </p>
          </motion.div>
        </div>
      </section>

      {/* FAQ Categories */}
      <section className="py-20 bg-white">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="space-y-12">
            {faqCategories.map((category, categoryIndex) => {
              const Icon = category.icon;
              return (
                <motion.div
                  key={categoryIndex}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.6, delay: categoryIndex * 0.1 }}
                >
                  <div className="flex items-center space-x-3 mb-6">
                    <div className="p-2 bg-blue-100 rounded-lg">
                      <Icon className="w-5 h-5 text-blue-600" />
                    </div>
                    <h2 className="text-2xl font-bold text-gray-900">{category.title}</h2>
                  </div>
                  
                  <Accordion type="single" collapsible className="space-y-4">
                    {category.faqs.map((faq, faqIndex) => (
                      <AccordionItem 
                        key={faqIndex} 
                        value={`${categoryIndex}-${faqIndex}`}
                        className="border border-gray-200 rounded-lg px-4"
                      >
                        <AccordionTrigger className="hover:no-underline py-4">
                          <span className="text-left font-medium">{faq.question}</span>
                        </AccordionTrigger>
                        <AccordionContent className="pb-4 text-gray-600">
                          {faq.answer}
                        </AccordionContent>
                      </AccordionItem>
                    ))}
                  </Accordion>
                </motion.div>
              );
            })}
          </div>
        </div>
      </section>

      {/* Support Options */}
      <section className="py-20 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
              Still Need Help?
            </h2>
            <p className="text-xl text-gray-600 max-w-2xl mx-auto">
              Our support team is here to help you succeed. Choose the best way to reach us.
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {supportOptions.map((option, index) => {
              const Icon = option.icon;
              return (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.6, delay: index * 0.1 }}
                  whileHover={{ y: -5 }}
                >
                  <Card className="h-full text-center hover:shadow-lg transition-shadow">
                    <CardHeader>
                      <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center mx-auto mb-4">
                        <Icon className="w-6 h-6 text-blue-600" />
                      </div>
                      <CardTitle className="text-lg">{option.title}</CardTitle>
                    </CardHeader>
                    <CardContent className="pt-0">
                      <p className="text-gray-600 mb-4">{option.description}</p>
                      <div className="flex items-center justify-center space-x-1 text-sm text-gray-500 mb-6">
                        <Clock className="w-3 h-3" />
                        <span>{option.availability}</span>
                      </div>
                      <Button className="w-full">
                        {option.action}
                        <ArrowRight className="w-4 h-4 ml-2" />
                      </Button>
                    </CardContent>
                  </Card>
                </motion.div>
              );
            })}
          </div>
        </div>
      </section>

      {/* Additional Resources */}
      <section className="py-20 bg-white">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
              Additional Resources
            </h2>
            <p className="text-xl text-gray-600">
              Explore more ways to get the most out of CareerPilot
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6 }}
            >
              <Card className="h-full">
                <CardContent className="p-6">
                  <h3 className="text-lg font-semibold mb-3">Getting Started Guide</h3>
                  <p className="text-gray-600 mb-4">
                    Step-by-step tutorials to help you set up your profile and get the most accurate career recommendations.
                  </p>
                  <Button variant="outline">
                    View Guide
                    <ArrowRight className="w-4 h-4 ml-2" />
                  </Button>
                </CardContent>
              </Card>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: 20 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6, delay: 0.1 }}
            >
              <Card className="h-full">
                <CardContent className="p-6">
                  <h3 className="text-lg font-semibold mb-3">Video Tutorials</h3>
                  <p className="text-gray-600 mb-4">
                    Watch our comprehensive video series covering all CareerPilot features and best practices.
                  </p>
                  <Button variant="outline">
                    Watch Videos
                    <ArrowRight className="w-4 h-4 ml-2" />
                  </Button>
                </CardContent>
              </Card>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: -20 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6, delay: 0.2 }}
            >
              <Card className="h-full">
                <CardContent className="p-6">
                  <h3 className="text-lg font-semibold mb-3">Community Forum</h3>
                  <p className="text-gray-600 mb-4">
                    Connect with other professionals, share experiences, and get advice from the CareerPilot community.
                  </p>
                  <Button variant="outline">
                    Join Community
                    <ArrowRight className="w-4 h-4 ml-2" />
                  </Button>
                </CardContent>
              </Card>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: 20 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6, delay: 0.3 }}
            >
              <Card className="h-full">
                <CardContent className="p-6">
                  <h3 className="text-lg font-semibold mb-3">API Documentation</h3>
                  <p className="text-gray-600 mb-4">
                    For developers looking to integrate CareerPilot's career intelligence into their applications.
                  </p>
                  <Button variant="outline">
                    View API Docs
                    <ArrowRight className="w-4 h-4 ml-2" />
                  </Button>
                </CardContent>
              </Card>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Contact CTA */}
      <section className="py-20 bg-gradient-to-r from-blue-600 to-purple-600">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center text-white">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <h2 className="text-3xl md:text-4xl font-bold mb-4">
              Didn't Find What You're Looking For?
            </h2>
            <p className="text-xl mb-8 opacity-90">
              Our support team is always ready to help you with any questions or concerns you might have.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Button size="lg" className="bg-white text-blue-600 hover:bg-gray-100">
                Contact Support
                <MessageSquare className="w-5 h-5 ml-2" />
              </Button>
              <Button size="lg" variant="outline" className="border-white text-white hover:bg-white hover:text-blue-600">
                Schedule a Demo
                <ArrowRight className="w-5 h-5 ml-2" />
              </Button>
            </div>
          </motion.div>
        </div>
      </section>
    </div>
  );
}