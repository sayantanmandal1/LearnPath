'use client';

import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { useAuth } from '../../contexts/AuthContext';
import { useRouter } from 'next/navigation';
import AnimatedButton from '../ui/AnimatedButton';
import GlassCard from '../ui/GlassCard';
import AnimatedBackground from '../ui/AnimatedBackground';
import Link from 'next/link';
import { 
  EyeIcon, 
  EyeSlashIcon,
  EnvelopeIcon,
  LockClosedIcon,
  UserIcon,
  CheckCircleIcon,
  XCircleIcon
} from '@heroicons/react/24/outline';
import anime from 'animejs';

const SignUpForm = () => {
  const { signUp, signInWithGoogle, signInWithGitHub, user, loading, checkUsernameAvailability } = useAuth();
  const router = useRouter();
  const [formData, setFormData] = useState({
    firstName: '',
    lastName: '',
    email: '',
    password: '',
    confirmPassword: '',
    username: '',
    agreeToTerms: false
  });
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [errors, setErrors] = useState({});
  const [passwordStrength, setPasswordStrength] = useState(0);
  const [usernameAvailable, setUsernameAvailable] = useState(null);
  const [checkingUsername, setCheckingUsername] = useState(false);

  useEffect(() => {
    if (user) {
      router.push('/dashboard');
    }
  }, [user, router]);

  useEffect(() => {
    // Animate form elements on mount
    anime({
      targets: '.form-element',
      translateY: [30, 0],
      opacity: [0, 1],
      duration: 800,
      delay: anime.stagger(100),
      easing: 'easeOutExpo'
    });
  }, []);

  useEffect(() => {
    // Check password strength
    const strength = calculatePasswordStrength(formData.password);
    setPasswordStrength(strength);
  }, [formData.password]);

  useEffect(() => {
    // Check username availability with debounce
    const timeoutId = setTimeout(async () => {
      if (formData.username && formData.username.length >= 3) {
        setCheckingUsername(true);
        const available = await checkUsernameAvailability(formData.username);
        setUsernameAvailable(available);
        setCheckingUsername(false);
      } else {
        setUsernameAvailable(null);
      }
    }, 500);

    return () => clearTimeout(timeoutId);
  }, [formData.username, checkUsernameAvailability]);

  const calculatePasswordStrength = (password) => {
    let strength = 0;
    if (password.length >= 8) strength += 25;
    if (/[a-z]/.test(password)) strength += 25;
    if (/[A-Z]/.test(password)) strength += 25;
    if (/[0-9]/.test(password)) strength += 12.5;
    if (/[^A-Za-z0-9]/.test(password)) strength += 12.5;
    return Math.min(strength, 100);
  };

  const getPasswordStrengthColor = (strength) => {
    if (strength < 25) return 'bg-red-500';
    if (strength < 50) return 'bg-orange-500';
    if (strength < 75) return 'bg-yellow-500';
    return 'bg-green-500';
  };

  const getPasswordStrengthText = (strength) => {
    if (strength < 25) return 'Weak';
    if (strength < 50) return 'Fair';
    if (strength < 75) return 'Good';
    return 'Strong';
  };

  const handleInputChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value
    }));
    
    // Clear error when user starts typing
    if (errors[name]) {
      setErrors(prev => ({
        ...prev,
        [name]: ''
      }));
    }
  };

  const validateForm = () => {
    const newErrors = {};
    
    if (!formData.firstName.trim()) {
      newErrors.firstName = 'First name is required';
    }
    
    if (!formData.lastName.trim()) {
      newErrors.lastName = 'Last name is required';
    }
    
    if (!formData.username.trim()) {
      newErrors.username = 'Username is required';
    } else if (formData.username.length < 3) {
      newErrors.username = 'Username must be at least 3 characters';
    } else if (!/^[a-zA-Z0-9_]+$/.test(formData.username)) {
      newErrors.username = 'Username can only contain letters, numbers, and underscores';
    } else if (usernameAvailable === false) {
      newErrors.username = 'Username is already taken';
    }
    
    if (!formData.email) {
      newErrors.email = 'Email is required';
    } else if (!/\S+@\S+\.\S+/.test(formData.email)) {
      newErrors.email = 'Email is invalid';
    }
    
    if (!formData.password) {
      newErrors.password = 'Password is required';
    } else if (formData.password.length < 8) {
      newErrors.password = 'Password must be at least 8 characters';
    } else if (passwordStrength < 50) {
      newErrors.password = 'Password is too weak';
    }
    
    if (!formData.confirmPassword) {
      newErrors.confirmPassword = 'Please confirm your password';
    } else if (formData.password !== formData.confirmPassword) {
      newErrors.confirmPassword = 'Passwords do not match';
    }
    
    if (!formData.agreeToTerms) {
      newErrors.agreeToTerms = 'You must agree to the terms and conditions';
    }
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!validateForm()) return;
    
    setIsLoading(true);
    
    try {
      const additionalData = {
        firstName: formData.firstName,
        lastName: formData.lastName,
        displayName: `${formData.firstName} ${formData.lastName}`,
        username: formData.username
      };
      
      const result = await signUp(formData.email, formData.password, additionalData);
      if (result.success) {
        router.push('/auth/verify-email');
      }
    } catch (error) {
      console.error('Sign up error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleGoogleSignUp = async () => {
    setIsLoading(true);
    try {
      const result = await signInWithGoogle();
      if (result.success) {
        router.push('/dashboard');
      }
    } catch (error) {
      console.error('Google sign up error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleGitHubSignUp = async () => {
    setIsLoading(true);
    try {
      const result = await signInWithGitHub();
      if (result.success) {
        router.push('/dashboard');
      }
    } catch (error) {
      console.error('GitHub sign up error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const inputVariants = {
    focus: {
      scale: 1.02,
      transition: { type: "spring", stiffness: 300, damping: 25 }
    }
  };

  const InputField = ({ 
    name, 
    type, 
    placeholder, 
    icon: Icon, 
    value, 
    onChange, 
    error,
    rightIcon,
    onRightIconClick,
    showValidation = false
  }) => (
    <motion.div 
      className="relative form-element"
      variants={inputVariants}
      whileFocus="focus"
    >
      <div className="relative">
        <Icon className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
        <input
          type={type}
          name={name}
          value={value}
          onChange={onChange}
          placeholder={placeholder}
          className={`
            w-full pl-10 pr-12 py-4 
            bg-white/5 backdrop-blur-sm
            border-2 rounded-xl
            text-white placeholder-gray-400
            focus:outline-none focus:ring-2 focus:ring-primary-500
            transition-all duration-300
            ${error 
              ? 'border-red-500 focus:border-red-500' 
              : showValidation && value
                ? usernameAvailable === true 
                  ? 'border-green-500 focus:border-green-500'
                  : usernameAvailable === false
                    ? 'border-red-500 focus:border-red-500'
                    : 'border-white/20 focus:border-primary-500'
                : 'border-white/20 focus:border-primary-500'
            }
          `}
        />
        {rightIcon && (
          <button
            type="button"
            onClick={onRightIconClick}
            className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-white transition-colors"
          >
            {rightIcon}
          </button>
        )}
        {showValidation && value && (
          <div className="absolute right-3 top-1/2 transform -translate-y-1/2">
            {checkingUsername ? (
              <motion.div
                className="w-5 h-5 border-2 border-primary-500 border-t-transparent rounded-full"
                animate={{ rotate: 360 }}
                transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
              />
            ) : usernameAvailable === true ? (
              <CheckCircleIcon className="w-5 h-5 text-green-500" />
            ) : usernameAvailable === false ? (
              <XCircleIcon className="w-5 h-5 text-red-500" />
            ) : null}
          </div>
        )}
      </div>
      {error && (
        <motion.p
          className="mt-2 text-sm text-red-400"
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ type: "spring", stiffness: 300, damping: 25 }}
        >
          {error}
        </motion.p>
      )}
    </motion.div>
  );

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <motion.div
          className="w-16 h-16 border-4 border-primary-500 border-t-transparent rounded-full"
          animate={{ rotate: 360 }}
          transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
        />
      </div>
    );
  }

  return (
    <div className="min-h-screen relative overflow-hidden bg-gradient-to-br from-dark-900 via-dark-800 to-dark-900">
      <AnimatedBackground variant="geometric" />
      
      <div className="relative z-10 min-h-screen flex items-center justify-center px-4 py-12">
        <motion.div
          className="w-full max-w-md"
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ type: "spring", stiffness: 300, damping: 30 }}
        >
          <GlassCard className="p-8" glow>
            {/* Header */}
            <motion.div 
              className="text-center mb-8 form-element"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
            >
              <motion.div
                className="w-16 h-16 bg-gradient-to-r from-primary-500 to-secondary-500 rounded-2xl flex items-center justify-center mx-auto mb-4"
                whileHover={{ rotate: 360, scale: 1.1 }}
                transition={{ duration: 0.5 }}
              >
                <span className="text-white font-bold text-2xl">AI</span>
              </motion.div>
              <h1 className="text-3xl font-bold text-white mb-2">Join CareerAI</h1>
              <p className="text-gray-400">Create your account and start your journey</p>
            </motion.div>

            {/* Social Sign Up */}
            <div className="space-y-3 mb-6">
              <AnimatedButton
                variant="glass"
                className="w-full form-element"
                onClick={handleGoogleSignUp}
                loading={isLoading}
                icon={
                  <svg className="w-5 h-5" viewBox="0 0 24 24">
                    <path fill="currentColor" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                    <path fill="currentColor" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                    <path fill="currentColor" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
                    <path fill="currentColor" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
                  </svg>
                }
              >
                Continue with Google
              </AnimatedButton>

              <AnimatedButton
                variant="glass"
                className="w-full form-element"
                onClick={handleGitHubSignUp}
                loading={isLoading}
                icon={
                  <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
                  </svg>
                }
              >
                Continue with GitHub
              </AnimatedButton>
            </div>

            {/* Divider */}
            <div className="relative mb-6 form-element">
              <div className="absolute inset-0 flex items-center">
                <div className="w-full border-t border-white/20"></div>
              </div>
              <div className="relative flex justify-center text-sm">
                <span className="px-2 bg-transparent text-gray-400">Or create with email</span>
              </div>
            </div>

            {/* Form */}
            <form onSubmit={handleSubmit} className="space-y-6">
              <div className="grid grid-cols-2 gap-4">
                <InputField
                  name="firstName"
                  type="text"
                  placeholder="First name"
                  icon={UserIcon}
                  value={formData.firstName}
                  onChange={handleInputChange}
                  error={errors.firstName}
                />

                <InputField
                  name="lastName"
                  type="text"
                  placeholder="Last name"
                  icon={UserIcon}
                  value={formData.lastName}
                  onChange={handleInputChange}
                  error={errors.lastName}
                />
              </div>

              <InputField
                name="username"
                type="text"
                placeholder="Choose a username"
                icon={UserIcon}
                value={formData.username}
                onChange={handleInputChange}
                error={errors.username}
                showValidation={true}
              />

              <InputField
                name="email"
                type="email"
                placeholder="Enter your email"
                icon={EnvelopeIcon}
                value={formData.email}
                onChange={handleInputChange}
                error={errors.email}
              />

              <div className="space-y-2">
                <InputField
                  name="password"
                  type={showPassword ? 'text' : 'password'}
                  placeholder="Create a password"
                  icon={LockClosedIcon}
                  value={formData.password}
                  onChange={handleInputChange}
                  error={errors.password}
                  rightIcon={showPassword ? <EyeSlashIcon className="w-5 h-5" /> : <EyeIcon className="w-5 h-5" />}
                  onRightIconClick={() => setShowPassword(!showPassword)}
                />
                
                {formData.password && (
                  <motion.div
                    className="space-y-2"
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    transition={{ duration: 0.3 }}
                  >
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-gray-400">Password strength:</span>
                      <span className={`font-medium ${
                        passwordStrength < 25 ? 'text-red-400' :
                        passwordStrength < 50 ? 'text-orange-400' :
                        passwordStrength < 75 ? 'text-yellow-400' :
                        'text-green-400'
                      }`}>
                        {getPasswordStrengthText(passwordStrength)}
                      </span>
                    </div>
                    <div className="w-full bg-gray-700 rounded-full h-2">
                      <motion.div
                        className={`h-2 rounded-full ${getPasswordStrengthColor(passwordStrength)}`}
                        initial={{ width: 0 }}
                        animate={{ width: `${passwordStrength}%` }}
                        transition={{ duration: 0.3 }}
                      />
                    </div>
                  </motion.div>
                )}
              </div>

              <InputField
                name="confirmPassword"
                type={showConfirmPassword ? 'text' : 'password'}
                placeholder="Confirm your password"
                icon={LockClosedIcon}
                value={formData.confirmPassword}
                onChange={handleInputChange}
                error={errors.confirmPassword}
                rightIcon={showConfirmPassword ? <EyeSlashIcon className="w-5 h-5" /> : <EyeIcon className="w-5 h-5" />}
                onRightIconClick={() => setShowConfirmPassword(!showConfirmPassword)}
              />

              <div className="form-element">
                <label className="flex items-start space-x-3">
                  <input
                    type="checkbox"
                    name="agreeToTerms"
                    checked={formData.agreeToTerms}
                    onChange={handleInputChange}
                    className="mt-1 rounded border-white/20 bg-white/5 text-primary-500 focus:ring-primary-500 focus:ring-offset-0"
                  />
                  <span className="text-sm text-gray-400 leading-relaxed">
                    I agree to the{' '}
                    <Link href="/terms" className="text-primary-400 hover:text-primary-300">
                      Terms of Service
                    </Link>
                    {' '}and{' '}
                    <Link href="/privacy" className="text-primary-400 hover:text-primary-300">
                      Privacy Policy
                    </Link>
                  </span>
                </label>
                {errors.agreeToTerms && (
                  <p className="mt-2 text-sm text-red-400">{errors.agreeToTerms}</p>
                )}
              </div>

              <AnimatedButton
                type="submit"
                variant="primary"
                className="w-full form-element"
                loading={isLoading}
                glow
                gradient
              >
                Create Account
              </AnimatedButton>
            </form>

            {/* Footer */}
            <motion.div 
              className="mt-6 text-center form-element"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.8 }}
            >
              <p className="text-gray-400">
                Already have an account?{' '}
                <Link href="/auth/signin">
                  <motion.span
                    className="text-primary-400 hover:text-primary-300 cursor-pointer font-medium"
                    whileHover={{ scale: 1.05 }}
                  >
                    Sign in
                  </motion.span>
                </Link>
              </p>
            </motion.div>
          </GlassCard>
        </motion.div>
      </div>
    </div>
  );
};

export default SignUpForm;