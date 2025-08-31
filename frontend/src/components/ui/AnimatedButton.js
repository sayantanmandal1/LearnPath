'use client';

import React from 'react';
import { motion } from 'framer-motion';

const AnimatedButton = ({
  children,
  variant = 'primary',
  size = 'md',
  loading = false,
  disabled = false,
  glow = false,
  gradient = false,
  icon,
  iconPosition = 'left',
  onClick,
  className = '',
  ...props
}) => {
  const baseClasses = 'relative inline-flex items-center justify-center font-medium rounded-xl transition-all duration-300 focus:outline-none focus:ring-4 focus:ring-opacity-50 overflow-hidden';
  
  const variants = {
    primary: `
      bg-gradient-to-r from-primary-500 to-primary-600 
      hover:from-primary-600 hover:to-primary-700
      text-white shadow-lg hover:shadow-xl
      focus:ring-primary-500
      ${glow ? 'shadow-glow hover:shadow-glow-lg' : ''}
    `,
    secondary: `
      bg-gradient-to-r from-secondary-500 to-secondary-600 
      hover:from-secondary-600 hover:to-secondary-700
      text-white shadow-lg hover:shadow-xl
      focus:ring-secondary-500
      ${glow ? 'shadow-neon-pink hover:shadow-neon-pink' : ''}
    `,
    accent: `
      bg-gradient-to-r from-accent-500 to-accent-600 
      hover:from-accent-600 hover:to-accent-700
      text-white shadow-lg hover:shadow-xl
      focus:ring-accent-500
    `,
    outline: `
      border-2 border-primary-500 text-primary-500 
      hover:bg-primary-500 hover:text-white
      focus:ring-primary-500
    `,
    ghost: `
      text-primary-500 hover:bg-primary-50 dark:hover:bg-primary-900/20
      focus:ring-primary-500
    `,
    glass: `
      backdrop-blur-xl bg-white/10 dark:bg-black/20 
      border border-white/20 dark:border-white/10
      text-white hover:bg-white/20 dark:hover:bg-black/30
      focus:ring-white/50
    `
  };

  const sizes = {
    sm: 'px-4 py-2 text-sm',
    md: 'px-6 py-3 text-base',
    lg: 'px-8 py-4 text-lg',
    xl: 'px-10 py-5 text-xl'
  };

  const gradientOverlay = gradient && (
    <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent transform -skew-x-12 -translate-x-full group-hover:translate-x-full transition-transform duration-1000" />
  );

  const loadingSpinner = loading && (
    <motion.div
      className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full mr-2"
      animate={{ rotate: 360 }}
      transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
    />
  );

  const buttonContent = (
    <>
      {loadingSpinner}
      {icon && iconPosition === 'left' && !loading && (
        <span className="mr-2">{icon}</span>
      )}
      <span className={loading ? 'opacity-70' : ''}>{children}</span>
      {icon && iconPosition === 'right' && !loading && (
        <span className="ml-2">{icon}</span>
      )}
    </>
  );

  return (
    <motion.button
      className={`
        ${baseClasses} 
        ${variants[variant]} 
        ${sizes[size]} 
        ${disabled || loading ? 'opacity-50 cursor-not-allowed' : 'group'}
        ${className}
      `}
      whileHover={!disabled && !loading ? { 
        scale: 1.05,
        transition: { type: "spring", stiffness: 400, damping: 25 }
      } : {}}
      whileTap={!disabled && !loading ? { 
        scale: 0.95,
        transition: { type: "spring", stiffness: 400, damping: 25 }
      } : {}}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{
        type: "spring",
        stiffness: 300,
        damping: 30
      }}
      disabled={disabled || loading}
      onClick={onClick}
      {...props}
    >
      {gradientOverlay}
      {buttonContent}
    </motion.button>
  );
};

export default AnimatedButton;