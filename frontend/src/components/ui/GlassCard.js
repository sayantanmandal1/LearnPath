'use client';

import React from 'react';
import { motion } from 'framer-motion';

const GlassCard = ({ 
  children, 
  className = '', 
  hover = true, 
  glow = false,
  gradient = false,
  onClick,
  ...props 
}) => {
  const baseClasses = `
    backdrop-blur-xl bg-white/10 dark:bg-black/20 
    border border-white/20 dark:border-white/10
    rounded-2xl shadow-xl
    ${glow ? 'shadow-glow' : ''}
    ${gradient ? 'bg-gradient-to-br from-white/20 to-white/5 dark:from-black/20 dark:to-black/5' : ''}
    ${onClick ? 'cursor-pointer' : ''}
  `;

  const hoverAnimation = hover ? {
    scale: 1.02,
    y: -5,
    boxShadow: glow 
      ? '0 20px 40px rgba(14, 165, 233, 0.3), 0 0 60px rgba(14, 165, 233, 0.2)'
      : '0 20px 40px rgba(0, 0, 0, 0.1)',
    transition: {
      type: "spring",
      stiffness: 300,
      damping: 20
    }
  } : {};

  const tapAnimation = onClick ? {
    scale: 0.98,
    transition: {
      type: "spring",
      stiffness: 400,
      damping: 25
    }
  } : {};

  return (
    <motion.div
      className={`${baseClasses} ${className}`}
      whileHover={hoverAnimation}
      whileTap={tapAnimation}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{
        type: "spring",
        stiffness: 300,
        damping: 30,
        delay: 0.1
      }}
      onClick={onClick}
      {...props}
    >
      {children}
    </motion.div>
  );
};

export default GlassCard;