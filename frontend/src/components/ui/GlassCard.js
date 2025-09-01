'use client';

import React from 'react';
import { motion } from 'framer-motion';

const GlassCard = ({ 
  children, 
  className = '', 
  hover = false, 
  glow = false,
  onClick,
  ...props 
}) => {
  const baseClasses = `
    backdrop-blur-xl 
    bg-white/10 
    dark:bg-black/20 
    border 
    border-white/20 
    dark:border-white/10 
    rounded-2xl 
    shadow-xl
    transition-all 
    duration-300
  `;

  const hoverClasses = hover ? `
    hover:scale-105 
    hover:-translate-y-2 
    hover:shadow-2xl 
    hover:bg-white/20 
    dark:hover:bg-black/30
    cursor-pointer
  ` : '';

  const glowClasses = glow ? `
    shadow-glow 
    hover:shadow-glow-lg
  ` : '';

  const combinedClasses = `${baseClasses} ${hoverClasses} ${glowClasses} ${className}`.trim();

  if (hover) {
    return (
      <motion.div
        whileHover={{ 
          scale: 1.02,
          y: -4
        }}
        transition={{ 
          type: "spring", 
          stiffness: 300, 
          damping: 20 
        }}
        className={combinedClasses}
        onClick={onClick}
        {...props}
      >
        {children}
      </motion.div>
    );
  }

  return (
    <div 
      className={combinedClasses}
      onClick={onClick}
      {...props}
    >
      {children}
    </div>
  );
};

export default GlassCard;