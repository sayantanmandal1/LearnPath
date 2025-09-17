import React from 'react';
import { motion } from 'motion/react';

export function LoadingSpinner() {
  return (
    <div className="min-h-screen bg-background flex items-center justify-center">
      <motion.div 
        className="flex flex-col items-center space-y-4"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.3 }}
      >
        <motion.div
          className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full"
          animate={{ rotate: 360 }}
          transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
        />
        <p className="text-muted-foreground text-sm">Loading CareerPilot...</p>
      </motion.div>
    </div>
  );
}