import React from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'motion/react';
import { Button } from './ui/button';
import { Home, ArrowLeft } from 'lucide-react';

export function NotFound() {
  return (
    <div className="min-h-screen bg-background flex items-center justify-center p-4">
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="max-w-md w-full text-center space-y-6"
      >
        <div className="space-y-4">
          <h1 className="text-6xl font-bold text-muted-foreground">404</h1>
          <h2 className="text-2xl font-medium text-foreground">Page Not Found</h2>
          <p className="text-muted-foreground">
            Sorry, we couldn't find the page you're looking for. 
            It might have been moved, deleted, or doesn't exist.
          </p>
        </div>
        
        <div className="flex flex-col sm:flex-row gap-3 justify-center">
          <Button asChild variant="default">
            <Link to="/" className="flex items-center space-x-2">
              <Home className="w-4 h-4" />
              <span>Go Home</span>
            </Link>
          </Button>
          
          <Button 
            variant="outline" 
            onClick={() => window.history.back()}
            className="flex items-center space-x-2"
          >
            <ArrowLeft className="w-4 h-4" />
            <span>Go Back</span>
          </Button>
        </div>
      </motion.div>
    </div>
  );
}