/**
 * Dashboard component - now uses the Enhanced Dashboard with real-time data integration
 * 
 * This component has been updated to implement task 13 requirements:
 * - Replace mock data with real-time analysis results from backend
 * - Implement live job recommendations with Indian market focus
 * - Create personalized learning path displays with progress tracking
 * - Add market insights integration with current industry trends
 */
import React from 'react';
import { EnhancedDashboard } from './EnhancedDashboard';

export function Dashboard() {
  return <EnhancedDashboard />;
}