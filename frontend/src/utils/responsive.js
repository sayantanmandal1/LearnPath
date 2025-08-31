// Responsive utility functions and breakpoints

export const breakpoints = {
  sm: '640px',
  md: '768px',
  lg: '1024px',
  xl: '1280px',
  '2xl': '1536px'
};

export const useResponsive = () => {
  if (typeof window === 'undefined') return { isMobile: false, isTablet: false, isDesktop: true };
  
  const width = window.innerWidth;
  
  return {
    isMobile: width < 768,
    isTablet: width >= 768 && width < 1024,
    isDesktop: width >= 1024,
    width
  };
};

export const getResponsiveGridCols = (itemCount, screenSize = 'desktop') => {
  if (screenSize === 'mobile') {
    return 1;
  } else if (screenSize === 'tablet') {
    return Math.min(itemCount, 2);
  } else {
    return Math.min(itemCount, 3);
  }
};

export const getResponsiveChartSize = (screenSize = 'desktop') => {
  switch (screenSize) {
    case 'mobile':
      return { width: 300, height: 300 };
    case 'tablet':
      return { width: 400, height: 400 };
    default:
      return { width: 500, height: 500 };
  }
};

export const responsiveClasses = {
  container: 'w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8',
  grid: {
    1: 'grid-cols-1',
    2: 'grid-cols-1 md:grid-cols-2',
    3: 'grid-cols-1 md:grid-cols-2 lg:grid-cols-3',
    4: 'grid-cols-1 md:grid-cols-2 lg:grid-cols-4'
  },
  text: {
    xs: 'text-xs sm:text-sm',
    sm: 'text-sm sm:text-base',
    base: 'text-base sm:text-lg',
    lg: 'text-lg sm:text-xl',
    xl: 'text-xl sm:text-2xl',
    '2xl': 'text-2xl sm:text-3xl',
    '3xl': 'text-3xl sm:text-4xl'
  },
  spacing: {
    xs: 'p-2 sm:p-3',
    sm: 'p-3 sm:p-4',
    base: 'p-4 sm:p-6',
    lg: 'p-6 sm:p-8',
    xl: 'p-8 sm:p-12'
  }
};