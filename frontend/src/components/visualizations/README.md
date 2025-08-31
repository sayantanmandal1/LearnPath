# Visualization Components

This directory contains professional visualization components for the AI Career Recommender dashboard, implementing interactive charts, animations, and responsive design.

## Components Overview

### 1. SkillRadarChart
Professional skill radar chart with dual rendering options (Recharts and D3.js).

**Features:**
- Interactive radar chart visualization
- Comparison mode for current vs required skills
- Smooth animations with Framer Motion
- D3.js integration for advanced customization
- Responsive design for all screen sizes
- Professional styling with dark mode support

**Props:**
- `skills`: Array of skill objects with name and level
- `title`: Chart title (default: "Skill Profile")
- `showComparison`: Boolean to show comparison radar
- `useD3`: Boolean to use D3.js rendering instead of Recharts
- `className`: Additional CSS classes

**Usage:**
```jsx
<SkillRadarChart
  skills={[
    { name: 'JavaScript', level: 0.9, required: 0.95 },
    { name: 'React', level: 0.8, required: 0.9 }
  ]}
  title="My Skills"
  showComparison={true}
  useD3={false}
/>
```

### 2. CareerRoadmapVisualization
Interactive timeline visualization for career progression.

**Features:**
- Horizontal timeline for desktop/tablet
- Vertical timeline for mobile
- Interactive step details on hover/click
- Progress tracking with animations
- Responsive design with different layouts
- Step status indicators (completed, current, upcoming)

**Props:**
- `roadmapData`: Array of career step objects
- `currentPosition`: Current position in the roadmap
- `className`: Additional CSS classes

**Usage:**
```jsx
<CareerRoadmapVisualization
  roadmapData={[
    {
      id: 1,
      title: "Junior Developer",
      description: "Entry-level position",
      duration: "6-12 months",
      skills: ["HTML", "CSS", "JavaScript"]
    }
  ]}
  currentPosition={1}
/>
```

### 3. SkillGapAnalysis
Comprehensive skill gap analysis with progress indicators and learning recommendations.

**Features:**
- Animated progress bars
- Priority-based color coding
- Expandable skill details
- Learning resource recommendations
- Sub-skill breakdown
- Overall readiness score

**Props:**
- `skillGaps`: Array of skill gap objects
- `targetRole`: Target job role name
- `className`: Additional CSS classes

**Usage:**
```jsx
<SkillGapAnalysis
  skillGaps={[
    {
      name: "AWS",
      currentLevel: 0.4,
      requiredLevel: 0.8,
      priority: "high",
      estimatedLearningTime: "3-4 months"
    }
  ]}
  targetRole="Senior Developer"
/>
```

### 4. JobCompatibilityDashboard
Advanced job matching dashboard with filtering and compatibility analysis.

**Features:**
- Real-time search and filtering
- Compatibility scoring with visual indicators
- Skill matching analysis
- Expandable job details
- Responsive grid layout
- Advanced filtering options

**Props:**
- `jobs`: Array of job objects
- `userSkills`: Array of user skill objects
- `className`: Additional CSS classes

**Usage:**
```jsx
<JobCompatibilityDashboard
  jobs={[
    {
      id: 1,
      title: "Senior Developer",
      company: "TechCorp",
      compatibilityScore: 0.85,
      requiredSkills: ["JavaScript", "React"]
    }
  ]}
  userSkills={[
    { name: "JavaScript", level: 0.9 }
  ]}
/>
```

### 5. VisualizationDashboard
Main dashboard component that orchestrates all visualizations.

**Features:**
- Tabbed interface for different views
- Fullscreen mode
- Export and share functionality
- Responsive layout management
- Smooth tab transitions
- Integrated data refresh

**Props:**
- `userData`: Complete user data object
- `className`: Additional CSS classes

**Usage:**
```jsx
<VisualizationDashboard
  userData={{
    skills: [...],
    skillGaps: [...],
    jobs: [...],
    careerRoadmap: [...]
  }}
/>
```

## Technical Implementation

### Dependencies
- **React 19.1.0**: Core framework
- **Framer Motion 11.0.3**: Animations and transitions
- **D3.js 7.8.5**: Advanced data visualization
- **Recharts 2.12.0**: Chart components
- **Chart.js 4.4.1**: Alternative charting library
- **Lucide React 0.344.0**: Icon library
- **Tailwind CSS 4**: Styling framework

### Animation Strategy
All components use Framer Motion for:
- Entrance animations with staggered delays
- Smooth transitions between states
- Interactive hover and click animations
- Progress bar animations
- Tab switching transitions

### Responsive Design
Components implement responsive design through:
- Tailwind CSS responsive classes
- Dynamic layout switching (horizontal/vertical)
- Adaptive chart sizing
- Mobile-first approach
- Touch-friendly interactions

### Performance Optimizations
- Lazy loading of heavy components
- Memoization of expensive calculations
- Efficient re-rendering strategies
- Optimized animation performance
- Responsive image loading

## File Structure
```
visualizations/
├── SkillRadarChart.js
├── CareerRoadmapVisualization.js
├── SkillGapAnalysis.js
├── JobCompatibilityDashboard.js
├── VisualizationDashboard.js
├── index.js
├── README.md
└── __tests__/
    └── SkillRadarChart.test.js
```

## Testing
Components include comprehensive tests covering:
- Rendering with various props
- User interactions
- Responsive behavior
- Animation completion
- Data transformation

Run tests with:
```bash
npm test
```

## Browser Support
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Accessibility
All components implement:
- ARIA labels and roles
- Keyboard navigation
- Screen reader support
- High contrast mode compatibility
- Focus management

## Contributing
When adding new visualization components:
1. Follow the established naming convention
2. Include comprehensive PropTypes or TypeScript definitions
3. Implement responsive design
4. Add smooth animations with Framer Motion
5. Include comprehensive tests
6. Update this README with component documentation