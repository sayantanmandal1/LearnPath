import { render, screen } from '@testing-library/react';
import SkillRadarChart from '../SkillRadarChart';

// Mock framer-motion
jest.mock('framer-motion', () => ({
  motion: {
    div: ({ children, ...props }) => <div {...props}>{children}</div>,
    h3: ({ children, ...props }) => <h3 {...props}>{children}</h3>,
  },
}));

// Mock recharts
jest.mock('recharts', () => ({
  ResponsiveContainer: ({ children }) => <div data-testid="responsive-container">{children}</div>,
  RadarChart: ({ children }) => <div data-testid="radar-chart">{children}</div>,
  Radar: () => <div data-testid="radar" />,
  PolarGrid: () => <div data-testid="polar-grid" />,
  PolarAngleAxis: () => <div data-testid="polar-angle-axis" />,
  PolarRadiusAxis: () => <div data-testid="polar-radius-axis" />,
  Tooltip: () => <div data-testid="tooltip" />,
  Legend: () => <div data-testid="legend" />,
}));

// Mock d3
jest.mock('d3', () => ({}));

const mockSkills = [
  { name: 'JavaScript', level: 0.9 },
  { name: 'React', level: 0.8 },
  { name: 'Node.js', level: 0.7 },
];

describe('SkillRadarChart', () => {
  it('renders with default props', () => {
    render(<SkillRadarChart skills={mockSkills} />);
    
    expect(screen.getByText('Skill Profile')).toBeInTheDocument();
    expect(screen.getByTestId('responsive-container')).toBeInTheDocument();
    expect(screen.getByTestId('radar-chart')).toBeInTheDocument();
  });

  it('renders with custom title', () => {
    render(<SkillRadarChart skills={mockSkills} title="Custom Skills" />);
    
    expect(screen.getByText('Custom Skills')).toBeInTheDocument();
  });

  it('displays skill indicators', () => {
    render(<SkillRadarChart skills={mockSkills} />);
    
    expect(screen.getByText('JavaScript')).toBeInTheDocument();
    expect(screen.getByText('React')).toBeInTheDocument();
    expect(screen.getByText('Node.js')).toBeInTheDocument();
  });

  it('shows comparison radar when enabled', () => {
    const skillsWithRequired = mockSkills.map(skill => ({
      ...skill,
      required: 0.9
    }));
    
    render(<SkillRadarChart skills={skillsWithRequired} showComparison={true} />);
    
    const radars = screen.getAllByTestId('radar');
    expect(radars).toHaveLength(2); // Current and required
  });
});