"""Seed skill taxonomy data

Revision ID: 003
Revises: 002
Create Date: 2024-01-03 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.sql import table, column
from uuid import uuid4

# revision identifiers, used by Alembic.
revision = '003'
down_revision = '002'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Define table structures for bulk insert
    skill_categories_table = table('skill_categories',
        column('id', sa.String),
        column('name', sa.String),
        column('parent_id', sa.String),
        column('description', sa.String),
        column('display_order', sa.Integer),
        column('is_active', sa.Boolean),
    )
    
    skills_table = table('skills',
        column('id', sa.String),
        column('name', sa.String),
        column('category', sa.String),
        column('subcategory', sa.String),
        column('description', sa.String),
        column('aliases', sa.String),
        column('market_demand', sa.Float),
        column('average_salary_impact', sa.Float),
        column('is_active', sa.Boolean),
    )
    
    # Insert skill categories
    programming_id = str(uuid4())
    frameworks_id = str(uuid4())
    databases_id = str(uuid4())
    cloud_id = str(uuid4())
    devops_id = str(uuid4())
    soft_skills_id = str(uuid4())
    tools_id = str(uuid4())
    design_id = str(uuid4())
    analytics_id = str(uuid4())
    project_mgmt_id = str(uuid4())
    
    categories_data = [
        {
            'id': programming_id,
            'name': 'Programming Languages',
            'parent_id': None,
            'description': 'Programming languages and scripting languages',
            'display_order': 1,
            'is_active': True
        },
        {
            'id': frameworks_id,
            'name': 'Frameworks & Libraries',
            'parent_id': None,
            'description': 'Software frameworks and libraries',
            'display_order': 2,
            'is_active': True
        },
        {
            'id': databases_id,
            'name': 'Databases',
            'parent_id': None,
            'description': 'Database systems and data storage technologies',
            'display_order': 3,
            'is_active': True
        },
        {
            'id': cloud_id,
            'name': 'Cloud Platforms',
            'parent_id': None,
            'description': 'Cloud computing platforms and services',
            'display_order': 4,
            'is_active': True
        },
        {
            'id': devops_id,
            'name': 'DevOps & Infrastructure',
            'parent_id': None,
            'description': 'DevOps tools and infrastructure management',
            'display_order': 5,
            'is_active': True
        },
        {
            'id': tools_id,
            'name': 'Development Tools',
            'parent_id': None,
            'description': 'Development and productivity tools',
            'display_order': 6,
            'is_active': True
        },
        {
            'id': analytics_id,
            'name': 'Data & Analytics',
            'parent_id': None,
            'description': 'Data analysis and machine learning tools',
            'display_order': 7,
            'is_active': True
        },
        {
            'id': design_id,
            'name': 'Design & UX',
            'parent_id': None,
            'description': 'Design and user experience tools',
            'display_order': 8,
            'is_active': True
        },
        {
            'id': soft_skills_id,
            'name': 'Soft Skills',
            'parent_id': None,
            'description': 'Communication and interpersonal skills',
            'display_order': 9,
            'is_active': True
        },
        {
            'id': project_mgmt_id,
            'name': 'Project Management',
            'parent_id': None,
            'description': 'Project management methodologies and tools',
            'display_order': 10,
            'is_active': True
        }
    ]
    
    op.bulk_insert(skill_categories_table, categories_data)
    
    # Insert core skills
    skills_data = [
        # Programming Languages
        {'id': str(uuid4()), 'name': 'Python', 'category': 'programming', 'subcategory': 'general_purpose', 'description': 'High-level programming language', 'aliases': 'python3,py', 'market_demand': 0.95, 'average_salary_impact': 15.0, 'is_active': True},
        {'id': str(uuid4()), 'name': 'JavaScript', 'category': 'programming', 'subcategory': 'web', 'description': 'Programming language for web development', 'aliases': 'js,javascript,ecmascript', 'market_demand': 0.98, 'average_salary_impact': 12.0, 'is_active': True},
        {'id': str(uuid4()), 'name': 'TypeScript', 'category': 'programming', 'subcategory': 'web', 'description': 'Typed superset of JavaScript', 'aliases': 'ts,typescript', 'market_demand': 0.85, 'average_salary_impact': 18.0, 'is_active': True},
        {'id': str(uuid4()), 'name': 'Java', 'category': 'programming', 'subcategory': 'enterprise', 'description': 'Object-oriented programming language', 'aliases': 'java,jvm', 'market_demand': 0.90, 'average_salary_impact': 14.0, 'is_active': True},
        {'id': str(uuid4()), 'name': 'C++', 'category': 'programming', 'subcategory': 'systems', 'description': 'Systems programming language', 'aliases': 'cpp,c++,cplusplus', 'market_demand': 0.70, 'average_salary_impact': 20.0, 'is_active': True},
        {'id': str(uuid4()), 'name': 'Go', 'category': 'programming', 'subcategory': 'systems', 'description': 'Systems programming language by Google', 'aliases': 'golang,go', 'market_demand': 0.75, 'average_salary_impact': 22.0, 'is_active': True},
        {'id': str(uuid4()), 'name': 'Rust', 'category': 'programming', 'subcategory': 'systems', 'description': 'Systems programming language focused on safety', 'aliases': 'rust,rustlang', 'market_demand': 0.65, 'average_salary_impact': 25.0, 'is_active': True},
        {'id': str(uuid4()), 'name': 'C#', 'category': 'programming', 'subcategory': 'enterprise', 'description': 'Microsoft programming language', 'aliases': 'csharp,c#,.net', 'market_demand': 0.80, 'average_salary_impact': 16.0, 'is_active': True},
        {'id': str(uuid4()), 'name': 'PHP', 'category': 'programming', 'subcategory': 'web', 'description': 'Server-side scripting language', 'aliases': 'php,php7,php8', 'market_demand': 0.75, 'average_salary_impact': 8.0, 'is_active': True},
        {'id': str(uuid4()), 'name': 'Ruby', 'category': 'programming', 'subcategory': 'web', 'description': 'Dynamic programming language', 'aliases': 'ruby,rb', 'market_demand': 0.60, 'average_salary_impact': 12.0, 'is_active': True},
        
        # Frameworks & Libraries
        {'id': str(uuid4()), 'name': 'React', 'category': 'frameworks', 'subcategory': 'frontend', 'description': 'JavaScript library for building user interfaces', 'aliases': 'reactjs,react.js', 'market_demand': 0.95, 'average_salary_impact': 15.0, 'is_active': True},
        {'id': str(uuid4()), 'name': 'Vue.js', 'category': 'frameworks', 'subcategory': 'frontend', 'description': 'Progressive JavaScript framework', 'aliases': 'vue,vuejs', 'market_demand': 0.75, 'average_salary_impact': 12.0, 'is_active': True},
        {'id': str(uuid4()), 'name': 'Angular', 'category': 'frameworks', 'subcategory': 'frontend', 'description': 'TypeScript-based web application framework', 'aliases': 'angular,angularjs', 'market_demand': 0.80, 'average_salary_impact': 14.0, 'is_active': True},
        {'id': str(uuid4()), 'name': 'Node.js', 'category': 'frameworks', 'subcategory': 'backend', 'description': 'JavaScript runtime for server-side development', 'aliases': 'nodejs,node', 'market_demand': 0.90, 'average_salary_impact': 16.0, 'is_active': True},
        {'id': str(uuid4()), 'name': 'Express.js', 'category': 'frameworks', 'subcategory': 'backend', 'description': 'Web framework for Node.js', 'aliases': 'express,expressjs', 'market_demand': 0.85, 'average_salary_impact': 12.0, 'is_active': True},
        {'id': str(uuid4()), 'name': 'Django', 'category': 'frameworks', 'subcategory': 'backend', 'description': 'Python web framework', 'aliases': 'django', 'market_demand': 0.80, 'average_salary_impact': 18.0, 'is_active': True},
        {'id': str(uuid4()), 'name': 'Flask', 'category': 'frameworks', 'subcategory': 'backend', 'description': 'Lightweight Python web framework', 'aliases': 'flask', 'market_demand': 0.70, 'average_salary_impact': 15.0, 'is_active': True},
        {'id': str(uuid4()), 'name': 'FastAPI', 'category': 'frameworks', 'subcategory': 'backend', 'description': 'Modern Python web framework for APIs', 'aliases': 'fastapi', 'market_demand': 0.75, 'average_salary_impact': 20.0, 'is_active': True},
        {'id': str(uuid4()), 'name': 'Spring Boot', 'category': 'frameworks', 'subcategory': 'backend', 'description': 'Java framework for microservices', 'aliases': 'spring,springboot', 'market_demand': 0.85, 'average_salary_impact': 18.0, 'is_active': True},
        {'id': str(uuid4()), 'name': 'Laravel', 'category': 'frameworks', 'subcategory': 'backend', 'description': 'PHP web application framework', 'aliases': 'laravel', 'market_demand': 0.70, 'average_salary_impact': 10.0, 'is_active': True},
        
        # Databases
        {'id': str(uuid4()), 'name': 'PostgreSQL', 'category': 'databases', 'subcategory': 'relational', 'description': 'Advanced open-source relational database', 'aliases': 'postgres,postgresql', 'market_demand': 0.85, 'average_salary_impact': 15.0, 'is_active': True},
        {'id': str(uuid4()), 'name': 'MySQL', 'category': 'databases', 'subcategory': 'relational', 'description': 'Popular open-source relational database', 'aliases': 'mysql', 'market_demand': 0.80, 'average_salary_impact': 10.0, 'is_active': True},
        {'id': str(uuid4()), 'name': 'MongoDB', 'category': 'databases', 'subcategory': 'nosql', 'description': 'Document-oriented NoSQL database', 'aliases': 'mongo,mongodb', 'market_demand': 0.75, 'average_salary_impact': 12.0, 'is_active': True},
        {'id': str(uuid4()), 'name': 'Redis', 'category': 'databases', 'subcategory': 'cache', 'description': 'In-memory data structure store', 'aliases': 'redis', 'market_demand': 0.70, 'average_salary_impact': 14.0, 'is_active': True},
        {'id': str(uuid4()), 'name': 'Elasticsearch', 'category': 'databases', 'subcategory': 'search', 'description': 'Distributed search and analytics engine', 'aliases': 'elasticsearch,elastic', 'market_demand': 0.65, 'average_salary_impact': 18.0, 'is_active': True},
        
        # Cloud Platforms
        {'id': str(uuid4()), 'name': 'AWS', 'category': 'cloud', 'subcategory': 'platform', 'description': 'Amazon Web Services cloud platform', 'aliases': 'aws,amazon web services', 'market_demand': 0.95, 'average_salary_impact': 20.0, 'is_active': True},
        {'id': str(uuid4()), 'name': 'Azure', 'category': 'cloud', 'subcategory': 'platform', 'description': 'Microsoft cloud platform', 'aliases': 'azure,microsoft azure', 'market_demand': 0.85, 'average_salary_impact': 18.0, 'is_active': True},
        {'id': str(uuid4()), 'name': 'Google Cloud', 'category': 'cloud', 'subcategory': 'platform', 'description': 'Google Cloud Platform', 'aliases': 'gcp,google cloud platform', 'market_demand': 0.75, 'average_salary_impact': 19.0, 'is_active': True},
        
        # DevOps & Infrastructure
        {'id': str(uuid4()), 'name': 'Docker', 'category': 'devops', 'subcategory': 'containerization', 'description': 'Containerization platform', 'aliases': 'docker', 'market_demand': 0.90, 'average_salary_impact': 16.0, 'is_active': True},
        {'id': str(uuid4()), 'name': 'Kubernetes', 'category': 'devops', 'subcategory': 'orchestration', 'description': 'Container orchestration platform', 'aliases': 'k8s,kubernetes', 'market_demand': 0.85, 'average_salary_impact': 22.0, 'is_active': True},
        {'id': str(uuid4()), 'name': 'Jenkins', 'category': 'devops', 'subcategory': 'ci_cd', 'description': 'Continuous integration and deployment tool', 'aliases': 'jenkins', 'market_demand': 0.75, 'average_salary_impact': 14.0, 'is_active': True},
        {'id': str(uuid4()), 'name': 'Terraform', 'category': 'devops', 'subcategory': 'infrastructure', 'description': 'Infrastructure as code tool', 'aliases': 'terraform', 'market_demand': 0.80, 'average_salary_impact': 20.0, 'is_active': True},
        
        # Development Tools
        {'id': str(uuid4()), 'name': 'Git', 'category': 'tools', 'subcategory': 'version_control', 'description': 'Distributed version control system', 'aliases': 'git,github,gitlab', 'market_demand': 0.98, 'average_salary_impact': 5.0, 'is_active': True},
        {'id': str(uuid4()), 'name': 'VS Code', 'category': 'tools', 'subcategory': 'editor', 'description': 'Code editor by Microsoft', 'aliases': 'vscode,visual studio code', 'market_demand': 0.90, 'average_salary_impact': 2.0, 'is_active': True},
        
        # Data & Analytics
        {'id': str(uuid4()), 'name': 'Pandas', 'category': 'analytics', 'subcategory': 'data_processing', 'description': 'Python data manipulation library', 'aliases': 'pandas', 'market_demand': 0.80, 'average_salary_impact': 15.0, 'is_active': True},
        {'id': str(uuid4()), 'name': 'NumPy', 'category': 'analytics', 'subcategory': 'data_processing', 'description': 'Python numerical computing library', 'aliases': 'numpy', 'market_demand': 0.75, 'average_salary_impact': 12.0, 'is_active': True},
        {'id': str(uuid4()), 'name': 'TensorFlow', 'category': 'analytics', 'subcategory': 'machine_learning', 'description': 'Machine learning framework', 'aliases': 'tensorflow,tf', 'market_demand': 0.70, 'average_salary_impact': 25.0, 'is_active': True},
        {'id': str(uuid4()), 'name': 'PyTorch', 'category': 'analytics', 'subcategory': 'machine_learning', 'description': 'Deep learning framework', 'aliases': 'pytorch,torch', 'market_demand': 0.75, 'average_salary_impact': 26.0, 'is_active': True},
        
        # Soft Skills
        {'id': str(uuid4()), 'name': 'Communication', 'category': 'soft_skills', 'subcategory': 'interpersonal', 'description': 'Effective communication skills', 'aliases': 'communication,verbal communication', 'market_demand': 0.95, 'average_salary_impact': 10.0, 'is_active': True},
        {'id': str(uuid4()), 'name': 'Leadership', 'category': 'soft_skills', 'subcategory': 'management', 'description': 'Leadership and team management skills', 'aliases': 'leadership,team leadership', 'market_demand': 0.85, 'average_salary_impact': 20.0, 'is_active': True},
        {'id': str(uuid4()), 'name': 'Problem Solving', 'category': 'soft_skills', 'subcategory': 'analytical', 'description': 'Analytical and problem-solving abilities', 'aliases': 'problem solving,analytical thinking', 'market_demand': 0.90, 'average_salary_impact': 15.0, 'is_active': True},
        
        # Project Management
        {'id': str(uuid4()), 'name': 'Agile', 'category': 'project_management', 'subcategory': 'methodology', 'description': 'Agile project management methodology', 'aliases': 'agile,scrum,kanban', 'market_demand': 0.85, 'average_salary_impact': 12.0, 'is_active': True},
        {'id': str(uuid4()), 'name': 'Jira', 'category': 'project_management', 'subcategory': 'tools', 'description': 'Project management and issue tracking tool', 'aliases': 'jira,atlassian jira', 'market_demand': 0.80, 'average_salary_impact': 8.0, 'is_active': True},
    ]
    
    op.bulk_insert(skills_table, skills_data)


def downgrade() -> None:
    # Delete all seeded data
    op.execute("DELETE FROM skills")
    op.execute("DELETE FROM skill_categories")