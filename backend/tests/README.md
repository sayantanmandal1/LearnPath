# Comprehensive Testing Suite

This directory contains a comprehensive testing suite for the AI Career Recommender system, covering all aspects from unit tests to end-to-end workflows.

## Test Structure

```
tests/
├── conftest.py                           # Test configuration and fixtures
├── utils/
│   └── test_data_generator.py           # Test data generation utilities
├── test_service_methods_comprehensive.py # Unit tests for all service methods
├── test_integration_comprehensive.py    # Integration tests for APIs and DB
├── test_performance_comprehensive.py    # Performance and load tests
├── test_e2e_workflows.py               # End-to-end workflow tests
└── README.md                           # This file
```

## Test Categories

### 🧪 Unit Tests (`test_service_methods_comprehensive.py`)
- **Purpose**: Test individual service methods and business logic
- **Scope**: All service classes, authentication, data processing
- **Markers**: `@pytest.mark.unit`
- **Coverage**: Aims for 90%+ code coverage
- **Run Time**: < 2 minutes

**Key Test Classes:**
- `TestProfileService`: Profile creation, data extraction, skill merging
- `TestRecommendationService`: Career recommendations, skill gap analysis
- `TestCareerTrajectoryService`: Career path generation, feasibility calculation
- `TestLearningPathService`: Learning path optimization, resource recommendation
- `TestAnalyticsService`: Skill radar, compatibility scoring, progress tracking
- `TestAuthService`: User authentication, token management

### 🤖 ML Algorithm Tests (`machinelearningmodel/tests/test_ml_algorithms_comprehensive.py`)
- **Purpose**: Test ML model accuracy and performance
- **Scope**: NLP engine, recommendation algorithms, skill classification
- **Markers**: `@pytest.mark.ml`
- **Accuracy Targets**: 85%+ for recommendations, 90%+ for skill extraction
- **Run Time**: 5-15 minutes (depending on model loading)

**Key Test Classes:**
- `TestNLPEngineAccuracy`: Skill extraction, embedding generation, text preprocessing
- `TestSkillClassifierAccuracy`: Skill categorization, similarity scoring
- `TestRecommendationEngineAccuracy`: Career recommendations, collaborative filtering
- `TestLearningPathOptimizerAccuracy`: Skill gap prioritization, learning sequencing
- `TestMLModelPerformanceMetrics`: Inference speed, memory usage, consistency

### 🔗 Integration Tests (`test_integration_comprehensive.py`)
- **Purpose**: Test API endpoints and database operations
- **Scope**: Complete request/response cycles, data persistence
- **Markers**: `@pytest.mark.integration`
- **Dependencies**: PostgreSQL, Redis
- **Run Time**: 3-8 minutes

**Key Test Classes:**
- `TestUserAuthenticationIntegration`: Registration, login, token validation
- `TestProfileManagementIntegration`: Profile CRUD, external data integration
- `TestRecommendationSystemIntegration`: End-to-end recommendation pipeline
- `TestAnalyticsIntegration`: Report generation, visualization data
- `TestExternalAPIIntegration`: GitHub, LeetCode, LinkedIn integration
- `TestDatabaseOperationsIntegration`: Cascade operations, data consistency

### ⚡ Performance Tests (`test_performance_comprehensive.py`)
- **Purpose**: Validate system performance and scalability
- **Scope**: API response times, concurrent users, memory usage
- **Markers**: `@pytest.mark.performance`
- **Thresholds**: < 3s for recommendations, < 1s for auth
- **Run Time**: 10-20 minutes

**Key Test Classes:**
- `TestAPIPerformance`: Endpoint response times, concurrent request handling
- `TestLoadTesting`: Multi-user scenarios, database connection pooling
- `TestMLModelPerformance`: Model inference speed, batch processing
- `TestScalabilityMetrics`: Throughput measurement, error rates under load

### 🎯 End-to-End Tests (`test_e2e_workflows.py`)
- **Purpose**: Test complete user workflows from start to finish
- **Scope**: Full user journeys, multi-user scenarios, system integration
- **Markers**: `@pytest.mark.e2e`
- **Environment**: Full system stack (API + DB + External services)
- **Run Time**: 15-30 minutes

**Key Test Classes:**
- `TestCompleteUserJourney`: Registration → Profile → Recommendations → Reports
- `TestMultiUserScenarios`: Concurrent users, data isolation
- `TestSystemIntegrationWorkflows`: Data pipelines, ML model integration

## Running Tests

### Prerequisites

1. **Install Dependencies**:
   ```bash
   cd backend
   pip install -r requirements-dev.txt
   ```

2. **Set Up Test Database**:
   ```bash
   # Using Docker
   docker run -d --name test-postgres -e POSTGRES_PASSWORD=testpass -p 5433:5432 postgres:15
   
   # Or use SQLite for quick testing (default in conftest.py)
   ```

3. **Set Up Redis**:
   ```bash
   docker run -d --name test-redis -p 6380:6379 redis:7
   ```

### Quick Test Commands

```bash
# Run all tests
python scripts/run_tests.py --all

# Run specific test categories
python scripts/run_tests.py --unit          # Unit tests only
python scripts/run_tests.py --ml            # ML algorithm tests
python scripts/run_tests.py --integration   # Integration tests
python scripts/run_tests.py --performance   # Performance tests
python scripts/run_tests.py --e2e           # End-to-end tests

# Quick mode (skip slow tests)
python scripts/run_tests.py --all --quick

# With coverage reporting
pytest tests/ --cov=app --cov-report=html
```

### Using pytest directly

```bash
# Run unit tests with coverage
pytest tests/test_service_methods_comprehensive.py -v --cov=app -m unit

# Run ML tests (excluding slow ones)
cd ../machinelearningmodel
pytest tests/test_ml_algorithms_comprehensive.py -v -m "ml and not slow"

# Run integration tests
pytest tests/test_integration_comprehensive.py -v -m integration

# Run performance tests
pytest tests/test_performance_comprehensive.py -v -m performance

# Run E2E tests
pytest tests/test_e2e_workflows.py -v -m e2e

# Run all tests with parallel execution
pytest tests/ -n auto --dist worksteal
```

## Test Configuration

### Environment Variables

```bash
# Test database
export DATABASE_URL="postgresql://user:pass@localhost:5433/testdb"
export REDIS_URL="redis://localhost:6380"
export TESTING=true

# External API mocking
export MOCK_EXTERNAL_APIS=true
export GITHUB_API_TOKEN="test-token"
export OPENAI_API_KEY="test-key"
```

### Pytest Markers

- `unit`: Fast unit tests (< 1s each)
- `integration`: Integration tests requiring external services
- `performance`: Performance and load tests
- `e2e`: End-to-end workflow tests
- `ml`: Machine learning model tests
- `slow`: Tests that take > 10s to run

### Test Data

Tests use a combination of:
- **Fixtures**: Defined in `conftest.py` for common test objects
- **Factories**: Using `factory_boy` for complex object creation
- **Generators**: Custom test data generators in `utils/test_data_generator.py`
- **Mocks**: External API responses and ML model outputs

## Coverage Requirements

| Component | Minimum Coverage | Target Coverage |
|-----------|------------------|-----------------|
| Service Layer | 85% | 95% |
| API Endpoints | 80% | 90% |
| ML Models | 70% | 85% |
| Database Models | 90% | 95% |
| Utilities | 80% | 90% |

## Performance Benchmarks

| Operation | Target Time | Max Time |
|-----------|-------------|----------|
| User Authentication | < 200ms | 500ms |
| Profile Creation | < 1s | 2s |
| Career Recommendations | < 2s | 3s |
| Learning Path Generation | < 3s | 5s |
| Skill Extraction (Resume) | < 1s | 2s |
| Analytics Report | < 2s | 4s |

## Continuous Integration

Tests are automatically run on:
- **Pull Requests**: Unit, Integration, and ML tests
- **Main Branch**: Full test suite including E2E
- **Nightly**: Performance tests and comprehensive ML accuracy tests
- **Weekly**: Security scans and dependency updates

### GitHub Actions Workflow

The `.github/workflows/comprehensive-testing.yml` file defines:
- Parallel test execution across multiple runners
- Database and Redis service containers
- Test result reporting and coverage uploads
- Performance benchmarking
- Security scanning with Bandit and Safety

## Troubleshooting

### Common Issues

1. **Database Connection Errors**:
   ```bash
   # Check if test database is running
   docker ps | grep postgres
   
   # Reset test database
   docker restart test-postgres
   ```

2. **Redis Connection Errors**:
   ```bash
   # Check Redis status
   docker ps | grep redis
   
   # Reset Redis
   docker restart test-redis
   ```

3. **ML Model Loading Issues**:
   ```bash
   # Clear model cache
   rm -rf ~/.cache/huggingface/
   
   # Reinstall ML dependencies
   pip install -r machinelearningmodel/requirements-test.txt
   ```

4. **Memory Issues in Performance Tests**:
   ```bash
   # Increase Docker memory limit
   # Or run tests with smaller datasets
   pytest tests/test_performance_comprehensive.py -k "not memory"
   ```

### Debug Mode

```bash
# Run tests with detailed output
pytest tests/ -v -s --tb=long

# Run single test with debugging
pytest tests/test_service_methods_comprehensive.py::TestProfileService::test_create_profile_success -v -s

# Profile test performance
pytest tests/ --benchmark-only --benchmark-sort=mean
```

## Contributing

When adding new tests:

1. **Follow naming conventions**: `test_<functionality>_<scenario>`
2. **Use appropriate markers**: Add `@pytest.mark.<category>`
3. **Include docstrings**: Explain what the test validates
4. **Mock external dependencies**: Use fixtures and mocks appropriately
5. **Test edge cases**: Include error conditions and boundary cases
6. **Update documentation**: Add new test classes to this README

### Test Quality Checklist

- [ ] Test has clear, descriptive name
- [ ] Test includes docstring explaining purpose
- [ ] Test uses appropriate fixtures and mocks
- [ ] Test covers both success and failure scenarios
- [ ] Test assertions are specific and meaningful
- [ ] Test runs in reasonable time (< 10s for unit tests)
- [ ] Test is deterministic (no random failures)
- [ ] Test cleans up after itself

## Reporting Issues

If you encounter test failures:

1. **Check the test output** for specific error messages
2. **Verify environment setup** (database, Redis, dependencies)
3. **Run the failing test in isolation** to reproduce
4. **Check recent changes** that might have affected the test
5. **Create an issue** with full error output and environment details

## Test Metrics Dashboard

The test suite generates comprehensive metrics:
- **Coverage Reports**: `htmlcov/index.html`
- **Performance Benchmarks**: `benchmark-results.json`
- **Test Results**: `test-results.xml` (JUnit format)
- **Security Reports**: `bandit-report.json`, `safety-report.json`

These are automatically uploaded to the CI/CD pipeline and can be viewed in the GitHub Actions artifacts.