# Requirements

## Runtime Dependencies

These are the core dependencies required to run the `coreason-assay` library and service.

- **Python**: `>=3.12, <3.15`
- **Pydantic**: `^2.12.5` - Data validation and settings management.
- **JSONSchema**: `^4.26.0` - Validation of JSON structures against schemas.
- **Typer**: `^0.21.1` - CLI application framework.
- **Pydantic Settings**: `^2.12.0` - Configuration management using environment variables.
- **HTTPX**: (Latest Stable) - Async HTTP client (standardized async stack).

### Server Dependencies (Service C)
Required for running the `coreason-assay` as a microservice (Server Mode).

- **FastAPI**: (Latest Stable) - Asynchronous web framework for building APIs.
- **Uvicorn**: (Latest Stable) - ASGI web server implementation.
- **Python-Multipart**: (Latest Stable) - Support for multipart/form-data requests (file uploads).

## Development Dependencies

Required for testing, linting, and documentation generation.

- **Pytest**: `^8.2.2` - Testing framework.
- **Ruff**: `^0.4.8` - Fast Python linter and formatter.
- **Pre-commit**: `^3.7.1` - Git hook management.
- **Pytest-cov**: `^5.0.0` - Coverage reporting for Pytest.
- **MkDocs**: `^1.6.0` - Documentation generator.
- **MkDocs Material**: `^9.5.26` - Material theme for MkDocs.
- **Pytest-Mock**: `^3.15.1` - Mocking fixtures for Pytest.
- **Mypy**: `^1.19.1` - Static type checker.
