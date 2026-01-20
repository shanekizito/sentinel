# Contributing to Sentinel

Thank you for your interest in contributing to the **Sentinel Sovereign** project. We welcome contributions from the community to help make this the most robust security infrastructure in the world.

## Development Workflow

1.  **Fork the Repository**: Click the "Fork" button on the top right of the GitHub page.
2.  **Clone your Fork**:
    ```bash
    git clone https://github.com/YOUR_USERNAME/sentinel.git
    cd sentinel
    ```
3.  **Create a Branch**:
    ```bash
    git checkout -b feature/amazing-new-feature
    ```
4.  **Make Changes**: Implement your feature or fix.
5.  **Test**: Run the test suite to ensure no regressions.
    ```bash
    cargo test --all-features
    ```
6.  **Commit**: Use descriptive commit messages.
    ```bash
    git commit -m "feat: Add support for Ruby parsing"
    ```
7.  **Push**:
    ```bash
    git push origin feature/amazing-new-feature
    ```
8.  **Open a Pull Request**: Go to the original repository and open a PR from your fork.

## Code Style

- **Rust**: We follow standard `rustfmt` guidelines. Run `cargo fmt` before committing.
- **Python**: We use `black` for Python formatting.

## Reporting Issues

If you find a bug or have a feature request, please open an Issue on GitHub with:
- A clear title.
- Steps to reproduce (for bugs).
- Motivation and detailed description (for features).

Thank you for building the future of sovereign security!
