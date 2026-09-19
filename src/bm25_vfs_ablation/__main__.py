"""Command-line entry point for the package."""


def main() -> None:
    from .cli import app

    app()


if __name__ == "__main__":
    main()
