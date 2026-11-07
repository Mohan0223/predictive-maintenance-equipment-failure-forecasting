"""Initialize the local SQLite database schema."""

from predictive_maintenance.database.repository import (
    DEFAULT_DATABASE_PATH,
    MaintenanceRepository,
)


def main() -> None:
    repository = MaintenanceRepository(DEFAULT_DATABASE_PATH)
    repository.initialize()
    print(f"SQLite database ready: {DEFAULT_DATABASE_PATH}")


if __name__ == "__main__":
    main()
