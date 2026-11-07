"""SQLite repository for predictive-maintenance application data."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATABASE_PATH = PROJECT_ROOT / "data" / "predictive_maintenance.db"
SCHEMA_PATH = PROJECT_ROOT / "sql" / "schema.sql"


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _optional_float(value: Any, field_name: str) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric or empty.") from exc
    if number != number or number in (float("inf"), float("-inf")):
        raise ValueError(f"{field_name} must be a finite number.")
    return number


class MaintenanceRepository:
    """Small SQLite data-access layer used by scripts and the dashboard."""

    def __init__(self, database_path: str | Path = DEFAULT_DATABASE_PATH) -> None:
        self.database_path = Path(database_path)

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """Open a connection with foreign keys and transaction handling."""
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        """Create the database schema if the tables do not exist."""
        if not SCHEMA_PATH.is_file():
            raise FileNotFoundError(f"SQL schema not found: {SCHEMA_PATH}")
        schema = SCHEMA_PATH.read_text(encoding="utf-8")
        with self.connection() as connection:
            connection.executescript(schema)
            sensor_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(sensor_readings)")
            }
            if "product_type" not in sensor_columns:
                connection.execute(
                    "ALTER TABLE sensor_readings ADD COLUMN product_type TEXT"
                )

    @staticmethod
    def _ensure_equipment(
        connection: sqlite3.Connection,
        equipment_code: str,
        equipment_name: str | None = None,
        equipment_type: str | None = None,
        location: str | None = None,
    ) -> int:
        code = equipment_code.strip()
        if not code:
            raise ValueError("Equipment code cannot be empty.")

        row = connection.execute(
            "SELECT equipment_id FROM equipment WHERE equipment_code = ?",
            (code,),
        ).fetchone()
        if row:
            return int(row["equipment_id"])

        cursor = connection.execute(
            """
            INSERT INTO equipment (
                equipment_code, equipment_name, equipment_type, location, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                code,
                (equipment_name or code).strip(),
                equipment_type,
                location,
                utc_now(),
            ),
        )
        return int(cursor.lastrowid)

    def add_equipment(
        self,
        equipment_code: str,
        equipment_name: str,
        equipment_type: str = "",
        location: str = "",
    ) -> int:
        """Add a new asset and return its database identifier."""
        normalized_code = equipment_code.strip()
        normalized_name = equipment_name.strip()
        if not normalized_code or not normalized_name:
            raise ValueError("Equipment code and name are required.")

        with self.connection() as connection:
            try:
                if connection.execute(
                    "SELECT 1 FROM equipment WHERE equipment_code = ?",
                    (normalized_code,),
                ).fetchone():
                    raise ValueError(
                        f"Equipment code {equipment_code!r} already exists."
                    )
                cursor = connection.execute(
                    """
                    INSERT INTO equipment (
                        equipment_code, equipment_name, equipment_type, location,
                        created_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        normalized_code,
                        normalized_name,
                        equipment_type or None,
                        location or None,
                        utc_now(),
                    ),
                )
                return int(cursor.lastrowid)
            except sqlite3.IntegrityError as exc:
                raise ValueError(
                    f"Equipment code {equipment_code!r} already exists."
                ) from exc

    def list_equipment(self) -> list[dict[str, Any]]:
        """Return all equipment records ordered by code."""
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT equipment_id, equipment_code, equipment_name,
                       equipment_type, location, created_at
                FROM equipment
                ORDER BY equipment_code
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def record_prediction(
        self,
        equipment_code: str,
        sensor_reading: dict[str, Any],
        prediction: dict[str, Any],
        *,
        model_name: str,
        model_version: str,
        data_source: str = "operator_input",
        equipment_name: str | None = None,
        equipment_type: str | None = None,
        location: str | None = None,
        recorded_at: str | None = None,
    ) -> dict[str, int | None]:
        """Atomically save a sensor reading, prediction, and any alert."""
        probability = float(prediction["failure_probability"])
        predicted_failure = int(prediction["predicted_failure"])
        risk = str(prediction["risk_level"])
        recommendation = str(prediction["recommended_action"])
        threshold = float(prediction.get("decision_threshold", 0.5))
        if predicted_failure not in (0, 1):
            raise ValueError("predicted_failure must be either 0 or 1.")
        if not 0.0 <= probability <= 1.0:
            raise ValueError("failure_probability must be between 0 and 1.")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("decision threshold must be between 0 and 1.")
        if risk not in {"Low", "Medium", "High"}:
            raise ValueError(f"Unknown risk level: {risk}")

        fields = {
            "product_type": str(sensor_reading.get("Type") or "").strip() or None,
            "air_temperature_k": _optional_float(
                sensor_reading.get("Air temperature [K]"), "Air temperature [K]"
            ),
            "process_temperature_k": _optional_float(
                sensor_reading.get("Process temperature [K]"),
                "Process temperature [K]",
            ),
            "vibration_value": _optional_float(
                sensor_reading.get("vibration"), "vibration"
            ),
            "pressure_value": _optional_float(
                sensor_reading.get("pressure"), "pressure"
            ),
            "rotational_speed_rpm": _optional_float(
                sensor_reading.get("Rotational speed [rpm]"),
                "Rotational speed [rpm]",
            ),
            "torque_nm": _optional_float(sensor_reading.get("Torque [Nm]"), "Torque [Nm]"),
            "tool_wear_min": _optional_float(
                sensor_reading.get("Tool wear [min]"), "Tool wear [min]"
            ),
        }
        timestamp = recorded_at or utc_now()

        with self.connection() as connection:
            equipment_id = self._ensure_equipment(
                connection,
                equipment_code,
                equipment_name,
                equipment_type,
                location,
            )
            cursor = connection.execute(
                """
                INSERT INTO sensor_readings (
                    equipment_id, recorded_at, data_source, product_type,
                    air_temperature_k, process_temperature_k, vibration_value, pressure_value,
                    rotational_speed_rpm, torque_nm, tool_wear_min, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    equipment_id,
                    timestamp,
                    data_source,
                    fields["product_type"],
                    fields["air_temperature_k"],
                    fields["process_temperature_k"],
                    fields["vibration_value"],
                    fields["pressure_value"],
                    fields["rotational_speed_rpm"],
                    fields["torque_nm"],
                    fields["tool_wear_min"],
                    utc_now(),
                ),
            )
            reading_id = int(cursor.lastrowid)

            cursor = connection.execute(
                """
                INSERT INTO predictions (
                    reading_id, equipment_id, model_name, model_version,
                    predicted_failure, failure_probability, decision_threshold,
                    risk_level, recommended_action, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    reading_id,
                    equipment_id,
                    model_name,
                    model_version,
                    predicted_failure,
                    probability,
                    threshold,
                    risk,
                    recommendation,
                    utc_now(),
                ),
            )
            prediction_id = int(cursor.lastrowid)

            alert_id: int | None = None
            if risk in {"Medium", "High"}:
                alert_cursor = connection.execute(
                    """
                    INSERT INTO maintenance_alerts (
                        prediction_id, equipment_id, severity, status,
                        recommendation, created_at
                    ) VALUES (?, ?, ?, 'Open', ?, ?)
                    """,
                    (
                        prediction_id,
                        equipment_id,
                        risk,
                        recommendation,
                        utc_now(),
                    ),
                )
                alert_id = int(alert_cursor.lastrowid)

        return {
            "equipment_id": equipment_id,
            "reading_id": reading_id,
            "prediction_id": prediction_id,
            "alert_id": alert_id,
        }

    def confirm_failure(
        self,
        equipment_code: str,
        confirmed_failure: bool,
        *,
        confirmed_by: str,
        confirmation_source: str = "operator_inspection",
        reading_id: int | None = None,
        notes: str = "",
    ) -> int:
        """Record a human-confirmed outcome separately from a prediction."""
        if not confirmed_by.strip():
            raise ValueError("confirmed_by is required for an actual outcome.")

        with self.connection() as connection:
            equipment = connection.execute(
                "SELECT equipment_id FROM equipment WHERE equipment_code = ?",
                (equipment_code.strip(),),
            ).fetchone()
            if equipment is None:
                raise ValueError(f"Unknown equipment code: {equipment_code!r}")
            equipment_id = int(equipment["equipment_id"])

            if reading_id is not None:
                reading = connection.execute(
                    """
                    SELECT reading_id FROM sensor_readings
                    WHERE reading_id = ? AND equipment_id = ?
                    """,
                    (reading_id, equipment_id),
                ).fetchone()
                if reading is None:
                    raise ValueError("Reading does not belong to this equipment.")

            cursor = connection.execute(
                """
                INSERT INTO failure_confirmations (
                    equipment_id, reading_id, confirmed_failure, confirmed_by,
                    confirmation_source, notes, confirmed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    equipment_id,
                    reading_id,
                    int(confirmed_failure),
                    confirmed_by.strip(),
                    confirmation_source,
                    notes.strip() or None,
                    utc_now(),
                ),
            )
            return int(cursor.lastrowid)

    def latest_equipment_status(self) -> list[dict[str, Any]]:
        """Return one latest reading/prediction row per equipment item."""
        query = """
            SELECT
                e.equipment_id, e.equipment_code, e.equipment_name,
                e.equipment_type, e.location,
                sr.reading_id, sr.recorded_at, sr.data_source, sr.product_type,
                sr.air_temperature_k, sr.process_temperature_k,
                sr.vibration_value, sr.pressure_value,
                sr.rotational_speed_rpm, sr.torque_nm, sr.tool_wear_min,
                p.prediction_id, p.predicted_failure,
                p.failure_probability, p.risk_level,
                p.recommended_action, p.created_at AS prediction_created_at
            FROM equipment e
            LEFT JOIN sensor_readings sr ON sr.reading_id = (
                SELECT sr2.reading_id FROM sensor_readings sr2
                WHERE sr2.equipment_id = e.equipment_id
                ORDER BY sr2.recorded_at DESC, sr2.reading_id DESC LIMIT 1
            )
            LEFT JOIN predictions p ON p.prediction_id = (
                SELECT p2.prediction_id FROM predictions p2
                WHERE p2.equipment_id = e.equipment_id
                ORDER BY p2.created_at DESC, p2.prediction_id DESC LIMIT 1
            )
            ORDER BY e.equipment_code
        """
        with self.connection() as connection:
            rows = connection.execute(query).fetchall()
        return [dict(row) for row in rows]

    def confirmation_counts(self) -> dict[str, int]:
        """Count human-confirmed outcomes, independently of predictions."""
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS total,
                       COALESCE(SUM(confirmed_failure), 0) AS failures,
                       COALESCE(SUM(1 - confirmed_failure), 0) AS non_failures
                FROM failure_confirmations
                """
            ).fetchone()
        return {
            "total": int(row["total"]),
            "failures": int(row["failures"]),
            "non_failures": int(row["non_failures"]),
        }

    def list_confirmations(
        self,
        equipment_code: str | None = None,
    ) -> list[dict[str, Any]]:
        """List operator-confirmed outcomes separately from predictions."""
        query = """
            SELECT fc.confirmation_id, fc.confirmed_failure, fc.confirmed_by,
                   fc.confirmation_source, fc.notes, fc.confirmed_at,
                   fc.reading_id, e.equipment_code, e.equipment_name
            FROM failure_confirmations fc
            JOIN equipment e ON e.equipment_id = fc.equipment_id
        """
        parameters: tuple[Any, ...] = ()
        if equipment_code is not None:
            query += " WHERE e.equipment_code = ?"
            parameters = (equipment_code,)
        query += " ORDER BY fc.confirmed_at DESC, fc.confirmation_id DESC"
        with self.connection() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [dict(row) for row in rows]

    def equipment_history(
        self,
        equipment_code: str,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """Return historical readings, latest prediction, and any confirmation."""
        if limit < 1:
            raise ValueError("limit must be at least 1.")
        query = """
            SELECT
                sr.reading_id, e.equipment_code, sr.recorded_at,
                sr.data_source, sr.product_type, sr.air_temperature_k,
                sr.process_temperature_k,
                sr.vibration_value, sr.pressure_value,
                sr.rotational_speed_rpm, sr.torque_nm, sr.tool_wear_min,
                p.predicted_failure, p.failure_probability, p.risk_level,
                p.recommended_action,
                fc.confirmed_failure, fc.confirmed_at, fc.confirmed_by
            FROM equipment e
            JOIN sensor_readings sr ON sr.equipment_id = e.equipment_id
            LEFT JOIN predictions p ON p.prediction_id = (
                SELECT p2.prediction_id FROM predictions p2
                WHERE p2.reading_id = sr.reading_id
                ORDER BY p2.created_at DESC, p2.prediction_id DESC LIMIT 1
            )
            LEFT JOIN failure_confirmations fc ON fc.confirmation_id = (
                SELECT fc2.confirmation_id FROM failure_confirmations fc2
                WHERE fc2.reading_id = sr.reading_id
                ORDER BY fc2.confirmed_at DESC, fc2.confirmation_id DESC LIMIT 1
            )
            WHERE e.equipment_code = ?
            ORDER BY sr.recorded_at DESC, sr.reading_id DESC
            LIMIT ?
        """
        with self.connection() as connection:
            rows = connection.execute(query, (equipment_code, limit)).fetchall()
        return pd.DataFrame([dict(row) for row in rows])

    def list_alerts(self, status: str | None = None) -> list[dict[str, Any]]:
        """Return maintenance alerts, optionally filtered by status."""
        allowed_statuses = {"Open", "Acknowledged", "Resolved"}
        if status is not None and status not in allowed_statuses:
            raise ValueError(f"Invalid alert status: {status}")

        query = """
            SELECT a.alert_id, a.severity, a.status, a.recommendation,
                   a.created_at, a.acknowledged_at, a.resolved_at,
                   e.equipment_code, e.equipment_name,
                   p.failure_probability, p.predicted_failure
            FROM maintenance_alerts a
            JOIN equipment e ON e.equipment_id = a.equipment_id
            JOIN predictions p ON p.prediction_id = a.prediction_id
        """
        parameters: tuple[Any, ...] = ()
        if status is not None:
            query += " WHERE a.status = ?"
            parameters = (status,)
        query += " ORDER BY a.created_at DESC, a.alert_id DESC"

        with self.connection() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [dict(row) for row in rows]

    def update_alert_status(self, alert_id: int, status: str) -> None:
        """Acknowledge or resolve an existing alert."""
        if status not in {"Acknowledged", "Resolved"}:
            raise ValueError("Status must be 'Acknowledged' or 'Resolved'.")

        timestamp = utc_now()
        with self.connection() as connection:
            cursor = connection.execute(
                """
                UPDATE maintenance_alerts
                SET status = ?,
                    acknowledged_at = CASE WHEN ? = 'Acknowledged'
                                           THEN ? ELSE acknowledged_at END,
                    resolved_at = CASE WHEN ? = 'Resolved'
                                       THEN ? ELSE resolved_at END
                WHERE alert_id = ?
                """,
                (status, status, timestamp, status, timestamp, alert_id),
            )
            if cursor.rowcount == 0:
                raise ValueError(f"Alert {alert_id} was not found.")
