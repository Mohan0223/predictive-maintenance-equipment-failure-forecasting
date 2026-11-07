PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS equipment (
    equipment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    equipment_code TEXT NOT NULL UNIQUE,
    equipment_name TEXT NOT NULL,
    equipment_type TEXT,
    location TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sensor_readings (
    reading_id INTEGER PRIMARY KEY AUTOINCREMENT,
    equipment_id INTEGER NOT NULL,
    recorded_at TEXT NOT NULL,
    data_source TEXT NOT NULL DEFAULT 'operator_input',
    product_type TEXT,
    air_temperature_k REAL,
    process_temperature_k REAL,
    vibration_value REAL,
    pressure_value REAL,
    rotational_speed_rpm REAL,
    torque_nm REAL,
    tool_wear_min REAL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (equipment_id) REFERENCES equipment(equipment_id)
        ON UPDATE CASCADE ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    reading_id INTEGER NOT NULL,
    equipment_id INTEGER NOT NULL,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    predicted_failure INTEGER NOT NULL CHECK (predicted_failure IN (0, 1)),
    failure_probability REAL NOT NULL
        CHECK (failure_probability >= 0.0 AND failure_probability <= 1.0),
    decision_threshold REAL NOT NULL
        CHECK (decision_threshold >= 0.0 AND decision_threshold <= 1.0),
    risk_level TEXT NOT NULL CHECK (risk_level IN ('Low', 'Medium', 'High')),
    recommended_action TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (reading_id) REFERENCES sensor_readings(reading_id)
        ON UPDATE CASCADE ON DELETE RESTRICT,
    FOREIGN KEY (equipment_id) REFERENCES equipment(equipment_id)
        ON UPDATE CASCADE ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS maintenance_alerts (
    alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id INTEGER NOT NULL UNIQUE,
    equipment_id INTEGER NOT NULL,
    severity TEXT NOT NULL CHECK (severity IN ('Medium', 'High')),
    status TEXT NOT NULL DEFAULT 'Open'
        CHECK (status IN ('Open', 'Acknowledged', 'Resolved')),
    recommendation TEXT NOT NULL,
    created_at TEXT NOT NULL,
    acknowledged_at TEXT,
    resolved_at TEXT,
    FOREIGN KEY (prediction_id) REFERENCES predictions(prediction_id)
        ON UPDATE CASCADE ON DELETE RESTRICT,
    FOREIGN KEY (equipment_id) REFERENCES equipment(equipment_id)
        ON UPDATE CASCADE ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS failure_confirmations (
    confirmation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    equipment_id INTEGER NOT NULL,
    reading_id INTEGER,
    confirmed_failure INTEGER NOT NULL CHECK (confirmed_failure IN (0, 1)),
    confirmed_by TEXT NOT NULL,
    confirmation_source TEXT NOT NULL,
    notes TEXT,
    confirmed_at TEXT NOT NULL,
    FOREIGN KEY (equipment_id) REFERENCES equipment(equipment_id)
        ON UPDATE CASCADE ON DELETE RESTRICT,
    FOREIGN KEY (reading_id) REFERENCES sensor_readings(reading_id)
        ON UPDATE CASCADE ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS idx_sensor_readings_equipment_time
    ON sensor_readings(equipment_id, recorded_at DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_equipment_time
    ON predictions(equipment_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_status_time
    ON maintenance_alerts(status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_confirmations_equipment_time
    ON failure_confirmations(equipment_id, confirmed_at DESC);
