# Supported Databases and Connection Strings

## Supported Databases

| Database | Driver | Connection String Format |
|---|---|---|
| PostgreSQL | psycopg2 | `postgresql://user:password@host:5432/dbname` |
| MySQL | pymysql | `mysql+pymysql://user:password@host:3306/dbname` |
| SQLite | built-in | `sqlite:///path/to/file.db` |
| SQL Server | pyodbc | `mssql+pyodbc://user:password@host/dbname?driver=ODBC+Driver+17` |
| BigQuery | sqlalchemy-bigquery | `bigquery://project/dataset` |

## Security Rules
- Connection strings are written to `sessions/{session_id}/.env` immediately on receipt
- They are never passed as function arguments beyond the ingestion module
- They are never written to session.json, logs, or the report
- The `.env` file is excluded from any export or sharing function

## Query Guidelines
- Always use a SELECT query — never INSERT, UPDATE, or DELETE
- Recommend the user adds a LIMIT clause during development to avoid loading millions of rows
- If no query is provided, default to `SELECT * FROM {table} LIMIT 10000` and inform the user
