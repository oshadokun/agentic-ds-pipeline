# Supported File Formats

## CSV and Delimited Files
| Extension | Delimiter | Notes |
|---|---|---|
| .csv | Comma (auto-detected) | Most common |
| .tsv | Tab | Auto-detected |
| .txt | Pipe or semicolon | Auto-detected |

Auto-detection tries: `,` → `;` → `\t` → `|` in order.
Accepts UTF-8, UTF-16, Latin-1 — encoding auto-detected via chardet.

## Excel (future support)
| Extension | Library |
|---|---|
| .xlsx | openpyxl |
| .xls | xlrd |

Note: For now, ask user to export as CSV first.

## JSON (via API)
- Flat list of objects → direct DataFrame conversion
- Nested dict with a list value → first list extracted
- Deeply nested JSON → flag to user, ask them to provide a flatter export
