/**
 * DataPreviewTable
 * Shows the first N rows of a dataset in a readable table.
 *
 * Props:
 *   rows     — array of row objects
 *   columns  — array of column name strings (optional — inferred from rows if omitted)
 *   maxRows  — max rows to display (default 8)
 *   caption  — accessible table caption
 */
export default function DataPreviewTable({ rows = [], columns, maxRows = 8, caption }) {
  if (!rows.length) return null

  const cols    = columns ?? Object.keys(rows[0])
  const visible = rows.slice(0, maxRows)
  const clipped = rows.length > maxRows

  return (
    <div className="mb-6">
      <div className="overflow-x-auto rounded-xl border border-gray-200 shadow-sm">
        <table className="w-full text-sm text-left border-collapse">
          {caption && (
            <caption className="sr-only">{caption}</caption>
          )}

          {/* Header */}
          <thead>
            <tr className="bg-gray-50 border-b border-gray-200">
              {cols.map(col => (
                <th
                  key={col}
                  scope="col"
                  className="px-4 py-3 text-xs font-semibold text-gray-500
                             uppercase tracking-wide whitespace-nowrap"
                >
                  {col}
                </th>
              ))}
            </tr>
          </thead>

          {/* Body */}
          <tbody>
            {visible.map((row, i) => (
              <tr
                key={i}
                className={`border-b border-gray-100 ${
                  i % 2 === 0 ? "bg-white" : "bg-gray-50/50"
                }`}
              >
                {cols.map(col => (
                  <td
                    key={col}
                    className="px-4 py-2.5 text-gray-700 max-w-xs truncate"
                    title={String(row[col] ?? "")}
                  >
                    {row[col] === null || row[col] === undefined
                      ? <span className="text-gray-300 italic">—</span>
                      : String(row[col])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {clipped && (
        <p className="text-xs text-gray-400 mt-2 pl-1">
          Showing {maxRows} of {rows.length.toLocaleString()} rows.
        </p>
      )}
    </div>
  )
}
