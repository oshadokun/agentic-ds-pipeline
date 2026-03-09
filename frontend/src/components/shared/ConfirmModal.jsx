import { useEffect } from "react"
import { AlertTriangle } from "lucide-react"

/**
 * ConfirmModal
 * Used before any irreversible action.
 *
 * Props:
 *   title          — modal heading
 *   message        — what is about to happen
 *   consequence    — what cannot be undone (shown in red box)
 *   confirmLabel   — label for the destructive button
 *   onConfirm      — called when user clicks confirm
 *   onCancel       — called when user clicks cancel or presses Escape
 */
export default function ConfirmModal({
  title, message, consequence, confirmLabel = "Confirm", onConfirm, onCancel
}) {
  // Close on Escape
  useEffect(() => {
    function onKey(e) { if (e.key === "Escape") onCancel() }
    document.addEventListener("keydown", onKey)
    return () => document.removeEventListener("keydown", onKey)
  }, [onCancel])

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="modal-title"
      className="fixed inset-0 bg-black/40 flex items-center justify-center
                 p-4 z-50 animate-fade-in"
      onClick={e => e.target === e.currentTarget && onCancel()}
    >
      <div className="bg-white rounded-2xl shadow-xl max-w-md w-full p-6 animate-scale-in">

        {/* Icon */}
        <div className="w-12 h-12 rounded-xl bg-amber-100 flex items-center
                        justify-center mb-4">
          <AlertTriangle size={24} className="text-amber-600" aria-hidden="true" />
        </div>

        {/* Content */}
        <h2 id="modal-title" className="text-xl font-serif text-gray-900 mb-2">
          {title}
        </h2>
        <p className="text-gray-600 text-sm leading-relaxed mb-3">{message}</p>

        {consequence && (
          <div className="bg-red-50 border border-red-200 rounded-xl px-4 py-3 mb-5">
            <p className="text-red-700 text-sm font-semibold mb-1">
              ⚠ This cannot be undone:
            </p>
            <p className="text-red-600 text-sm leading-relaxed">{consequence}</p>
          </div>
        )}

        {/* Actions */}
        <div className="flex gap-3">
          <button
            onClick={onCancel}
            className="flex-1 py-2.5 rounded-xl border border-gray-200
                       text-gray-700 text-sm font-medium hover:bg-gray-50
                       transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={onConfirm}
            className="flex-1 py-2.5 rounded-xl bg-red-600 text-white
                       text-sm font-medium hover:bg-red-700 transition-colors"
          >
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  )
}
