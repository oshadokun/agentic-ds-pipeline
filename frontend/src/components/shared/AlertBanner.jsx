import { XCircle, AlertTriangle, Info, CheckCircle, X } from "lucide-react"

const STYLES = {
  error:   "bg-red-50 border-red-300 text-red-800",
  warning: "bg-amber-50 border-amber-300 text-amber-800",
  info:    "bg-blue-50 border-blue-300 text-blue-800",
  success: "bg-green-50 border-green-300 text-green-700"
}

const ICONS = {
  error:   XCircle,
  warning: AlertTriangle,
  info:    Info,
  success: CheckCircle
}

/**
 * AlertBanner
 * Props:
 *   type     — "error" | "warning" | "info" | "success"
 *   title    — optional bold heading
 *   message  — main text (required)
 *   action   — optional { label, onClick }
 *   onClose  — if provided, shows a close button
 */
export default function AlertBanner({
  type = "info", title, message, action, onClose
}) {
  const Icon = ICONS[type] ?? Info

  return (
    <div
      role={type === "error" ? "alert" : "status"}
      className={`flex gap-3 p-4 rounded-xl border ${STYLES[type]} mb-4 animate-slide-up`}
    >
      <Icon size={20} className="flex-shrink-0 mt-0.5" aria-hidden="true" />

      <div className="flex-1 min-w-0">
        {title && (
          <p className="font-semibold text-sm mb-1">{title}</p>
        )}
        <p className="text-sm leading-relaxed">{message}</p>
        {action && (
          <button
            onClick={action.onClick}
            className="mt-2 text-sm font-medium underline underline-offset-2
                       hover:no-underline transition-all"
          >
            {action.label}
          </button>
        )}
      </div>

      {onClose && (
        <button
          onClick={onClose}
          aria-label="Dismiss"
          className="flex-shrink-0 opacity-60 hover:opacity-100 transition-opacity p-0.5"
        >
          <X size={16} />
        </button>
      )}
    </div>
  )
}
