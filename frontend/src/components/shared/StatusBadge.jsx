import { CheckCircle, Circle, AlertCircle, Clock, Loader2 } from "lucide-react"

const CONFIG = {
  complete: {
    label: "Complete",
    icon:  CheckCircle,
    style: "bg-green-50 text-green-700 border-green-200"
  },
  in_progress: {
    label: "In progress",
    icon:  Loader2,
    style: "bg-blue-50 text-blue-700 border-blue-200",
    spin:  true
  },
  pending: {
    label: "Not started",
    icon:  Circle,
    style: "bg-gray-50 text-gray-400 border-gray-200"
  },
  failed: {
    label: "Failed",
    icon:  AlertCircle,
    style: "bg-red-50 text-red-700 border-red-200"
  },
  decisions_required: {
    label: "Decision needed",
    icon:  Clock,
    style: "bg-amber-50 text-amber-700 border-amber-200"
  }
}

/**
 * StatusBadge
 * Props:
 *   status — "complete" | "in_progress" | "pending" | "failed" | "decisions_required"
 *   label  — override the default label
 *   size   — "sm" | "md" (default "md")
 */
export default function StatusBadge({ status = "pending", label, size = "md" }) {
  const cfg  = CONFIG[status] ?? CONFIG.pending
  const Icon = cfg.icon
  const text = label ?? cfg.label

  const sizeClass = size === "sm"
    ? "text-xs px-2 py-0.5 gap-1"
    : "text-xs px-2.5 py-1 gap-1.5"

  return (
    <span
      className={`
        inline-flex items-center font-medium rounded-full border
        ${cfg.style} ${sizeClass}
      `}
      aria-label={`Status: ${text}`}
    >
      <Icon
        size={size === "sm" ? 11 : 13}
        className={cfg.spin ? "animate-spin" : ""}
        aria-hidden="true"
      />
      {text}
    </span>
  )
}
