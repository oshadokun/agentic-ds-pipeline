import {
  Brain, Shield, ShieldAlert, ChevronRight
} from "lucide-react"
import { useSession } from "../../contexts/SessionContext"

export default function Header() {
  const { session } = useSession()

  const privacyAcknowledged = session?.privacy?.user_acknowledged
  const hasSensitiveCols    = (session?.privacy?.sensitive_columns_found ?? []).length > 0

  return (
    <header className="h-14 bg-[#1B3A5C] text-white flex items-center px-6 gap-4 flex-shrink-0 z-20 shadow-md">
      {/* Logo + name */}
      <div className="flex items-center gap-2.5 flex-shrink-0">
        <div className="w-7 h-7 rounded-lg bg-white/10 flex items-center justify-center">
          <Brain size={16} className="text-white" />
        </div>
        <span className="font-serif text-lg tracking-tight">Data Science Pipeline</span>
      </div>

      <div className="flex-1" />

      {/* Session info */}
      {session && (
        <div className="flex items-center gap-3 text-sm">
          {/* Goal excerpt */}
          <span className="text-blue-200 hidden md:block max-w-xs truncate">
            {session.goal?.description
              ? `"${session.goal.description.slice(0, 60)}${session.goal.description.length > 60 ? "…" : ""}"`
              : "Session active"}
          </span>

          {/* Privacy indicator */}
          {hasSensitiveCols && (
            <div
              className={`flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium ${
                privacyAcknowledged
                  ? "bg-green-800/40 text-green-200"
                  : "bg-amber-600/40 text-amber-200"
              }`}
              title={
                privacyAcknowledged
                  ? "Privacy decisions confirmed"
                  : "Privacy decisions required before proceeding"
              }
            >
              {privacyAcknowledged
                ? <><Shield size={12} /> Privacy confirmed</>
                : <><ShieldAlert size={12} /> Privacy review needed</>}
            </div>
          )}
        </div>
      )}
    </header>
  )
}
