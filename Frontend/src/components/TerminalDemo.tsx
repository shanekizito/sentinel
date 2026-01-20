import { useEffect, useState, useRef } from 'react';
import { cn } from "@/lib/utils";
import { Terminal, ShieldCheck, Activity, Lock, Wifi, Search, AlertCircle, CheckCircle2 } from 'lucide-react';
import { motion } from 'framer-motion';

interface LogEntry {
  id: string;
  timestamp: string;
  type: 'info' | 'success' | 'warning' | 'error' | 'process';
  message: string;
  details?: string;
}

const SAMPLE_LOGS: LogEntry[] = [
  { id: '1', timestamp: '10:42:01', type: 'info', message: 'Initializing Sentinel secure environment...' },
  { id: '2', timestamp: '10:42:01', type: 'process', message: 'Verifying TLS 1.3 handshake compatibility' },
  { id: '3', timestamp: '10:42:02', type: 'success', message: 'Connection established securely', details: 'AES-256-GCM' },
  { id: '4', timestamp: '10:42:02', type: 'info', message: 'Scanning codebase for vulnerable patterns...' },
  { id: '5', timestamp: '10:42:03', type: 'process', message: 'Analyzing commit d4f8a2...', details: 'Git Hook Triggered' },
  { id: '6', timestamp: '10:42:03', type: 'warning', message: 'Detected HARDCODED_SECRET in routes/user.ts', details: 'Line 24: AWS_ACCESS_KEY_ID' },
  { id: '7', timestamp: '10:42:04', type: 'info', message: 'Initiating auto-remediation protocol...' },
  { id: '8', timestamp: '10:42:04', type: 'success', message: 'Secret rotated to Vault', details: 'Action verified' },
  { id: '9', timestamp: '10:42:05', type: 'process', message: 'Re-running compliance checks' },
  { id: '10', timestamp: '10:42:05', type: 'success', message: 'Systems secure. No threats detected.', details: 'SOC2 Compliant' },
  { id: '11', timestamp: '10:42:06', type: 'info', message: 'Sentinel Core active. Monitoring realtime.' },
];

export const TerminalDemo = () => {
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [activeIndex, setActiveIndex] = useState(0);
  const [scanProgress, setScanProgress] = useState(0);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (activeIndex >= SAMPLE_LOGS.length) {
      const timeout = setTimeout(() => {
        setLogs([]);
        setActiveIndex(0);
        setScanProgress(0);
      }, 5000);
      return () => clearTimeout(timeout);
    }

    const delay = Math.random() * 600 + 400;
    const timeout = setTimeout(() => {
      setLogs(prev => [...prev, SAMPLE_LOGS[activeIndex]]);
      setActiveIndex(prev => prev + 1);
      setScanProgress(prev => Math.min(prev + (100 / SAMPLE_LOGS.length), 100));
    }, delay);

    return () => clearTimeout(timeout);
  }, [activeIndex]);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs]);

  return (
    <div className="w-full max-w-2xl mx-auto perspective-1000 group">
      {/* Industrial Frame */}
      <div className="relative rounded-xl overflow-hidden bg-white border border-border shadow-2xl transform transition-all duration-500 hover:scale-[1.01]">

        {/* Header Bar - Light Industrial */}
        <div className="flex items-center justify-between px-4 py-3 bg-[#F5F5F3] border-b border-border relative z-10">
          <div className="flex items-center gap-4">
            {/* Window Controls */}
            <div className="flex gap-1.5">
              <div className="w-3 h-3 rounded-full bg-border" />
              <div className="w-3 h-3 rounded-full bg-border" />
              <div className="w-3 h-3 rounded-full bg-border" />
            </div>

            <div className="flex items-center gap-2 pl-2 border-l border-gray-300">
              <ShieldCheck className="w-3.5 h-3.5 text-primary" />
              <span className="text-xs font-mono text-foreground font-semibold tracking-tight">SENTINEL_CORE.SYS</span>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2 px-2 py-0.5 rounded-full bg-white border border-border/60 shadow-sm">
              <Activity className="w-3 h-3 text-primary animate-pulse" />
              <span className="text-[10px] font-mono text-primary font-bold">ONLINE</span>
            </div>
          </div>
        </div>

        {/* Scanner HUD - Light Mode */}
        <div className="h-0.5 bg-gray-200 w-full relative overflow-hidden">
          <motion.div
            className="absolute inset-y-0 left-0 bg-primary"
            initial={{ width: "0%" }}
            animate={{ width: `${scanProgress}%` }}
            transition={{ type: "spring", stiffness: 50 }}
          />
        </div>

        {/* Content Area - Dark High Contrast for Code */}
        <div
          ref={scrollRef}
          className="p-6 h-[400px] overflow-y-auto font-mono text-sm scrollbar-none bg-[#0F1412] relative"
        >
          {/* Subtle Grid inside Terminal */}
          <div className="absolute inset-0 bg-grid-fine opacity-5 pointer-events-none" />

          <div className="relative z-10 space-y-3">
            {logs.map((log) => (
              <div key={log.id} className="flex gap-3 animate-fade-in group/line border-l-2 border-transparent hover:border-white/10 pl-2 transition-all">
                <span className="text-white/30 shrink-0 select-none text-xs pt-0.5">[{log.timestamp}]</span>
                <div className="flex-1">
                  <div className="flex items-start gap-2">
                    <span className={cn("font-bold mt-0.5",
                      log.type === 'success' ? 'text-primary' :
                        log.type === 'warning' ? 'text-amber-400' :
                          log.type === 'error' ? 'text-red-400' :
                            log.type === 'process' ? 'text-blue-400' :
                              'text-gray-400'
                    )}>
                      {log.type === 'process' && '~'}
                      {log.type === 'success' && <CheckCircle2 className="w-3.5 h-3.5" />}
                      {log.type === 'info' && 'i'}
                      {log.type === 'warning' && <AlertCircle className="w-3.5 h-3.5" />}
                      {log.type === 'error' && '!'}
                    </span>
                    <span className="text-gray-200 font-medium tracking-tight">{log.message}</span>
                  </div>
                  {log.details && (
                    <div className="ml-6 mt-1 text-xs text-white/40 font-mono flex items-center gap-1">
                      <span className="w-2 h-[1px] bg-white/20" />
                      {log.details}
                    </div>
                  )}
                </div>
              </div>
            ))}

            {/* Active Line */}
            <div className="flex gap-3 pt-2 opacity-80 pl-2">
              <span className="text-white/20 shrink-0 select-none text-xs">[{new Date().toLocaleTimeString('en-US', { hour12: false })}]</span>
              <div className="flex items-center gap-2">
                <span className="text-primary font-bold">➜</span>
                <span className="w-2 h-4 bg-primary animate-blink" />
              </div>
            </div>
          </div>
        </div>

        {/* Footer Status - Light Industrial */}
        <div className="px-4 py-2 bg-[#F5F5F3] border-t border-border flex items-center justify-between text-[10px] text-muted-foreground font-mono uppercase tracking-widest relative z-10">
          <div className="flex gap-4">
            <span className="flex items-center gap-1.5 font-semibold">
              <Search className="w-3 h-3" />
              Threat Hunting
            </span>
            <span className="text-gray-400">Target: /usr/src/app</span>
          </div>
          <div className="flex items-center gap-2 text-primary font-semibold">
            <Wifi className="w-3 h-3" />
            <span>Secure Link</span>
          </div>
        </div>
      </div>
    </div>
  );
};
