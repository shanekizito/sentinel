import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Globe, Shield, Activity, Map, Zap, Crosshair } from "lucide-react";

interface Threat {
    id: string;
    x: number;
    y: number;
    type: "injection" | "ddos" | "malware" | "zero-day";
    country: string;
}

const ThreatMonitor = () => {
    const [threats, setThreats] = useState<Threat[]>([]);
    const [blockedCount, setBlockedCount] = useState(14205);

    useEffect(() => {
        const interval = setInterval(() => {
            // Add a new random threat
            const newThreat: Threat = {
                id: Math.random().toString(36).substr(2, 9),
                x: Math.random() * 100,
                y: Math.random() * 100,
                type: ["injection", "ddos", "malware", "zero-day"][Math.floor(Math.random() * 4)] as any,
                country: ["US", "DE", "JP", "BR", "CN", "RU"][Math.floor(Math.random() * 6)],
            };

            setThreats((prev) => [...prev.slice(-15), newThreat]);
            setBlockedCount((prev) => prev + Math.floor(Math.random() * 3) + 1);
        }, 800);

        return () => clearInterval(interval);
    }, []);

    return (
        <div className="w-full max-w-5xl mx-auto p-3 bg-white rounded-xl border border-border shadow-xl">
            {/* Header - Light & Clean */}
            <div className="flex items-center justify-between px-2 pb-3">
                <div className="flex items-center gap-2">
                    <div className="w-2 h-8 bg-primary rounded-full" />
                    <div>
                        <h3 className="text-sm font-bold text-foreground font-display tracking-tight leading-none">THREAT_INTEL</h3>
                        <span className="text-[10px] text-muted-foreground uppercase tracking-widest font-mono">Realtime Global Feed</span>
                    </div>
                </div>

                <div className="flex gap-4 text-right">
                    <div>
                        <span className="block text-[10px] text-muted-foreground uppercase font-mono">Status</span>
                        <span className="text-primary text-xs font-bold font-mono">ACTIVE</span>
                    </div>
                    <div>
                        <span className="block text-[10px] text-muted-foreground uppercase font-mono">Threats Blocked</span>
                        <span className="text-foreground text-xs font-bold font-mono tabular-nums">{blockedCount.toLocaleString()}</span>
                    </div>
                </div>
            </div>

            <div className="bg-[#050505] rounded-lg overflow-hidden relative min-h-[450px] flex flex-col border border-border/10 shadow-inner">

                {/* Dark Map Area */}
                <div className="flex-1 relative overflow-hidden bg-[url('https://upload.wikimedia.org/wikipedia/commons/e/ec/World_map_blank_without_borders.svg')] bg-no-repeat bg-center bg-contain opacity-20 invert filter">

                    {/* Dark Grid Overlay */}
                    <div className="absolute inset-0 bg-grid-isometric opacity-[0.05] pointer-events-none" />

                    {/* Radar Scan Effect - Green */}
                    <div className="absolute inset-0 animate-[spin_8s_linear_infinite] opacity-20 pointer-events-none origin-center">
                        <div className="w-full h-full border-r border-primary/50 bg-gradient-to-l from-primary/10 to-transparent" style={{ clipPath: 'polygon(50% 50%, 100% 0, 100% 50%)' }} />
                    </div>

                    <AnimatePresence>
                        {threats.map((threat) => (
                            <motion.div
                                key={threat.id}
                                initial={{ opacity: 0, scale: 0 }}
                                animate={{ opacity: 1, scale: 1 }}
                                exit={{ opacity: 0, scale: 1.5 }}
                                transition={{ duration: 0.5 }}
                                className="absolute w-4 h-4"
                                style={{ left: `${threat.x}%`, top: `${threat.y}%` }}
                            >
                                {/* Target Reticle Animation - High Contrast Green on Black */}
                                <span className="absolute inline-flex h-full w-full rounded-full border border-primary opacity-75 animate-ping"></span>
                                <span className="relative inline-flex rounded-full h-2 w-2 top-1 left-1 bg-primary shadow-[0_0_10px_#00A676]"></span>

                                {/* Label */}
                                <motion.div
                                    initial={{ opacity: 0, y: 5 }}
                                    animate={{ opacity: 1, y: -20 }}
                                    exit={{ opacity: 0 }}
                                    className="absolute left-1/2 -translate-x-1/2 -top-1 whitespace-nowrap bg-primary/20 border border-primary/40 px-1.5 py-0.5 rounded text-[9px] text-primary font-mono font-bold pointer-events-none z-20 backdrop-blur-md"
                                >
                                    {threat.type.toUpperCase()}
                                </motion.div>
                            </motion.div>
                        ))}
                    </AnimatePresence>
                </div>

                {/* Footer Stats - Dark HUD Style */}
                <div className="grid grid-cols-4 border-t border-white/10 bg-[#0A0A0A] divide-x divide-white/5">
                    {[
                        { label: "Active Nodes", value: "8,942", icon: Activity },
                        { label: "Bandwidth", value: "4.2 TB/s", icon: Zap },
                        { label: "Global Latency", value: "24ms", icon: Map },
                        { label: "Success Rate", value: "99.99%", icon: Shield },
                    ].map((stat) => (
                        <div key={stat.label} className="p-4 flex items-center justify-between group hover:bg-white/5 transition-colors cursor-default">
                            <div>
                                <p className="text-[10px] text-gray-500 uppercase tracking-widest mb-1 font-mono">{stat.label}</p>
                                <p className="text-sm font-mono font-medium text-gray-300 group-hover:text-primary transition-colors">{stat.value}</p>
                            </div>
                            <stat.icon className="w-4 h-4 text-gray-700 group-hover:text-primary transition-colors" />
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
};

export default ThreatMonitor;
