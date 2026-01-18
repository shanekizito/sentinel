import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Check, X, Code2, AlertTriangle, ShieldCheck } from "lucide-react";

const CodeDiffViewer = () => {
    const [activeTab, setActiveTab] = useState<"insecure" | "secure">("insecure");

    const insecureCode = `// 🚨 VULNERABLE CODE
app.post('/api/user', (req, res) => {
  const { username, role } = req.body;
  
  // No input validation
  // No role sanitization
  
  db.query(
    "INSERT INTO users (name, role) VALUES ('" + username + "', '" + role + "')"
  );
  
  res.status(200).json({ success: true });
});`;

    const secureCode = `// ✅ SENTINEL SECURED
import { z } from "zod";

const UserSchema = z.object({
  username: z.string().min(3).max(20),
  role: z.enum(["user", "guest"]), // Restricted roles
{{ ... }}
  );
  
  res.status(200).json({ success: true });
});`;

    return (
        <div className="w-full max-w-4xl mx-auto rounded-xl overflow-hidden shadow-2xl border border-border bg-white font-mono text-sm relative group">
            {/* Top Bar - Clean Industrial Light */}
            <div className="flex items-center justify-between px-4 py-3 bg-[#F5F5F3] border-b border-border relative z-10">
                <div className="flex items-center gap-4">
                    <div className="flex gap-1.5 mr-2">
                        <div className="w-3 h-3 rounded-full border border-gray-300 bg-gray-100" />
                        <div className="w-3 h-3 rounded-full border border-gray-300 bg-gray-100" />
                        <div className="w-3 h-3 rounded-full border border-gray-300 bg-gray-100" />
                    </div>

                    <div className="flex bg-gray-200/50 p-1 rounded-lg gap-1">
                        <button
                            onClick={() => setActiveTab("insecure")}
                            className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all flex items-center gap-2 ${activeTab === "insecure"
                                    ? "bg-white text-red-600 shadow-sm border border-border"
                                    : "text-gray-500 hover:text-gray-700"
                                }`}
                        >
                            <AlertTriangle className="w-3.5 h-3.5" />
                            Insecure.ts
                        </button>
                        <button
                            onClick={() => setActiveTab("secure")}
                            className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all flex items-center gap-2 ${activeTab === "secure"
                                    ? "bg-white text-emerald-600 shadow-sm border border-border"
                                    : "text-gray-500 hover:text-gray-700"
                                }`}
                        >
                            <ShieldCheck className="w-3.5 h-3.5" />
                            Secured_by_Sentinel.ts
                        </button>
                    </div>
                </div>

                <div className="text-xs text-gray-400 font-medium hidden sm:flex items-center gap-2">
                    <Code2 className="w-3.5 h-3.5" />
                    <span>TypeScript</span>
                </div>
            </div>

            {/* Code Area - DARK MODE for Contrast */}
            <div className="relative p-6 min-h-[300px] overflow-x-auto bg-[#09090B] text-gray-300">
                {/* Subtle inner grid for tech feel */}
                <div className="absolute inset-0 bg-grid-fine opacity-[0.03] pointer-events-none" />

                <div className="absolute top-0 left-0 w-10 h-full bg-[#0F0F11] border-r border-white/5" />
                <AnimatePresence mode="wait">
                    <motion.pre
                        key={activeTab}
                        initial={{ opacity: 0, y: 5 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -5 }}
                        transition={{ duration: 0.2 }}
                        className="font-mono text-[13px] leading-relaxed relative z-10 pl-4"
                    >
                        {activeTab === "insecure" ? (
                            <code className="block">
                                {insecureCode.split("\n").map((line, i) => (
                                    <div key={i} className="table-row">
                                        <span className="table-cell select-none text-gray-600 text-right pr-6 w-8">{i + 1}</span>
                                        <span className={`table-cell ${line.includes("INSERT INTO") || line.includes("req.body") ? "text-red-300 bg-red-900/20 w-full font-medium" : "text-gray-400"
                                            }`}>
                                            {line}
                                        </span>
                                    </div>
                                ))}
                            </code>
                        ) : (
                            <code className="block">
                                {secureCode.split("\n").map((line, i) => (
                                    <div key={i} className="table-row">
                                        <span className="table-cell select-none text-gray-600 text-right pr-6 w-8">{i + 1}</span>
                                        <span className={`table-cell ${line.includes("z.object") || line.includes("safeParse") || line.includes("$1") ? "text-emerald-300 bg-emerald-900/20 w-full font-medium" : "text-gray-400"
                                            }`}>
                                            {line}
                                        </span>
                                    </div>
                                ))}
                            </code>
                        )}
                    </motion.pre>
                </AnimatePresence>

                {/* Floating Badge */}
                <div className="absolute bottom-4 right-4 pointer-events-none">
                    {activeTab === "insecure" ? (
                        <div className="px-3 py-1.5 rounded-md bg-red-950/80 border border-red-500/30 text-red-200 text-xs font-bold flex items-center gap-1.5 backdrop-blur-md shadow-lg shadow-black/40">
                            <X className="w-3.5 h-3.5" />
                            CRITICAL VULNERABILITIES DETECTED
                        </div>
                    ) : (
                        <div className="px-3 py-1.5 rounded-md bg-emerald-950/80 border border-emerald-500/30 text-emerald-200 text-xs font-bold flex items-center gap-1.5 backdrop-blur-md shadow-lg shadow-black/40">
                            <Check className="w-3.5 h-3.5" />
                            AUTO-REMEDIATION APPLIED
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default CodeDiffViewer;
