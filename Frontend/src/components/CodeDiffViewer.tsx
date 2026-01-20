import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Check, X, Code2, AlertTriangle, ShieldCheck, Terminal, Cpu, Lock } from "lucide-react";

const CodeDiffViewer = () => {
    const [activeTab, setActiveTab] = useState<"insecure" | "secure">("insecure");

    const insecureCode = `// ⚠️ VULNERABLE: Session Key Generation
// CWE-330: Use of Insufficiently Random Values

use rand::prelude::*;

pub fn generate_session_key() -> String {
    // ❌ WEAKNESS: Non-cryptographic RNG
    // Predictable seed from system time
    let mut rng = rand::thread_rng();
    
    // ❌ VULNERABILITY: 32-bit entropy is insufficient 
    // for session security in high-risk environments
    let key: u32 = rng.gen();
    
    // ❌ LOGIC BUG: Key space too small for 
    // brute-force resistance
    format!("sess_{:x}", key)
}`;

    const secureCode = `// ✅ SENTINEL SECURED: PQC Kyber-768 Encap
// FIPS 203 Compliant Key Encapsulation

use pqc_kyber::*;
use sentinel_core::verify;

// 🔒 FORMAL VERIFICATION: SMT-Proved Safe
#[sentinel::verified(entropy >= 256)]
pub fn generate_session_key(pk: &[u8]) -> Result<[u8; 32], KEMError> {
    // ✅ KEM: Quantum-safe encapsulation
    let (ct, ss) = kyber768::encapsulate(&pk, &mut thread_rng())?;

    // ✅ VERIFY: Constant-time comparison
    // prevents timing side-channels
    verify::assert_constant_time(&ss);

    Ok(ss)
}`;

    return (
        <div className="w-full max-w-5xl mx-auto font-mono text-sm relative">
            {/* Architectural Frame System */}
            <div className="border-2 border-gray-900 bg-white relative shadow-[0_32px_64px_-16px_rgba(0,0,0,0.1)] transition-shadow duration-500 hover:shadow-[0_48px_80px_-20px_rgba(0,0,0,0.15)]">

                {/* Corner Accents */}
                <div className="absolute -top-1 -left-1 w-3 h-3 border-t-2 border-l-2 border-primary bg-white z-20" />
                <div className="absolute -top-1 -right-1 w-3 h-3 border-t-2 border-r-2 border-primary bg-white z-20" />
                <div className="absolute -bottom-1 -left-1 w-3 h-3 border-b-2 border-l-2 border-primary bg-white z-20" />
                <div className="absolute -bottom-1 -right-1 w-3 h-3 border-b-2 border-r-2 border-primary bg-white z-20" />

                {/* Header: Industrial Control Panel */}
                <div className="flex flex-col md:flex-row items-center justify-between px-6 py-4 bg-gray-50 border-b-2 border-gray-900">

                    {/* Left: Tab Switcher (Physical Buttons) */}
                    <div className="flex items-center gap-4 w-full md:w-auto">
                        <div className="flex gap-px bg-gray-900 border-2 border-gray-900 p-px">
                            <button
                                onClick={() => setActiveTab("insecure")}
                                className={`px-4 py-2 text-xs font-bold uppercase tracking-wider transition-all flex items-center gap-2 ${activeTab === "insecure"
                                    ? "bg-red-50 text-red-700"
                                    : "bg-white text-gray-400 hover:text-gray-900 hover:bg-gray-100"
                                    }`}
                            >
                                <AlertTriangle className="w-3.5 h-3.5" />
                                Vulnerable
                            </button>
                            <button
                                onClick={() => setActiveTab("secure")}
                                className={`px-4 py-2 text-xs font-bold uppercase tracking-wider transition-all flex items-center gap-2 ${activeTab === "secure"
                                    ? "bg-primary/10 text-primary"
                                    : "bg-white text-gray-400 hover:text-gray-900 hover:bg-gray-100"
                                    }`}
                            >
                                <ShieldCheck className="w-3.5 h-3.5" />
                                Secured
                            </button>
                        </div>
                    </div>

                    {/* Right: Technical Meta */}
                    <div className="flex items-center gap-8 mt-4 md:mt-0 text-xs font-bold text-gray-400 uppercase tracking-widest">
                        <div className="flex items-center gap-2">
                            <Code2 className="w-4 h-4 text-gray-900" />
                            <span>Rust 1.78</span>
                        </div>
                        <div className="hidden sm:flex items-center gap-2">
                            <Cpu className="w-4 h-4 text-gray-900" />
                            <span>PQC-Kyber</span>
                        </div>
                        <div className="hidden sm:flex items-center gap-2">
                            <Terminal className="w-4 h-4 text-gray-900" />
                            <span>O(1)</span>
                        </div>
                    </div>
                </div>

                {/* Code Area: Blueprint Mode */}
                <div className="relative p-0 overflow-hidden bg-[#0a0a0c]">

                    {/* Inner Structural Lines */}
                    <div className="absolute left-[3.5rem] top-0 bottom-0 w-px bg-white/10" />
                    <div className="absolute inset-0 bg-[linear-gradient(to_right,transparent_99%,rgba(255,255,255,0.03)_1%)] bg-[size:2rem_100%] pointer-events-none" />

                    <div className="max-h-[400px] overflow-auto custom-scrollbar">
                        <AnimatePresence mode="wait">
                            <motion.div
                                key={activeTab}
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                exit={{ opacity: 0 }}
                                transition={{ duration: 0.15 }}
                                className="font-mono text-[13px] leading-7 p-6"
                            >
                                {activeTab === "insecure" ? (
                                    <code>
                                        {insecureCode.split("\n").map((line, i) => (
                                            <div key={i} className="flex">
                                                <span className="w-10 text-gray-700 text-right pr-4 select-none shrink-0">{i + 1}</span>
                                                <span className={`whitespace-pre ${line.includes("❌") ? "text-red-400 font-bold bg-red-950/30 w-full" :
                                                    line.includes("⚠️") ? "text-amber-400" :
                                                        line.startsWith("//") ? "text-gray-500 italic" : "text-gray-300"
                                                    }`}>
                                                    {line}
                                                </span>
                                            </div>
                                        ))}
                                    </code>
                                ) : (
                                    <code>
                                        {secureCode.split("\n").map((line, i) => (
                                            <div key={i} className="flex">
                                                <span className="w-10 text-gray-700 text-right pr-4 select-none shrink-0">{i + 1}</span>
                                                <span className={`whitespace-pre ${line.includes("✅") ? "text-primary font-bold bg-primary/20 w-full" :
                                                    line.includes("🔒") ? "text-primary" :
                                                        line.startsWith("//") ? "text-gray-500 italic" : "text-gray-300"
                                                    }`}>
                                                    {line}
                                                </span>
                                            </div>
                                        ))}
                                    </code>
                                )}
                            </motion.div>
                        </AnimatePresence>
                    </div>

                    {/* Footer Status Bar */}
                    <div className="flex items-center justify-between px-6 py-2 bg-gray-900 border-t border-white/10 text-[10px] font-mono uppercase tracking-wider text-gray-400">
                        <div className="flex items-center gap-4">
                            <span>Status: {activeTab === "insecure" ? <span className="text-red-500 font-bold">VULNERABLE</span> : <span className="text-primary font-bold">SECURE</span>}</span>
                            <span className="hidden sm:inline">Check: SHA-256</span>
                        </div>
                        <div className="flex items-center gap-2">
                            {activeTab === "secure" && <Lock className="w-3 h-3 text-primary" />}
                            <span>{activeTab === "insecure" ? "Scan Required" : "Verified Safe"}</span>
                        </div>
                    </div>

                </div>
            </div>
        </div >
    );
};

export default CodeDiffViewer;
