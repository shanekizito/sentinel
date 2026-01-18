import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import ThreatMonitor from "@/components/ThreatMonitor";
import { Shield, Search, FileCode, Server, Database, CheckCircle2, Terminal, EyeOff, FileKey, Fingerprint, Lock } from "lucide-react";

const Security = () => {
    return (
        <div className="min-h-screen bg-background flex flex-col font-sans selection:bg-emerald-100 selection:text-emerald-900">
            <Navbar />

            <main className="flex-grow pt-32">
                {/* Hero Section - Denser, Tighter Typography */}
                <section className="container mx-auto px-6 mb-24">
                    <div className="max-w-4xl">
                        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-50 border border-emerald-200 text-xs font-bold text-emerald-700 mb-8 tracking-wide uppercase">
                            <Shield className="w-3.5 h-3.5" />
                            <span>Defense in Depth Architecture</span>
                        </div>
                        <h1 className="font-display text-6xl md:text-7xl font-bold mb-8 text-foreground tracking-tighter leading-[0.9]">
                            The Hybrid Engine: <br />
                            <span className="text-muted-foreground/80">Deterministic Precision + <br />GenAI Adaptability.</span>
                        </h1>
                        <p className="text-2xl text-muted-foreground max-w-2xl leading-relaxed text-balance font-light">
                            Sentinel combines static analysis (AST) for 100% known-pattern detection with a fine-tuned LLM for context-aware logic vulnerabilities.
                        </p>
                    </div>
                </section>

                {/* Architecture Diagram - Glass Panel Polish */}
                <section className="bg-secondary/30 border-y border-border py-24 mb-24 relative overflow-hidden">
                    <div className="absolute inset-0 bg-grid-fine opacity-[0.03] pointer-events-none" />
                    <div className="container mx-auto px-6 relative z-10">
                        <div className="grid lg:grid-cols-2 gap-20 items-center">
                            <div className="space-y-10">
                                <h2 className="font-display text-4xl font-bold tracking-tight">How the Engine Works</h2>
                                <div className="space-y-4">
                                    {[
                                        {
                                            title: "1. AST Parsing",
                                            desc: "Code is converted into an Abstract Syntax Tree. Sentinel maps data flow paths to identify tainted inputs reaching sinks.",
                                            icon: <FileCode className="w-5 h-5 text-blue-600" />
                                        },
                                        {
                                            title: "2. Heuristic Scans",
                                            desc: "6,000+ deterministic rules check for hardcoded secrets, misconfigurations, and known CVE patterns instantly.",
                                            icon: <Search className="w-5 h-5 text-purple-600" />
                                        },
                                        {
                                            title: "3. AI Contextualization",
                                            desc: "For complex logic (e.g., 'Is this user authorized?'), our varying-parameter LLM analyzes function intent and scope.",
                                            icon: <Server className="w-5 h-5 text-emerald-600" />
                                        }
                                    ].map((step) => (
                                        <div key={step.title} className="flex gap-5 p-6 bg-white/80 backdrop-blur-sm rounded-xl border border-white/50 shadow-sm hover:border-primary/20 transition-all duration-300">
                                            <div className="mt-1 shrink-0 bg-secondary p-2 rounded-lg">{step.icon}</div>
                                            <div>
                                                <h3 className="font-bold text-lg text-foreground mb-2">{step.title}</h3>
                                                <p className="text-sm text-muted-foreground leading-relaxed">{step.desc}</p>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>

                            {/* Visual Representation of Engine - Premium Look */}
                            <div className="relative bg-gradient-to-br from-white to-secondary/50 rounded-3xl border border-white/60 shadow-2xl p-10 aspect-square flex flex-col justify-center backdrop-blur-xl">
                                <div className="absolute inset-0 bg-dot-pattern opacity-[0.1] pointer-events-none" />

                                <div className="relative z-10 space-y-6">
                                    <div className="flex justify-center">
                                        <div className="bg-gray-900 text-white px-8 py-4 rounded-xl font-mono text-sm border border-gray-700 shadow-xl flex items-center gap-3">
                                            <Terminal className="w-4 h-4 text-gray-400" />
                                            Source Code Input
                                        </div>
                                    </div>
                                    <div className="h-12 w-px bg-border mx-auto border-l-2 border-dashed border-gray-300" />
                                    <div className="grid grid-cols-2 gap-6">
                                        <div className="p-6 bg-blue-50/50 border border-blue-100 rounded-2xl text-center backdrop-blur-sm shadow-sm">
                                            <div className="font-bold text-blue-900 mb-2">Static Analysis</div>
                                            <div className="text-xs text-blue-600 font-mono bg-blue-100/50 px-2 py-1 rounded inline-block">AST / RegEx</div>
                                        </div>
                                        <div className="p-6 bg-emerald-50/50 border border-emerald-100 rounded-2xl text-center backdrop-blur-sm shadow-sm">
                                            <div className="font-bold text-emerald-900 mb-2">AI Reasoning</div>
                                            <div className="text-xs text-emerald-600 font-mono bg-emerald-100/50 px-2 py-1 rounded inline-block">LLM / Context</div>
                                        </div>
                                    </div>
                                    <div className="h-12 w-px bg-border mx-auto border-l-2 border-dashed border-gray-300" />
                                    <div className="flex justify-center">
                                        <div className="bg-white border-2 border-emerald-500 text-emerald-800 px-8 py-4 rounded-xl font-bold shadow-xl flex items-center gap-3">
                                            <CheckCircle2 className="w-5 h-5 text-emerald-500" />
                                            Verified Patch
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </section>

                {/* Privacy By Design Section - NEW */}
                <section className="container mx-auto px-6 mb-32">
                    <div className="bg-[#0A0A0A] rounded-3xl p-12 md:p-16 text-white relative overflow-hidden">
                        <div className="absolute top-0 right-0 w-96 h-96 bg-emerald-500/10 rounded-full blur-3xl -translate-y-1/2 translate-x-1/2" />

                        <div className="relative z-10">
                            <h2 className="font-display text-4xl font-bold mb-12 flex items-center gap-4">
                                <Lock className="w-8 h-8 text-emerald-400" />
                                Privacy by Design
                            </h2>

                            <div className="grid md:grid-cols-3 gap-12">
                                <div className="space-y-4">
                                    <div className="w-12 h-12 rounded-full bg-white/10 flex items-center justify-center mb-4 text-emerald-400">
                                        <EyeOff className="w-6 h-6" />
                                    </div>
                                    <h3 className="font-bold text-xl">Zero Retention</h3>
                                    <p className="text-gray-400 leading-relaxed">
                                        Code snippets sent to the LLM are ephemeral. They are processed in memory and immediately discarded. We never train models on your data.
                                    </p>
                                </div>
                                <div className="space-y-4">
                                    <div className="w-12 h-12 rounded-full bg-white/10 flex items-center justify-center mb-4 text-emerald-400">
                                        <FileKey className="w-6 h-6" />
                                    </div>
                                    <h3 className="font-bold text-xl">PII Redaction</h3>
                                    <p className="text-gray-400 leading-relaxed">
                                        Before analysis, all potential PII (emails, IPs, keys) is locally redacted and replaced with synthetic tokens.
                                    </p>
                                </div>
                                <div className="space-y-4">
                                    <div className="w-12 h-12 rounded-full bg-white/10 flex items-center justify-center mb-4 text-emerald-400">
                                        <Fingerprint className="w-6 h-6" />
                                    </div>
                                    <h3 className="font-bold text-xl">SOC2 Type II</h3>
                                    <p className="text-gray-400 leading-relaxed">
                                        Our infrastructure is audited annually. We encrypt all data at rest (AES-256) and in transit (TLS 1.3).
                                    </p>
                                </div>
                            </div>
                        </div>
                    </div>
                </section>

                {/* Attack Vector Matrix - Enhanced */}
                <section className="container mx-auto px-6 mb-32">
                    <div className="text-center max-w-3xl mx-auto mb-20">
                        <h2 className="font-display text-4xl font-bold mb-6">Coverage Matrix</h2>
                        <p className="text-xl text-muted-foreground font-light">Trained on the OWASP Top 10, CWE Top 25, and proprietary zero-day datasets.</p>
                    </div>

                    <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
                        {[
                            { name: "SQL Injection", cvss: "9.8", code: "SELECT * FROM users WHERE id = " + "{input}" },
                            { name: "XSS (Cross-Site Scripting)", cvss: "6.1", code: "<div dangerouslySetInnerHTML={{ __html: input }} />" },
                            { name: "Broken Access Control", cvss: "8.6", code: "if (user.id === params.id) { ... }" },
                            { name: "Sensitive Data Exposure", cvss: "7.5", code: "console.log(process.env.STRIPE_KEY)" },
                            { name: "SSRF", cvss: "9.1", code: "fetch(req.query.url)" },
                            { name: "Insecure Deserialization", cvss: "9.8", code: "pickle.load(user_input)" },
                        ].map((vuln) => (
                            <div key={vuln.name} className="group border border-border rounded-xl bg-white p-8 hover:border-primary/50 transition-all duration-300 hover:shadow-lg relative overflow-hidden">
                                <div className="absolute top-0 right-0 w-24 h-24 bg-gradient-to-br from-primary/5 to-transparent rounded-bl-full -mr-4 -mt-4 transition-transform group-hover:scale-150" />

                                <div className="flex items-center justify-between mb-6 relative">
                                    <h3 className="font-bold text-lg text-foreground">{vuln.name}</h3>
                                    <span className="font-mono text-xs font-bold text-muted-foreground bg-secondary px-2 py-1 rounded">CVSS {vuln.cvss}</span>
                                </div>
                                <div className="bg-gray-950 rounded-lg p-4 relative overflow-hidden shadow-inner border border-gray-800">
                                    <div className="flex gap-1.5 mb-3 opacity-50">
                                        <div className="w-2.5 h-2.5 rounded-full bg-red-500" />
                                        <div className="w-2.5 h-2.5 rounded-full bg-yellow-500" />
                                        <div className="w-2.5 h-2.5 rounded-full bg-green-500" />
                                    </div>
                                    <code className="text-xs font-mono text-gray-400 block overflow-x-auto pb-1">
                                        {vuln.code}
                                    </code>
                                </div>
                            </div>
                        ))}
                    </div>
                </section>

                {/* Compliance Table - Refined */}
                <section className="py-24 border-t border-border bg-[#F9F9F7]">
                    <div className="container mx-auto px-6">
                        <div className="grid lg:grid-cols-4 gap-12">
                            <div className="lg:col-span-1">
                                <h2 className="font-display text-4xl font-bold mb-6">Compliance on Autopilot</h2>
                                <p className="text-muted-foreground text-base leading-relaxed">
                                    Don't manually map code changes to controls. Sentinel tags every PR with the compliance standards it impacts.
                                </p>
                            </div>
                            <div className="lg:col-span-3">
                                <div className="overflow-hidden rounded-2xl border border-border shadow-sm bg-white">
                                    <table className="w-full text-sm text-left">
                                        <thead className="bg-secondary/50 text-muted-foreground font-mono text-xs uppercase tracking-wider">
                                            <tr>
                                                <th className="px-8 py-5 font-bold">Standard</th>
                                                <th className="px-8 py-5 font-bold">Control ID</th>
                                                <th className="px-8 py-5 font-bold">Sentinel Action</th>
                                            </tr>
                                        </thead>
                                        <tbody className="divide-y divide-border">
                                            {[
                                                { std: "SOC2", id: "CC6.1", action: "Automatic vulnerability scanning on every commit." },
                                                { std: "SOC2", id: "CC6.8", action: "Prevents unauthorized code execution & injection." },
                                                { std: "HIPAA", id: "164.312(a)(1)", action: "Detects unencrypted PHI logging." },
                                                { std: "ISO 27001", id: "A.14.2.1", action: "Ensures secure development policy enforcement." },
                                                { std: "GDPR", id: "Art. 32", action: "Identifies personal data exposure risks." },
                                            ].map((row, i) => (
                                                <tr key={i} className="hover:bg-gray-50 transition-colors group">
                                                    <td className="px-8 py-5 font-bold text-foreground group-hover:text-primary transition-colors">{row.std}</td>
                                                    <td className="px-8 py-5 font-mono text-muted-foreground text-xs">{row.id}</td>
                                                    <td className="px-8 py-5 text-gray-600">{row.action}</td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        </div>
                    </div>
                </section>

                {/* Threat Monitor Section */}
                <section className="container mx-auto px-6 py-24">
                    <div className="mb-10 text-center">
                        <div className="inline-flex items-center gap-2 mb-4 text-emerald-600 font-bold text-xs uppercase tracking-widest border border-emerald-100 bg-emerald-50 px-3 py-1 rounded-full">
                            <Database className="w-3 h-3" />
                            Live Intelligence
                        </div>
                        <h2 className="font-display text-4xl font-bold">Global Threat Stream</h2>
                    </div>
                    <ThreatMonitor />
                </section>

            </main>
            <Footer />
        </div>
    );
};

export default Security;
