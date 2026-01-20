import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import CTA from "@/components/CTA";
import ThreatMonitor from "@/components/ThreatMonitor";
import { Shield, Search, FileCode, Server, Database, CheckCircle2, Terminal, EyeOff, FileKey, Fingerprint, Lock, ArrowRight } from "lucide-react";

const Security = () => {
    return (
        <div className="min-h-screen bg-white flex flex-col font-sans">
            <Navbar />

            <main className="flex-grow pt-32">
                {/* Hero Section */}
                <section className="container mx-auto px-6 mb-24 relative">
                    <div className="absolute top-0 right-0 w-64 h-64 border-r-2 border-t-2 border-gray-100 -mr-6 -mt-6" />

                    <div className="max-w-5xl">
                        <div className="inline-flex items-center gap-3 px-1 py-1 border-l-4 border-primary pl-4 mb-8">
                            <span className="text-xs font-mono uppercase tracking-[0.2em] text-gray-500">
                                Defense Architecture
                            </span>
                        </div>
                        <h1 className="font-display text-6xl md:text-7xl font-bold mb-8 text-gray-900 tracking-tight leading-[0.95]">
                            Deterministic Precision. <br />
                            <span className="text-primary">GenAI Adaptability.</span>
                        </h1>
                        <p className="text-xl text-gray-600 max-w-2xl leading-relaxed text-balance font-light border-l-2 border-gray-200 pl-6">
                            Sentinel combines static analysis (AST) for 100% known-pattern detection with a fine-tuned LLM for context-aware logic vulnerabilities.
                        </p>
                    </div>
                </section>

                {/* Architecture Diagram - Blueprint Style */}
                <section className="bg-gray-50 border-y-2 border-gray-900 py-24 mb-24 relative overflow-hidden">
                    {/* Grid Background */}
                    <div className="absolute inset-0 bg-[linear-gradient(to_right,rgba(0,0,0,0.05)_1px,transparent_1px),linear-gradient(to_bottom,rgba(0,0,0,0.05)_1px,transparent_1px)] bg-[size:2rem_2rem]" />

                    <div className="container mx-auto px-6 relative z-10">
                        <div className="grid lg:grid-cols-2 gap-20 items-center">
                            <div className="space-y-12">
                                <h2 className="font-display text-4xl font-bold tracking-tight text-gray-900 border-l-4 border-gray-900 pl-6">How the Engine Works</h2>
                                <div className="space-y-px bg-gray-900 border-2 border-gray-900">
                                    {[
                                        {
                                            title: "1. AST Parsing",
                                            desc: "Code is converted into an Abstract Syntax Tree. Sentinel maps data flow paths to identify tainted inputs reaching sinks.",
                                            icon: <FileCode className="w-5 h-5" />
                                        },
                                        {
                                            title: "2. Heuristic Scans",
                                            desc: "6,000+ deterministic rules check for hardcoded secrets, misconfigurations, and known CVE patterns instantly.",
                                            icon: <Search className="w-5 h-5" />
                                        },
                                        {
                                            title: "3. AI Contextualization",
                                            desc: "For complex logic (e.g., 'Is this user authorized?'), our varying-parameter LLM analyzes function intent and scope.",
                                            icon: <Server className="w-5 h-5" />
                                        }
                                    ].map((step) => (
                                        <div key={step.title} className="flex gap-6 p-8 bg-white group hover:bg-gray-50 transition-colors">
                                            <div className="w-10 h-10 border-2 border-gray-900 flex items-center justify-center shrink-0 group-hover:bg-primary group-hover:border-primary group-hover:text-white transition-colors">
                                                {step.icon}
                                            </div>
                                            <div>
                                                <h3 className="font-bold text-lg text-gray-900 mb-2 font-display">{step.title}</h3>
                                                <p className="text-sm text-gray-600 leading-relaxed">{step.desc}</p>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>

                            {/* Visual Representation of Engine - New Architectural Look */}
                            <div className="relative bg-white border-2 border-gray-900 p-8">
                                <div className="absolute -top-2 -left-2 w-8 h-8 border-t-4 border-l-4 border-primary" />
                                <div className="absolute -bottom-2 -right-2 w-8 h-8 border-b-4 border-r-4 border-primary" />

                                <div className="space-y-8 relative z-10">
                                    <div className="flex justify-center">
                                        <div className="bg-gray-900 text-white px-8 py-3 font-mono text-sm flex items-center gap-3 w-full justify-center border-2 border-gray-900">
                                            <Terminal className="w-4 h-4 text-gray-400" />
                                            Source Code Input
                                        </div>
                                    </div>

                                    <div className="flex justify-center">
                                        <div className="h-8 w-px bg-gray-300" />
                                    </div>

                                    <div className="grid grid-cols-2 gap-4">
                                        <div className="p-6 bg-gray-50 border-2 border-gray-200 text-center">
                                            <div className="font-bold text-gray-900 mb-2">Static Analysis</div>
                                            <div className="text-xs text-gray-500 font-mono uppercase tracking-wider">AST / RegEx</div>
                                        </div>
                                        <div className="p-6 bg-gray-50 border-2 border-gray-200 text-center">
                                            <div className="font-bold text-gray-900 mb-2">AI Reasoning</div>
                                            <div className="text-xs text-gray-500 font-mono uppercase tracking-wider">LLM / Context</div>
                                        </div>
                                    </div>

                                    <div className="flex justify-center">
                                        <div className="h-8 w-px bg-gray-300" />
                                    </div>

                                    <div className="flex justify-center">
                                        <div className="bg-primary text-white px-8 py-4 font-bold flex items-center gap-3 w-full justify-center">
                                            <CheckCircle2 className="w-5 h-5 text-white" />
                                            Verified Patch
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </section>

                {/* Privacy By Design Section - Structural Dark Mode */}
                <section className="container mx-auto px-6 mb-32">
                    <div className="bg-[#0a0a0c] border-2 border-gray-900 p-12 md:p-16 text-white relative overflow-hidden">
                        {/* Inner Grid */}
                        <div className="absolute inset-0 bg-[linear-gradient(to_right,rgba(255,255,255,0.05)_1px,transparent_1px),linear-gradient(to_bottom,rgba(255,255,255,0.05)_1px,transparent_1px)] bg-[size:4rem_4rem]" />

                        <div className="relative z-10">
                            <h2 className="font-display text-4xl font-bold mb-12 flex items-center gap-4 border-l-4 border-primary pl-6">
                                <Lock className="w-8 h-8 text-primary" />
                                Privacy by Design
                            </h2>

                            <div className="grid md:grid-cols-3 gap-px bg-gray-800 border border-gray-800">
                                <div className="space-y-4 bg-[#0a0a0c] p-8">
                                    <div className="w-12 h-12 border border-gray-700 flex items-center justify-center mb-4 text-primary">
                                        <EyeOff className="w-6 h-6" />
                                    </div>
                                    <h3 className="font-bold text-xl text-white">Zero Retention</h3>
                                    <p className="text-gray-400 leading-relaxed text-sm">
                                        Code snippets sent to the LLM are ephemeral. Processed in memory and immediately discarded. Never trained on your data.
                                    </p>
                                </div>
                                <div className="space-y-4 bg-[#0a0a0c] p-8">
                                    <div className="w-12 h-12 border border-gray-700 flex items-center justify-center mb-4 text-primary">
                                        <FileKey className="w-6 h-6" />
                                    </div>
                                    <h3 className="font-bold text-xl text-white">PII Redaction</h3>
                                    <p className="text-gray-400 leading-relaxed text-sm">
                                        Before analysis, all potential PII (emails, IPs, keys) is locally redacted and replaced with synthetic tokens.
                                    </p>
                                </div>
                                <div className="space-y-4 bg-[#0a0a0c] p-8">
                                    <div className="w-12 h-12 border border-gray-700 flex items-center justify-center mb-4 text-primary">
                                        <Fingerprint className="w-6 h-6" />
                                    </div>
                                    <h3 className="font-bold text-xl text-white">SOC2 Type II</h3>
                                    <p className="text-gray-400 leading-relaxed text-sm">
                                        Our infrastructure is audited annually. We encrypt all data at rest (AES-256) and in transit (TLS 1.3).
                                    </p>
                                </div>
                            </div>
                        </div>
                    </div>
                </section>

                {/* Attack Vector Matrix - Technical */}
                <section className="container mx-auto px-6 mb-32">
                    <div className="max-w-3xl mb-12 border-l-4 border-gray-900 pl-6">
                        <h2 className="font-display text-4xl font-bold text-gray-900 mb-2">Coverage Matrix</h2>
                        <p className="text-xl text-gray-500 font-light">Trained on the OWASP Top 10, CWE Top 25, and proprietary zero-day datasets.</p>
                    </div>

                    <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-px bg-gray-200 border-2 border-gray-200">
                        {[
                            { name: "SQL Injection", cvss: "9.8", code: "SELECT * FROM users WHERE id = {input}" },
                            { name: "XSS (Cross-Site Scripting)", cvss: "6.1", code: "<div dangerouslySetInnerHTML={{ __html: input }} />" },
                            { name: "Broken Access Control", cvss: "8.6", code: "if (user.id === params.id) { ... }" },
                            { name: "Sensitive Data Exposure", cvss: "7.5", code: "console.log(process.env.STRIPE_KEY)" },
                            { name: "SSRF", cvss: "9.1", code: "fetch(req.query.url)" },
                            { name: "Insecure Deserialization", cvss: "9.8", code: "pickle.load(user_input)" },
                        ].map((vuln) => (
                            <div key={vuln.name} className="group bg-white p-8 relative hover:bg-gray-50 transition-colors">
                                <div className="flex items-center justify-between mb-6">
                                    <h3 className="font-bold text-lg text-gray-900">{vuln.name}</h3>
                                    <span className="font-mono text-xs font-bold text-primary border border-primary/20 bg-primary/5 px-2 py-1">CVSS {vuln.cvss}</span>
                                </div>
                                <div className="bg-[#0a0a0c] p-4 border border-gray-900">
                                    <div className="flex gap-1.5 mb-3 opacity-30">
                                        <div className="w-2 h-2 rounded-full bg-white" />
                                        <div className="w-2 h-2 rounded-full bg-white" />
                                        <div className="w-2 h-2 rounded-full bg-white" />
                                    </div>
                                    <code className="text-xs font-mono text-gray-400 block overflow-x-auto pb-1">
                                        {vuln.code}
                                    </code>
                                </div>
                            </div>
                        ))}
                    </div>
                </section>

                <CTA />
            </main>
            <Footer />
        </div>
    );
};

export default Security;
