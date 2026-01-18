import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import { Download, FileJson, BrainCircuit, Wand2, ShieldCheck, GitMerge, ChevronDown } from "lucide-react";
import { useState } from "react";

// Simple "Code Splitter" component for the page
const CodeCompare = () => {
    return (
        <div className="grid md:grid-cols-2 gap-4 bg-[#1e1e1e] rounded-xl border border-gray-800 overflow-hidden text-xs font-mono">
            <div className="p-4 border-b md:border-b-0 md:border-r border-gray-800">
                <div className="text-red-400 font-bold mb-3 flex items-center justify-between">
                    <span>BEFORE</span>
                    <span className="text-[10px] bg-red-900/30 px-2 py-0.5 rounded text-red-300">Vulnerable</span>
                </div>
                <div className="text-gray-400 space-y-1">
                    <p>app.get('/user', (req, res) ={">"} {"{"}</p>
                    <p className="pl-4 text-white bg-red-900/20 -mx-4 px-4 py-1 border-l-2 border-red-500">
                        const query = "SELECT * FROM users WHERE id = " + req.query.id;
                    </p>
                    <p className="pl-4">db.exec(query, (err, rows) ={">"} {"{"}</p>
                    <p className="pl-8">res.json(rows);</p>
                    <p className="pl-4">{"}"});</p>
                    <p>{"}"});</p>
                </div>
            </div>
            <div className="p-4 bg-[#1e1e1e]">
                <div className="text-emerald-400 font-bold mb-3 flex items-center justify-between">
                    <span>AFTER</span>
                    <span className="text-[10px] bg-emerald-900/30 px-2 py-0.5 rounded text-emerald-300">Patched</span>
                </div>
                <div className="text-gray-400 space-y-1">
                    <p>app.get('/user', (req, res) ={">"} {"{"}</p>
                    <p className="pl-4 text-white bg-emerald-900/20 -mx-4 px-4 py-1 border-l-2 border-emerald-500">
                        const query = "SELECT * FROM users WHERE id = ?";
                    </p>
                    <p className="pl-4 text-white bg-emerald-900/20 -mx-4 px-4 py-1 border-l-2 border-emerald-500">
                        db.exec(query, [req.query.id], (err, rows) ={">"} {"{"}</p>
                    <p className="pl-8">res.json(rows);</p>
                    <p className="pl-4">{"}"});</p>
                    <p>{"}"});</p>
                </div>
            </div>
        </div>
    );
};

const HowItWorksPage = () => {
    const [openFaq, setOpenFaq] = useState<number | null>(null);

    const toggleFaq = (index: number) => {
        setOpenFaq(openFaq === index ? null : index);
    };

    return (
        <div className="min-h-screen bg-background flex flex-col font-sans selection:bg-purple-100 selection:text-purple-900">
            <Navbar />

            <main className="flex-grow pt-32">
                {/* Hero */}
                <section className="container mx-auto px-6 mb-24 text-center">
                    <div className="max-w-3xl mx-auto">
                        <div className="inline-block px-3 py-1 rounded-full bg-purple-50 border border-purple-200 text-xs font-bold text-purple-700 mb-8 uppercase tracking-wide">
                            Process Breakdown
                        </div>
                        <h1 className="font-display text-6xl md:text-7xl font-bold mb-8 text-foreground tracking-tighter leading-[0.9]">
                            The Lifecycle of a Fix
                        </h1>
                        <p className="text-2xl text-muted-foreground leading-relaxed font-light text-balance">
                            From the moment you push code to the final merge, see exactly what happens inside Sentinel's brain.
                        </p>
                    </div>
                </section>

                {/* Vertical Step-by-Step */}
                <section className="container mx-auto px-6 mb-32 max-w-4xl">
                    <div className="relative border-l-2 border-border ml-4 md:ml-12 space-y-20">
                        {[
                            {
                                step: "01",
                                title: "Ingestion & Context Mapping",
                                desc: "Sentinel connects to your repo (GitHub/GitLab) and clones the delta. It builds a dependency graph to understand not just the changed file, but everything importing it.",
                                icon: <Download className="w-6 h-6 text-white" />,
                                color: "bg-blue-600",
                                tech: "Git Hooks / AST Parser"
                            },
                            {
                                step: "02",
                                title: "Static Analysis (SAST)",
                                desc: "The engine runs deterministic checks against 6,000+ known patterns (Regex, Semgrep rules). This catches low-hanging fruit like hardcoded keys and basic injection flaws instantly.",
                                icon: <FileJson className="w-6 h-6 text-white" />,
                                color: "bg-indigo-600",
                                tech: "Signatures / RegEx"
                            },
                            {
                                step: "03",
                                title: "AI Semantic Review",
                                desc: "For logic that passes SAST, our fine-tuned LLM analyzes the AST for intent. It asks: 'Does this user ID check match the session ID?' identifying IDOR and complex logic bugs.",
                                icon: <BrainCircuit className="w-6 h-6 text-white" />,
                                color: "bg-purple-600",
                                tech: "Transformer Models"
                            },
                            {
                                step: "04",
                                title: "Synthesis & Patching",
                                desc: "Instead of just flagging the line, Sentinel rewrites it. It generates a secure implementation, preserving your coding style (tabs/spaces, variable naming conventions).",
                                icon: <Wand2 className="w-6 h-6 text-white" />,
                                color: "bg-pink-600",
                                tech: "Generative AI",
                                extra: <CodeCompare />
                            },
                            {
                                step: "05",
                                title: "Verification Testing",
                                desc: "Sentinel spins up a sandboxed runner to compile the patched code and run your existing unit tests. If tests fail, it self-corrects and retries the patch.",
                                icon: <ShieldCheck className="w-6 h-6 text-white" />,
                                color: "bg-emerald-600",
                                tech: "Sandboxed Runner"
                            },
                            {
                                step: "06",
                                title: "Pull Request Generation",
                                desc: "A clean PR is opened with a description of the vulnerability, the fix, and proof of verification. You get a notification to review and merge.",
                                icon: <GitMerge className="w-6 h-6 text-white" />,
                                color: "bg-gray-800",
                                tech: "GitHub API"
                            }
                        ].map((item, index) => (
                            <div key={item.step} className="relative pl-8 md:pl-16">
                                {/* Node Marker */}
                                <div className={`absolute -left-[9px] top-0 w-4 h-4 rounded-full border-2 border-white ring-2 ${item.color.replace('bg-', 'ring-')} bg-white shadow-sm`}></div>

                                <div className="flex flex-col md:flex-row gap-8 items-start">
                                    {/* Icon Box */}
                                    <div className={`w-16 h-16 rounded-2xl ${item.color} flex items-center justify-center shrink-0 shadow-lg shadow-gray-200 ring-2 ring-white`}>
                                        {item.icon}
                                    </div>

                                    {/* Content */}
                                    <div className="flex-1 bg-white p-8 rounded-3xl border border-border shadow-sm hover:shadow-lg transition-shadow">
                                        <div className="flex justify-between items-start mb-4">
                                            <span className="text-xs font-bold text-muted-foreground uppercase tracking-wider">Step {item.step}</span>
                                            <span className="text-[10px] font-mono bg-secondary px-3 py-1 rounded-full text-foreground border border-border font-bold">{item.tech}</span>
                                        </div>
                                        <h3 className="text-2xl font-bold text-foreground mb-4">{item.title}</h3>
                                        <p className="text-muted-foreground leading-relaxed text-lg mb-6">{item.desc}</p>

                                        {/* Inset Content (Code Compare) */}
                                        {item.extra && (
                                            <div className="mt-8 pt-8 border-t border-border">
                                                {item.extra}
                                            </div>
                                        )}
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                </section>

                {/* Technical FAQ */}
                <section className="bg-[#F9F9F7] py-24 border-t border-border">
                    <div className="container mx-auto px-6 max-w-3xl">
                        <h2 className="font-display text-4xl font-bold text-center mb-16">Technical FAQ</h2>

                        <div className="space-y-4">
                            {[
                                { q: "Does Sentinel send my source code to OpenAI?", a: "No. Sentinel uses a locally-hosted or private-cloud inference model by default. For our SaaS offering, code snippets are processed in memory and never stored for training." },
                                { q: "What happens if a patch breaks the build?", a: "The Verification Agent runs your test suite ('npm test', 'make test'). If a regression is detected, the patch is discarded and a 'Human Review Needed' flag is raised on the PR." },
                                { q: "How long does a typical scan take?", a: "Deterministic scans take milliseconds. AI-contextual scans typically take 15-30 seconds per file changed. The average end-to-end webhook latency is 45 seconds." },
                                { q: "Can I define custom security policies?", a: "Yes. You can supply a `sentinel.policy.yaml` file to enforce specific organizational rules (e.g., 'No AWS keys in environment variables', 'All public endpoints must have rate limiting')." },
                            ].map((item, i) => (
                                <div key={i} className="bg-white border border-border rounded-xl overflow-hidden">
                                    <button
                                        onClick={() => toggleFaq(i)}
                                        className="w-full flex items-center justify-between p-6 text-left font-bold text-foreground hover:bg-gray-50 transition-colors"
                                    >
                                        {item.q}
                                        <ChevronDown className={`w-5 h-5 transition-transform ${openFaq === i ? "rotate-180" : ""}`} />
                                    </button>
                                    {openFaq === i && (
                                        <div className="p-6 pt-0 text-muted-foreground leading-relaxed border-t border-border/50 bg-gray-50/50">
                                            {item.a}
                                        </div>
                                    )}
                                </div>
                            ))}
                        </div>
                    </div>
                </section>

            </main>
            <Footer />
        </div>
    );
};

export default HowItWorksPage;
