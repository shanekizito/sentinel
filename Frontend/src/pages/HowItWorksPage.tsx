import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import CTA from "@/components/CTA";
import { Download, FileJson, BrainCircuit, Wand2, ShieldCheck, GitMerge, ChevronDown } from "lucide-react";
import { useState } from "react";

// Architectural Code Splitter
const CodeCompare = () => {
    return (
        <div className="border-2 border-gray-900 bg-[#0a0a0c] font-mono text-xs relative">
            <div className="absolute top-0 left-1/2 -ml-px w-[2px] h-full bg-gray-800 z-10 hidden md:block" />

            <div className="grid md:grid-cols-2">
                {/* Vulnerable Side */}
                <div className="p-6 border-b-2 md:border-b-0 border-gray-800 relative">
                    <div className="flex items-center justify-between mb-4 pb-2 border-b border-gray-800">
                        <span className="text-red-500 font-bold tracking-wider">VULNERABLE</span>
                        <span className="text-gray-500">app.ts</span>
                    </div>
                    <div className="space-y-2 text-gray-500">
                        <p>app.get('/user', (req, res) ={">"} {"{"}</p>
                        <div className="bg-red-900/10 border-l-2 border-red-500 pl-3 py-1 -ml-3 text-red-400">
                            const query = "SELECT * FROM users WHERE id = " + req.query.id;
                        </div>
                        <p className="pl-4">db.exec(query, (err, rows) ={">"} {"{"}</p>
                        <p className="pl-8">res.json(rows);</p>
                        <p className="pl-4">{"}"});</p>
                        <p>{"}"});</p>
                    </div>
                </div>

                {/* Secure Side */}
                <div className="p-6 bg-[#0a0a0c] relative">
                    <div className="flex items-center justify-between mb-4 pb-2 border-b border-gray-800">
                        <span className="text-primary font-bold tracking-wider">SECURE</span>
                        <span className="text-gray-500">app_fixed.ts</span>
                    </div>
                    <div className="space-y-2 text-gray-500">
                        <p>app.get('/user', (req, res) ={">"} {"{"}</p>
                        <div className="bg-primary/10 border-l-2 border-primary pl-3 py-1 -ml-3 text-primary">
                            const query = "SELECT * FROM users WHERE id = ?";
                        </div>
                        <div className="bg-primary/10 border-l-2 border-primary pl-3 py-1 -ml-3 text-primary">
                            db.exec(query, [req.query.id], (err, rows) ={">"} {"{"}
                        </div>
                        <p className="pl-8">res.json(rows);</p>
                        <p className="pl-4">{"}"});</p>
                        <p>{"}"});</p>
                    </div>
                </div>
            </div>
        </div >
    );
};

const HowItWorksPage = () => {
    const [openFaq, setOpenFaq] = useState<number | null>(null);

    const toggleFaq = (index: number) => {
        setOpenFaq(openFaq === index ? null : index);
    };

    return (
        <div className="min-h-screen bg-white flex flex-col font-sans">
            <Navbar />

            <main className="flex-grow pt-32">
                {/* Hero */}
                <section className="container mx-auto px-6 mb-24 relative">
                    {/* Structural Grid */}
                    <div className="absolute top-0 right-6 w-32 h-32 border-t-2 border-r-2 border-gray-200" />

                    <div className="max-w-4xl">
                        <div className="inline-flex items-center gap-3 px-1 py-1 border-l-4 border-primary pl-4 mb-8">
                            <span className="text-xs font-mono uppercase tracking-[0.2em] text-gray-500">
                                Process Breakdown
                            </span>
                        </div>
                        <h1 className="font-display text-6xl md:text-7xl font-bold mb-8 text-gray-900 tracking-tight leading-[0.95]">
                            The Lifecycle of a Fix
                        </h1>
                        <p className="text-xl text-gray-600 leading-relaxed font-light max-w-2xl border-l-2 border-gray-100 pl-6">
                            From the moment you push code to the final merge, see exactly what happens inside Sentinel's brain.
                        </p>
                    </div>
                </section>

                {/* Vertical Step-by-Step */}
                <section className="container mx-auto px-6 mb-32 max-w-5xl">
                    <div className="relative border-l-2 border-gray-200 ml-4 md:ml-0 space-y-px">
                        {[
                            {
                                step: "01",
                                title: "Ingestion & Context Mapping",
                                desc: "Sentinel connects to your repo (GitHub/GitLab) and clones the delta. It builds a dependency graph to understand not just the changed file, but everything importing it.",
                                icon: <Download className="w-5 h-5" />,
                                tech: "Git Hooks / AST Parser"
                            },
                            {
                                step: "02",
                                title: "Static Analysis (SAST)",
                                desc: "The engine runs deterministic checks against 6,000+ known patterns (Regex, Semgrep rules). This catches low-hanging fruit like hardcoded keys and basic injection flaws instantly.",
                                icon: <FileJson className="w-5 h-5" />,
                                tech: "Signatures / RegEx"
                            },
                            {
                                step: "03",
                                title: "AI Semantic Review",
                                desc: "For logic that passes SAST, our fine-tuned LLM analyzes the AST for intent. It asks: 'Does this user ID check match the session ID?' identifying IDOR and complex logic bugs.",
                                icon: <BrainCircuit className="w-5 h-5" />,
                                tech: "Transformer Models"
                            },
                            {
                                step: "04",
                                title: "Synthesis & Patching",
                                desc: "Instead of just flagging the line, Sentinel rewrites it. It generates a secure implementation, preserving your coding style (tabs/spaces, variable naming conventions).",
                                icon: <Wand2 className="w-5 h-5" />,
                                tech: "Generative AI",
                                extra: <CodeCompare />
                            },
                            {
                                step: "05",
                                title: "Verification Testing",
                                desc: "Sentinel spins up a sandboxed runner to compile the patched code and run your existing unit tests. If tests fail, it self-corrects and retries the patch.",
                                icon: <ShieldCheck className="w-5 h-5" />,
                                tech: "Sandboxed Runner"
                            },
                            {
                                step: "06",
                                title: "Pull Request Generation",
                                desc: "A clean PR is opened with a description of the vulnerability, the fix, and proof of verification. You get a notification to review and merge.",
                                icon: <GitMerge className="w-5 h-5" />,
                                tech: "GitHub API"
                            }
                        ].map((item, index) => (
                            <div key={item.step} className="relative md:pl-12 group">
                                {/* Node Marker */}
                                <div className="absolute -left-[9px] top-8 w-4 h-4 bg-white border-2 border-gray-900 group-hover:bg-primary group-hover:border-primary transition-colors z-10" />

                                <div className="flex flex-col md:flex-row gap-8 items-start border-2 border-transparent hover:border-gray-900 p-8 transition-all hover:bg-gray-50">
                                    {/* Icon Box */}
                                    <div className="w-12 h-12 border-2 border-gray-900 bg-white flex items-center justify-center shrink-0 group-hover:bg-primary group-hover:border-primary transition-colors text-gray-900 group-hover:text-white">
                                        {item.icon}
                                    </div>

                                    {/* Content */}
                                    <div className="flex-1">
                                        <div className="flex justify-between items-start mb-2">
                                            <span className="text-xs font-bold text-primary uppercase tracking-widest">Step {item.step}</span>
                                            <span className="text-[10px] font-mono text-gray-400 uppercase tracking-widest border border-gray-200 px-2 py-1">{item.tech}</span>
                                        </div>
                                        <h3 className="text-2xl font-bold text-gray-900 mb-4 font-display">{item.title}</h3>
                                        <p className="text-gray-600 leading-relaxed text-lg mb-6">{item.desc}</p>

                                        {/* Inset Content (Code Compare) */}
                                        {item.extra && (
                                            <div className="mt-8 pt-8">
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
                <section className="bg-gray-50 py-24 border-t-2 border-gray-900 relative">
                    {/* Grid Lines */}
                    <div className="absolute inset-0 pointer-events-none opacity-50">
                        <div className="absolute left-1/4 top-0 bottom-0 w-px bg-gray-200" />
                        <div className="absolute right-1/4 top-0 bottom-0 w-px bg-gray-200" />
                    </div>

                    <div className="container mx-auto px-6 max-w-4xl relative z-10">
                        <h2 className="font-display text-4xl font-bold text-gray-900 mb-16 border-l-4 border-gray-900 pl-6">Technical FAQ</h2>

                        <div className="space-y-px bg-gray-900 border-2 border-gray-900">
                            {[
                                { q: "Does Sentinel send my source code to OpenAI?", a: "No. Sentinel uses a locally-hosted or private-cloud inference model by default. For our SaaS offering, code snippets are processed in memory and never stored for training." },
                                { q: "What happens if a patch breaks the build?", a: "The Verification Agent runs your test suite ('npm test', 'make test'). If a regression is detected, the patch is discarded and a 'Human Review Needed' flag is raised on the PR." },
                                { q: "How long does a typical scan take?", a: "Deterministic scans take milliseconds. AI-contextual scans typically take 15-30 seconds per file changed. The average end-to-end webhook latency is 45 seconds." },
                                { q: "Can I define custom security policies?", a: "Yes. You can supply a `sentinel.policy.yaml` file to enforce specific organizational rules (e.g., 'No AWS keys in environment variables', 'All public endpoints must have rate limiting')." },
                            ].map((item, i) => (
                                <div key={i} className="bg-white group">
                                    <button
                                        onClick={() => toggleFaq(i)}
                                        className="w-full flex items-center justify-between p-6 text-left font-bold text-gray-900 hover:bg-gray-50 transition-colors"
                                    >
                                        <span className="text-lg">{item.q}</span>
                                        <ChevronDown className={`w-5 h-5 text-primary transition-transform ${openFaq === i ? "rotate-180" : ""}`} />
                                    </button>
                                    {openFaq === i && (
                                        <div className="p-6 pt-0 text-gray-600 leading-relaxed border-t border-gray-100 bg-gray-50 font-light">
                                            {item.a}
                                        </div>
                                    )}
                                </div>
                            ))}
                        </div>
                    </div>
                </section>

                <CTA />
            </main>
            <Footer />
        </div>
    );
};

export default HowItWorksPage;
