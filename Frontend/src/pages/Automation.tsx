import { Link } from "react-router-dom";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import CTA from "@/components/CTA";
import { GitBranch, Terminal, Server, Shield, Laptop, Workflow, Check, Layers, Code2, Cloud, Command, Box, Cpu, ArrowRight } from "lucide-react";

const Automation = () => {
    return (
        <div className="min-h-screen bg-white flex flex-col font-sans">
            <Navbar />

            <main className="flex-grow pt-32">
                {/* Hero */}
                <section className="container mx-auto px-6 mb-24 relative">
                    {/* Structural Grid */}
                    <div className="absolute top-0 right-6 w-32 h-32 border-r-2 border-t-2 border-gray-100" />

                    <div className="max-w-5xl">
                        <div className="inline-flex items-center gap-3 px-1 py-1 border-l-4 border-primary pl-4 mb-8">
                            <span className="text-xs font-mono uppercase tracking-[0.2em] text-gray-500">
                                Zero-Touch Integration
                            </span>
                        </div>
                        <h1 className="font-display text-6xl md:text-7xl font-bold mb-8 text-gray-900 tracking-tight leading-[0.95]">
                            Security That Fits <br />
                            <span className="text-primary">Your Workflow.</span>
                        </h1>
                        <p className="text-xl text-gray-600 max-w-2xl leading-relaxed text-balance font-light border-l-2 border-gray-200 pl-6">
                            From local git hooks to air-gapped server clusters, Sentinel runs wherever your code lives. No dashboard switching required.
                        </p>
                    </div>
                </section>

                {/* CI/CD Pipeline Visualizer - Industrial Style */}
                <section className="bg-gray-50 border-y-2 border-gray-900 py-24 mb-24 relative overflow-hidden">
                    <div className="absolute inset-0 bg-[linear-gradient(to_right,rgba(0,0,0,0.05)_1px,transparent_1px),linear-gradient(to_bottom,rgba(0,0,0,0.05)_1px,transparent_1px)] bg-[size:2rem_2rem]" />

                    <div className="container mx-auto px-6 relative z-10">
                        <div className="mb-20">
                            <h2 className="font-display text-4xl font-bold text-gray-900 border-l-4 border-gray-900 pl-6 mb-2">The Automated Lifecycle</h2>
                            <p className="text-gray-600 text-lg pl-7">How a single commit triggers a global defense chain.</p>
                        </div>

                        {/* Pipeline Steps */}
                        <div className="relative max-w-6xl mx-auto">
                            {/* Connector Line */}
                            <div className="hidden md:block absolute top-[28px] left-10 right-10 h-0.5 bg-gray-300 -z-10" />

                            <div className="grid md:grid-cols-4 gap-8">
                                {[
                                    { title: "Commit", icon: <GitBranch className="w-6 h-6" />, desc: "Developer pushes code to feature branch." },
                                    { title: "Scan", icon: <Terminal className="w-6 h-6" />, desc: "Sentinel Action runs analysis in < 20s.", badge: "20ms" },
                                    { title: "Remediate", icon: <Shield className="w-6 h-6" />, desc: "Vulnerabilities blocked. Fix PR created." },
                                    { title: "Merge", icon: <Check className="w-6 h-6" />, desc: "Clean code merged to main branch." }
                                ].map((step, i) => (
                                    <div key={step.title} className="group flex flex-col items-center text-center bg-white md:bg-transparent p-6 md:p-0 border-2 md:border-none border-gray-200 relative">
                                        <div className="w-14 h-14 bg-white border-2 border-gray-900 flex items-center justify-center mb-6 z-10 relative group-hover:bg-primary group-hover:border-primary group-hover:text-white transition-colors text-gray-900">
                                            {step.icon}
                                            {/* Badge for time */}
                                            {step.badge && <span className="absolute -top-3 -right-3 bg-primary text-white text-[10px] font-bold px-2 py-0.5 border border-primary text-xs font-mono uppercase tracking-wider">{step.badge}</span>}
                                        </div>
                                        <h3 className="font-bold text-gray-900 text-xl mb-3 font-display">{step.title}</h3>
                                        <p className="text-sm text-gray-600 leading-relaxed max-w-[220px]">{step.desc}</p>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>
                </section>

                {/* Ecosystem Grid - Architectural */}
                <section className="container mx-auto px-6 mb-32">
                    <div className="border-2 border-gray-900 p-12 lg:p-16 relative">
                        {/* Corner Accents */}
                        <div className="absolute -top-2 -left-2 w-8 h-8 border-t-4 border-l-4 border-primary bg-white" />
                        <div className="absolute -bottom-2 -right-2 w-8 h-8 border-b-4 border-r-4 border-primary bg-white" />

                        <div className="grid lg:grid-cols-3 gap-16 items-center">
                            <div className="lg:col-span-1">
                                <h2 className="font-display text-4xl font-bold mb-6 text-gray-900">Works with Everything</h2>
                                <p className="text-lg text-gray-600 leading-relaxed mb-8">
                                    We support 30+ languages and frameworks out of the box. Drop Sentinel into any environment.
                                </p>
                                <Link to="/docs" className="inline-flex h-12 px-6 border-2 border-gray-900 text-gray-900 font-bold uppercase tracking-wider hover:bg-gray-900 hover:text-white transition-colors items-center gap-2 text-sm w-fit">
                                    View Integration Docs <ArrowRight className="w-4 h-4" />
                                </Link>
                            </div>

                            <div className="lg:col-span-2">
                                <div className="grid grid-cols-2 md:grid-cols-4 gap-px bg-gray-200 border-2 border-gray-200">
                                    {[
                                        { name: "GitHub", icon: <Code2 className="w-6 h-6" /> },
                                        { name: "GitLab", icon: <GitBranch className="w-6 h-6" /> },
                                        { name: "AWS", icon: <Cloud className="w-6 h-6" /> },
                                        { name: "Docker", icon: <Box className="w-6 h-6" /> },
                                        { name: "Kubernetes", icon: <Layers className="w-6 h-6" /> },
                                        { name: "Jenkins", icon: <Command className="w-6 h-6" /> },
                                        { name: "Jira", icon: <Layers className="w-6 h-6" /> },
                                        { name: "Slack", icon: <Terminal className="w-6 h-6" /> },
                                    ].map((item) => (
                                        <div key={item.name} className="flex flex-col items-center justify-center p-8 bg-white hover:bg-gray-50 transition-colors group">
                                            <div className="text-gray-400 mb-3 group-hover:text-primary transition-colors">{item.icon}</div>
                                            <span className="font-bold text-sm text-gray-900">{item.name}</span>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </div>
                    </div>
                </section>

                {/* Deployment Models */}
                <section className="container mx-auto px-6 mb-24">
                    <div className="grid lg:grid-cols-2 gap-16 lg:gap-24">
                        {/* Card 1: Cloud & IDE */}
                        <div className="space-y-8">
                            <h2 className="font-display text-3xl font-bold text-gray-900">Developer Experience</h2>
                            <p className="text-gray-600 text-lg">Catch bugs before they even leave your machine. Real-time feedback loops.</p>

                            <div className="bg-[#0a0a0c] border-2 border-gray-900 p-1 relative">
                                <div className="flex items-center px-4 py-2 border-b border-gray-800 bg-gray-900/50">
                                    <div className="flex gap-1.5 mr-4 opacity-50">
                                        <div className="w-2.5 h-2.5 rounded-full bg-red-500" />
                                        <div className="w-2.5 h-2.5 rounded-full bg-yellow-500" />
                                        <div className="w-2.5 h-2.5 rounded-full bg-primary" />
                                    </div>
                                    <span className="text-xs text-gray-400 font-mono">user_controller.ts</span>
                                </div>
                                <div className="p-6 font-mono text-sm leading-relaxed text-gray-400">
                                    <div className="opacity-50">1  <span className="text-purple-400">const</span> <span className="text-blue-400">user</span> = <span className="text-purple-400">await</span> db.<span className="text-yellow-200">query</span>(</div>
                                    <div className="relative pl-4 border-l-2 border-red-500/30 my-2">
                                        <div className="text-red-400 bg-red-900/10 inline-block px-1">2    `SELECT * FROM users WHERE id = ${"{"}req.id{"}"}`</div>
                                    </div>
                                    <div className="opacity-50">3  );</div>

                                    <div className="mt-4 p-3 border border-red-500/30 bg-red-900/10 text-red-300 text-xs">
                                        <div className="font-bold flex items-center gap-2 mb-1">
                                            <Shield className="w-3 h-3" /> SQL Injection Detected
                                        </div>
                                        Use parameterized queries to prevent injection.
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* Card 2: Enterprise & Air Gap */}
                        <div className="space-y-8">
                            <h2 className="font-display text-3xl font-bold text-gray-900">Enterprise Mode</h2>
                            <p className="text-gray-600 text-lg">For banking, defense, and healthcare environments requiring total isolation.</p>

                            <div className="space-y-4">
                                <div className="p-8 bg-white border-2 border-gray-200 flex gap-6 items-start hover:border-gray-900 transition-colors group">
                                    <div className="w-12 h-12 bg-gray-100 flex items-center justify-center shrink-0 border border-gray-200 group-hover:bg-gray-900 group-hover:text-white transition-colors">
                                        <Server className="w-6 h-6" />
                                    </div>
                                    <div>
                                        <h3 className="font-bold text-xl text-gray-900 mb-2">Self-Hosted Container</h3>
                                        <p className="text-sm text-gray-600 mb-4 leading-relaxed">Run the entire Sentinel analysis engine as a Docker container within your VPC. No data ever leaves your network.</p>
                                        <code className="text-xs bg-gray-100 px-3 py-1.5 text-gray-900 font-mono border border-gray-200 block w-fit">docker run -d sentinel/enterprise:latest</code>
                                    </div>
                                </div>

                                <div className="p-8 bg-white border-2 border-gray-200 flex gap-6 items-start hover:border-gray-900 transition-colors group">
                                    <div className="w-12 h-12 bg-gray-100 flex items-center justify-center shrink-0 border border-gray-200 group-hover:bg-primary group-hover:text-white transition-colors">
                                        <Cpu className="w-6 h-6" />
                                    </div>
                                    <div>
                                        <h3 className="font-bold text-xl text-gray-900 mb-2">Air-Gapped Offline Mode</h3>
                                        <p className="text-sm text-gray-600 leading-relaxed">
                                            Fully bundled signature databases and local LLM weights. Requires zero internet connectivity to function. Updates via signed binaries.
                                        </p>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </section>

                <CTA />
            </main>
            <Footer />
        </div>
    );
};

export default Automation;
