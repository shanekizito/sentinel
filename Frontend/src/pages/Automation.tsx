import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import { GitBranch, Terminal, Server, Shield, Laptop, Workflow, Check, Layers, Code2, Cloud, Command, Box, Cpu } from "lucide-react";

const Automation = () => {
    return (
        <div className="min-h-screen bg-background flex flex-col font-sans selection:bg-blue-100 selection:text-blue-900">
            <Navbar />

            <main className="flex-grow pt-32">
                {/* Hero */}
                <section className="container mx-auto px-6 mb-24">
                    <div className="max-w-4xl">
                        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-50 border border-blue-200 text-xs font-bold text-blue-700 mb-8 tracking-wide uppercase">
                            <Workflow className="w-3.5 h-3.5" />
                            <span>Zero-Touch Integration</span>
                        </div>
                        <h1 className="font-display text-6xl md:text-7xl font-bold mb-8 text-foreground tracking-tighter leading-[0.9]">
                            Security That Fits <br />
                            <span className="text-muted-foreground/80">Your Workflow. Not The Other Way Around.</span>
                        </h1>
                        <p className="text-2xl text-muted-foreground max-w-2xl leading-relaxed text-balance font-light">
                            From local git hooks to air-gapped server clusters, Sentinel runs wherever your code lives. No dashboard switching required.
                        </p>
                    </div>
                </section>

                {/* CI/CD Pipeline Visualizer - Animated Polish */}
                <section className="bg-white border-y border-border py-24 mb-24 relative overflow-hidden">
                    <div className="absolute inset-0 bg-grid-isometric opacity-[0.03] pointer-events-none" />

                    {/* Running Beam Animation */}
                    <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-transparent via-blue-500 to-transparent opacity-20 animate-marquee" />

                    <div className="container mx-auto px-6 relative z-10">
                        <div className="text-center mb-20">
                            <h2 className="font-display text-4xl font-bold mb-4">The Automated Lifecycle</h2>
                            <p className="text-muted-foreground text-lg">How a single commit triggers a global defense chain.</p>
                        </div>

                        {/* Pipeline Steps */}
                        <div className="relative max-w-6xl mx-auto">
                            {/* Connector Line with Pulse */}
                            <div className="hidden md:block absolute top-[28px] left-10 right-10 h-0.5 bg-gray-100 -z-10 overflow-hidden">
                                <div className="absolute inset-0 bg-gradient-to-r from-transparent via-blue-500 to-transparent w-1/2 h-full animate-marquee opacity-50" />
                            </div>

                            <div className="grid md:grid-cols-4 gap-8">
                                {[
                                    { title: "Commit", icon: <GitBranch className="w-6 h-6 text-gray-700" />, desc: "Developer pushes code to feature branch." },
                                    { title: "Scan", icon: <Terminal className="w-6 h-6 text-blue-600" />, desc: "Sentinel Action runs analysis in < 20s.", badge: "20ms" },
                                    { title: "Remediate", icon: <Shield className="w-6 h-6 text-purple-600" />, desc: "Vulnerabilities blocked. Fix PR created." },
                                    { title: "Merge", icon: <Check className="w-6 h-6 text-emerald-600" />, desc: "Clean code merged to main branch." }
                                ].map((step, i) => (
                                    <div key={step.title} className="group flex flex-col items-center text-center bg-white md:bg-transparent p-6 md:p-0 rounded-xl border md:border-none border-border relative transition-transform hover:-translate-y-1">
                                        <div className="w-14 h-14 rounded-2xl bg-white border-2 border-border flex items-center justify-center mb-6 shadow-sm z-10 relative group-hover:border-blue-400 group-hover:shadow-lg transition-all duration-300">
                                            {step.icon}
                                            {/* Badge for time */}
                                            {step.badge && <span className="absolute -top-3 -right-3 bg-blue-600 text-white text-[10px] font-bold px-2 py-0.5 rounded-full shadow-md animate-pulse">{step.badge}</span>}
                                        </div>
                                        <h3 className="font-bold text-foreground text-xl mb-3">{step.title}</h3>
                                        <p className="text-sm text-muted-foreground leading-relaxed max-w-[220px]">{step.desc}</p>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>
                </section>

                {/* Ecosystem Grid - NEW */}
                <section className="container mx-auto px-6 mb-32">
                    <div className="bg-gray-50 border border-border rounded-3xl p-12 lg:p-16">
                        <div className="grid lg:grid-cols-3 gap-16 items-center">
                            <div className="lg:col-span-1">
                                <h2 className="font-display text-4xl font-bold mb-6">Works with Everything</h2>
                                <p className="text-lg text-muted-foreground leading-relaxed mb-8">
                                    We support 30+ languages and frameworks out of the box. Drop Sentinel into any environment.
                                </p>
                                <button className="btn-secondary bg-white">View Integration Docs</button>
                            </div>

                            <div className="lg:col-span-2">
                                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
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
                                        <div key={item.name} className="flex flex-col items-center justify-center p-6 bg-white border border-border rounded-xl hover:border-blue-300 hover:shadow-md transition-all cursor-default">
                                            <div className="text-gray-400 mb-3">{item.icon}</div>
                                            <span className="font-bold text-sm text-foreground">{item.name}</span>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </div>
                    </div>
                </section>

                {/* Deployment Models - Refined */}
                <section className="container mx-auto px-6 mb-24">
                    <div className="grid lg:grid-cols-2 gap-16 lg:gap-24">
                        {/* Card 1: Cloud & IDE */}
                        <div className="space-y-8">
                            <h2 className="font-display text-3xl font-bold">Developer Experience</h2>
                            <p className="text-muted-foreground text-lg">Catch bugs before they even leave your machine. Real-time feedback loops.</p>

                            <div className="bg-[#1e1e1e] rounded-xl overflow-hidden shadow-2xl border border-gray-800 ring-4 ring-gray-100">
                                {/* Fake VS Code Header */}
                                <div className="flex items-center px-4 py-3 bg-[#252526] border-b border-[#333]">
                                    <div className="flex gap-1.5 mr-4">
                                        <div className="w-3 h-3 rounded-full bg-[#FF5F56]" />
                                        <div className="w-3 h-3 rounded-full bg-[#FFBD2E]" />
                                        <div className="w-3 h-3 rounded-full bg-[#27C93F]" />
                                    </div>
                                    <span className="text-xs text-gray-400 font-medium">user_controller.ts</span>
                                </div>
                                <div className="p-6 font-mono text-sm leading-relaxed">
                                    <div className="text-gray-500">1  <span className="text-[#C586C0]">const</span> <span className="text-[#9CDCFE]">user</span> = <span className="text-[#C586C0]">await</span> db.<span className="text-[#DCDCAA]">query</span>(</div>
                                    <div className="text-gray-500 relative pl-4 border-l-2 border-transparent">
                                        2    <span className="text-[#CE9178]">{`\`SELECT * FROM users WHERE id = <span className="text-white bg-red-900/50 px-1 border border-red-500/50 rounded ring-2 ring-red-500/20">\${req.id}</span>\``}</span>
                                        {/* Tooltip */}
                                        <div className="absolute left-16 top-8 w-72 bg-[#252526] border border-red-500/40 p-4 rounded-lg shadow-2xl z-20">
                                            <div className="flex items-center gap-2 text-red-400 font-bold text-xs mb-2 uppercase tracking-wide">
                                                <Shield className="w-3 h-3" />
                                                SQL Injection Detected
                                            </div>
                                            <p className="text-xs text-gray-300 mb-3 leading-relaxed">Unsanitized input detected in raw SQL query. Use parameterized queries.</p>
                                            <button className="w-full bg-blue-600 hover:bg-blue-500 text-white text-xs font-bold py-2 rounded transition-colors">Apply Fix (Ctrl+.)</button>
                                        </div>
                                    </div>
                                    <div className="text-gray-500">3  );</div>
                                </div>
                            </div>

                            <div className="flex items-center gap-6 pt-4">
                                <div className="flex items-center gap-2 text-sm font-bold text-foreground">
                                    <Laptop className="w-4 h-4" />
                                    IDE Plugins
                                </div>
                                <div className="h-4 w-px bg-border" />
                                <div className="flex items-center gap-2 text-sm font-bold text-foreground">
                                    <Terminal className="w-4 h-4" />
                                    CLI Tool
                                </div>
                            </div>
                        </div>

                        {/* Card 2: Enterprise & Air Gap */}
                        <div className="space-y-8">
                            <h2 className="font-display text-3xl font-bold">Enterprise Mode</h2>
                            <p className="text-muted-foreground text-lg">For banking, defense, and healthcare environments requiring total isolation.</p>

                            <div className="space-y-4">
                                <div className="p-8 bg-white border border-border rounded-2xl flex gap-6 items-start hover:border-blue-300 transition-all shadow-sm">
                                    <div className="w-12 h-12 rounded-xl bg-blue-50 flex items-center justify-center shrink-0">
                                        <Server className="w-6 h-6 text-blue-600" />
                                    </div>
                                    <div>
                                        <h3 className="font-bold text-xl text-foreground mb-2">Self-Hosted Container</h3>
                                        <p className="text-sm text-muted-foreground mb-4 leading-relaxed">Run the entire Sentinel analysis engine as a Docker container within your VPC. No data ever leaves your network.</p>
                                        <code className="text-xs bg-gray-100 px-3 py-1.5 rounded-lg text-gray-700 font-mono border border-gray-200">docker run -d sentinel/enterprise:latest</code>
                                    </div>
                                </div>

                                <div className="p-8 bg-white border border-border rounded-2xl flex gap-6 items-start hover:border-emerald-300 transition-all shadow-sm">
                                    <div className="w-12 h-12 rounded-xl bg-emerald-50 flex items-center justify-center shrink-0">
                                        <Cpu className="w-6 h-6 text-emerald-600" />
                                    </div>
                                    <div>
                                        <h3 className="font-bold text-xl text-foreground mb-2">Air-Gapped Offline Mode</h3>
                                        <p className="text-sm text-muted-foreground leading-relaxed">
                                            Fully bundled signature databases and local LLM weights. Requires zero internet connectivity to function. Updates via signed binaries.
                                        </p>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </section>

            </main>
            <Footer />
        </div>
    );
};

export default Automation;
