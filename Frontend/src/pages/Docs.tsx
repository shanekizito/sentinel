import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import { FileCode, Terminal, Book, Code2, ChevronRight, Search } from "lucide-react";

const Docs = () => {
    return (
        <div className="min-h-screen bg-background flex flex-col">
            <Navbar />
            <main className="flex-grow pt-24 pb-20">
                <div className="container mx-auto px-6">
                    {/* Header */}
                    <div className="flex flex-col md:flex-row md:items-center justify-between gap-8 mb-16">
                        <div>
                            <h1 className="font-display text-4xl font-bold mb-4">Documentation</h1>
                            <p className="text-muted-foreground text-lg max-w-xl">
                                Integration guides, API references, and security policy configuration for Sentinel V2.
                            </p>
                        </div>
                        <div className="relative w-full md:w-96">
                            <input
                                type="text"
                                placeholder="Search documentation..."
                                className="w-full pl-10 pr-4 py-3 bg-white border border-border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary/20 transition-all font-mono text-sm"
                            />
                            <Search className="w-4 h-4 text-muted-foreground absolute left-3 top-1/2 -translate-y-1/2" />
                            <div className="absolute right-3 top-1/2 -translate-y-1/2 flex items-center gap-1">
                                <span className="text-[10px] bg-secondary px-1.5 py-0.5 rounded border border-border text-muted-foreground font-mono">CTRL</span>
                                <span className="text-[10px] bg-secondary px-1.5 py-0.5 rounded border border-border text-muted-foreground font-mono">K</span>
                            </div>
                        </div>
                    </div>

                    {/* Quick Start Grid */}
                    <div className="grid md:grid-cols-3 gap-6 mb-20">
                        {[
                            { title: "Quick Start", icon: <Terminal className="w-5 h-5" />, desc: "Deploy Sentinel in 5 minutes." },
                            { title: "API Reference", icon: <Code2 className="w-5 h-5" />, desc: "Complete REST and GraphQL API." },
                            { title: "Configuration", icon: <FileCode className="w-5 h-5" />, desc: "sentinel.config.js options." },
                        ].map((item, i) => (
                            <div key={i} className="group p-6 bg-white border border-border rounded-xl hover:border-primary/50 transition-all cursor-pointer hover:shadow-sm">
                                <div className="w-10 h-10 rounded-lg bg-primary/5 border border-primary/10 flex items-center justify-center text-primary mb-4 group-hover:bg-primary group-hover:text-white transition-colors">
                                    {item.icon}
                                </div>
                                <h3 className="font-bold text-foreground mb-2 flex items-center gap-2">
                                    {item.title}
                                    <ChevronRight className="w-4 h-4 opacity-0 group-hover:opacity-100 transition-opacity -translate-x-2 group-hover:translate-x-0" />
                                </h3>
                                <p className="text-sm text-muted-foreground">{item.desc}</p>
                            </div>
                        ))}
                    </div>

                    {/* Documentation Content */}
                    <div className="grid lg:grid-cols-4 gap-12">
                        {/* Sidebar */}
                        <div className="hidden lg:block space-y-8">
                            {[
                                { cat: "Getting Started", items: ["Installation", "Authentication", "First Scan"] },
                                { cat: "Core Concepts", items: ["Threat Model", "Policy Engine", "Reporting"] },
                                { cat: "Integrations", items: ["GitHub Actions", "GitLab CI", "Jenkins", "Jira"] },
                                { cat: "SDKs", items: ["Python", "Node.js", "Go"] },
                            ].map((section) => (
                                <div key={section.cat}>
                                    <h4 className="font-mono text-xs font-bold text-foreground uppercase tracking-widest mb-4">{section.cat}</h4>
                                    <ul className="space-y-3 border-l border-border pl-4">
                                        {section.items.map(item => (
                                            <li key={item} className="text-sm text-muted-foreground hover:text-primary cursor-pointer transition-colors block">
                                                {item}
                                            </li>
                                        ))}
                                    </ul>
                                </div>
                            ))}
                        </div>

                        {/* Main Content Preview */}
                        <div className="lg:col-span-3 space-y-12">
                            <div className="p-8 bg-white border border-border rounded-2xl">
                                <div className="flex items-center gap-2 mb-6 text-sm text-muted-foreground">
                                    <span>Docs</span>
                                    <ChevronRight className="w-3 h-3" />
                                    <span>Getting Started</span>
                                    <ChevronRight className="w-3 h-3" />
                                    <span className="text-foreground font-medium">Installation</span>
                                </div>

                                <h2 className="text-3xl font-display font-bold mb-6">Installation</h2>
                                <p className="text-lg text-muted-foreground leading-relaxed mb-8">
                                    Sentinel acts as a standalone binary or a containerized agent.
                                    The easiest way to get started is via our CLI installer.
                                </p>

                                <div className="bg-[#0f1115] rounded-lg border border-border/50 p-6 font-mono text-sm text-gray-300 relative group overflow-hidden">
                                    <div className="absolute top-3 right-3 opacity-0 group-hover:opacity-100 transition-opacity">
                                        <button className="text-xs bg-white/10 hover:bg-white/20 px-2 py-1 rounded text-white">Copy</button>
                                    </div>
                                    <div className="flex items-center gap-2 border-b border-white/10 pb-4 mb-4 text-xs text-gray-500">
                                        <div className="flex gap-1.5">
                                            <div className="w-2.5 h-2.5 rounded-full bg-red-500/50" />
                                            <div className="w-2.5 h-2.5 rounded-full bg-yellow-500/50" />
                                            <div className="w-2.5 h-2.5 rounded-full bg-green-500/50" />
                                        </div>
                                        <span className="ml-2">bash</span>
                                    </div>
                                    <p>
                                        <span className="text-emerald-400">$</span> curl -sSL https://get.sentinel.sec/install.sh | bash
                                    </p>
                                    <p className="mt-2 text-gray-500"># Verifying signature...</p>
                                    <p className="text-gray-500"># Installing executable to /usr/local/bin/sentinel</p>
                                    <p className="mt-2 text-emerald-300">✓ Sentinel v2.4.0 installed successfully.</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </main>
            <Footer />
        </div>
    );
};

export default Docs;
