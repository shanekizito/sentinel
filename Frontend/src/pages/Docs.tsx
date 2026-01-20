import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import { FileCode, Terminal, Book, Code2, ChevronRight, Search, Command } from "lucide-react";

const Docs = () => {
    return (
        <div className="min-h-screen bg-white flex flex-col">
            <Navbar />
            <main className="flex-grow pt-24 pb-20">
                <div className="container mx-auto px-6">
                    {/* Header */}
                    <div className="flex flex-col md:flex-row md:items-end justify-between gap-8 mb-16 border-b-2 border-gray-900 pb-8">
                        <div className="max-w-xl">
                            <h1 className="font-display text-5xl font-bold mb-4 text-gray-900 leading-none">Documentation</h1>
                            <p className="text-gray-600 text-lg">
                                Integration guides, API references, and security policy configuration for Sentinel V2.
                            </p>
                        </div>
                        <div className="relative w-full md:w-96">
                            <input
                                type="text"
                                placeholder="Search documentation..."
                                className="w-full pl-10 pr-4 py-3 bg-white border-2 border-gray-200 focus:border-primary focus:outline-none transition-colors font-mono text-xs text-gray-900 placeholder:text-gray-400"
                            />
                            <Search className="w-4 h-4 text-gray-500 absolute left-3 top-1/2 -translate-y-1/2" />
                            <div className="absolute right-3 top-1/2 -translate-y-1/2 flex items-center gap-1">
                                <span className="text-[10px] bg-gray-100 px-1.5 py-0.5 border border-gray-200 text-gray-500 font-mono">CTRL K</span>
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
                            <div key={i} className="group p-8 bg-white border-2 border-gray-200 hover:border-gray-900 transition-colors cursor-pointer relative">
                                <div className="absolute top-0 right-0 w-0 h-0 border-t-[8px] border-r-[8px] border-transparent group-hover:border-primary transition-all duration-300" />

                                <div className="w-10 h-10 border-2 border-gray-900 flex items-center justify-center text-gray-900 mb-6 group-hover:bg-primary group-hover:border-primary group-hover:text-white transition-colors">
                                    {item.icon}
                                </div>
                                <h3 className="font-bold text-gray-900 mb-2 flex items-center gap-2 text-lg font-display">
                                    {item.title}
                                    <ChevronRight className="w-4 h-4 opacity-0 group-hover:opacity-100 transition-opacity -translate-x-2 group-hover:translate-x-0 text-primary" />
                                </h3>
                                <p className="text-sm text-gray-600">{item.desc}</p>
                            </div>
                        ))}
                    </div>

                    {/* Documentation Content */}
                    <div className="grid lg:grid-cols-4 gap-12">
                        {/* Sidebar */}
                        <div className="hidden lg:block space-y-10">
                            {[
                                { cat: "Getting Started", items: ["Installation", "Authentication", "First Scan"] },
                                { cat: "Core Concepts", items: ["Threat Model", "Policy Engine", "Reporting"] },
                                { cat: "Integrations", items: ["GitHub Actions", "GitLab CI", "Jenkins", "Jira"] },
                                { cat: "SDKs", items: ["Python", "Node.js", "Go"] },
                            ].map((section) => (
                                <div key={section.cat}>
                                    <h4 className="font-mono text-xs font-bold text-gray-900 uppercase tracking-widest mb-4 border-b border-gray-200 pb-2">{section.cat}</h4>
                                    <ul className="space-y-2">
                                        {section.items.map(item => (
                                            <li key={item} className="text-sm text-gray-500 hover:text-primary cursor-pointer transition-colors block pl-2 border-l-2 border-transparent hover:border-primary">
                                                {item}
                                            </li>
                                        ))}
                                    </ul>
                                </div>
                            ))}
                        </div>

                        {/* Main Content Preview */}
                        <div className="lg:col-span-3 space-y-12">
                            <div className="p-10 border-2 border-gray-200 bg-white relative">
                                {/* Breadcrumb */}
                                <div className="flex items-center gap-2 mb-8 text-xs font-mono uppercase tracking-wider text-gray-400">
                                    <span>Docs</span>
                                    <ChevronRight className="w-3 h-3" />
                                    <span>Getting Started</span>
                                    <ChevronRight className="w-3 h-3" />
                                    <span className="text-gray-900 font-bold">Installation</span>
                                </div>

                                <h2 className="text-4xl font-display font-bold mb-6 text-gray-900">Installation</h2>
                                <p className="text-lg text-gray-600 leading-relaxed mb-8 max-w-2xl">
                                    Sentinel acts as a standalone binary or a containerized agent.
                                    The easiest way to get started is via our CLI installer.
                                </p>

                                <div className="bg-[#0a0a0c] border-2 border-gray-900 p-6 font-mono text-sm text-gray-300 relative group">
                                    <div className="absolute top-4 right-4 opacity-0 group-hover:opacity-100 transition-opacity">
                                        <button className="text-[10px] uppercase tracking-wider font-bold bg-white text-gray-900 hover:bg-white/90 px-3 py-1 border border-transparent">Copy</button>
                                    </div>
                                    <div className="flex items-center gap-2 border-b border-gray-800 pb-4 mb-4 text-xs text-gray-500">
                                        <Terminal className="w-3 h-3" />
                                        <span className="ml-1 uppercase tracking-wider">bash</span>
                                    </div>
                                    <div className="space-y-2">
                                        <p>
                                            <span className="text-primary">$</span> curl -sSL https://get.sentinel.sec/install.sh | bash
                                        </p>
                                        <p className="text-gray-600"># Verifying signature...</p>
                                        <p className="text-gray-600"># Installing executable to /usr/local/bin/sentinel</p>
                                        <p className="text-primary">✓ Sentinel v2.4.0 installed successfully.</p>
                                    </div>
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
