import { useState } from "react";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import { DocsContent } from "@/components/DocsContent";
import { FileCode, Terminal, Book, Code2, ChevronRight, Search, LayoutTemplate } from "lucide-react";

const Docs = () => {
    const [activeSection, setActiveSection] = useState("installation");
    const [searchQuery, setSearchQuery] = useState("");

    const sidebarSections = [
        {
            cat: "Start Here",
            items: [
                { id: "installation", label: "Installation" },
                { id: "first-scan", label: "First Scan" },
                { id: "authentication", label: "Authentication" }
            ]
        },
        {
            cat: "Core Concepts",
            items: [
                { id: "architecture", label: "System Architecture" },
                { id: "threat-model", label: "Threat Model" },
                { id: "policy-engine", label: "Policy Engine" }
            ]
        },
        {
            cat: "API Reference",
            items: [
                { id: "rest-api", label: "REST API" },
                { id: "graphql", label: "GraphQL" },
                { id: "sdk-python", label: "Python SDK" }
            ]
        }
    ];

    return (
        <div className="min-h-screen bg-white flex flex-col font-sans">
            <Navbar />
            <main className="flex-grow pt-28 pb-20">
                <div className="container mx-auto px-6">
                    {/* Header */}
                    <div className="flex flex-col md:flex-row md:items-end justify-between gap-8 mb-12 border-b border-gray-200 pb-8">
                        <div className="max-w-xl">
                            <h1 className="font-display text-4xl font-bold mb-3 text-gray-900 leading-none tracking-tight">Documentation</h1>
                            <p className="text-gray-500 text-lg font-light">
                                The sovereign manual for the Sentinel V2 infrastructure.
                            </p>
                        </div>
                        <div className="relative w-full md:w-80 group">
                            <input
                                type="text"
                                placeholder="Search docs..."
                                value={searchQuery}
                                onChange={(e) => setSearchQuery(e.target.value)}
                                className="w-full pl-10 pr-4 py-2.5 bg-gray-50 border border-gray-200 focus:border-gray-900 focus:ring-0 focus:outline-none transition-all font-mono text-xs text-gray-900 placeholder:text-gray-400 rounded-lg"
                            />
                            <Search className="w-4 h-4 text-gray-400 absolute left-3 top-1/2 -translate-y-1/2 group-focus-within:text-gray-900 transition-colors" />
                            <div className="absolute right-3 top-1/2 -translate-y-1/2 flex items-center gap-1">
                                <span className="text-[10px] bg-white px-1.5 py-0.5 border border-gray-200 rounded text-gray-400 font-mono">/</span>
                            </div>
                        </div>
                    </div>

                    <div className="grid lg:grid-cols-12 gap-12">
                        {/* Sidebar */}
                        <div className="hidden lg:block lg:col-span-3 space-y-10 sticky top-32 self-start h-[calc(100vh-10rem)] overflow-y-auto pr-4 scrollbar-hide">
                            {sidebarSections.map((section) => (
                                <div key={section.cat}>
                                    <h4 className="font-mono text-[10px] font-bold text-gray-400 uppercase tracking-widest mb-4 pl-3">{section.cat}</h4>
                                    <ul className="space-y-1">
                                        {section.items.map(item => (
                                            <li key={item.id}>
                                                <button
                                                    onClick={() => setActiveSection(item.id)}
                                                    className={`w-full text-left text-sm py-1.5 px-3 rounded-md transition-all duration-200 ${activeSection === item.id
                                                            ? "bg-gray-100 text-gray-900 font-medium"
                                                            : "text-gray-500 hover:text-gray-900 hover:bg-gray-50"
                                                        }`}
                                                >
                                                    {item.label}
                                                </button>
                                            </li>
                                        ))}
                                    </ul>
                                </div>
                            ))}
                        </div>

                        {/* Main Content Area */}
                        <div className="lg:col-span-9 min-h-[500px]">
                            {/* Breadcrumb */}
                            <div className="flex items-center gap-2 mb-8 text-[10px] font-mono uppercase tracking-wider text-gray-400">
                                <Book className="w-3 h-3" />
                                <span>Docs</span>
                                <ChevronRight className="w-3 h-3" />
                                <span className="text-gray-900 font-bold">
                                    {sidebarSections.flatMap(s => s.items).find(i => i.id === activeSection)?.label || "Select Topic"}
                                </span>
                            </div>

                            <div className="bg-white">
                                <DocsContent section={activeSection} />
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
