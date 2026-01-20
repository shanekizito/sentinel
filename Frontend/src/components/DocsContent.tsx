import { Terminal, Check, Copy } from "lucide-react";

interface DocsContentProps {
    section: string;
}

const CodeBlock = ({ code, language = "bash" }: { code: string; language?: string }) => (
    <div className="bg-[#0a0a0c] border border-gray-800 rounded-lg p-4 font-mono text-sm text-gray-300 relative group my-6">
        <div className="absolute top-4 right-4 opacity-0 group-hover:opacity-100 transition-opacity">
            <button className="text-[10px] uppercase tracking-wider font-bold bg-white/10 hover:bg-white/20 text-white px-2 py-1 rounded transition-colors flex items-center gap-1">
                <Copy className="w-3 h-3" /> Copy
            </button>
        </div>
        <div className="flex items-center gap-2 border-b border-gray-800 pb-2 mb-4 text-xs text-gray-500 unselectable">
            <Terminal className="w-3 h-3" />
            <span className="uppercase tracking-wider">{language}</span>
        </div>
        <pre className="overflow-x-auto">
            <code>{code}</code>
        </pre>
    </div>
);

const SectionHeader = ({ title, subtitle }: { title: string; subtitle: string }) => (
    <div className="mb-10 border-b border-gray-200 pb-6">
        <h2 className="text-4xl font-display font-bold text-gray-900 mb-4">{title}</h2>
        <p className="text-xl text-gray-500 font-light">{subtitle}</p>
    </div>
);

export const DocsContent = ({ section }: DocsContentProps) => {
    switch (section) {
        case "installation":
            return (
                <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
                    <SectionHeader
                        title="Installation"
                        subtitle="Get Sentinel running on your local machine in minutes."
                    />

                    <div className="prose prose-lg text-gray-600 max-w-none">
                        <p>
                            Sentinel is distributed as a standalone binary with zero runtime dependencies
                            for the core engine. You can install it via our CLI installer or build from source.
                        </p>

                        <h3 className="text-2xl font-bold text-gray-900 mt-10 mb-6">Prerequisites</h3>
                        <ul className="space-y-3 mb-8">
                            {[
                                "Rust Toolchain (1.75+) - for reliable safety guarantees",
                                "Python 3.10+ - for neural engine extensions",
                                "Git - for version control integration"
                            ].map((item, i) => (
                                <li key={i} className="flex items-start gap-3">
                                    <div className="mt-1.5 w-1.5 h-1.5 rounded-full bg-primary flex-shrink-0" />
                                    <span>{item}</span>
                                </li>
                            ))}
                        </ul>

                        <h3 className="text-2xl font-bold text-gray-900 mt-10 mb-6">Quick Install</h3>
                        <p>The fastest way to install Sentinel is using the verified install script:</p>

                        <CodeBlock code="curl -sSL https://get.sentinel.sec/install.sh | bash" />

                        <div className="bg-blue-50 border-l-4 border-primary p-6 my-8">
                            <h4 className="font-bold text-gray-900 mb-2">Build from Source</h4>
                            <p className="text-sm">
                                If you prefer to build from source, ensure you have the Rust toolchain installed.
                                Clone the repository and run <code className="bg-blue-100 px-1 py-0.5 rounded text-blue-800 font-mono text-xs">cargo build --release</code>.
                            </p>
                        </div>
                    </div>
                </div>
            );

        case "first-scan":
            return (
                <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
                    <SectionHeader
                        title="Your First Scan"
                        subtitle="Analyze a codebase and generate your first security report."
                    />

                    <div className="prose prose-lg text-gray-600 max-w-none">
                        <p>
                            Once installed, Sentinel interacts with your repositories via the command line.
                            Let's perform a dry-run analysis on a local project.
                        </p>

                        <h3 className="text-2xl font-bold text-gray-900 mt-10 mb-6">1. Initialize Sentinel</h3>
                        <p>Navigate to your project root and initialize the Sentinel configuration:</p>
                        <CodeBlock code="sentinel init" />

                        <h3 className="text-2xl font-bold text-gray-900 mt-10 mb-6">2. Run Ingestion</h3>
                        <p>
                            Sentinel will parse your codebase into a Code Property Graph (CPG).
                            This process is extremely fast thanks to our dedicated Rust parser.
                        </p>
                        <CodeBlock code="./sentinel ingest --source ./src --output ./data/shards" />

                        <h3 className="text-2xl font-bold text-gray-900 mt-10 mb-6">3. View Results</h3>
                        <p>
                            After ingestion, the dashboard will automatically spin up on localhost.
                        </p>
                        <CodeBlock code="./sentinel serve --port 8080" />
                    </div>
                </div>
            );

        case "architecture":
            return (
                <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
                    <SectionHeader
                        title="System Architecture"
                        subtitle="Deep dive into the Neural-Symbolic Hybrid Engine."
                    />

                    <div className="grid gap-8 mb-12">
                        <div className="bg-gray-50 p-8 border border-gray-200 rounded-xl">
                            <h3 className="text-xl font-bold text-gray-900 mb-4">The Logic Core (Symbolic)</h3>
                            <p className="text-gray-600 mb-4">
                                At the heart of Sentinel lies a formal verification engine powered by Z3 and CVC5 solvers.
                                Unlike traditional linters that use regex, Sentinel proves mathematical correctness.
                            </p>
                            <ul className="grid sm:grid-cols-2 gap-4">
                                {["Path Sensitivity", "Context Sensitivity", "Flow Sensitivity", "Object Sensitivity"].map(item => (
                                    <div key={item} className="flex items-center gap-2 text-sm font-mono text-gray-600 bg-white px-3 py-2 border border-gray-200 rounded">
                                        <Check className="w-4 h-4 text-primary" />
                                        {item}
                                    </div>
                                ))}
                            </ul>
                        </div>

                        <div className="bg-gray-900 text-white p-8 border border-gray-800 rounded-xl">
                            <h3 className="text-xl font-bold text-white mb-4">The Neural Net (Probabilistic)</h3>
                            <p className="text-gray-400 mb-4">
                                For patterns too complex for symbolic logic (like "business logic flaws"), we utilize
                                a Graph Convolutional Network (GCN) trained on 100M+ lines of vulnerable code.
                            </p>
                            <div className="flex items-center gap-4 text-xs font-mono uppercase tracking-widest text-gray-500">
                                <span>PyTorch</span>
                                <span className="w-1 h-1 bg-gray-700 rounded-full" />
                                <span>Geometric</span>
                                <span className="w-1 h-1 bg-gray-700 rounded-full" />
                                <span>CUDA 12</span>
                            </div>
                        </div>
                    </div>

                    <div className="prose prose-lg text-gray-600 max-w-none">
                        <h3 className="text-2xl font-bold text-gray-900 mb-6">The Hybrid Bridge</h3>
                        <p>
                            Sentinel connects these two worlds via a strictly typed gRPC bridge.
                            Findings from the Neural Net are passed to the Symbolic Solver for verification,
                            eliminating false positives.
                        </p>
                    </div>
                </div>
            );

        default:
            return (
                <div className="flex flex-col items-center justify-center h-96 text-center">
                    <div className="bg-gray-100 p-4 rounded-full mb-6">
                        <Terminal className="w-8 h-8 text-gray-400" />
                    </div>
                    <h3 className="text-xl font-bold text-gray-900 mb-2">Select a Topic</h3>
                    <p className="text-gray-500 max-w-md">
                        Choose a section from the sidebar to view documentation.
                    </p>
                </div>
            );
    }
};
