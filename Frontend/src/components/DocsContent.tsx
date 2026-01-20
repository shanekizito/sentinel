import { Terminal, Check, Copy, Shield, Lock, AlertTriangle, FileCode, Server, Database, Boxes, Network, Workflow } from "lucide-react";

interface DocsContentProps {
    section: string;
}

const CodeBlock = ({ code, language = "bash" }: { code: string; language?: string }) => (
    <div className="bg-[#0a0a0c] border border-gray-800 rounded-lg p-4 font-mono text-sm text-gray-300 relative group my-6 overflow-hidden">
        <div className="absolute top-4 right-4 opacity-0 group-hover:opacity-100 transition-opacity z-10">
            <button className="text-[10px] uppercase tracking-wider font-bold bg-white/10 hover:bg-white/20 text-white px-2 py-1 rounded transition-colors flex items-center gap-1 backdrop-blur-sm">
                <Copy className="w-3 h-3" /> Copy
            </button>
        </div>
        <div className="flex items-center gap-2 border-b border-gray-800 pb-2 mb-4 text-xs text-gray-500 unselectable">
            <Terminal className="w-3 h-3" />
            <span className="uppercase tracking-wider">{language}</span>
        </div>
        <pre className="overflow-x-auto scrollbar-thin scrollbar-thumb-gray-800 scrollbar-track-transparent pb-2">
            <code>{code}</code>
        </pre>
    </div>
);

const SectionHeader = ({ title, subtitle, icon: Icon }: { title: string; subtitle: string; icon?: any }) => (
    <div className="mb-10 border-b border-gray-200 pb-6">
        <div className="flex items-center gap-4 mb-4">
            {Icon && <div className="p-3 bg-gray-100 rounded-xl"><Icon className="w-8 h-8 text-gray-900" /></div>}
            <h2 className="text-4xl font-display font-bold text-gray-900">{title}</h2>
        </div>
        <p className="text-xl text-gray-500 font-light max-w-3xl leading-relaxed">{subtitle}</p>
    </div>
);

const InfoBox = ({ title, children, type = "info" }: { title: string; children: React.ReactNode; type?: "info" | "warning" }) => (
    <div className={`border-l-4 p-6 my-8 rounded-r-lg ${type === "warning" ? "bg-amber-50 border-amber-500" : "bg-blue-50 border-primary"}`}>
        <h4 className={`font-bold mb-2 flex items-center gap-2 ${type === "warning" ? "text-amber-900" : "text-gray-900"}`}>
            {type === "warning" ? <AlertTriangle className="w-4 h-4" /> : <Shield className="w-4 h-4" />}
            {title}
        </h4>
        <div className={`text-sm ${type === "warning" ? "text-amber-800" : "text-gray-700"}`}>
            {children}
        </div>
    </div>
);

export const DocsContent = ({ section }: DocsContentProps) => {
    switch (section) {
        // --- START HERE ---
        case "installation":
            return (
                <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
                    <SectionHeader
                        title="Installation"
                        subtitle="Sentinel is designed as a standalone binary with zero runtime dependencies. It can be deployed on local workstations, CI/CD runners, or minimal containers."
                        icon={Terminal}
                    />

                    <div className="prose prose-lg text-gray-600 max-w-none">
                        <h3 className="text-2xl font-bold text-gray-900 mt-10 mb-6">System Requirements</h3>
                        <div className="grid md:grid-cols-2 gap-6 mb-8">
                            <div className="border border-gray-200 p-6 rounded-xl bg-gray-50">
                                <h4 className="font-bold text-gray-900 mb-4 flex items-center gap-2"><Server className="w-4 h-4" /> Recommended Specs</h4>
                                <ul className="space-y-2 text-sm">
                                    <li><strong>CPU:</strong> 4+ Cores (AVX-512 preferred for massive scans)</li>
                                    <li><strong>RAM:</strong> 16GB (Sentinel uses `mmap` for low footprint)</li>
                                    <li><strong>Storage:</strong> NVMe SSD (for rapid graph traversal)</li>
                                </ul>
                            </div>
                            <div className="border border-gray-200 p-6 rounded-xl bg-gray-50">
                                <h4 className="font-bold text-gray-900 mb-4 flex items-center gap-2"><Boxes className="w-4 h-4" /> Supported OS</h4>
                                <ul className="space-y-2 text-sm">
                                    <li>Linux (kernel 5.4+) - Top Performance</li>
                                    <li>macOS (Apple Silicon & Intel)</li>
                                    <li>Windows 11 (WSL2 Recommended)</li>
                                </ul>
                            </div>
                        </div>

                        <h3 className="text-2xl font-bold text-gray-900 mt-10 mb-6">Automated Install</h3>
                        <p>The universal installer detects your architecture and OS, downloading the correct optimized binary.</p>
                        <CodeBlock code="curl -sSL https://get.sentinel.sec/install.sh | bash" />

                        <div className="my-8">
                            <h4 className="font-bold text-gray-900 mb-2">Build from Source</h4>
                            <p>For air-gapped environments or maximum optimization:</p>
                            <CodeBlock code={`git clone https://github.com/shanekizito/sentinel.git\ncd sentinel/Core/sentinel-cli\ncargo build --release`} />
                        </div>
                    </div>
                </div>
            );

        case "first-scan":
            return (
                <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
                    <SectionHeader
                        title="Your First Scan"
                        subtitle="Perform an initial security audit on a codebase. Sentinel supports polyglot repositories out of the box."
                        icon={Check}
                    />
                    <div className="prose prose-lg text-gray-600 max-w-none">
                        <h3 className="text-2xl font-bold text-gray-900 mt-8 mb-4">1. Initialize Project</h3>
                        <p>Run this in the root of your target repository. It creates a `.sentinel.yaml` configuration file.</p>
                        <CodeBlock code="sentinel init" />

                        <h3 className="text-2xl font-bold text-gray-900 mt-8 mb-4">2. Ingest & Analyze</h3>
                        <p>This command performs two steps: parsing source code into a CPG (Code Property Graph) and running the Neural-Symbolic analysis engine.</p>
                        <CodeBlock code="./sentinel ingest --source ./ --output ./sentinel_db" />

                        <InfoBox title="Zero-Config Mode">
                            By default, Sentinel automatically detects languages (Rust, Python, TS/JS, Go) and applies the relevant policy packs. No manual rule configuration is needed for the first run.
                        </InfoBox>
                    </div>
                </div>
            );

        case "authentication":
            return (
                <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
                    <SectionHeader
                        title="Authentication"
                        subtitle="Secure your Sentinel instance with Post-Quantum Cryptography (PQC) identity management."
                        icon={Lock}
                    />
                    <div className="prose prose-lg text-gray-600 max-w-none">
                        <p>Sentinel does not use traditional passwords. Instead, it relies on cryptographic identity keys (Kyber-1024) to authenticate users and sign audit logs.</p>

                        <h3 className="text-2xl font-bold text-gray-900 mt-8 mb-4">Generating Identity</h3>
                        <CodeBlock code="sentinel auth gen-key --name 'Admin User'" />
                        <p>This will produce a `sentinel.key` file. <strong>Guards this key with your life.</strong> It allows root access to the policy engine.</p>

                        <h3 className="text-2xl font-bold text-gray-900 mt-8 mb-4">Service Accounts</h3>
                        <p>For CI/CD pipelines (GitHub Actions, Jenkins), generate a restricted service token:</p>
                        <CodeBlock code="sentinel auth issuer --role CI_RUNNER --ttl 30d" />
                    </div>
                </div>
            );

        // --- CORE CONCEPTS ---
        case "architecture":
            return (
                <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
                    <SectionHeader
                        title="System Architecture"
                        subtitle="Understanding the Neuro-Symbolic Hybrid Engine that powers Sentinel."
                        icon={Network}
                    />

                    <div className="bg-gray-50 border border-gray-200 rounded-2xl p-8 mb-10 text-center">
                        <div className="inline-block bg-white p-6 rounded-xl shadow-sm border border-gray-100">
                            {/* Simple Mermaid Logic Diagram Representation */}
                            <div className="flex items-center gap-4 text-sm font-bold text-gray-600">
                                <div className="p-3 border rounded bg-gray-50">Raw Code</div>
                                <div className="h-0.5 w-8 bg-gray-300"></div>
                                <div className="p-3 border rounded bg-blue-50 text-blue-900">GCN Encoder</div>
                                <div className="h-0.5 w-8 bg-gray-300"></div>
                                <div className="p-3 border rounded bg-purple-50 text-purple-900">Symbolic Verifier</div>
                                <div className="h-0.5 w-8 bg-gray-300"></div>
                                <div className="p-3 border rounded bg-green-50 text-green-900">Proven Fix</div>
                            </div>
                        </div>
                        <p className="mt-4 text-sm text-gray-500">Figure 1. The Sentinel Hybrid Pipeline</p>
                    </div>

                    <div className="grid md:grid-cols-2 gap-8 mb-12">
                        <div>
                            <h3 className="text-xl font-bold text-gray-900 mb-3 text-purple-600">Symbolic Layer (The Judge)</h3>
                            <p className="text-gray-600 mb-4">Uses Z3 and CVC5 SMT solvers to mathematically prove the existence of bugs. If the solver says "SAT", the bug is 100% real.</p>
                            <ul className="text-sm space-y-1 text-gray-500">
                                <li>• Path Constraint Solving</li>
                                <li>• Taint Tracking (Source-to-Sink)</li>
                                <li>• Integer Overflow Proofs</li>
                            </ul>
                        </div>
                        <div>
                            <h3 className="text-xl font-bold text-gray-900 mb-3 text-blue-600">Neural Layer (The Intuition)</h3>
                            <p className="text-gray-600 mb-4">Uses Graph Neural Networks (GCN) to intuitively find "business logic" flaws that have no strict mathematical definition.</p>
                            <ul className="text-sm space-y-1 text-gray-500">
                                <li>• Pattern Recognition</li>
                                <li>• Anomaly Detection</li>
                                <li>• Variable Naming Analysis</li>
                            </ul>
                        </div>
                    </div>
                </div>
            );

        case "threat-model":
            return (
                <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
                    <SectionHeader
                        title="Threat Model"
                        subtitle="What Sentinel protects against. Our engine covers the entire OWASP Top 10 and CWE Top 25."
                        icon={Shield}
                    />

                    <div className="space-y-6">
                        {[
                            { code: "IR-101", name: "SQL Injection", desc: "Taint analysis tracks user input flowing into raw SQL queries.", proof: "Proved via CVC5 solver." },
                            { code: "IR-102", name: "Command Injection", desc: "Detects unsanitized input reaching `exec()` or `system()`.", proof: "Proved via Z3." },
                            { code: "IR-116", name: "Concurrency Deadlocks", desc: "Identifies circular mutex waits in multi-threaded code.", proof: "Resource allocation graph analysis." },
                            { code: "IR-121", name: "Weak Cryptography", desc: "Flags use of MD5, SHA1, or weak random number generators.", proof: "Pattern matching + entropy checks." }
                        ].map(threat => (
                            <div key={threat.code} className="border border-gray-200 rounded-lg p-6 hover:border-primary transition-colors cursor-default">
                                <div className="flex items-center justify-between mb-2">
                                    <h3 className="font-bold text-gray-900">{threat.name}</h3>
                                    <span className="font-mono text-xs bg-gray-100 px-2 py-1 rounded text-gray-600">{threat.code}</span>
                                </div>
                                <p className="text-gray-600 text-sm mb-3">{threat.desc}</p>
                                <div className="text-xs font-mono text-primary flex items-center gap-1">
                                    <Check className="w-3 h-3" /> {threat.proof}
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            );

        case "policy-engine":
            return (
                <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
                    <SectionHeader
                        title="Policy Engine"
                        subtitle="Define the 'Law of the Code'. Policies enforce security invariants that block pull requests if violated."
                        icon={Database}
                    />

                    <div className="prose prose-lg text-gray-600 max-w-none">
                        <p>Policies are defined in `sentinel.policy.yaml`. They allow you to customize the strictness of the sovereign guardian.</p>

                        <CodeBlock language="yaml" code={`version: "2.0"\npolicies:\n  - name: "No High Severity Bugs"\n    enforcement: BLOCK\n    rules:\n      - min_severity: HIGH\n      - certainty: PROVEN\n\n  - name: "Crypto Compliance"\n    enforcement: WARN\n    description: "Ensure only approved crypto libraries are used"\n    monitor:\n      - "sentinel-crypto/pqc.rs"`} />

                        <h3 className="text-2xl font-bold text-gray-900 mt-8 mb-4">Enforcement Modes</h3>
                        <ul className="grid grid-cols-3 gap-4 text-center">
                            <li className="bg-red-50 p-4 rounded border border-red-100">
                                <span className="block font-bold text-red-900 mb-1">BLOCK</span>
                                <span className="text-xs text-red-700">Fails CI pipeline. Code cannot merge.</span>
                            </li>
                            <li className="bg-amber-50 p-4 rounded border border-amber-100">
                                <span className="block font-bold text-amber-900 mb-1">WARN</span>
                                <span className="text-xs text-amber-700">Alerts developers but allows merge.</span>
                            </li>
                            <li className="bg-blue-50 p-4 rounded border border-blue-100">
                                <span className="block font-bold text-blue-900 mb-1">AUDIT</span>
                                <span className="text-xs text-blue-700">Silent logging for compliance reports.</span>
                            </li>
                        </ul>
                    </div>
                </div>
            );

        // --- API & SDK ---
        case "rest-api":
            return (
                <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
                    <SectionHeader
                        title="REST API"
                        subtitle="Programmatic access to the Sentinel platform. All endpoints are versioned and require PQC-signed headers."
                        icon={Server}
                    />

                    <div className="space-y-12">
                        <div>
                            <h3 className="text-lg font-bold text-gray-900 mb-4 flex items-center gap-2">
                                <span className="bg-green-100 text-green-800 text-xs px-2 py-1 rounded">POST</span>
                                /v1/scan/submit
                            </h3>
                            <p className="text-gray-600 mb-4">Submit a new codebase artifact for analysis.</p>
                            <CodeBlock language="json" code={`{\n  "repo_url": "https://github.com/org/repo.git",\n  "commit_sha": "d7a8f8c...",\n  "priority": "HIGH"\n}`} />
                        </div>

                        <div>
                            <h3 className="text-lg font-bold text-gray-900 mb-4 flex items-center gap-2">
                                <span className="bg-blue-100 text-blue-800 text-xs px-2 py-1 rounded">GET</span>
                                /v1/findings/{`{scan_id}`}
                            </h3>
                            <p className="text-gray-600 mb-4">Retrieve paginated results for a scan. Includes CPG node IDs for deep linking.</p>
                        </div>
                    </div>
                </div>
            );

        case "graphql":
            return (
                <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
                    <SectionHeader
                        title="GraphQL API"
                        subtitle="Query the Knowledge Graph directly. The most powerful way to explore code relationships."
                        icon={Workflow}
                    />
                    <div className="prose prose-lg text-gray-600">
                        <p>Sentinel exposes the entire Code Property Graph (CPG) via GraphQL. You can write queries to find complex patterns like <em>"Show me all functions that call a database but never sanitize input."</em></p>

                        <CodeBlock language="graphql" code={`query {\n  vulnerabilities(severity: CRITICAL) {\n    type\n    location {\n      file\n      line\n    }\n    dataFlow {\n      source\n      sink\n    }\n  }\n}`} />
                    </div>
                </div>
            );

        case "sdk-python":
            return (
                <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
                    <SectionHeader
                        title="Python SDK"
                        subtitle="Build custom analysis tools on top of the Sentinel Engine using our official Python bindings."
                        icon={FileCode}
                    />

                    <div className="prose prose-lg text-gray-600">
                        <CodeBlock language="bash" code="pip install sentinel-sdk" />

                        <h3 className="text-2xl font-bold text-gray-900 mt-8 mb-4">Example: Custom Rule</h3>
                        <CodeBlock language="python" code={`from sentinel import Engine, Query\n\ndef find_unencrypted_logging(graph):\n    # Find all logging calls\n    logs = graph.query(Query.calls("logger.info"))\n    \n    # Filter for sensitive keywords (password, key)\n    leaks = logs.filter(lambda x: "password" in x.arguments)\n    \n    return leaks\n\nengine = Engine.connect()\nresults = engine.run(find_unencrypted_logging)`} />
                    </div>
                </div>
            );

        default:
            return (
                <div className="flex flex-col items-center justify-center h-96 text-center">
                    <div className="bg-gray-100 p-6 rounded-full mb-6">
                        <Book className="w-12 h-12 text-gray-400" />
                    </div>
                    <h3 className="text-2xl font-bold text-gray-900 mb-2 font-display">Select a Topic</h3>
                    <p className="text-gray-500 max-w-md mx-auto leading-relaxed">
                        Explore the documentation sidebar to learn about installation, architecture, threat models, and more.
                    </p>
                </div>
            );
    }
};
