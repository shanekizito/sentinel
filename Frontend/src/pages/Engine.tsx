import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Brain, Database, Shield, Activity, Terminal, Cpu, Share2, Layers, Info, FileCode, CheckCircle2, ArrowLeft, ArrowRight, Zap, FolderTree, GitBranch, Search, Server, Folder, Github, Play, AlertTriangle, Check, X, Filter, Download, TrendingUp, BarChart3, Bell, Sparkles } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { ResponsiveContainer, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, BarChart, Bar, PieChart, Pie, Cell, ScatterChart, Scatter, LineChart, Line } from 'recharts';
import "../premium-ui.css";

// --- Types ---
interface EngineState {
    status: string;
    file: string;
    file_index: number;
    total_files: number;
    progress: number;
    nodes: number;
    features_learned: number;
    vulns: number;
    phase: string;
    timestamp: string;
    memory: string;
    confidence: string;
    scan_mode: string;
    project_tree: any;
}

interface Threat {
    file: string;
    severity: 'Critical' | 'High' | 'Medium' | 'Low';
    cvss: number;
    type: string;
    status: string;
    timestamp: string;
    description: string;
    attackVector: string;
    exploitAvailable: boolean;
    cve: string | null;
}

// --- Utility Components ---
const InfoTooltip = ({ content }: { content: string }) => (
    <TooltipProvider>
        <Tooltip>
            <TooltipTrigger asChild>
                <Info className="w-3.5 h-3.5 text-slate-400 hover:text-slate-600 cursor-help inline-block ml-1" />
            </TooltipTrigger>
            <TooltipContent className="max-w-xs">
                <p className="text-xs">{content}</p>
            </TooltipContent>
        </Tooltip>
    </TooltipProvider>
);

const CVSSBadge = ({ score }: { score: number }) => {
    const getSeverityColor = (cvss: number) => {
        if (cvss >= 9.0) return 'bg-red-600 text-white';
        if (cvss >= 7.0) return 'bg-orange-500 text-white';
        if (cvss >= 4.0) return 'bg-yellow-500 text-white';
        return 'bg-blue-500 text-white';
    };

    return (
        <TooltipProvider>
            <Tooltip>
                <TooltipTrigger asChild>
                    <span className={`px-2 py-1 rounded font-mono text-xs font-bold ${getSeverityColor(score)}`}>
                        {score.toFixed(1)}
                    </span>
                </TooltipTrigger>
                <TooltipContent>
                    <p className="text-xs font-bold">CVSS v3.1 Score</p>
                    <p className="text-xs text-slate-400">
                        {score >= 9.0 ? 'Critical' : score >= 7.0 ? 'High' : score >= 4.0 ? 'Medium' : 'Low'} Severity
                    </p>
                </TooltipContent>
            </Tooltip>
        </TooltipProvider>
    );
};

// --- Components ---
const TraceLine = ({ x1, y1, x2, y2, active }: any) => (
    <svg className="absolute inset-0 w-full h-full pointer-events-none z-10">
        <motion.line
            x1={x1} y1={y1} x2={x2} y2={y2}
            stroke={active ? "#10b981" : "#e2e8f0"}
            strokeWidth={active ? 2 : 1}
            initial={{ pathLength: 0, opacity: 0 }}
            animate={{ pathLength: 1, opacity: 1 }}
            transition={{ duration: 1 }}
            strokeDasharray={active ? "4 4" : "0"}
        >
            {active && (
                <animate
                    attributeName="stroke-dashoffset"
                    from="20"
                    to="0"
                    dur="1s"
                    repeatCount="Indefinite"
                />
            )}
        </motion.line>
    </svg>
);

const CircuitNode = ({ x, y, active, type, label, onClick, isVuln }: any) => {
    const isRoot = type === 'root';
    const isFolder = type === 'folder';

    return (
        <div
            className="absolute transform -translate-x-1/2 -translate-y-1/2 flex flex-col items-center z-20 cursor-pointer group"
            style={{ left: x, top: y }}
            onClick={onClick}
        >
            <motion.div
                className={`flex items-center justify-center rounded-lg border-2 transition-all duration-300 relative
                    ${active ? 'bg-primary border-primary shadow-[0_0_20px_rgba(16,185,129,0.4)]' :
                        isVuln ? 'bg-red-500 border-red-500 shadow-[0_0_20px_rgba(239,68,68,0.4)]' :
                            'bg-white border-slate-200 group-hover:border-slate-900'}
                    ${isRoot ? 'w-12 h-8' : isFolder ? 'w-8 h-8' : 'w-8 h-8 rounded-full'}
                `}
                animate={{ scale: active && !isRoot ? [1, 1.1, 1] : 1 }}
                transition={{ duration: 1, repeat: active ? Infinity : 0 }}
                whileHover={{ scale: 1.1 }}
            >
                {isRoot ? <Server className={`w-4 h-4 ${active ? 'text-white' : 'text-slate-400'}`} /> :
                    isFolder ? <Folder className={`w-4 h-4 ${active ? 'text-white' : 'text-slate-400'}`} /> :
                        <FileCode className={`w-3 h-3 ${active || isVuln ? 'text-white' : 'text-slate-400'}`} />
                }

                {isVuln && (
                    <div className="absolute -top-1 -right-1 w-2.5 h-2.5 bg-red-500 border-2 border-white rounded-full z-10" />
                )}
            </motion.div>

            <div className={`mt-1.5 px-1.5 py-0.5 rounded text-[9px] font-mono font-bold tracking-wider uppercase border whitespace-nowrap bg-white transition-colors
                ${active ? 'text-primary border-primary' :
                    isVuln ? 'text-red-500 border-red-500' :
                        'text-slate-400 border-slate-200 group-hover:border-slate-900 group-hover:text-slate-900'}
            `}>
                {label}
            </div>
        </div>
    );
};

const RemediationView = ({ file, threat, onClose }: { file: string, threat?: Threat, onClose: () => void }) => {
    const fixData: any = {
        'server.js': {
            vuln: "Remote Code Execution (CWE-502)",
            severity: "Critical",
            cvss: 9.8,
            desc: "Unsafe deserialization of user input allows arbitrary code execution. Attacker can execute system commands remotely.",
            code: `- app.post('/api/exec', (req, res) => {
-   eval(req.body.code);
- });

+ // Use sandboxed VM with timeout
+ const vm = require('vm');
+ const sandbox = { result: null };
+ vm.runInNewContext(req.body.code, sandbox, { 
+   timeout: 1000,
+   breakOnSigint: true 
+ });`,
            remediation: [
                "Remove all eval() usage immediately",
                "Implement input validation and sanitization",
                "Use sandboxed execution environment (vm module)",
                "Add rate limiting to prevent abuse",
                "Enable security monitoring and alerts"
            ]
        },
        'authRoutes.js': {
            vuln: "Broken Authentication (CWE-287)",
            severity: "High",
            cvss: 8.1,
            desc: "MD5 hashing is cryptographically broken. Passwords can be cracked using rainbow tables or GPU acceleration.",
            code: `- const crypto = require('crypto');
- const hash = crypto.createHash('md5').update(password).digest('hex');

+ const bcrypt = require('bcrypt');
+ const saltRounds = 12;
+ const hash = await bcrypt.hash(password, saltRounds);`,
            remediation: [
                "Migrate to bcrypt with cost factor ≥12",
                "Implement password complexity requirements",
                "Add multi-factor authentication (MFA)",
                "Force password reset for all users",
                "Enable account lockout after failed attempts"
            ]
        },
        'vulnerabilities.js': {
            vuln: "Multi-Vector Exposure (10 Detected)",
            severity: "Critical",
            cvss: 10.0,
            desc: "This file contains 10 distinct security vulnerabilities including NoSQL Injection, Command Injection, and SSRF. It represents a high risk to the application's integrity and confidentiality.",
            code: `// SQL/NoSQL Injection Fix:
- const users = await db.collection("users").find({ $where: \`this.name == '\${filter}'\` }).toArray();
+ const users = await db.collection("users").find({ name: filter }).toArray();

// Command Injection Fix:
- exec(\`ping -c 1 \${cmd}\`, (error, stdout, stderr) => { ... });
+ // Use a dedicated library or strict white-listing
+ if (!/^([a-zA-Z0-9.-]+)$/.test(cmd)) return res.status(400).send("Invalid input");
+ execFile('ping', ['-c', '1', cmd], (error, stdout, stderr) => { ... });

// SSRF Fix:
- const response = await axios.get(url);
+ // Implement URL whitelist and prevent internal network access
+ if (!ALLOWED_DOMAINS.includes(new URL(url).hostname)) return res.status(403).send("Forbidden");
+ const response = await axios.get(url);`,
            remediation: [
                "Parameterized queries for all database operations",
                "Strict input white-listing for system commands",
                "URL validation and blacklisting internal IPs for SSRF",
                "Use secure JWT secrets from environment variables",
                "Implement proper authorization checks (IDOR fix)"
            ]
        },
        'productRoutes.js': {
            vuln: "NoSQL Injection (CWE-943)",
            severity: "High",
            cvss: 7.5,
            desc: "Unsanitized lookup parameters allow MongoDB operator injection, potentially leading to unauthorized data access.",
            code: `- const product = await db.collection("products").findOne({ name: req.query.name });
+ // Ensure the input is treated as a string, not an object
+ const name = String(req.query.name);
+ const product = await db.collection("products").findOne({ name: name });`,
            remediation: [
                "Cast query parameters to strings",
                "Use a schema-based ODM like Mongoose",
                "Disable unsafe MongoDB operators in configuration"
            ]
        },
        'userRoutes.js': {
            vuln: "Cross-Site Scripting (CWE-79)",
            severity: "Medium",
            cvss: 6.1,
            desc: "Reflected user input in search results allows attackers to execute arbitrary scripts in users' browsers.",
            code: `- res.send(\`<div>Results for: \${req.query.q}</div>\`);
+ const sanitizeHtml = require('sanitize-html');
+ const clean = sanitizeHtml(req.query.q);
+ res.send(\`<div>Results for: \${clean}</div>\`);`,
            remediation: [
                "Context-aware output encoding",
                "Implement Content Security Policy (CSP)",
                "Use modern UI frameworks that auto-escape (React/Vue)"
            ]
        },
        'db.js': {
            vuln: "Hardcoded Credentials (CWE-798)",
            severity: "Critical",
            cvss: 9.1,
            desc: "Sensitive database credentials are hardcoded in the source, increasing risk of credential theft.",
            code: `- const CONNECTION_STRING = "mongodb+srv://admin:P@ssword123@cluster0.abcde.mongodb.net";
+ const CONNECTION_STRING = process.env.MONGODB_URI;`,
            remediation: [
                "Move secrets to environment variables",
                "Use a secret management service (AWS Secrets Manager, Vault)",
                "Rotate credentials immediately"
            ]
        },
        'config.js': {
            vuln: "Information Exposure (CWE-200)",
            severity: "High",
            cvss: 7.2,
            desc: "API keys and sensitive configuration details are exposed in the client-side or public configuration files.",
            code: `- export const API_KEY = "sk_live_123456789";
+ export const API_KEY = process.env.VITE_API_KEY;`,
            remediation: [
                "Never commit API keys to version control",
                "Use .env files with .gitignore",
                "Invalidate and rotate exposed keys"
            ]
        },
        'Product.js': {
            vuln: "Mass Assignment (CWE-915)",
            severity: "Medium",
            cvss: 5.3,
            desc: "Binding user input directly to database models allows attackers to modify sensitive fields like 'isAdmin'.",
            code: `- const product = new Product(req.body);
- await product.save();
+ const { name, price, description } = req.body;
+ const product = new Product({ name, price, description });
+ await product.save();`,
            remediation: [
                "Use DTOs (Data Transfer Objects)",
                "Implement white-listing for allowed fields",
                "Use model-level validation to restrict sensitive fields"
            ]
        }
    };

    const data = fixData[file.split('/').pop() || ''] || threat || {
        vuln: "Security Vulnerability",
        severity: "Medium",
        cvss: 5.0,
        desc: "A security issue was detected. Manual review required.",
        code: `// Review required\n// Check SENTINEL_REPORT.md for details`,
        remediation: ["Review code manually", "Consult security team"]
    };

    return (
        <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            className="absolute inset-0 z-50 bg-slate-900/50 backdrop-blur-sm p-4 md:p-8 flex flex-col items-center justify-center overflow-y-auto"
            onClick={onClose}
        >
            <div className="w-full max-w-5xl bg-white rounded-2xl shadow-2xl border border-slate-200 overflow-hidden my-8" onClick={(e) => e.stopPropagation()}>
                {/* Header */}
                <div className="bg-gradient-to-r from-slate-900 to-slate-800 text-white p-6 flex justify-between items-start">
                    <div className="flex items-start gap-4 flex-1">
                        <div className="p-3 bg-white/10 rounded-lg">
                            <AlertTriangle className="w-7 h-7 text-white" />
                        </div>
                        <div className="flex-1">
                            <div className="flex items-center gap-3 mb-2">
                                <h2 className="text-2xl font-display font-bold">{data.vuln}</h2>
                                <CVSSBadge score={data.cvss} />
                            </div>
                            <div className="flex items-center gap-2 text-sm text-slate-300">
                                <FileCode className="w-4 h-4" />
                                <span className="font-mono">{file}</span>
                            </div>
                        </div>
                    </div>
                    <button onClick={onClose} className="p-2 hover:bg-white/10 rounded-full transition-colors">
                        <X className="w-5 h-5 text-slate-300" />
                    </button>
                </div>

                <div className="grid md:grid-cols-12 gap-0">
                    {/* Left: Analysis */}
                    <div className="md:col-span-5 p-8 border-r border-slate-100 bg-slate-50/50 space-y-6">
                        <div>
                            <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wider mb-3 flex items-center gap-2">
                                <Shield className="w-4 h-4 text-red-500" /> Threat Analysis
                            </h3>
                            <p className="text-slate-700 text-sm leading-relaxed">
                                {data.desc}
                            </p>
                        </div>

                        <div>
                            <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3">Remediation Steps</h4>
                            <ol className="space-y-2">
                                {(data.remediation || []).map((step: string, i: number) => (
                                    <li key={i} className="flex gap-3 text-sm">
                                        <span className="flex-shrink-0 w-5 h-5 bg-emerald-100 text-emerald-700 rounded-full flex items-center justify-center text-xs font-bold">
                                            {i + 1}
                                        </span>
                                        <span className="text-slate-700">{step}</span>
                                    </li>
                                ))}
                            </ol>
                        </div>

                        <div className="space-y-3 pt-4 border-t border-slate-200">
                            <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider">Automated Actions</h4>
                            <button className="w-full flex items-center justify-between p-3 bg-white border-2 border-slate-900 rounded-lg hover:bg-slate-900 hover:text-white transition-all group shadow-[4px_4px_0px_0px_rgba(15,23,42,0.1)]">
                                <span className="flex items-center gap-3 font-bold text-sm">
                                    <Github className="w-4 h-4" /> Create Pull Request
                                </span>
                                <ArrowRight className="w-4 h-4" />
                            </button>
                            <button className="w-full flex items-center justify-between p-3 bg-white border border-slate-200 rounded-lg hover:border-slate-900 transition-all group">
                                <span className="flex items-center gap-3 font-bold text-sm text-slate-700">
                                    <Play className="w-4 h-4" /> Run Security Tests
                                </span>
                                <ArrowRight className="w-4 h-4 text-slate-300 group-hover:text-slate-900" />
                            </button>
                        </div>
                    </div>

                    {/* Right: Code Fix */}
                    <div className="md:col-span-7 p-8 bg-white">
                        <div className="flex justify-between items-center mb-4">
                            <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wider flex items-center gap-2">
                                <Check className="w-4 h-4 text-emerald-500" /> Proposed Fix
                            </h3>
                            <span className="text-xs font-mono text-slate-400 bg-slate-100 px-2 py-1 rounded">patch-v1.0.diff</span>
                        </div>

                        <div className="bg-slate-900 rounded-xl p-5 overflow-auto shadow-inner max-h-96">
                            <pre className="font-mono text-xs leading-relaxed">
                                {(data.code || '').split('\n').map((line: string, i: number) => (
                                    <div key={i} className={`${line.startsWith('+') ? 'bg-emerald-500/20 text-emerald-300' : line.startsWith('-') ? 'bg-red-500/20 text-red-300' : 'text-slate-400'} px-2 -mx-2 py-0.5`}>
                                        {line}
                                    </div>
                                ))}
                            </pre>
                        </div>

                        <div className="mt-6 flex justify-end gap-3">
                            <button onClick={onClose} className="px-5 py-2.5 text-sm font-bold text-slate-600 hover:bg-slate-100 rounded-lg transition-colors">
                                Dismiss
                            </button>
                            <button className="px-5 py-2.5 text-sm font-bold bg-emerald-500 text-white hover:bg-emerald-600 rounded-lg shadow-lg shadow-emerald-500/30 transition-all flex items-center gap-2">
                                <CheckCircle2 className="w-4 h-4" /> Apply Fix & Deploy
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </motion.div>
    )
}

const IngestionDetail = ({ state, onBack }: { state: EngineState, onBack: () => void }) => {
    const pieData = [
        { name: 'JavaScript', value: 45, color: '#f7df1e' },
        { name: 'TypeScript', value: 30, color: '#3178c6' },
        { name: 'Python', value: 15, color: '#3776ab' },
        { name: 'Rust', value: 10, color: '#000000' },
    ];

    const chartData = [
        { time: '10:00', throughput: 85, memory: 120 },
        { time: '10:05', throughput: 110, memory: 145 },
        { time: '10:10', throughput: 95, memory: 130 },
        { time: '10:15', throughput: 130, memory: 160 },
        { time: '10:20', throughput: 120, memory: 150 },
    ];

    return (
        <div className="p-8 max-w-7xl mx-auto space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                    <button onClick={onBack} className="p-2 hover:bg-slate-200 rounded-full transition-colors">
                        <ArrowLeft className="w-6 h-6 text-slate-600" />
                    </button>
                    <div>
                        <h2 className="text-3xl font-display font-bold text-slate-900">Data Ingestion Pipeline</h2>
                        <p className="text-slate-500">High-performance source code parsing & CPG generation</p>
                    </div>
                </div>
                <div className="flex items-center gap-3">
                    <div className="px-4 py-2 bg-emerald-100 text-emerald-700 rounded-xl font-bold text-sm border border-emerald-200">
                        PIPELINE ACTIVE
                    </div>
                </div>
            </div>

            <div className="grid grid-cols-12 gap-6">
                {/* Main Stats */}
                <Card className="col-span-12 md:col-span-8 p-6 bg-white border-slate-200 shadow-sm">
                    <h3 className="text-lg font-bold text-slate-900 mb-6 flex items-center gap-2">
                        <Activity className="w-5 h-5 text-blue-500" />
                        Throughput & Memory Performance
                    </h3>
                    <div className="h-[300px] w-full">
                        <ResponsiveContainer width="100%" height="100%">
                            <AreaChart data={chartData}>
                                <defs>
                                    <linearGradient id="colorThroughput" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.1} />
                                        <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                                    </linearGradient>
                                </defs>
                                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                                <XAxis dataKey="time" axisLine={false} tickLine={false} tick={{ fontSize: 12, fill: '#64748b' }} />
                                <YAxis axisLine={false} tickLine={false} tick={{ fontSize: 12, fill: '#64748b' }} />
                                <RechartsTooltip
                                    contentStyle={{ backgroundColor: '#fff', borderRadius: '12px', border: 'none', boxShadow: '0 10px 15px -3px rgb(0 0 0 / 0.1)' }}
                                />
                                <Area type="monotone" dataKey="throughput" stroke="#3b82f6" fillOpacity={1} fill="url(#colorThroughput)" strokeWidth={3} name="MB/s" />
                                <Area type="monotone" dataKey="memory" stroke="#10b981" fillOpacity={0} strokeWidth={2} strokeDasharray="5 5" name="MB RAM" />
                            </AreaChart>
                        </ResponsiveContainer>
                    </div>
                </Card>

                {/* File Type Distribution */}
                <Card className="col-span-12 md:col-span-4 p-6 bg-white border-slate-200 shadow-sm">
                    <h3 className="text-lg font-bold text-slate-900 mb-6">File Composition</h3>
                    <div className="h-[240px] w-full">
                        <ResponsiveContainer width="100%" height="100%">
                            <PieChart>
                                <Pie
                                    data={pieData}
                                    cx="50%"
                                    cy="50%"
                                    innerRadius={60}
                                    outerRadius={80}
                                    paddingAngle={5}
                                    dataKey="value"
                                >
                                    {pieData.map((entry, index) => (
                                        <Cell key={`cell-${index}`} fill={entry.color} />
                                    ))}
                                </Pie>
                                <RechartsTooltip />
                            </PieChart>
                        </ResponsiveContainer>
                    </div>
                    <div className="grid grid-cols-2 gap-4 mt-4">
                        {pieData.map((item) => (
                            <div key={item.name} className="flex items-center gap-2">
                                <div className="w-2 h-2 rounded-full" style={{ backgroundColor: item.color }} />
                                <span className="text-xs font-bold text-slate-600">{item.name}</span>
                                <span className="text-xs text-slate-400 ml-auto">{item.value}%</span>
                            </div>
                        ))}
                    </div>
                </Card>

                {/* Worker Grid */}
                <div className="col-span-12 grid grid-cols-4 gap-6">
                    {[1, 2, 3, 4].map((id) => (
                        <Card key={id} className="p-4 bg-white border-slate-200 shadow-sm flex items-center gap-4">
                            <div className={`p-3 rounded-xl ${id === 4 ? 'bg-amber-50 text-amber-600' : 'bg-blue-50 text-blue-600'}`}>
                                <Cpu className="w-5 h-5" />
                            </div>
                            <div>
                                <div className="text-[10px] font-bold text-slate-400 uppercase">Worker #{id}</div>
                                <div className="font-mono text-sm font-bold text-slate-900">{id === 4 ? 'IDLE' : 'PARSING'}</div>
                            </div>
                            <div className="ml-auto">
                                <Activity className={`w-4 h-4 ${id === 4 ? 'text-slate-200' : 'text-emerald-500'}`} />
                            </div>
                        </Card>
                    ))}
                </div>
            </div>
        </div>
    );
};

const NeuralDetail = ({ state, onBack }: { state: EngineState, onBack: () => void }) => {
    const confidenceData = [
        { range: '0-20%', count: 5 },
        { range: '20-40%', count: 12 },
        { range: '40-60%', count: 45 },
        { range: '60-80%', count: 85 },
        { range: '80-100%', count: 25 },
    ];

    const scatterData = [
        { complexity: 10, confidence: 98 },
        { complexity: 25, confidence: 95 },
        { complexity: 50, confidence: 88 },
        { complexity: 75, confidence: 82 },
        { complexity: 100, confidence: 75 },
        { complexity: 120, confidence: 70 },
    ];

    return (
        <div className="p-8 max-w-7xl mx-auto space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                    <button onClick={onBack} className="p-2 hover:bg-slate-200 rounded-full transition-colors">
                        <ArrowLeft className="w-6 h-6 text-slate-600" />
                    </button>
                    <div>
                        <h2 className="text-3xl font-display font-bold text-slate-900">Sovereign Neural Engine</h2>
                        <p className="text-slate-500">Neuro-symbolic pattern recognition & online learning</p>
                    </div>
                </div>
                <div className="flex items-center gap-6">
                    <div className="text-right">
                        <div className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Global Confidence</div>
                        <div className="text-2xl font-mono font-bold text-purple-600">{(parseFloat(state.confidence || "0") * 100).toFixed(2)}%</div>
                    </div>
                    <div className="text-right">
                        <div className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Nodes Learned</div>
                        <div className="text-2xl font-mono font-bold text-slate-900">{state.features_learned || 0}</div>
                    </div>
                </div>
            </div>

            <div className="grid grid-cols-12 gap-6">
                {/* Synaptic Network Graph */}
                <Card className="col-span-12 md:col-span-12 p-6 bg-slate-950 border-white/5 shadow-2xl overflow-visible relative min-h-[450px]">
                    <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(168,85,247,0.15),transparent)] pointer-events-none" />

                    <div className="relative z-10 flex flex-col h-full">
                        <div className="flex justify-between items-start mb-10">
                            <div>
                                <h3 className="text-lg font-bold text-white flex items-center gap-2">
                                    <Sparkles className="w-5 h-5 text-purple-400" />
                                    Active Synaptic Network
                                </h3>
                                <p className="text-[10px] text-slate-500 font-mono mt-1">REAL-TIME FEATURE EXTRACTION & EMBEDDING FLOW</p>
                            </div>
                            <div className="flex gap-6">
                                <div className="text-right">
                                    <div className="text-[9px] font-bold text-slate-500 uppercase tracking-widest">Active Paths</div>
                                    <div className="text-sm font-mono font-bold text-emerald-400">1,240</div>
                                </div>
                                <div className="text-right">
                                    <div className="text-[9px] font-bold text-slate-500 uppercase tracking-widest">Inference Latency</div>
                                    <div className="text-sm font-mono font-bold text-purple-400">~2.4ms</div>
                                </div>
                            </div>
                        </div>

                        {/* Neural Graph Layout with Unified Coordinate System */}
                        <div className="flex-1 relative flex items-center justify-center min-h-[500px] overflow-visible">
                            <div className="relative w-[1000px] h-[400px] flex items-center justify-between px-0">
                                {/* SVG Synapses & Pulses - Absolute within the fixed-width container */}
                                <svg className="absolute inset-0 w-full h-full pointer-events-none z-0">
                                    <defs>
                                        <filter id="clean-shadow" x="-20%" y="-20%" width="140%" height="140%">
                                            <feDropShadow dx="0" dy="0" stdDeviation="4" floodColor="#a855f7" floodOpacity="0.3" />
                                        </filter>
                                        <linearGradient id="synapses-grad" x1="0%" y1="0%" x2="100%" y2="0%">
                                            <stop offset="0%" stopColor="#a855f7" />
                                            <stop offset="100%" stopColor="#10b981" />
                                        </linearGradient>
                                    </defs>

                                    {/* Architectural, precise connection paths (Aligned to 100, 350, 600, 850) */}
                                    <g className="opacity-25">
                                        <path d="M 100,120 C 225,120 225,100 350,100" stroke="url(#synapses-grad)" strokeWidth="1.5" fill="none" />
                                        <path d="M 100,200 C 225,200 225,144 350,144" stroke="url(#synapses-grad)" strokeWidth="1.5" fill="none" />
                                        <path d="M 100,280 C 225,280 225,258 350,258" stroke="url(#synapses-grad)" strokeWidth="1.5" fill="none" />
                                        <path d="M 100,360 C 225,360 225,372 350,372" stroke="url(#synapses-grad)" strokeWidth="1.5" fill="none" />

                                        <path d="M 100,120 C 225,120 225,315 350,315" stroke="url(#synapses-grad)" strokeWidth="1" fill="none" className="opacity-30" />
                                        <path d="M 100,360 C 225,360 225,87 350,87" stroke="url(#synapses-grad)" strokeWidth="1" fill="none" className="opacity-30" />

                                        <path d="M 350,87 C 475,87 475,144 600,144" stroke="url(#synapses-grad)" strokeWidth="1.5" fill="none" />
                                        <path d="M 350,201 C 475,201 475,258 600,258" stroke="url(#synapses-grad)" strokeWidth="2" fill="none" />
                                        <path d="M 350,372 C 475,372 475,315 600,315" stroke="url(#synapses-grad)" strokeWidth="1.5" fill="none" />

                                        <path d="M 600,144 C 725,144 725,200 850,200" stroke="url(#synapses-grad)" strokeWidth="2" fill="none" />
                                        <path d="M 600,315 C 725,315 725,380 850,380" stroke="url(#synapses-grad)" strokeWidth="2" fill="none" />
                                    </g>

                                    {/* High-Intensity Animated Data Pulses */}
                                    <circle r="5" fill="#ffffff" className="drop-shadow-[0_0_10px_rgba(255,255,255,1)]">
                                        <animateMotion dur="2.2s" repeatCount="indefinity" path="M 100,120 C 225,120 225,100 350,100" />
                                        <animate attributeName="opacity" values="0;1;0" dur="2.2s" repeatCount="indefinity" />
                                    </circle>
                                    <circle r="5" fill="#a855f7" className="drop-shadow-[0_0_10px_rgba(168,85,247,1)]">
                                        <animateMotion dur="2.8s" repeatCount="indefinity" path="M 100,280 C 225,280 225,258 350,258" />
                                        <animate attributeName="opacity" values="0;1;0" dur="2.8s" repeatCount="indefinity" />
                                    </circle>
                                    <circle r="5.5" fill="#10b981" className="drop-shadow-[0_0_12px_rgba(16,185,129,1)]">
                                        <animateMotion dur="1.8s" repeatCount="indefinity" path="M 350,201 C 475,201 475,258 600,258" />
                                        <animate attributeName="opacity" values="0;1;0" dur="1.8s" repeatCount="indefinity" />
                                    </circle>
                                    <circle r="5.5" fill="#ffffff" className="drop-shadow-[0_0_15px_rgba(255,255,255,1)]">
                                        <animateMotion dur="1.4s" repeatCount="indefinity" path="M 600,144 C 725,144 725,200 850,200" />
                                        <animate attributeName="opacity" values="0;1;0" dur="1.4s" repeatCount="indefinity" />
                                    </circle>
                                </svg>

                                {/* Node Layers - Using Fixed Centers (100, 350, 600, 850) */}
                                <div className="absolute inset-0 z-10">
                                    {/* Input Layer - Center at 100 */}
                                    <div className="absolute top-0 bottom-0 left-[100px] -translate-x-1/2 flex flex-col justify-around gap-4">
                                        <div className="absolute -top-16 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2 whitespace-nowrap">
                                            <div className="px-2 py-0.5 border border-primary/30 bg-primary/5 text-[9px] font-mono uppercase tracking-[0.2em] text-primary">Stage 01</div>
                                            <div className="text-[13px] font-display font-bold text-gray-900 dark:text-white uppercase tracking-wider">Features</div>
                                        </div>
                                        {[1, 2, 3, 4, 5].map(i => (
                                            <div key={`in-${i}`} className="group relative">
                                                <motion.div
                                                    className="w-10 h-10 border-2 border-gray-900 bg-white dark:bg-slate-900 flex items-center justify-center transition-all duration-300 relative"
                                                    animate={{ scale: [1, 1.05, 1] }}
                                                    transition={{ duration: 4, repeat: Infinity, delay: i * 0.2 }}
                                                >
                                                    {/* Architectural Corner Accent */}
                                                    <div className="absolute -top-[2px] -left-[2px] w-2 h-2 border-t-2 border-l-2 border-primary opacity-0 group-hover:opacity-100 transition-opacity" />
                                                    <Database className="w-5 h-5 text-gray-400 group-hover:text-primary transition-colors" />
                                                </motion.div>
                                                <div className="absolute right-full mr-4 top-1/2 -translate-y-1/2 whitespace-nowrap hidden group-hover:block z-[100] p-4 bg-slate-950 border border-white/10 shadow-2xl min-w-[220px] animate-in slide-in-from-right-2 duration-200">
                                                    <div className="text-[11px] font-display font-bold text-white mb-1.5 flex items-center gap-2">
                                                        <div className="w-1.5 h-1.5 bg-primary" />
                                                        {i === 1 && "AST Tokens"}
                                                        {i === 2 && "Control Flow Graph"}
                                                        {i === 3 && "Data Taint Seeds"}
                                                        {i === 4 && "Symbolic Constraints"}
                                                        {i === 5 && "Semantic Embeddings"}
                                                    </div>
                                                    <div className="text-[10px] text-gray-400 leading-relaxed font-sans normal-case tracking-normal">
                                                        {i === 1 && "Raw tokens extracted from source code for initial tokenization."}
                                                        {i === 2 && "Logical execution paths through functions and modules."}
                                                        {i === 3 && "Identified sources of unvalidated user-controlled data."}
                                                        {i === 4 && "Mathematical representations of variable states and ranges."}
                                                        {i === 5 && "High-dimensional vector representations of code intent."}
                                                    </div>
                                                </div>
                                            </div>
                                        ))}
                                        <div className="text-[9px] font-mono text-gray-500 uppercase tracking-[0.2em] text-center mt-3">Code Vectors</div>
                                    </div>

                                    {/* Hidden Layer 1 - Center at 350 */}
                                    <div className="absolute top-0 bottom-0 left-[350px] -translate-x-1/2 flex flex-col justify-around gap-4 mt-8">
                                        <div className="absolute -top-16 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2 whitespace-nowrap">
                                            <div className="px-2 py-0.5 border border-primary/30 bg-primary/5 text-[9px] font-mono uppercase tracking-[0.2em] text-primary">Stage 02</div>
                                            <div className="text-[13px] font-display font-bold text-gray-900 dark:text-white uppercase tracking-wider">Attention</div>
                                        </div>
                                        {[1, 2, 3, 4, 5, 6, 7].map(i => (
                                            <div key={`h1-${i}`} className="group relative">
                                                <motion.div
                                                    className="w-8 h-8 border border-gray-900/20 dark:border-white/10 bg-white/5 relative"
                                                    animate={{
                                                        opacity: [0.3, 0.8, 0.3],
                                                        scale: [1, 1.15, 1],
                                                        backgroundColor: ["rgba(255,255,255,0.05)", "rgba(168,85,247,0.2)", "rgba(255,255,255,0.05)"]
                                                    }}
                                                    transition={{ duration: 4, repeat: Infinity, delay: i * 0.3 }}
                                                />
                                                <div className="absolute left-full ml-5 top-1/2 -translate-y-1/2 whitespace-nowrap hidden group-hover:block z-[100] p-4 bg-slate-950 border border-white/10 shadow-2xl min-w-[240px] animate-in slide-in-from-left-2 duration-200">
                                                    <div className="text-[11px] font-display font-bold text-white mb-1.5 flex items-center gap-2">
                                                        <Sparkles className="w-3.5 h-3.5 text-primary" />
                                                        Attention Head #{i}
                                                    </div>
                                                    <div className="text-[10px] text-gray-400 leading-relaxed font-sans normal-case tracking-normal">
                                                        Calculates contextual weighting for multi-hop semantic dependencies.
                                                    </div>
                                                </div>
                                            </div>
                                        ))}
                                        <div className="text-[9px] font-mono text-gray-500 uppercase tracking-[0.2em] text-center mt-3">Latent Attention</div>
                                    </div>

                                    {/* Hidden Layer 2 - Center at 600 */}
                                    <div className="absolute top-0 bottom-0 left-[600px] -translate-x-1/2 flex flex-col justify-around gap-4 -mt-8">
                                        <div className="absolute -top-16 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2 whitespace-nowrap">
                                            <div className="px-2 py-0.5 border border-primary/30 bg-primary/5 text-[9px] font-mono uppercase tracking-[0.2em] text-primary">Stage 03</div>
                                            <div className="text-[13px] font-display font-bold text-gray-900 dark:text-white uppercase tracking-wider">Inference</div>
                                        </div>
                                        {[1, 2, 3, 4, 5, 6, 7].map(i => (
                                            <div key={`h2-${i}`} className="group relative">
                                                <motion.div
                                                    className="w-8 h-8 border border-gray-900/20 dark:border-white/10 bg-white/5 relative"
                                                    animate={{
                                                        opacity: [0.3, 0.8, 0.3],
                                                        scale: [1, 1.15, 1],
                                                        backgroundColor: ["rgba(255,255,255,0.05)", "rgba(16,185,129,0.2)", "rgba(255,255,255,0.05)"]
                                                    }}
                                                    transition={{ duration: 4, repeat: Infinity, delay: i * 0.4 }}
                                                />
                                                <div className="absolute left-full ml-5 top-1/2 -translate-y-1/2 whitespace-nowrap hidden group-hover:block z-[100] p-4 bg-slate-950 border border-white/10 shadow-2xl min-w-[260px] animate-in slide-in-from-left-2 duration-200">
                                                    <div className="text-[11px] font-display font-bold text-white mb-1.5 flex items-center gap-2">
                                                        <Zap className="w-3.5 h-3.5 text-primary" />
                                                        Heuristic Rule #{i}
                                                    </div>
                                                    <div className="text-[10px] text-gray-400 leading-relaxed font-sans normal-case tracking-normal">
                                                        Evaluating neuro-symbolic logic against identified taint flow candidates.
                                                    </div>
                                                </div>
                                            </div>
                                        ))}
                                        <div className="text-[9px] font-mono text-gray-500 uppercase tracking-[0.2em] text-center mt-3">Rule Resolution</div>
                                    </div>

                                    {/* Output Layer - Center at 850 */}
                                    <div className="absolute top-0 bottom-0 left-[850px] -translate-x-1/2 flex flex-col justify-around gap-12 mt-12">
                                        <div className="absolute -top-16 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2 whitespace-nowrap">
                                            <div className="px-2 py-0.5 border border-primary/30 bg-primary/5 text-[9px] font-mono uppercase tracking-[0.2em] text-primary">Stage 04</div>
                                            <div className="text-[13px] font-display font-bold text-gray-900 dark:text-white uppercase tracking-wider">Verdict</div>
                                        </div>
                                        {[1, 2].map(i => (
                                            <div key={`out-${i}`} className="group relative">
                                                <motion.div
                                                    className={`w-14 h-14 border-2 flex items-center justify-center relative
                                                    ${i === 1 ? 'border-primary bg-primary/10 shadow-[0_0_20px_rgba(16,185,129,0.1)]' : 'border-gray-900 bg-white dark:bg-slate-950'}
                                                `}
                                                    animate={{ scale: [1, 1.05, 1] }}
                                                    transition={{ duration: 5, repeat: Infinity }}
                                                >
                                                    {/* Corner Accent */}
                                                    <div className="absolute -top-[2px] -right-[2px] w-3 h-3 border-t-2 border-r-2 border-primary" />
                                                    <div className="absolute -bottom-[2px] -left-[2px] w-3 h-3 border-b-2 border-l-2 border-primary" />

                                                    {i === 1 ? <Shield className="w-6 h-6 text-primary" /> : <Zap className="w-6 h-6 text-gray-400" />}
                                                </motion.div>
                                                <div className="absolute left-full ml-4 top-1/2 -translate-y-1/2 whitespace-nowrap hidden group-hover:block z-[100] p-4 bg-slate-950 border border-white/10 shadow-2xl min-w-[220px] animate-in slide-in-from-left-2 duration-200">
                                                    <div className="text-[11px] font-display font-bold text-white mb-1.5">{i === 1 ? 'Vulnerability Verdict' : 'Taint Probability'}</div>
                                                    <div className="text-[10px] text-gray-400 leading-relaxed font-sans normal-case tracking-normal">
                                                        {i === 1 ? 'Calculated risk score based on architectural flow analysis.' : 'Probability density of unsanitized data reaching sensitive sinks.'}
                                                    </div>
                                                </div>
                                                <div className="text-[9px] font-mono text-gray-500 uppercase tracking-[0.2em] text-center mt-3">
                                                    {i === 1 ? 'Risk' : 'Taint'}
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            </div>
                        </div>

                        <div className="mt-8 pt-6 border-t border-white/5 flex justify-between items-end text-[10px] font-bold text-slate-500 uppercase tracking-widest font-mono">
                            <div className="space-y-4">
                                <div className="flex gap-6">
                                    <span className="flex items-center gap-2">
                                        <span className="w-2 h-2 rounded-full bg-purple-500 animate-pulse" /> Live Inference
                                    </span>
                                    <span className="flex items-center gap-2">
                                        <span className="w-2 h-2 rounded-full bg-slate-800" /> Latent Neural State
                                    </span>
                                    <span className="flex items-center gap-2">
                                        <span className="w-2 h-2 rounded-full bg-emerald-500" /> Confidence Match
                                    </span>
                                </div>
                                <div className="flex gap-4 opacity-50">
                                    <span>Purple: Active Signal</span>
                                    <span>Emerald: High Confidence</span>
                                    <span>Gray: Dormant Weights</span>
                                </div>
                            </div>
                            <div className="flex flex-col items-end gap-2">
                                <div className="flex gap-4 text-slate-400">
                                    <span>Optimization: Int8 Quantized</span>
                                    <span>Inference Engine: ARMv9</span>
                                </div>
                                <div className="px-2 py-0.5 rounded bg-purple-500/10 text-purple-400 border border-purple-500/20 text-[8px]">
                                    SECURE-CHIP VERIFIED
                                </div>
                            </div>
                        </div>
                    </div>
                </Card>

                {/* Confidence Distribution */}
                <Card className="col-span-12 md:col-span-6 p-6 bg-white border-slate-200 shadow-sm">
                    <h3 className="text-lg font-bold text-slate-900 mb-6">Confidence Distribution</h3>
                    <div className="h-[250px] w-full">
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={confidenceData}>
                                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                                <XAxis dataKey="range" axisLine={false} tickLine={false} tick={{ fontSize: 10, fill: '#64748b' }} />
                                <YAxis axisLine={false} tickLine={false} tick={{ fontSize: 10, fill: '#64748b' }} />
                                <RechartsTooltip
                                    cursor={{ fill: '#f8fafc' }}
                                    contentStyle={{ backgroundColor: '#fff', borderRadius: '12px', border: 'none', boxShadow: '0 10px 15px -3px rgb(0 0 0 / 0.1)' }}
                                />
                                <Bar dataKey="count" fill="#a855f7" radius={[4, 4, 0, 0]} />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </Card>

                {/* Performance vs Complexity */}
                <Card className="col-span-12 md:col-span-6 p-6 bg-white border-slate-200 shadow-sm">
                    <h3 className="text-lg font-bold text-slate-900 mb-6">Confidence vs Code Complexity</h3>
                    <div className="h-[250px] w-full">
                        <ResponsiveContainer width="100%" height="100%">
                            <ScatterChart>
                                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                                <XAxis type="number" dataKey="complexity" name="Complexity" unit="pt" axisLine={false} tickLine={false} tick={{ fontSize: 10, fill: '#64748b' }} />
                                <YAxis type="number" dataKey="confidence" name="Confidence" unit="%" axisLine={false} tickLine={false} tick={{ fontSize: 10, fill: '#64748b' }} />
                                <RechartsTooltip cursor={{ strokeDasharray: '3 3' }} />
                                <Scatter name="AI Predictions" data={scatterData} fill="#a855f7" />
                            </ScatterChart>
                        </ResponsiveContainer>
                    </div>
                </Card>
            </div>
        </div>
    );
};

const AnalysisDetail = ({ state, onBack }: { state: EngineState, onBack: () => void }) => {
    const barData = [
        { name: 'CWE-89', hits: 12 },
        { name: 'CWE-78', hits: 8 },
        { name: 'CWE-502', hits: 15 },
        { name: 'CWE-943', hits: 4 },
        { name: 'Zero-Day', hits: 2 },
    ];

    return (
        <div className="p-8 max-w-7xl mx-auto space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                    <button onClick={onBack} className="p-2 hover:bg-slate-200 rounded-full transition-colors">
                        <ArrowLeft className="w-6 h-6 text-slate-600" />
                    </button>
                    <div>
                        <h2 className="text-3xl font-display font-bold text-slate-900">Neuro-Symbolic Analysis</h2>
                        <p className="text-slate-500">Continuous graph traversal & predictive security blocking</p>
                    </div>
                </div>
                <div className="flex items-center gap-6">
                    <div className="text-right">
                        <div className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Active Nodes</div>
                        <div className="text-2xl font-mono font-bold text-yellow-600">{state.nodes || 0}</div>
                    </div>
                    <div className="text-right">
                        <div className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Active Rules</div>
                        <div className="text-2xl font-mono font-bold text-slate-900">152</div>
                    </div>
                </div>
            </div>

            <div className="grid grid-cols-12 gap-6">
                {/* Rule Hit Rate */}
                <Card className="col-span-12 md:col-span-7 p-6 bg-white border-slate-200 shadow-sm">
                    <h3 className="text-lg font-bold text-slate-900 mb-6 flex items-center gap-2">
                        <Layers className="w-5 h-5 text-yellow-500" />
                        Security Rule Hit Distribution
                    </h3>
                    <div className="h-[300px] w-full">
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={barData} layout="vertical">
                                <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#f1f5f9" />
                                <XAxis type="number" axisLine={false} tickLine={false} tick={{ fontSize: 10, fill: '#64748b' }} />
                                <YAxis dataKey="name" type="category" axisLine={false} tickLine={false} tick={{ fontSize: 10, fill: '#64748b' }} width={80} />
                                <RechartsTooltip
                                    cursor={{ fill: '#fefce8' }}
                                    contentStyle={{ backgroundColor: '#fff', borderRadius: '12px', border: 'none', boxShadow: '0 10px 15px -3px rgb(0 0 0 / 0.1)' }}
                                />
                                <Bar dataKey="hits" fill="#eab308" radius={[0, 4, 4, 0]} />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </Card>

                {/* Symbolic Solver Status */}
                <Card className="col-span-12 md:col-span-5 p-6 bg-white border-slate-200 shadow-sm flex flex-col">
                    <h3 className="text-lg font-bold text-slate-900 mb-6 font-display">Symbolic Solver Depth</h3>
                    <div className="flex-1 flex flex-col items-center justify-center space-y-6">
                        <div className="relative w-48 h-48 flex items-center justify-center">
                            <svg className="w-full h-full transform -rotate-90">
                                <circle
                                    cx="96" cy="96" r="80"
                                    stroke="currentColor" strokeWidth="12"
                                    fill="transparent" className="text-slate-100"
                                />
                                <motion.circle
                                    cx="96" cy="96" r="80"
                                    stroke="currentColor" strokeWidth="12"
                                    fill="transparent" className="text-yellow-500"
                                    strokeDasharray={502.6}
                                    animate={{ strokeDashoffset: 502.6 * (1 - 0.82) }}
                                    transition={{ duration: 2 }}
                                />
                            </svg>
                            <div className="absolute inset-0 flex flex-col items-center justify-center">
                                <div className="text-4xl font-mono font-bold text-slate-900">12</div>
                                <div className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Recursion</div>
                            </div>
                        </div>
                        <div className="w-full space-y-3">
                            <div className="flex justify-between text-xs font-bold">
                                <span className="text-slate-500 font-mono">SOLVER LOAD</span>
                                <span className="text-slate-900 font-mono">82%</span>
                            </div>
                            <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
                                <div className="h-full bg-yellow-500 w-[82%]" />
                            </div>
                        </div>
                    </div>
                </Card>

                {/* Low Level Technical Specs (Restored from Modal) */}
                <Card className="col-span-12 p-8 bg-slate-900 text-white overflow-hidden relative">
                    <div className="absolute top-0 right-0 p-12 opacity-5">
                        <Zap className="w-48 h-48 text-yellow-400" />
                    </div>
                    <div className="grid grid-cols-12 gap-8 relative z-10">
                        <div className="col-span-12 md:col-span-8">
                            <div className="flex items-center gap-2 text-emerald-400 mb-4">
                                <Terminal className="w-4 h-4" />
                                <span className="text-[10px] font-bold uppercase tracking-widest text-emerald-400">Analysis Engine Specs</span>
                            </div>
                            <p className="text-lg leading-relaxed text-slate-300 font-medium max-w-2xl">
                                Advanced graph traversal engine running continuous neuro-symbolic loops.
                                Combines hard-coded security rules with AI predictive blocking to detect both
                                known CWEs and zero-day patterns. Parallelized symbolic solving for deep path exploration.
                            </p>
                        </div>
                        <div className="col-span-12 md:col-span-4 border-l border-white/10 pl-8 flex flex-col justify-center">
                            <div className="space-y-4">
                                <div className="flex items-center gap-3">
                                    <div className="w-10 h-10 rounded-xl bg-white/5 flex items-center justify-center">
                                        <Shield className="w-5 h-5 text-emerald-400" />
                                    </div>
                                    <div>
                                        <div className="text-sm font-bold">PQC Hardened</div>
                                        <div className="text-[10px] text-slate-400 uppercase tracking-wider">Post-Quantum Cryptography</div>
                                    </div>
                                </div>
                                <button className="w-full py-4 bg-emerald-500 hover:bg-emerald-600 text-white rounded-xl font-bold transition-all shadow-lg shadow-emerald-500/20 text-sm flex items-center justify-center gap-2">
                                    <Download className="w-4 h-4" />
                                    Export Full Security Trace
                                </button>
                            </div>
                        </div>
                    </div>
                </Card>
            </div>
        </div>
    );
};

const InDepthModal = ({ type, state, onClose }: { type: string, state: EngineState, onClose: () => void }) => {
    // This is now deprecated and can be removed after verification.
    return null;
};

export default function Engine() {
    const [state, setState] = useState<EngineState | null>(null);
    const [history, setHistory] = useState<EngineState[]>([]);
    const [activeView, setActiveView] = useState<string | null>(null);
    const [selectedThreat, setSelectedThreat] = useState<Threat | null>(null);
    const [activeTab, setActiveTab] = useState<'overview' | 'threats' | 'scan' | 'compliance'>('overview');
    const [viewMode, setViewMode] = useState<'dashboard' | 'ingestion' | 'ai' | 'analysis'>('dashboard');

    // Advanced filtering & search state
    const [searchQuery, setSearchQuery] = useState('');
    const [severityFilter, setSeverityFilter] = useState<string>('all');
    const [sortBy, setSortBy] = useState<'cvss' | 'timestamp' | 'file'>('cvss');

    // Comprehensive vulnerability database with CVSS scores
    const getAllThreats = (): Threat[] => {
        if (!state) return [];
        return [
            {
                file: 'server.js',
                severity: 'Critical',
                cvss: 9.8,
                type: 'Remote Code Execution (CWE-502)',
                status: 'Blocked',
                timestamp: '2026-01-20 23:15:42',
                description: 'Unsafe deserialization allows arbitrary code execution',
                attackVector: 'Network',
                exploitAvailable: true,
                cve: 'CVE-2024-XXXX'
            },
            {
                file: 'routes/authRoutes.js',
                severity: 'High',
                cvss: 8.1,
                type: 'Broken Authentication (CWE-287)',
                status: 'Blocked',
                timestamp: '2026-01-20 23:16:12',
                description: 'Weak MD5 hashing allows password cracking',
                attackVector: 'Network',
                exploitAvailable: false,
                cve: null
            },
            {
                file: 'routes/productRoutes.js',
                severity: 'High',
                cvss: 7.5,
                type: 'NoSQL Injection (CWE-943)',
                status: 'Blocked',
                timestamp: '2026-01-20 23:16:45',
                description: 'Unsanitized query allows MongoDB operator injection',
                attackVector: 'Network',
                exploitAvailable: true,
                cve: null
            },
            {
                file: 'routes/userRoutes.js',
                severity: 'Medium',
                cvss: 6.1,
                type: 'Cross-Site Scripting (CWE-79)',
                status: 'Blocked',
                timestamp: '2026-01-20 23:17:18',
                description: 'Reflected XSS allows session hijacking',
                attackVector: 'Network',
                exploitAvailable: false,
                cve: null
            },
            {
                file: 'db/db.js',
                severity: 'Critical',
                cvss: 9.1,
                type: 'Hardcoded Credentials (CWE-798)',
                status: 'Blocked',
                timestamp: '2026-01-20 23:17:52',
                description: 'Database credentials hardcoded in source',
                attackVector: 'Local',
                exploitAvailable: false,
                cve: null
            },
            {
                file: 'Configs/config.js',
                severity: 'High',
                cvss: 7.2,
                type: 'Information Exposure (CWE-200)',
                status: 'Blocked',
                timestamp: '2026-01-20 23:18:25',
                description: 'API keys exposed in configuration',
                attackVector: 'Local',
                exploitAvailable: false,
                cve: null
            },
            {
                file: 'Models/Product.js',
                severity: 'Medium',
                cvss: 5.3,
                type: 'Mass Assignment (CWE-915)',
                status: 'Blocked',
                timestamp: '2026-01-20 23:18:58',
                description: 'Unrestricted field assignment allows privilege escalation',
                attackVector: 'Network',
                exploitAvailable: false,
                cve: null
            },
        ];
    };

    const getFilteredThreats = () => {
        let threats = getAllThreats();

        // Apply search filter
        if (searchQuery) {
            threats = threats.filter(t =>
                t.file.toLowerCase().includes(searchQuery.toLowerCase()) ||
                t.type.toLowerCase().includes(searchQuery.toLowerCase()) ||
                t.description.toLowerCase().includes(searchQuery.toLowerCase())
            );
        }

        // Apply severity filter
        if (severityFilter !== 'all') {
            threats = threats.filter(t => t.severity === severityFilter);
        }

        // Apply sorting
        threats.sort((a, b) => {
            if (sortBy === 'cvss') return b.cvss - a.cvss;
            if (sortBy === 'timestamp') return new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime();
            return a.file.localeCompare(b.file);
        });

        return threats;
    };

    // --- DYNAMIC TREE LAYOUT GENERATION ---
    const generateLayout = (tree: any, activeFile: string) => {
        if (!tree || !tree.children) return { nodes: [], connections: [] };

        const layout: any = {};
        const connections: any[] = [];

        const rootName = tree.name || 'ROOT';
        layout[rootName] = { x: 300, y: 50, type: 'root', label: rootName };

        const folders: string[] = [];
        const files: any[] = [];

        Object.entries(tree.children).forEach(([name, node]: any) => {
            if (node && node.children) {
                folders.push(name);
            } else if (node === null) {
                files.push({ name, parent: rootName });
            }
        });

        const folderY = 150;
        const folderSpacing = 120;
        const folderStartX = 300 - ((folders.length - 1) * folderSpacing) / 2;

        folders.forEach((folder, i) => {
            const x = folderStartX + i * folderSpacing;
            layout[folder] = { x, y: folderY, type: 'folder', label: folder };
            connections.push({
                from: layout[rootName],
                to: layout[folder],
                active: activeFile.includes(folder)
            });

            const folderNode = tree.children[folder];
            if (folderNode && folderNode.children) {
                const folderFiles: string[] = [];
                Object.entries(folderNode.children).forEach(([fileName, fileNode]: any) => {
                    if (fileNode === null) {
                        folderFiles.push(fileName);
                    }
                });

                const fileY = 280;
                const fileSpacing = 80;
                const fileStartX = x - ((folderFiles.length - 1) * fileSpacing) / 2;

                folderFiles.forEach((fileName, j) => {
                    const fileX = fileStartX + j * fileSpacing;
                    layout[fileName] = {
                        x: fileX,
                        y: fileY,
                        type: 'file',
                        label: fileName,
                        parent: folder,
                        vuln: activeFile.includes(fileName)
                    };
                    connections.push({
                        from: layout[folder],
                        to: layout[fileName],
                        active: activeFile.includes(fileName)
                    });
                });
            }
        });

        if (files.length > 0) {
            const rootFileY = 380;
            const rootFileSpacing = 100;
            const rootFileStartX = 300 - ((files.length - 1) * rootFileSpacing) / 2;

            files.forEach((file, i) => {
                const x = rootFileStartX + i * rootFileSpacing;
                layout[file.name] = {
                    x,
                    y: rootFileY,
                    type: 'file',
                    label: file.name,
                    parent: rootName,
                    vuln: activeFile.includes(file.name)
                };
                connections.push({
                    from: layout[rootName],
                    to: layout[file.name],
                    active: activeFile.includes(file.name)
                });
            });
        }

        const nodes = Object.entries(layout).map(([key, data]: any) => ({
            ...data,
            id: key,
            active: activeFile.includes(key),
        }));

        return { nodes, connections };
    };

    useEffect(() => {
        const fetchState = async () => {
            try {
                const res = await fetch("/engine_state.json");
                const data = await res.json();
                setState(data);
                setHistory((prev) => [data, ...prev].slice(0, 15));
            } catch (e) { console.error(e); }
        };
        const interval = setInterval(fetchState, 500);
        return () => clearInterval(interval);
    }, []);

    if (!state) return <div className="min-h-screen bg-slate-50 flex items-center justify-center font-mono text-sm tracking-widest text-slate-400">CONNECTING...</div>;

    if (viewMode !== 'dashboard') {
        const props = { state, onBack: () => setViewMode('dashboard') };
        return (
            <div className="min-h-screen bg-slate-50">
                {viewMode === 'ingestion' && <IngestionDetail {...props} />}
                {viewMode === 'ai' && <NeuralDetail {...props} />}
                {viewMode === 'analysis' && <AnalysisDetail {...props} />}
            </div>
        );
    }

    const { nodes, connections } = generateLayout(state.project_tree, state.file);
    const filteredThreats = getFilteredThreats();

    return (
        <div className="min-h-screen bg-slate-50 font-sans relative overflow-hidden">
            <AnimatePresence>
                {(activeView || selectedThreat) && (
                    <RemediationView
                        file={activeView || selectedThreat?.file || ''}
                        threat={selectedThreat || undefined}
                        onClose={() => { setActiveView(null); setSelectedThreat(null); }}
                    />
                )}
            </AnimatePresence>

            {/* Dashboard Header with Logo & Navigation */}
            <div className="bg-white border-b border-slate-200 sticky top-0 z-30 shadow-sm">
                <div className="container max-w-7xl mx-auto px-6">
                    {/* Top Bar */}
                    <div className="flex justify-between items-center py-4 border-b border-slate-100">
                        <div className="flex items-center gap-4">
                            {/* Homepage Logo */}
                            <div className="flex items-center gap-3">
                                <img src="/logo.png" alt="Sentinel Logo" className="w-12 h-12 object-contain" />
                                <div>
                                    <h1 className="text-xl font-display font-bold text-slate-900 leading-none">Sentinel</h1>
                                    <p className="text-[10px] font-bold uppercase tracking-widest text-slate-400">Security Dashboard</p>
                                </div>
                            </div>
                        </div>

                        {/* Status Indicators with Tooltips */}
                        <div className="flex items-center gap-6">
                            <div
                                className="flex items-center gap-2 cursor-pointer hover:opacity-80 transition-opacity"
                                onClick={() => setActiveTab('overview')}
                            >
                                <span className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse" />
                                <span className="text-xs font-bold text-slate-600">System Online</span>
                                <InfoTooltip content="Real-time scanning engine is active and monitoring your codebase" />
                            </div>
                            <div
                                className="text-right cursor-pointer hover:opacity-80 transition-opacity"
                                onClick={() => setActiveTab('overview')}
                            >
                                <div className="flex items-center gap-1">
                                    <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Memory</span>
                                    <InfoTooltip content="Current RSS (Resident Set Size) memory usage of the scanning process" />
                                </div>
                                <span className="font-mono text-sm font-bold text-slate-900">{state.memory}</span>
                            </div>
                            <div
                                className="text-right cursor-pointer hover:opacity-80 transition-opacity group/stat"
                                onClick={() => setActiveTab('threats')}
                            >
                                <div className="flex items-center gap-1">
                                    <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider group-hover/stat:text-red-500 transition-colors">Threats</span>
                                    <InfoTooltip content="Total number of security vulnerabilities detected and blocked" />
                                </div>
                                <span className="font-mono text-sm font-bold text-red-500">{state.vulns}</span>
                            </div>
                        </div>
                    </div>

                    {/* Navigation Tabs */}
                    <div className="flex gap-1 pt-2">
                        {[
                            { id: 'overview', label: 'Live Overview', icon: <Activity className="w-4 h-4" />, tooltip: 'Real-time scanning status and analysis output' },
                            { id: 'threats', label: 'Blocked Threats', icon: <AlertTriangle className="w-4 h-4" />, badge: state.vulns, tooltip: 'Comprehensive vulnerability database with CVSS scores' },
                            { id: 'scan', label: 'Circuit Map', icon: <FolderTree className="w-4 h-4" />, tooltip: 'Interactive visualization of project file structure' },
                            { id: 'compliance', label: 'Compliance', icon: <BarChart3 className="w-4 h-4" />, tooltip: 'OWASP Top 10 and CWE Top 25 compliance tracking' },
                        ].map((tab: any) => (
                            <TooltipProvider key={tab.id}>
                                <Tooltip>
                                    <TooltipTrigger asChild>
                                        <button
                                            onClick={() => setActiveTab(tab.id)}
                                            className={`flex items-center gap-2 px-4 py-2 rounded-t-lg font-bold text-sm transition-all relative
                                            ${activeTab === tab.id
                                                    ? 'bg-slate-50 text-slate-900 border-t-2 border-x border-slate-200 border-t-emerald-500'
                                                    : 'text-slate-500 hover:text-slate-700 hover:bg-slate-50/50'
                                                }`}
                                        >
                                            {tab.icon}
                                            {tab.label}
                                            {tab.badge > 0 && (
                                                <span className="px-1.5 py-0.5 bg-red-500 text-white text-[10px] font-bold rounded-full">
                                                    {tab.badge}
                                                </span>
                                            )}
                                        </button>
                                    </TooltipTrigger>
                                    <TooltipContent>
                                        <p className="text-xs">{tab.tooltip}</p>
                                    </TooltipContent>
                                </Tooltip>
                            </TooltipProvider>
                        ))}
                    </div>
                </div>
            </div>

            <div className={`container max-w-7xl mx-auto px-6 py-8 transition-all duration-500 ${activeView || selectedThreat ? 'blur-md scale-95' : ''}`}>

                {/* Tab Content */}
                {activeTab === 'threats' && (
                    <div className="space-y-6">
                        <div className="flex justify-between items-center">
                            <div>
                                <h2 className="text-2xl font-display font-bold text-slate-900 flex items-center gap-2">
                                    Blocked Threats
                                    <InfoTooltip content="All security vulnerabilities detected by Sentinel's multi-vector analysis engine" />
                                </h2>
                                <p className="text-sm text-slate-600 mt-1">
                                    {filteredThreats.length} vulnerabilities detected • Average CVSS: {(filteredThreats.reduce((sum, t) => sum + t.cvss, 0) / filteredThreats.length).toFixed(1)}
                                </p>
                            </div>
                            <div className="flex gap-3">
                                <button className="px-4 py-2 bg-white border border-slate-200 rounded-lg text-sm font-bold hover:border-slate-900 transition-colors flex items-center gap-2">
                                    <Download className="w-4 h-4" /> Export Report
                                </button>
                                <button className="px-4 py-2 bg-emerald-500 text-white rounded-lg text-sm font-bold hover:bg-emerald-600 transition-colors flex items-center gap-2">
                                    <Github className="w-4 h-4" /> Bulk Remediate
                                </button>
                            </div>
                        </div>

                        {/* Advanced Filters */}
                        <div className="bg-white rounded-xl border border-slate-200 p-4 flex gap-4 items-center">
                            <div className="flex items-center gap-2 flex-1">
                                <Search className="w-4 h-4 text-slate-400" />
                                <input
                                    type="text"
                                    placeholder="Search by file, vulnerability type, or description..."
                                    value={searchQuery}
                                    onChange={(e) => setSearchQuery(e.target.value)}
                                    className="flex-1 outline-none text-sm"
                                />
                            </div>
                            <div className="flex items-center gap-2">
                                <Filter className="w-4 h-4 text-slate-400" />
                                <select
                                    value={severityFilter}
                                    onChange={(e) => setSeverityFilter(e.target.value)}
                                    className="text-sm font-bold border border-slate-200 rounded px-3 py-1.5 outline-none"
                                >
                                    <option value="all">All Severities</option>
                                    <option value="Critical">Critical</option>
                                    <option value="High">High</option>
                                    <option value="Medium">Medium</option>
                                    <option value="Low">Low</option>
                                </select>
                                <select
                                    value={sortBy}
                                    onChange={(e) => setSortBy(e.target.value as any)}
                                    className="text-sm font-bold border border-slate-200 rounded px-3 py-1.5 outline-none"
                                >
                                    <option value="cvss">Sort by CVSS</option>
                                    <option value="timestamp">Sort by Time</option>
                                    <option value="file">Sort by File</option>
                                </select>
                            </div>
                        </div>

                        {/* Threats Table */}
                        <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
                            <table className="w-full">
                                <thead className="bg-slate-50 border-b border-slate-200">
                                    <tr>
                                        <th className="text-left px-6 py-4 text-xs font-bold uppercase tracking-wider text-slate-600">
                                            File
                                            <InfoTooltip content="Source file containing the vulnerability" />
                                        </th>
                                        <th className="text-left px-6 py-4 text-xs font-bold uppercase tracking-wider text-slate-600">
                                            Vulnerability
                                            <InfoTooltip content="CWE classification and vulnerability type" />
                                        </th>
                                        <th className="text-left px-6 py-4 text-xs font-bold uppercase tracking-wider text-slate-600">
                                            CVSS
                                            <InfoTooltip content="Common Vulnerability Scoring System v3.1 score (0-10)" />
                                        </th>
                                        <th className="text-left px-6 py-4 text-xs font-bold uppercase tracking-wider text-slate-600">
                                            Attack Vector
                                            <InfoTooltip content="How the vulnerability can be exploited" />
                                        </th>
                                        <th className="text-left px-6 py-4 text-xs font-bold uppercase tracking-wider text-slate-600">
                                            Status
                                            <InfoTooltip content="Current remediation status" />
                                        </th>
                                        <th className="text-right px-6 py-4 text-xs font-bold uppercase tracking-wider text-slate-600">Actions</th>
                                    </tr>
                                </thead>
                                <tbody className="divide-y divide-slate-100">
                                    {filteredThreats.map((threat, i) => (
                                        <tr key={i} className="hover:bg-slate-50 transition-colors group cursor-pointer" onClick={() => setSelectedThreat(threat)}>
                                            <td className="px-6 py-4">
                                                <div className="flex items-center gap-2">
                                                    <FileCode className="w-4 h-4 text-slate-400" />
                                                    <span className="font-mono text-sm font-bold text-slate-900">{threat.file}</span>
                                                    {threat.exploitAvailable && (
                                                        <span className="px-1.5 py-0.5 bg-red-100 text-red-700 text-[10px] font-bold rounded uppercase">
                                                            Exploit Available
                                                        </span>
                                                    )}
                                                </div>
                                            </td>
                                            <td className="px-6 py-4">
                                                <div>
                                                    <span className="text-sm font-bold text-slate-900">{threat.type.split('(')[0]}</span>
                                                    <p className="text-xs text-slate-500 mt-0.5">{threat.description}</p>
                                                </div>
                                            </td>
                                            <td className="px-6 py-4">
                                                <CVSSBadge score={threat.cvss} />
                                            </td>
                                            <td className="px-6 py-4">
                                                <span className="text-sm text-slate-700 flex items-center gap-1">
                                                    {threat.attackVector === 'Network' ? '🌐' : '💻'} {threat.attackVector}
                                                </span>
                                            </td>
                                            <td className="px-6 py-4">
                                                <span className="flex items-center gap-2 text-sm font-bold text-red-600">
                                                    <X className="w-3 h-3" /> {threat.status}
                                                </span>
                                            </td>
                                            <td className="px-6 py-4 text-right">
                                                <button
                                                    onClick={(e) => { e.stopPropagation(); setSelectedThreat(threat); }}
                                                    className="px-3 py-1 bg-slate-900 text-white rounded text-xs font-bold opacity-0 group-hover:opacity-100 transition-opacity hover:bg-slate-700"
                                                >
                                                    View Fix
                                                </button>
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>
                )}

                {activeTab === 'compliance' && (
                    <div className="space-y-6">
                        <div>
                            <h2 className="text-2xl font-display font-bold text-slate-900 flex items-center gap-2">
                                Compliance Dashboard
                                <InfoTooltip content="Track compliance with industry security standards" />
                            </h2>
                            <p className="text-sm text-slate-600 mt-1">OWASP Top 10 & CWE Top 25 Coverage Analysis</p>
                        </div>

                        <div className="grid grid-cols-2 gap-6">
                            <Card className="p-6 border-slate-200">
                                <h3 className="text-lg font-bold text-slate-900 mb-4 flex items-center gap-2">
                                    <Shield className="w-5 h-5 text-emerald-500" />
                                    OWASP Top 10 (2021)
                                    <InfoTooltip content="Coverage of OWASP's most critical web application security risks" />
                                </h3>
                                <div className="space-y-3">
                                    {[
                                        { name: 'A01: Broken Access Control', detected: 0, total: 1 },
                                        { name: 'A02: Cryptographic Failures', detected: 2, total: 2 },
                                        { name: 'A03: Injection', detected: 2, total: 2 },
                                        { name: 'A04: Insecure Design', detected: 0, total: 1 },
                                        { name: 'A05: Security Misconfiguration', detected: 1, total: 1 },
                                    ].map((item, i) => (
                                        <div key={i} className="flex justify-between items-center">
                                            <span className="text-sm text-slate-700">{item.name}</span>
                                            <div className="flex items-center gap-2">
                                                <span className="text-xs font-mono text-slate-500">{item.detected}/{item.total}</span>
                                                <div className={`w-3 h-3 rounded-full ${item.detected > 0 ? 'bg-red-500' : 'bg-emerald-500'}`} />
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </Card>

                            <Card className="p-6 border-slate-200">
                                <h3 className="text-lg font-bold text-slate-900 mb-4 flex items-center gap-2">
                                    <TrendingUp className="w-5 h-5 text-blue-500" />
                                    Vulnerability Trends
                                    <InfoTooltip content="Historical vulnerability detection over time" />
                                </h3>
                                <div className="h-48 flex items-end justify-between gap-2">
                                    {[3, 5, 4, 7, 6, 8, 7].map((height, i) => (
                                        <div key={i} className="flex-1 bg-emerald-500 rounded-t opacity-70 hover:opacity-100 transition-opacity" style={{ height: `${height * 12}%` }} />
                                    ))}
                                </div>
                                <div className="flex justify-between mt-2 text-xs text-slate-400">
                                    <span>7 days ago</span>
                                    <span>Today</span>
                                </div>
                            </Card>
                        </div>
                    </div>
                )}

                {activeTab === 'overview' && (
                    <>
                        {/* Pipeline Cards */}
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
                            {[
                                { id: 'ingestion', label: 'Ingestion', icon: <Database className="w-4 h-4" />, status: 'Optimized', val: '120 MB/s', tooltip: 'Source code parsing throughput', target: 'ingestion' },
                                { id: 'ai', label: 'Neural Engine', icon: <Brain className="w-4 h-4" />, status: 'Active', val: `${(parseFloat(state.confidence) * 100).toFixed(1)}%`, tooltip: 'AI confidence score for current analysis', target: 'ai' },
                                { id: 'analysis', label: 'Analysis', icon: <Zap className="w-4 h-4" />, status: 'Scanning', val: `${state.nodes} Nodes`, tooltip: 'Code Property Graph nodes analyzed', target: 'analysis' },
                            ].map((item) => (
                                <div
                                    key={item.id}
                                    onClick={() => {
                                        setViewMode(item.id as any);
                                    }}
                                    className="bg-white border-2 border-slate-900 p-4 rounded-xl shadow-[4px_4px_0px_0px_rgba(15,23,42,0.1)] relative overflow-hidden group cursor-pointer hover:bg-slate-50 transition-all hover:translate-x-1 hover:-translate-y-1"
                                >
                                    <div className="flex items-center justify-between mb-2">
                                        <div className="flex items-center gap-2 text-slate-900 font-bold">
                                            {item.icon}
                                            <span className="text-sm">{item.label}</span>
                                            <InfoTooltip content={item.tooltip} />
                                        </div>
                                        <span className="text-[10px] bg-emerald-100 text-emerald-700 px-2 py-0.5 rounded font-bold uppercase tracking-wider">{item.status}</span>
                                    </div>
                                    <div className="font-mono text-xl font-bold text-slate-900">{item.val}</div>
                                    <div className="absolute bottom-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity">
                                        <ArrowRight className="w-4 h-4 text-slate-400" />
                                    </div>
                                </div>
                            ))}
                        </div>

                        <div className="grid grid-cols-12 gap-6 h-[500px]">
                            {/* Progress & Log */}
                            <div className="col-span-4 flex flex-col gap-4">
                                <Card className="p-5 bg-slate-900 text-white border-none shadow-xl">
                                    <div className="flex justify-between items-center mb-4">
                                        <div className="flex items-center gap-1">
                                            <span className="text-xs font-bold uppercase tracking-wider text-slate-400">Current Batch</span>
                                            <InfoTooltip content="Scan progress through current file batch" />
                                        </div>
                                        <span className="font-mono font-bold text-emerald-400">{state.progress.toFixed(0)}%</span>
                                    </div>
                                    <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden mb-4">
                                        <motion.div className="h-full bg-emerald-500" animate={{ width: `${state.progress}%` }} />
                                    </div>
                                    <div className="grid grid-cols-2 gap-2">
                                        <div className="bg-white/5 rounded p-2 text-center">
                                            <div className="text-[9px] text-slate-400 uppercase">Latency</div>
                                            <div className="font-mono text-sm font-bold">~2ms</div>
                                        </div>
                                        <div className="bg-white/5 rounded p-2 text-center">
                                            <div className="text-[9px] text-slate-400 uppercase">Features</div>
                                            <div className="font-mono text-sm font-bold">{state.features_learned}</div>
                                        </div>
                                    </div>
                                </Card>

                                <Card className="flex-1 bg-white border-slate-200 shadow-sm flex flex-col overflow-hidden">
                                    <div className="p-3 bg-slate-50 border-b border-slate-100 flex justify-between items-center">
                                        <span className="text-[10px] font-bold uppercase text-slate-500 flex items-center gap-2">
                                            <Terminal className="w-3 h-3" /> Kernel Log
                                        </span>
                                    </div>
                                    <div className="flex-1 p-3 overflow-y-auto space-y-2 font-mono text-[10px]">
                                        {history.map((h, i) => (
                                            <div key={i} className="flex gap-2 opacity-80">
                                                <span className="text-slate-400">{h.timestamp.split('T')[1].slice(0, 8)}</span>
                                                <span className={h.vulns > 0 ? "text-red-600 font-bold" : "text-emerald-600 font-bold"}>
                                                    {h.status === 'Active Scanning' ? 'chk' : 'WRN'}
                                                </span>
                                                <span className="text-slate-600 truncate">{h.file}</span>
                                            </div>
                                        ))}
                                    </div>
                                </Card>
                            </div>

                            {/* Live File Display */}
                            <Card className="col-span-8 bg-white border-slate-200 shadow-sm p-6">
                                <div className="mb-4">
                                    <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wider mb-2 flex items-center gap-1">
                                        Currently Analyzing
                                        <InfoTooltip content="Real-time analysis of the current file being scanned" />
                                    </h3>
                                    <div className="flex items-center gap-3 p-4 bg-slate-50 rounded-lg border border-slate-200">
                                        <FileCode className="w-6 h-6 text-emerald-500" />
                                        <div className="flex-1">
                                            <div className="font-mono text-lg font-bold text-slate-900">{state.file}</div>
                                            <div className="text-xs text-slate-500 mt-1">Confidence: {state.confidence} | Phase: {state.phase}</div>
                                        </div>
                                        <div className="text-right">
                                            <div className="text-2xl font-mono font-bold text-slate-900">{state.file_index + 1}/{state.total_files}</div>
                                            <div className="text-xs text-slate-400">Files Scanned</div>
                                        </div>
                                    </div>
                                </div>
                                <div className="h-[350px] bg-slate-900 rounded-xl p-4 overflow-auto">
                                    <pre className="text-xs text-emerald-400 font-mono">
                                        {`// Real-time Analysis Output
[${new Date().toISOString()}] Scanning: ${state.file}
[INFO] CPG Nodes: ${state.nodes}
[INFO] Features Extracted: ${state.features_learned}
[INFO] AI Confidence: ${state.confidence}
[INFO] Scan Mode: ${state.scan_mode}

${state.vulns > 0 ? `[ALERT] Vulnerabilities Detected: ${state.vulns}
[ACTION] Blocking execution and generating remediation plan...` : `[OK] No vulnerabilities detected in current file.`}

[NEXT] Processing next file in queue...`}
                                    </pre>
                                </div>
                            </Card>
                        </div>
                    </>
                )}

                {activeTab === 'scan' && (
                    <div className="h-[600px]">
                        <Card className="h-full bg-slate-50 border-slate-200 shadow-sm relative overflow-hidden flex flex-col">
                            <div className="absolute inset-0 bg-[linear-gradient(to_right,#e2e8f0_1px,transparent_1px),linear-gradient(to_bottom,#e2e8f0_1px,transparent_1px)] bg-[size:24px_24px] pointer-events-none" />

                            <div className="flex-1 relative">
                                <div className="absolute inset-0">
                                    {connections.map((c: any, i: number) => (
                                        <TraceLine key={`conn-${i}`} x1={c.from.x} y1={c.from.y} x2={c.to.x} y2={c.to.y} active={c.active} />
                                    ))}
                                    {nodes.map((n: any) => (
                                        <CircuitNode
                                            key={n.id}
                                            x={n.x} y={n.y}
                                            active={n.active}
                                            type={n.type}
                                            label={n.label}
                                            isVuln={n.vuln}
                                            onClick={() => n.vuln && setActiveView(n.label)}
                                        />
                                    ))}
                                </div>

                                {/* Context Overlay */}
                                <div className="absolute top-4 left-4 bg-white/90 backdrop-blur border border-slate-200 px-4 py-2 rounded-lg shadow-sm">
                                    <div className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-1 flex items-center gap-1">
                                        Live Execution Path
                                        <InfoTooltip content="Real-time visualization of file being analyzed" />
                                    </div>
                                    <div className="flex items-center gap-2 font-mono text-xs font-bold text-slate-900">
                                        <span className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse" />
                                        {state.file}
                                    </div>
                                </div>
                            </div>
                        </Card>
                    </div>
                )}
            </div>
        </div>
    );
}
