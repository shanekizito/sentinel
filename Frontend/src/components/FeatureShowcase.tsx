import { ReactNode } from "react";
import { cn } from "@/lib/utils";
import { Shield, Zap, Globe, GitPullRequest, Terminal, FileCode, CheckCircle2, Lock, Layout, Workflow, Server } from "lucide-react";

// Tech Icon Container
const TechIcon = ({ icon }: { icon: ReactNode }) => (
    <div className="w-10 h-10 rounded-lg bg-[#F5F5F3] border border-border flex items-center justify-center text-foreground group-hover:bg-primary/10 group-hover:text-primary transition-colors duration-300">
        {icon}
    </div>
);

export const FeatureShowcase = () => {
    return (
        <section className="py-24 relative overflow-hidden bg-[#FDFDFB]">
            <div className="container mx-auto px-6 relative z-10 space-y-32">

                {/* Section 1: Auto-Remediation (Split Layout - NO CARD) */}
                <div className="flex flex-col lg:flex-row items-center gap-16">
                    <div className="flex-1 space-y-8">
                        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-50 border border-emerald-200 text-xs font-medium text-emerald-700">
                            <Zap className="w-3 h-3" />
                            <span>Autonomous Fixes</span>
                        </div>

                        <h2 className="text-4xl lg:text-5xl font-display font-bold text-foreground leading-[1.1]">
                            Sentinel writes the patch.<br /> You just merge.
                        </h2>

                        <p className="text-lg text-muted-foreground leading-relaxed max-w-xl">
                            Stop wasting engineering cycles on repetitive vulnerability fixes. Sentinel analyzes the context, generates a secure implementation, and opens a Pull Request instantly.
                        </p>

                        <div className="flex gap-4 pt-4">
                            <div className="flex flex-col gap-1">
                                <span className="text-2xl font-bold font-mono text-foreground">1.2s</span>
                                <span className="text-xs text-muted-foreground uppercase tracking-wider">Mean Time to Fix</span>
                            </div>
                            <div className="w-[1px] h-12 bg-border" />
                            <div className="flex flex-col gap-1">
                                <span className="text-2xl font-bold font-mono text-foreground">94%</span>
                                <span className="text-xs text-muted-foreground uppercase tracking-wider">Acceptance Rate</span>
                            </div>
                        </div>
                    </div>

                    {/* Visual - Floating */}
                    <div className="flex-1 w-full">
                        <div className="relative rounded-xl border border-border bg-white shadow-2xl p-6 rotate-1 hover:rotate-0 transition-transform duration-700 max-w-lg mx-auto lg:mr-0">
                            <div className="flex items-center gap-3 mb-6 border-b border-border pb-4">
                                <div className="w-8 h-8 rounded-full bg-purple-100 flex items-center justify-center">
                                    <GitPullRequest className="w-4 h-4 text-purple-600" />
                                </div>
                                <div>
                                    <div className="text-sm font-bold text-foreground">fix/sql-injection-user-query</div>
                                    <div className="text-xs text-muted-foreground font-mono">#42 • Open 12s ago by sentinel-bot</div>
                                </div>
                            </div>

                            <div className="space-y-3 font-mono text-xs">
                                <div className="p-3 bg-red-50 text-red-700 rounded border border-red-100 line-through opacity-70">
                                    - const query = "SELECT * FROM users WHERE id = " + req.id;
                                </div>
                                <div className="p-3 bg-emerald-50 text-emerald-700 rounded border border-emerald-100 font-medium">
                                    + const query = "SELECT * FROM users WHERE id = $1";
                                    <br />+ const values = [req.id];
                                </div>
                            </div>

                            <div className="mt-6 flex justify-between items-center">
                                <div className="flex -space-x-2">
                                    {[1, 2, 3].map(i => <div key={i} className="w-6 h-6 rounded-full bg-gray-200 border-2 border-white" />)}
                                </div>
                                <button className="px-4 py-2 bg-foreground text-background text-xs font-bold rounded hover:bg-foreground/90 transition-colors">
                                    Review Changes
                                </button>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Section 2: Compliance (List Layout - NO CARD) */}
                <div className="max-w-4xl mx-auto">
                    <div className="text-center mb-16">
                        <h2 className="text-3xl font-display font-bold text-foreground mb-4">Compliance on Autopilot</h2>
                        <p className="text-muted-foreground">Map your security posture to global standards automatically.</p>
                    </div>

                    <div className="grid md:grid-cols-2 gap-x-12 gap-y-8">
                        {[
                            { title: "SOC2 Type II", desc: "Access controls, audit logging, and change management mapped.", status: "Ready" },
                            { title: "HIPAA", desc: "PHI data flow monitoring and encryption verification.", status: "Verified" },
                            { title: "ISO 27001", desc: "Information security management systems controls.", status: "Ready" },
                            { title: "GDPR", desc: "Data privacy, consent management, and right-to-be-forgotten.", status: "Verified" },
                        ].map((item) => (
                            <div key={item.title} className="flex gap-4 items-start group">
                                <div className="w-8 h-8 rounded-full bg-emerald-100 flex items-center justify-center shrink-0 mt-1">
                                    <CheckCircle2 className="w-4 h-4 text-emerald-600" />
                                </div>
                                <div>
                                    <h3 className="text-lg font-bold text-foreground flex items-center gap-3">
                                        {item.title}
                                        <span className="px-2 py-0.5 rounded-full bg-gray-100 text-[10px] text-gray-500 font-mono uppercase tracking-wide border border-border">
                                            {item.status}
                                        </span>
                                    </h3>
                                    <p className="text-sm text-muted-foreground mt-1 leading-relaxed group-hover:text-foreground transition-colors">
                                        {item.desc}
                                    </p>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Section 3: Feature Clusters (Tight Icon Layout - NO BIG CARDS) */}
                <div className="grid md:grid-cols-3 gap-8 pt-12 border-t border-border">
                    {[
                        {
                            icon: <Globe className="w-5 h-5" />,
                            title: "Cloud Security",
                            desc: "Monitor IAM, S3, and API Gateways."
                        },
                        {
                            icon: <Lock className="w-5 h-5" />,
                            title: "Secret Detection",
                            desc: "Stop API keys from leaking to prod."
                        },
                        {
                            icon: <Terminal className="w-5 h-5" />,
                            title: "CI/CD Integration",
                            desc: "Native controls for GitHub & GitLab."
                        },
                        {
                            icon: <FileCode className="w-5 h-5" />,
                            title: "Custom Rules",
                            desc: "Write policy as code with Python."
                        },
                        {
                            icon: <Layout className="w-5 h-5" />,
                            title: "Dependency Graph",
                            desc: "Visualize how data flows through libs."
                        },
                        {
                            icon: <Server className="w-5 h-5" />,
                            title: "Container Scan",
                            desc: "Vulnerability analysis for Docker/K8s."
                        },
                    ].map((feature, i) => (
                        <div key={i} className="flex gap-4 group cursor-default">
                            <TechIcon icon={feature.icon} />
                            <div>
                                <h4 className="font-bold text-foreground text-sm mb-1 group-hover:text-primary transition-colors">{feature.title}</h4>
                                <p className="text-xs text-muted-foreground leading-relaxed pr-4">{feature.desc}</p>
                            </div>
                        </div>
                    ))}
                </div>

            </div>
        </section>
    );
};
