import { ReactNode } from "react";
import { cn } from "@/lib/utils";
import { Shield, Zap, Globe, GitPullRequest, Terminal, FileCode, CheckCircle2, Lock } from "lucide-react";

interface BentoItemProps {
    title: string;
    description: string;
    icon: ReactNode;
    className?: string;
    graphic?: ReactNode;
    stats?: string;
}

const BentoItem = ({ title, description, icon, className, graphic, stats }: BentoItemProps) => (
    <div className={cn(
        "blueprint-card p-6 flex flex-col justify-between h-full group",
        className
    )}>
        <div className="space-y-4 z-10 relative">
            <div className="flex items-center justify-between">
                <div className="w-10 h-10 rounded-lg bg-secondary border border-border flex items-center justify-center text-foreground group-hover:scale-110 transition-transform group-hover:bg-primary/10 group-hover:text-primary">
                    {icon}
                </div>
                {stats && (
                    <span className="text-[10px] font-mono font-bold text-primary bg-primary/5 px-2 py-1 rounded border border-primary/20">
                        {stats}
                    </span>
                )}
            </div>

            <div>
                <h3 className="text-xl font-display font-bold text-foreground mb-2 group-hover:text-primary transition-colors duration-300">{title}</h3>
                <p className="text-muted-foreground text-sm leading-relaxed">{description}</p>
            </div>
        </div>
        {graphic && <div className="mt-6 border-t border-border/50 pt-4">{graphic}</div>}
    </div>
);

export const FeatureBentoGrid = () => {
    return (
        <section className="py-24 relative overflow-hidden bg-[#FDFDFB]">
            <div className="absolute inset-0 bg-grid-fine opacity-40 pointer-events-none" />

            <div className="container mx-auto px-6 relative z-10">
                <div className="text-center max-w-2xl mx-auto mb-16">
                    <h2 className="font-display text-4xl font-bold mb-4 text-foreground">Complete Security Coverage</h2>
                    <p className="text-muted-foreground text-lg">Sentinel protects every layer of your AI-generated stack, from raw prompts to production deployments.</p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-6 auto-rows-[380px]">
                    {/* Large Featured Item: Auto-Remediation */}
                    <div className="md:col-span-2 row-span-1 blueprint-card p-8 group">
                        <div className="absolute inset-0 bg-grid-isometric opacity-[0.05] pointer-events-none" />

                        <div className="relative z-20 h-full flex flex-col sm:flex-row gap-8 items-center">
                            <div className="flex-1 space-y-6">
                                <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-primary/5 border border-primary/20 text-xs font-medium text-primary">
                                    <Zap className="w-3 h-3" />
                                    <span>Autonomous Fixes</span>
                                </div>
                                <h3 className="text-3xl font-display font-bold leading-tight text-foreground">
                                    It doesn't just find bugs.<br /> It writes the patch.
                                </h3>
                                <p className="text-muted-foreground leading-relaxed">
                                    Sentinel analyzes vulnerable code, generates secure alternatives, and opens a PR for you to review.
                                </p>
                            </div>

                            <div className="w-full sm:w-[320px] bg-white rounded-lg border border-border shadow-lg p-4 rotate-1 group-hover:rotate-0 transition-transform duration-500">
                                <div className="flex items-center gap-2 mb-3 border-b border-border pb-2">
                                    <GitPullRequest className="w-4 h-4 text-purple-500" />
                                    <span className="text-xs text-gray-500 font-mono">fix/sql-injection-patch</span>
                                </div>
                                <div className="space-y-2 font-mono text-[10px]">
                                    <div className="text-red-400 line-through opacity-70">- const query = "SELECT * FROM users..."</div>
                                    <div className="text-primary font-medium bg-primary/5 w-full block rounded">+ const query = "SELECT * FROM users WHERE id = $1"</div>
                                    <div className="mt-3 flex gap-2">
                                        <div className="px-2 py-1 bg-primary/10 text-primary rounded text-[9px] font-bold">TESTS PASS</div>
                                        <div className="px-2 py-1 bg-blue-50 text-blue-600 rounded text-[9px]">SCANNING...</div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Vertical Item: Compliance */}
                    <BentoItem
                        className="md:row-span-2 bg-[#FAfaf9]"
                        title="Compliance Mapping"
                        description="Automatically map headers and auth flows to SOC2, HIPAA, and GDPR requirements."
                        icon={<Shield className="w-5 h-5" />}
                        stats="100% Coverage"
                        graphic={
                            <div className="relative h-full min-h-[160px] w-full bg-white rounded border border-border p-3 flex flex-col gap-2">
                                {['SOC2 Type II', 'HIPAA', 'ISO 27001', 'GDPR'].map((std, i) => (
                                    <div key={std} className="flex items-center justify-between text-xs p-2 rounded bg-secondary/30 border border-secondary">
                                        <span className="text-foreground font-medium">{std}</span>
                                        <CheckCircle2 className="w-3.5 h-3.5 text-primary" />
                                    </div>
                                ))}
                            </div>
                        }
                    />

                    {/* Standard Items */}
                    <BentoItem
                        title="SaaS & Cloud Security"
                        description="Monitor IAM policies, S3 buckets, and API gateways for misconfigurations."
                        icon={<Globe className="w-5 h-5" />}
                        graphic={
                            <div className="flex gap-2 justify-center py-2 opacity-50 grayscale group-hover:grayscale-0 group-hover:opacity-100 transition-all">
                                <Globe className="w-8 h-8 text-blue-500" />
                                <Lock className="w-8 h-8 text-orange-500" />
                            </div>
                        }
                    />

                    <BentoItem
                        title="Secret Detection"
                        description="Prevent API keys and tokens from leaking into production builds."
                        icon={<Lock className="w-5 h-5" />}
                        stats="0 Leaks"
                    />

                    <BentoItem
                        title="Custom Rule Engine"
                        description="Define organization-specific policies with Rego or Python."
                        icon={<FileCode className="w-5 h-5" />}
                    />

                    <BentoItem
                        title="CLI & CI/CD"
                        description="Native integrations for GitHub Actions, GitLab CI, and local terminals."
                        icon={<Terminal className="w-5 h-5" />}
                    />
                </div>
            </div>
        </section>
    );
};
