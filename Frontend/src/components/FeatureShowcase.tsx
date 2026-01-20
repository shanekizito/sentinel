import { Zap, GitPullRequest, Globe, Lock, Terminal, FileCode, Server, Layout, Cpu, Shield, Clock } from "lucide-react";

export const FeatureShowcase = () => {
    return (
        <section className="py-24 bg-gray-50 border-y-2 border-gray-900 relative overflow-hidden">
            {/* Grid Lines */}
            <div className="absolute inset-0 pointer-events-none">
                <div className="absolute left-0 right-0 top-1/2 h-px bg-gray-200" />
            </div>

            <div className="container mx-auto px-6 relative z-10">

                {/* Top Section - Workflow Focused */}
                <div className="grid lg:grid-cols-2 gap-20 items-center mb-24">
                    {/* Text */}
                    <div className="space-y-10">
                        <div className="space-y-6">
                            <div className="inline-block border-l-4 border-primary pl-4">
                                <div className="text-xs font-mono uppercase tracking-[0.2em] text-gray-500 mb-2">
                                    Complete Automation
                                </div>
                            </div>
                            <h2 className="font-display text-5xl lg:text-6xl font-bold tracking-tight text-gray-900 leading-[1.05]">
                                We find it.
                                <br />
                                We fix it.
                                <br />
                                You merge.
                            </h2>
                        </div>

                        <p className="text-xl text-gray-600 leading-relaxed max-w-lg font-light border-l-2 border-gray-200 pl-6">
                            Stop spending hours chasing bugs. Sentinel finds vulnerabilities, writes the production-ready patch, and proves it breaks nothing.
                        </p>

                        {/* Process Steps - Simple Action Verbs */}
                        <div className="space-y-px bg-gray-900 border-2 border-gray-900 max-w-md">
                            {[
                                { step: '01', action: 'Scan', desc: 'Checks every commit instantly' },
                                { step: '02', action: 'Fix', desc: 'Auto-generates secure code' },
                                { step: '03', action: 'Verify', desc: 'Ensures logic is preserved' },
                                { step: '04', action: 'Deploy', desc: 'Ready for production' }
                            ].map((item) => (
                                <div key={item.step} className="bg-white px-6 py-3 flex items-center justify-between">
                                    <div className="flex items-center gap-4">
                                        <span className="font-mono text-xs text-gray-400">{item.step}</span>
                                        <span className="font-bold text-sm text-gray-900">{item.action}</span>
                                    </div>
                                    <span className="text-xs text-gray-500">{item.desc}</span>
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* Image Frame */}
                    <div className="relative">
                        <div className="border-2 border-gray-900 p-4 bg-white">
                            <div className="absolute -top-2 -left-2 w-16 h-16 border-t-4 border-l-4 border-primary" />
                            <div className="absolute -bottom-2 -right-2 w-16 h-16 border-b-4 border-r-4 border-primary" />

                            <div className="border border-gray-200 bg-gray-50">
                                <img
                                    src="/code-analysis.png"
                                    alt="Automated Workflow"
                                    className="w-full h-auto"
                                />
                            </div>
                        </div>

                        {/* Floating Annotation */}
                        <div className="absolute -bottom-6 -right-6 bg-white border-2 border-gray-900 px-6 py-4">
                            <div className="flex items-center gap-3">
                                <GitPullRequest className="w-5 h-5 text-gray-900" />
                                <div>
                                    <div className="text-sm font-bold text-gray-900">PR Ready</div>
                                    <div className="text-xs text-gray-500 font-mono">1 Click Merge</div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Feature Grid - Benefits not Specs */}
                <div className="max-w-6xl mx-auto">
                    <div className="mb-12 border-l-4 border-gray-900 pl-8">
                        <h3 className="font-display text-3xl font-bold text-gray-900 mb-2">Built for Scale</h3>
                        <p className="text-gray-600">Enterprise-grade security that just works.</p>
                    </div>

                    <div className="grid md:grid-cols-3 gap-px bg-gray-900 border-2 border-gray-900">
                        {[
                            {
                                icon: <Clock className="w-5 h-5" />,
                                title: "Blazing Fast",
                                desc: "Scans thousands of files in milliseconds so you're never waiting.",
                                tag: "Velocity"
                            },
                            {
                                icon: <Shield className="w-5 h-5" />,
                                title: "Future-Proof Encryption",
                                desc: "Your data is protected by the highest standard of modern cryptography.",
                                tag: "Security"
                            },
                            {
                                icon: <Server className="w-5 h-5" />,
                                title: "Global Scale",
                                desc: "Works seamlessly whether you have 10 developers or 10,000.",
                                tag: "Scalability"
                            },
                            {
                                icon: <Globe className="w-5 h-5" />,
                                title: "Zero Data Retention",
                                desc: "We analyze your code and forget it. No IP ever stored on our servers.",
                                tag: "Privacy"
                            },
                            {
                                icon: <Lock className="w-5 h-5" />,
                                title: "Private by Design",
                                desc: "Analysis happens in isolated secure environments for total privacy.",
                                tag: "Isolation"
                            },
                            {
                                icon: <Terminal className="w-5 h-5" />,
                                title: "Safe Testing",
                                desc: "Every fix is tested in a sandboxed environment before it reaches you.",
                                tag: "Quality"
                            },
                        ].map((feature, i) => (
                            <div key={i} className="bg-white p-8 group hover:bg-gray-50 transition-colors">
                                <div className="w-10 h-10 border-2 border-gray-900 flex items-center justify-center mb-4 group-hover:border-primary group-hover:bg-primary transition-colors">
                                    <div className="text-gray-900 group-hover:text-white transition-colors">
                                        {feature.icon}
                                    </div>
                                </div>
                                <div className="mb-2 text-[10px] uppercase tracking-wider text-primary font-bold">
                                    {feature.tag}
                                </div>
                                <h4 className="font-bold text-lg mb-2 text-gray-900">{feature.title}</h4>
                                <p className="text-sm text-gray-600 leading-relaxed">{feature.desc}</p>
                            </div>
                        ))}
                    </div>
                </div>

            </div>
        </section>
    );
};
