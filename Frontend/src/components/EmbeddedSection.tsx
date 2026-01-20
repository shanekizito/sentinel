import { MemoryStick, Brain, History } from "lucide-react";

const EmbeddedSection = () => {
    return (
        <section className="py-24 bg-[#F5F5F3] border-t border-border">
            <div className="container mx-auto px-6">
                <div className="grid lg:grid-cols-2 gap-20 items-center">
                    {/* Visual First on Desktop */}
                    <div className="order-2 lg:order-1 relative h-80 bg-white rounded-3xl border border-border shadow-sm flex items-center justify-center p-10">
                        <div className="absolute inset-0 bg-dot-pattern opacity-[0.1]" />
                        <div className="text-center relative z-10">
                            <div className="flex justify-center gap-4 mb-6">
                                <div className="bg-gray-100 p-3 rounded-lg"><History className="w-6 h-6 text-gray-500" /></div>
                                <div className="bg-gray-100 p-3 rounded-lg"><Brain className="w-6 h-6 text-gray-500" /></div>
                            </div>
                            <div className="bg-primary/5 px-6 py-2 rounded-full border border-primary/20 text-primary font-bold text-sm mb-2">
                                Context Retained
                            </div>
                            <p className="text-xs text-gray-500 max-w-[200px] mx-auto">
                                Sentinel remembers regression #1024 from 3 months ago.
                            </p>
                        </div>
                    </div>

                    <div className="order-1 lg:order-2">
                        <h2 className="font-display text-4xl font-bold mb-6 text-foreground">
                            A Security Agent That Remembers
                        </h2>
                        <p className="text-xl text-muted-foreground leading-relaxed font-light mb-8">
                            Sentinel can live directly inside your project. It acts as persistent memory for your security stack.
                        </p>

                        <div className="space-y-4">
                            {[
                                "Remembers past vulnerabilities",
                                "Tracks previous fixes",
                                "Flags AI re-introduction patterns"
                            ].map((item) => (
                                <div key={item} className="flex items-center gap-3">
                                    <div className="w-1.5 h-1.5 rounded-full bg-primary" />
                                    <span className="font-medium text-foreground">{item}</span>
                                </div>
                            ))}
                        </div>

                        <div className="mt-8 p-4 bg-white border-l-4 border-primary shadow-sm rounded-r-lg">
                            <p className="text-sm text-gray-600 italic">
                                "Most tools forget. Sentinel accumulates understanding."
                            </p>
                        </div>
                    </div>
                </div>
            </div>
        </section>
    );
};

export default EmbeddedSection;
