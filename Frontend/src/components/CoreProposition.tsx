import { RefreshCw, Bot, Shield, ArrowRight, Terminal, Cpu, CheckCircle2 } from "lucide-react";
import { Link } from "react-router-dom";

const CoreProposition = () => {
  return (
    <section className="py-24 bg-white relative border-t-2 border-gray-900">
      {/* Structural Grid */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute left-[15%] top-0 bottom-0 w-px bg-gray-100" />
        <div className="absolute right-[15%] top-0 bottom-0 w-px bg-gray-100" />
      </div>

      <div className="container mx-auto px-6">

        {/* Section Header - User Benefit Focused */}
        <div className="max-w-4xl mb-16">
          <div className="border-l-4 border-primary pl-8 py-2">
            <h2 className="font-display text-4xl md:text-5xl font-bold tracking-tight text-gray-900 mb-4">
              Security that fits
              <br />
              your workflow.
            </h2>
            <p className="text-xl text-gray-600 leading-relaxed font-light max-w-2xl">
              Sentinel connects directly to your repo and works in the background. No dashboards to manage, no false alarms to triage.
            </p>
          </div>
        </div>

        {/* Grid System - Workflow Benefits */}
        <div className="grid md:grid-cols-12 gap-px bg-gray-900 border-2 border-gray-900">

          {/* Card 1: Context Awareness */}
          <div className="md:col-span-8 bg-white p-10 relative">
            <div className="absolute top-4 left-4 w-8 h-8 border-t-2 border-l-2 border-primary" />

            <div className="space-y-6">
              <div className="w-12 h-12 bg-gray-900 flex items-center justify-center">
                <Cpu className="w-6 h-6 text-white" />
              </div>
              <h3 className="font-display text-2xl font-bold text-gray-900">Deep Code Understanding</h3>
              <p className="text-lg text-gray-600 leading-relaxed max-w-md">
                Most tools treat code like text. Sentinel understands how your application actually runs—tracking data flow across functions to find complex bugs others miss.
              </p>

              {/* Benefit Stack */}
              <div className="mt-8 space-y-3 font-mono text-sm">
                <div className="flex items-center justify-between border-b border-gray-100 pb-2">
                  <span className="text-gray-600">Context</span>
                  <span className="text-gray-900 font-bold">Full Project Awareness</span>
                </div>
                <div className="flex items-center justify-between border-b border-gray-100 pb-2">
                  <span className="text-gray-600">Accuracy</span>
                  <span className="text-gray-900 font-bold">No Noise / False Positives</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-600">Speed</span>
                  <span className="text-gray-900 font-bold">Instant Feedback</span>
                </div>
              </div>
            </div>
          </div>

          {/* Card 2: AI + Verification */}
          <div className="md:col-span-4 md:row-span-2 bg-white p-10 flex flex-col relative">
            {/* Corner Accent */}
            <div className="absolute top-4 right-4 w-8 h-8 border-t-2 border-r-2 border-primary" />

            <div className="space-y-6 flex-1">
              <div className="w-12 h-12 bg-gray-900 flex items-center justify-center">
                <Bot className="w-6 h-6 text-white" />
              </div>
              <h3 className="font-display text-2xl font-bold text-gray-900">Smart & Safe</h3>
              <p className="text-lg text-gray-600 leading-relaxed">
                Generative AI writes the fix. Mathematical logic proves it's safe. You get the best of both worlds.
              </p>
            </div>

            {/* AI Visual */}
            <div className="mt-auto pt-8 border-t border-gray-100">
              <div className="aspect-square bg-gray-50 border-2 border-gray-100 flex items-center justify-center relative overflow-hidden">
                {/* Technical Grid Background */}
                <div className="absolute inset-0 bg-[linear-gradient(to_right,rgba(0,0,0,0.05)_1px,transparent_1px),linear-gradient(to_bottom,rgba(0,0,0,0.05)_1px,transparent_1px)] bg-[size:24px_24px]" />
                <img src="/ai-brain.png" alt="AI Logic" className="w-3/4 opacity-90 relative z-10" />
              </div>
              <div className="mt-4 flex items-center gap-2 text-xs font-mono text-primary font-bold uppercase tracking-wider">
                <CheckCircle2 className="w-4 h-4" />
                <span>Logic Verified</span>
              </div>
            </div>
          </div>

          {/* Card 3: Trusted Security */}
          <div className="md:col-span-8 bg-white p-10 relative">
            <div className="absolute bottom-4 right-4 w-8 h-8 border-b-2 border-r-2 border-primary" />

            <div className="flex flex-col md:flex-row gap-10">
              <div className="flex-1 space-y-6">
                <div className="w-12 h-12 bg-gray-900 flex items-center justify-center">
                  <Shield className="w-6 h-6 text-white" />
                </div>
                <h3 className="font-display text-2xl font-bold text-gray-900">Guaranteed Correctness</h3>
                <p className="text-lg text-gray-600 leading-relaxed">
                  We don't just "guess" at vulnerabilities. We prove them mathematically before alerting you. If Sentinel reports it, it's real.
                </p>
                <Link to="/security" className="inline-flex items-center gap-2 text-sm font-bold text-gray-900 uppercase tracking-wider border-b-2 border-gray-900 pb-1 hover:border-primary hover:text-primary transition-colors">
                  How it works <ArrowRight className="w-4 h-4" />
                </Link>
              </div>

              {/* Verification Engines */}
              <div className="flex flex-col gap-px bg-gray-900 border-2 border-gray-900 min-w-[200px]">
                {[
                  { name: 'Syntax Check', status: 'Passed' },
                  { name: 'Type Safety', status: 'Passed' },
                  { name: 'Logic Proof', status: 'Verified' }
                ].map((item) => (
                  <div key={item.name} className="bg-white px-4 py-3">
                    <div className="flex items-center justify-between mb-1">
                      <span className="font-mono text-sm font-bold text-gray-900">{item.name}</span>
                      <div className="w-2 h-2 bg-primary" />
                    </div>
                    <div className="text-xs text-gray-500">{item.status}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>

        </div>
      </div>
    </section>
  );
};

export default CoreProposition;