import { RefreshCw, Bot, CircuitBoard, ArrowRight, GitCommitHorizontal, CheckCircle2 } from "lucide-react";
import { Link } from "react-router-dom";

const CoreProposition = () => {
  return (
    <section className="py-32 bg-white relative overflow-hidden">
      {/* Background Trace Lines */}
      <div className="absolute inset-0 bg-grid-black/[0.02] -z-10" />

      <div className="container mx-auto px-6">
        <div className="max-w-4xl mx-auto text-center mb-20">
          <h2 className="font-display text-5xl font-bold mb-6 tracking-tight text-balance">
            Sentinel Is Not a Tool You Remember to Run
          </h2>
          <p className="text-2xl text-muted-foreground leading-relaxed font-light">
            You don’t “use” Sentinel. You <strong className="text-foreground font-medium">live with it.</strong>
          </p>
        </div>

        {/* Bento Grid Layout */}
        <div className="grid md:grid-cols-3 gap-6 max-w-6xl mx-auto">

          {/* Cell 1: Continuous (Large Span) */}
          <div className="md:col-span-2 bg-[#F9F9F9] rounded-3xl p-10 border border-border relative overflow-hidden group hover:border-blue-200 transition-colors">
            <div className="absolute top-0 right-0 p-10 opacity-10">
              <RefreshCw className="w-64 h-64 text-blue-500" />
            </div>

            <div className="relative z-10 h-full flex flex-col justify-between">
              <div>
                <div className="w-14 h-14 rounded-2xl bg-white shadow-sm border border-gray-100 flex items-center justify-center mb-6">
                  <RefreshCw className="w-7 h-7 text-blue-600" />
                </div>
                <h3 className="text-3xl font-bold mb-4">Continuous Infrastructure</h3>
                <p className="text-xl text-muted-foreground leading-relaxed max-w-md">
                  It runs on every git push, after AI generation, on schedules, and before deployment.
                </p>
              </div>

              {/* Mock Git Timeline */}
              <div className="mt-12 flex items-center gap-4 text-sm font-mono text-gray-500 bg-white/50 p-4 rounded-xl border border-gray-100 backdrop-blur-sm w-fit">
                <div className="flex items-center gap-2">
                  <GitCommitHorizontal className="w-4 h-4" />
                  <span>feat: add user auth</span>
                </div>
                <ArrowRight className="w-4 h-4 text-gray-300" />
                <div className="flex items-center gap-2 text-blue-600 font-bold">
                  <CheckCircle2 className="w-4 h-4" />
                  <span>Sentinel: Clean</span>
                </div>
              </div>
            </div>
          </div>

          {/* Cell 2: Autonomous (Tall) */}
          <div className="md:row-span-2 bg-gradient-to-b from-gray-900 to-black rounded-3xl p-10 border border-gray-800 text-white relative overflow-hidden group">
            <div className="absolute inset-0 bg-grid-white/[0.05]" />

            <div className="relative z-10 flex flex-col h-full">
              <div className="w-14 h-14 rounded-2xl bg-white/10 border border-white/10 flex items-center justify-center mb-6 backdrop-blur-md">
                <Bot className="w-7 h-7 text-white" />
              </div>
              <h3 className="text-3xl font-bold mb-4">Autonomous</h3>
              <p className="text-gray-400 leading-relaxed mb-12">
                Sentinel detects vulnerabilities, writes the fix, and generates regression tests. Humans approve. Sentinel does the work.
              </p>

              {/* Bot Visual */}
              <div className="mt-auto bg-gray-800/50 rounded-xl p-4 border border-gray-700 font-mono text-xs space-y-3">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
                  <span className="text-gray-300">Scanning diff...</span>
                </div>
                <div className="text-emerald-400">{">"} SQL Injection found</div>
                <div className="text-blue-400">{">"} Generating patch...</div>
                <div className="text-purple-400">{">"} Verifying fix...</div>
                <Link to="/automation" className="block text-center bg-white/10 hover:bg-white/20 py-2 rounded text-white font-bold transition-colors mt-4">
                  View Automation Flow
                </Link>
              </div>
            </div>
          </div>

          {/* Cell 3: Embedded (Standard) */}
          <div className="md:col-span-2 bg-emerald-50/50 rounded-3xl p-10 border border-emerald-100 relative overflow-hidden group hover:bg-emerald-50 transition-colors">
            <div className="absolute -right-20 -bottom-20 opacity-10">
              <CircuitBoard className="w-80 h-80 text-emerald-600" />
            </div>

            <div className="relative z-10 flex flex-col md:flex-row gap-10 items-center">
              <div className="flex-1">
                <div className="w-14 h-14 rounded-2xl bg-white shadow-sm border border-emerald-100 flex items-center justify-center mb-6">
                  <CircuitBoard className="w-7 h-7 text-emerald-600" />
                </div>
                <h3 className="text-3xl font-bold mb-4">Embedded Reality</h3>
                <p className="text-xl text-muted-foreground leading-relaxed">
                  Security doesn't live outside the system it protects. Sentinel runs in CI, locally, or even offline within your VPC.
                </p>
                <Link to="/security" className="inline-flex items-center gap-2 text-emerald-700 font-bold mt-6 hover:gap-3 transition-all">
                  View Architecture <ArrowRight className="w-4 h-4" />
                </Link>
              </div>

              {/* Graphic */}
              <div className="w-full md:w-1/3 aspect-square bg-white rounded-2xl shadow-sm border border-emerald-100 flex items-center justify-center relative">
                <div className="absolute inset-2 border-2 border-dashed border-emerald-100 rounded-xl" />
                <div className="text-center">
                  <div className="text-4xl font-black text-emerald-900/10">VPC</div>
                  <div className="text-xs font-bold text-emerald-600 uppercase tracking-widest mt-2">Local Mode</div>
                </div>
              </div>
            </div>
          </div>

        </div >
      </div >
    </section >
  );
};

export default CoreProposition;