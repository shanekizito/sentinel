import { ArrowRight, Play, ShieldCheck, FileCode, CheckCircle2 } from "lucide-react";

const Hero = () => {
  return (
    <div className="relative pt-32 pb-20 md:pt-48 md:pb-32 overflow-hidden bg-background">
      {/* Background Elements */}
      <div className="absolute inset-0 -z-10">
        <div className="absolute inset-0 bg-grid-black/[0.02]" />
        <div className="absolute top-0 left-1/4 w-96 h-96 bg-primary/5 rounded-full blur-3xl opacity-50" />
        <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-blue-500/5 rounded-full blur-3xl opacity-50" />
      </div>

      <div className="container mx-auto px-6">
        <div className="flex flex-col lg:flex-row items-center gap-16 lg:gap-24">

          {/* Text Content */}
          <div className="flex-1 max-w-2xl">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-secondary border border-border text-xs font-semibold text-foreground mb-8 cursor-default hover:bg-secondary/80 transition-colors">
              AI-Native Security Agent
              <ArrowRight className="w-3 h-3" />
            </div>

            <h1 className="font-display text-5xl md:text-7xl font-bold mb-8 tracking-tighter text-balance text-foreground leading-[1.1]">
              Security for Code Written at <span className="text-transparent bg-clip-text bg-gradient-to-r from-emerald-600 to-teal-500">AI Speed</span>
            </h1>

            <p className="text-xl md:text-2xl text-muted-foreground mb-10 leading-relaxed max-w-xl font-light">
              Sentinel is an autonomous security agent that continuously audits, fixes, and tests AI-generated codebases — automatically, asynchronously, and without slowing development.
            </p>

            <div className="flex flex-col sm:flex-row gap-4">
              <button className="btn-primary h-14 px-8 text-lg w-full sm:w-auto shadow-xl shadow-emerald-900/10 whitespace-nowrap">
                Start Free Scan
              </button>
              <button className="btn-secondary h-14 px-8 text-lg w-full sm:w-auto flex items-center justify-center gap-2 group border border-border bg-white/50 backdrop-blur-sm whitespace-nowrap">
                <Play className="w-4 h-4 fill-current group-hover:scale-110 transition-transform" />
                View Demo
              </button>
            </div>

            <div className="mt-12 flex items-center gap-6 text-sm text-muted-foreground font-medium">
              <div className="flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-emerald-500" />
                <span>SOC2 Compliant</span>
              </div>
              <div className="flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-emerald-500" />
                <span>Zero-Retention</span>
              </div>
            </div>
          </div>

          {/* New "Orbital" Visual */}
          <div className="flex-1 w-full max-w-lg lg:max-w-xl relative">
            {/* Central Core */}
            <div className="relative z-10 w-64 h-64 mx-auto bg-white rounded-full border-2 border-emerald-100 shadow-2xl flex items-center justify-center relative overflow-hidden backdrop-blur-xl">
              <div className="absolute inset-0 bg-emerald-50/30 animate-pulse-slow" />
              <div className="text-center relative z-10">
                <ShieldCheck className="w-16 h-16 text-emerald-600 mx-auto mb-2" />
                <span className="text-xs font-bold text-emerald-800 uppercase tracking-widest block">Sentinel Core</span>
                <span className="text-[10px] text-emerald-600 mt-1 block">Active Protection</span>
              </div>
            </div>

            {/* Orbital Rings */}
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[140%] h-[140%] border border-dashed border-gray-200 rounded-full -z-10 opacity-60 animate-spin-slow" />
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[180%] h-[180%] border border-gray-100 rounded-full -z-20 opacity-40" />

            {/* Orbiting Elements (Simulated) */}
            <div className="absolute top-0 right-10 bg-white p-3 rounded-lg shadow-lg border border-gray-100 flex items-center gap-3 animate-float-delayed">
              <div className="w-8 h-8 bg-blue-50 rounded flex items-center justify-center">
                <FileCode className="w-4 h-4 text-blue-600" />
              </div>
              <div>
                <div className="text-xs font-bold text-gray-800">PR #402</div>
                <div className="text-[10px] text-gray-500">Scanning...</div>
              </div>
            </div>

            <div className="absolute bottom-10 left-0 bg-white p-3 rounded-lg shadow-lg border border-gray-100 flex items-center gap-3 animate-float">
              <div className="w-8 h-8 bg-purple-50 rounded flex items-center justify-center">
                <ShieldCheck className="w-4 h-4 text-purple-600" />
              </div>
              <div>
                <div className="text-xs font-bold text-gray-800">Auto-Patched</div>
                <div className="text-[10px] text-gray-500">SQL Injection prevented</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Hero;