import { ArrowRight, Terminal, Shield } from "lucide-react";
import { Link } from "react-router-dom";

const CTA = () => {
  return (
    <section className="py-32 bg-[#050505] text-white relative overflow-hidden">
      {/* Background Glow & Grid */}
      <div className="absolute inset-0 bg-grid-white/[0.03] pointer-events-none" />
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] bg-emerald-500/10 rounded-full blur-[120px] pointer-events-none" />

      <div className="container mx-auto px-6 relative z-10 text-center">

        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 font-mono text-xs font-bold mb-8">
          <Shield className="w-3 h-3" />
          SECURE BY DEFAULT
        </div>

        <h2 className="font-display text-5xl md:text-7xl font-bold mb-8 tracking-tighter text-balance">
          Make Security Automatic
        </h2>

        <p className="text-xl md:text-2xl text-gray-400 mb-12 max-w-2xl mx-auto font-light leading-relaxed text-balance">
          Run Sentinel once — then let it protect your codebase continuously.
        </p>

        <div className="flex flex-col sm:flex-row gap-6 justify-center items-center">
          <button className="bg-emerald-500 hover:bg-emerald-400 text-black h-14 px-8 text-lg font-bold rounded-lg transition-all transform hover:scale-105 shadow-[0_0_20px_rgba(16,185,129,0.3)] whitespace-nowrap">
            Start Free Scan
          </button>
          <button className="h-14 px-8 text-lg font-bold border border-white/20 rounded-lg hover:bg-white/10 transition-colors flex items-center justify-center gap-2 whitespace-nowrap">
            <Terminal className="w-5 h-5" />
            Join Early Access
          </button>
        </div>

        <div className="mt-20 pt-10 border-t border-white/10 flex flex-col md:flex-row justify-between items-center gap-6 text-sm text-gray-500 font-medium">
          <div className="flex gap-8">
            <span className="flex items-center gap-2 text-gray-400">
              <div className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
              No credit card
            </span>
            <span className="flex items-center gap-2 text-gray-400">
              <div className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
              No lock-in
            </span>
            <span className="flex items-center gap-2 text-gray-400">
              <div className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
              Full control
            </span>
          </div>
          <div className="font-mono text-emerald-500/40 tracking-wider">
            sentinel --daemon --watch
          </div>
        </div>
      </div>
    </section>
  );
};

export default CTA;