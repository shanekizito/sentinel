import { ArrowRight, Terminal, CheckCircle2 } from "lucide-react";

const CTA = () => {
  return (
    <section className="py-32 bg-white relative border-y-2 border-gray-900">
      {/* Structural Background Grid */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute left-1/2 top-0 bottom-0 w-px bg-gray-100 -translate-x-1/2" />
        <div className="absolute top-1/2 left-0 right-0 h-px bg-gray-100 -translate-y-1/2" />
      </div>

      {/* Corner Accents - The "Architectural" Signature */}
      <div className="absolute top-0 left-0 w-4 h-4 border-t-2 border-l-2 border-gray-900" />
      <div className="absolute top-0 right-0 w-4 h-4 border-t-2 border-r-2 border-gray-900" />
      <div className="absolute bottom-0 left-0 w-4 h-4 border-b-2 border-l-2 border-gray-900" />
      <div className="absolute bottom-0 right-0 w-4 h-4 border-b-2 border-r-2 border-gray-900" />

      <div className="container mx-auto px-6 relative z-10">
        <div className="max-w-4xl mx-auto text-center">

          {/* Status Badge - Text Only, No Glow */}
          <div className="inline-flex items-center gap-2 mb-8 uppercase tracking-[0.2em] text-xs font-mono font-bold text-primary">
            <span className="w-2 h-2 bg-primary" />
            Production Ready
          </div>

          {/* Heading - Clean, Big, High Contrast */}
          <h2 className="font-display text-5xl md:text-7xl font-bold text-gray-900 tracking-tight leading-[1.05] mb-8">
            Secure your codebase
            <br />
            <span className="text-primary">before you ship.</span>
          </h2>

          <p className="text-xl text-gray-600 max-w-2xl mx-auto mb-12">
            Enterprise-grade static analysis for modern engineering teams.
            <br className="hidden md:block" />
            Install in seconds. Zero false positives.
          </p>

          {/* Buttons - Solid, Hard Edges, No Shadows */}
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
            <button className="h-14 px-8 bg-primary text-white font-bold text-sm uppercase tracking-wider hover:bg-primary/90 transition-colors flex items-center gap-2 min-w-[200px] justify-center">
              Start Free Scan
              <ArrowRight className="w-4 h-4" />
            </button>

            <button className="h-14 px-8 border-2 border-gray-900 text-gray-900 font-bold text-sm uppercase tracking-wider hover:bg-gray-900 hover:text-white transition-colors flex items-center gap-2 min-w-[200px] justify-center">
              <Terminal className="w-4 h-4" />
              Documentation
            </button>
          </div>

          {/* Footer - Simple Text */}
          <div className="mt-16 pt-8 border-t border-gray-100 flex flex-wrap items-center justify-center gap-x-8 gap-y-4 text-xs font-mono text-gray-400 uppercase tracking-wider">
            <div className="flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-primary" />
              <span>Free 14-day Pro Trial</span>
            </div>
            <div className="flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-primary" />
              <span>Unlimited Repos</span>
            </div>
            <div className="flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-primary" />
              <span>Cancel Anytime</span>
            </div>
          </div>

        </div>
      </div>
    </section>
  );
};

export default CTA;